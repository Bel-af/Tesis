#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ecg_processor.py - Procesador de ECG para API Flask

Orquesta el pipeline completo de prediccion DNS:
1. Lectura y preprocesamiento
2. Deteccion de picos R
3. Analisis RR
4. Extraccion de caracteristicas
5. Prediccion DNS
6. Generacion de reporte

Este modulo NO modifica la base de datos, solo procesa ECGs.
"""

import logging
import os
import sys
import numpy as np
import glob
import re

# Configurar logging solo si no hay handlers configurados (evitar conflictos con Flask)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Agregar ecg_pipeline al path
ecg_pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'ecg_pipeline')
if ecg_pipeline_path not in sys.path:
    sys.path.insert(0, ecg_pipeline_path)

try:
    from preprocessing import process_ecg_file, select_best_derivation
    from peak_detection import detect_r_peaks
    from rr_analysis import calculate_rr_intervals, analyze_rr_intervals
    from features import extract_features
    
    # Intentar usar modelo ML si existe, fallback a reglas
    try:
        from model_ml import predict_dns_ml as predict_dns_probability
        logger.info("✓ Usando modelo ML para predicciones DNS")
    except (ImportError, FileNotFoundError):
        from model import predict_dns_probability
        logger.info("✓ Usando modelo basado en reglas (ML no disponible)")
        
except ImportError as e:
    logger.error("Error importando modulos del pipeline DNS: {}".format(str(e)))
    logger.error("Verificar que ecg_pipeline/ existe en: {}".format(ecg_pipeline_path))
    raise


class ECGProcessor:
    """Procesador de archivos ECG para deteccion de DNS"""
    
    def __init__(self):
        self.last_result = None

    def _parse_csv_numeric_matrix_loose(self, csv_filepath):
        """Parsea CSV sintético tolerando filas no numéricas y sin metadatos EasyG."""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        content = None

        for enc in encodings:
            try:
                with open(csv_filepath, 'r', encoding=enc) as f:
                    content = f.readlines()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            return None

        rows = []
        expected_cols = None
        for line in content:
            line = line.replace('\xa0', ' ').strip()
            if not line or line.startswith('%'):
                continue

            parts = [line]
            for d in [',', ';', '\t']:
                if d in line:
                    parts = [p.strip() for p in line.split(d)]
                    break

            values = []
            valid_row = True
            for p in parts:
                if not p:
                    continue
                try:
                    values.append(float(p.replace(',', '.')))
                except ValueError:
                    valid_row = False
                    break

            if not valid_row or not values:
                continue

            if expected_cols is None:
                expected_cols = len(values)
            if len(values) != expected_cols:
                continue

            rows.append(values)

        if not rows:
            return None

        return np.array(rows, dtype=np.float32)

    def _looks_synthetic(self, csv_filepath):
        base_name = os.path.basename(csv_filepath).lower()
        return bool(re.search(r'(augmented|synthetic|simul|noise|noise)', base_name))

    def _find_companion_metadata_file(self, csv_filepath):
        directory = os.path.dirname(csv_filepath)
        base_name = os.path.splitext(os.path.basename(csv_filepath))[0]

        stripped_name = re.sub(r'([_-]?(augmented|synthetic|simul|simulation|noise|noisy|scaled))+', '', base_name, flags=re.IGNORECASE)
        stripped_name = stripped_name.strip('_- ') or base_name

        candidate_patterns = [
            os.path.join(directory, f"{stripped_name}.csv"),
            os.path.join(directory, f"{stripped_name}_*.csv"),
            os.path.join(directory, f"*{stripped_name}*.csv"),
        ]

        seen = set()
        candidates = []
        for pattern in candidate_patterns:
            for path in glob.glob(pattern):
                if path in seen or os.path.abspath(path) == os.path.abspath(csv_filepath):
                    continue
                seen.add(path)
                candidates.append(path)

        for path in candidates:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for _ in range(30):
                        line = f.readline()
                        if not line:
                            break
                        if line.lstrip().startswith('%'):
                            return path
                        if line.strip() and not line.lstrip().startswith('%'):
                            # Si ya entró a datos sin metadatos, seguimos con el siguiente candidato.
                            break
            except Exception:
                continue

        return None

    def _extract_metadata_from_file(self, csv_filepath):
        metadata = {}
        try:
            with open(csv_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    line = line.replace('\xa0', ' ').strip()
                    if not line.startswith('%'):
                        continue

                    match = re.search(r'Cantidad de muestras\s*=\s*(\d+)', line)
                    if match:
                        metadata['num_samples'] = int(match.group(1))

                    match = re.search(r'LSB\s*\[uV\]\s*=\s*([\d.,]+)', line)
                    if match:
                        metadata['lsb_uv'] = float(match.group(1).replace(',', '.'))

                    match = re.search(r'Frecuencia de muestreo efectiva\s*\[Hz\]\s*=\s*(\d+)', line)
                    if match:
                        metadata['fs_hz'] = int(match.group(1))

                    match = re.search(r'Orden\s*=\s*(.+)$', line)
                    if match:
                        deriv_str = match.group(1).strip()
                        metadata['derivations'] = [d.strip() for d in deriv_str.split(',') if d.strip()]

            if metadata:
                return metadata
        except Exception:
            return {}

        return {}

    def _metadata_for_synthetic_file(self, csv_filepath, fs_hz):
        companion = self._find_companion_metadata_file(csv_filepath)
        if companion:
            metadata = self._extract_metadata_from_file(companion)
            if metadata:
                metadata.setdefault('fs_hz', fs_hz)
                metadata.setdefault('lsb_uv', 1.0)
                metadata.setdefault('derivations', [])
                metadata.setdefault('num_samples', 0)
                metadata['source_file'] = companion
                return metadata

        return {
            'num_samples': 0,
            'lsb_uv': 1.0,
            'fs_hz': int(fs_hz),
            'derivations': [],
            'data_type': 'SYNTHETIC_FALLBACK',
        }

    def _process_synthetic_csv(self, csv_filepath, fs_hz=300):
        """Fallback sintético para no bloquear backend si faltan metadatos EasyG."""
        companion = self._find_companion_metadata_file(csv_filepath)
        if companion:
            try:
                logger.info("Usando archivo compañero para recuperar la señal original: {}".format(companion))
                result = process_ecg_file(companion)
                metadata = dict(result.get('metadata', {}) or {})
                metadata['synthetic_source_file'] = csv_filepath
                metadata['source_file'] = companion
                result['metadata'] = metadata
                result['synthetic_fallback'] = True
                return result
            except Exception as e:
                logger.warning("No se pudo procesar el archivo compañero '{}': {}".format(companion, e))

        raw = self._parse_csv_numeric_matrix_loose(csv_filepath)
        if raw is None or raw.shape[0] < 16:
            return None

        metadata = self._metadata_for_synthetic_file(csv_filepath, fs_hz)
        derivations = metadata.get('derivations') or [f"D{i+1}" for i in range(raw.shape[1])]
        fs_value = int(metadata.get('fs_hz', fs_hz) or fs_hz)
        signal_norm, selected_deriv = select_best_derivation(raw, derivations, fs_hz)
        duration_s = float(raw.shape[0]) / float(fs_value)

        return {
            'signal_normalized': signal_norm,
            'signal_raw_all': raw,
            'metadata': {
                'num_samples': int(metadata.get('num_samples', raw.shape[0]) or raw.shape[0]),
                'lsb_uv': float(metadata.get('lsb_uv', 1.0) or 1.0),
                'fs_hz': fs_value,
                'derivations': derivations,
                'data_type': metadata.get('data_type', 'SYNTHETIC_FALLBACK'),
                'duration_s': duration_s,
                'source_file': metadata.get('source_file'),
            },
            'duration_s': duration_s,
            'fs_hz': fs_value,
            'selected_derivation': selected_deriv,
        }

    def _build_synthetic_inconclusive_result(self, metadata, duration_s, fs_hz, signal_quality=35.0):
        """Resultado seguro para sintéticos cuando la señal no permite análisis completo."""
        return {
            'dns_probability_pct': 35.0,
            'confidence_score_pct': 35.0,
            'hr_mean_bpm': 0.0,
            'hr_zone': 'N/A',
            'rr_max_s': 0.0,
            'sdnn_s': 0.0,
            'rmssd_s': 0.0,
            'pauses_gt3s_per_min': 0.0,
            'pauses_gt4s_per_min': 0.0,
            'count_rr_gt3s': 0,
            'count_rr_gt4s': 0,
            'duration_s': float(duration_s),
            'signal_quality_score': float(signal_quality),
            'num_samples': int(metadata.get('num_samples', 0)),
            'fs_hz': int(fs_hz),
            'derivations': metadata.get('derivations', []),
            'metadata': metadata,
            'decision_threshold_pct': 50.0,
            'is_dns_positive': False,
            'synthetic_fallback': True,
        }
    
    def process_ecg_file(self, csv_filepath):
        """
        Procesa un archivo ECG y retorna resultado del analisis DNS.
        
        Args:
            csv_filepath: Ruta al archivo CSV EasyG
            
        Returns:
            dict con estructura:
            {
                'dns_probability_pct': float (0-100),
                'confidence_score_pct': float (0-100),
                'hr_mean_bpm': float,
                'rr_max_s': float,
                'sdnn_s': float,
                'rmssd_s': float,
                'pauses_gt3s_per_min': float,
                'pauses_gt4s_per_min': float,
                'duration_s': float,
                'signal_quality_score': float,
                'metadata': dict,
                'error': str (solo si fallo)
            }
        """
        
        try:
            logger.info("Procesando ECG: {}".format(csv_filepath))

            # Paso 1: Preprocesamiento
            try:
                result = process_ecg_file(csv_filepath)
            except Exception as parse_error:
                fallback_result = self._process_synthetic_csv(csv_filepath, fs_hz=300)
                if fallback_result is None:
                    raise parse_error
                logger.info("Aplicado fallback sintético para archivo sin metadatos EasyG")
                result = fallback_result

            metadata = result['metadata']
            signal_norm = result['signal_normalized']
            duration_s = result['duration_s']
            fs_hz = result['fs_hz']
            
            logger.info("ECG cargado: {:.2f}s, {} Hz".format(duration_s, fs_hz))
            
            # Paso 2: Deteccion de picos R
            peaks_seconds = detect_r_peaks(signal_norm, fs_hz)
            logger.info("{} picos R detectados".format(len(peaks_seconds)))
            
            if len(peaks_seconds) < 3:
                logger.warning("Muy pocos picos detectados")
                if metadata.get('data_type') == 'SYNTHETIC_FALLBACK':
                    return self._build_synthetic_inconclusive_result(metadata, duration_s, fs_hz, signal_quality=30.0)
                return {
                    'error': 'Señal de baja calidad: muy pocos picos R detectados',
                    'dns_probability_pct': 0,
                    'confidence_score_pct': 0
                }
            
            # Paso 3: Analisis RR
            rr_intervals_s, valid_rr_fraction = calculate_rr_intervals(peaks_seconds)
            rr_analysis = analyze_rr_intervals(rr_intervals_s, duration_s)
            
            logger.info("RR media: {:.3f}s, validez: {:.1f}%".format(
                rr_analysis.get('rr_mean_s', 0),
                valid_rr_fraction * 100
            ))
            
            # Paso 4: Extraccion de caracteristicas
            features = extract_features(rr_analysis, valid_rr_fraction, duration_s)
            
            # Paso 5: Prediccion DNS
            prediction = predict_dns_probability(features)
            if isinstance(prediction, dict):
                dns_probability_pct = float(prediction.get('dns_probability_pct', 0))
                confidence_score_pct = float(
                    prediction.get('confidence_score_pct', prediction.get('confidence_score', 0))
                )
                decision_threshold_pct = float(prediction.get('decision_threshold_pct', 50.0))
                is_dns_positive = bool(prediction.get('is_dns_positive', dns_probability_pct >= decision_threshold_pct))
                model_type = prediction.get('model_type', 'ml')
                model_explanation = prediction.get('explanation')
                literature_override = bool(prediction.get('literature_override', False))
            else:
                dns_probability_pct, confidence_score_pct = prediction
                decision_threshold_pct = 50.0
                is_dns_positive = bool(dns_probability_pct >= decision_threshold_pct)
                model_type = 'rules'
                model_explanation = None
                literature_override = False
            
            logger.info("Prediccion DNS: {:.1f}% (confianza: {:.1f}%)".format(
                dns_probability_pct, confidence_score_pct
            ))
            
            # Construir resultado
            resultado = {
                'dns_probability_pct': dns_probability_pct,
                'confidence_score_pct': confidence_score_pct,
                'hr_mean_bpm': features.get('hr_mean_bpm', 0),
                'hr_zone': features.get('hr_zone', 'N/A'),
                'rr_max_s': features.get('rr_max_s', 0),
                'sdnn_s': features.get('sdnn_s', 0),
                'rmssd_s': features.get('rmssd_s', 0),
                'pauses_gt3s_per_min': rr_analysis.get('pauses_per_min_gt3s', 0),
                'pauses_gt4s_per_min': rr_analysis.get('pauses_per_min_gt4s', 0),
                'duration_s': duration_s,
                'signal_quality_score': features.get('signal_quality_score', 0),
                'num_samples': metadata.get('num_samples', 0),
                'fs_hz': fs_hz,
                'derivations': metadata.get('derivations', []),
                'metadata': metadata
            }
            resultado['decision_threshold_pct'] = decision_threshold_pct
            resultado['is_dns_positive'] = is_dns_positive
            resultado['model_type'] = model_type
            resultado['literature_override'] = literature_override
            if model_explanation:
                resultado['model_explanation'] = model_explanation
            
            self.last_result = resultado
            return resultado
            
        except Exception as e:
            logger.error("Error procesando ECG: {}".format(str(e)), exc_info=True)

            # Último resguardo: si parece CSV sintético legible, no devolver error técnico.
            raw = self._parse_csv_numeric_matrix_loose(csv_filepath)
            if raw is not None and raw.shape[0] > 0:
                metadata = {
                    'num_samples': int(raw.shape[0]),
                    'lsb_uv': 1.0,
                    'fs_hz': 300,
                    'derivations': [f"D{i+1}" for i in range(raw.shape[1])],
                    'data_type': 'SYNTHETIC_FALLBACK',
                    'duration_s': float(raw.shape[0]) / 300.0,
                }
                return self._build_synthetic_inconclusive_result(
                    metadata,
                    metadata['duration_s'],
                    metadata['fs_hz'],
                    signal_quality=25.0,
                )

            return {
                'error': str(e),
                'dns_probability_pct': 0,
                'confidence_score_pct': 0
            }
    
    def get_clinical_interpretation(self, result):
        """
        Genera interpretacion clinica en lenguaje natural.
        
        Args:
            result: dict retornado por process_ecg_file()
            
        Returns:
            dict con 'estado' y 'recomendaciones'
        """
        
        if 'error' in result:
            return {
                'estado': 'error',
                'recomendaciones': ['No se pudo procesar el ECG: {}'.format(result['error'])]
            }
        
        dns_prob = result.get('dns_probability_pct', 0)
        confidence = result.get('confidence_score_pct', 0)
        decision_threshold = float(result.get('decision_threshold_pct', 50.0))
        hr_mean = result.get('hr_mean_bpm', 0)
        pauses_gt3s = result.get('pauses_gt3s_per_min', 0)
        pauses_gt4s = result.get('pauses_gt4s_per_min', 0)
        
        recomendaciones = []
        
        # Clasificacion segun probabilidad DNS
        if dns_prob >= decision_threshold + 20:
            estado = 'DNS altamente probable'
            recomendaciones.append('Consulta cardiologica urgente recomendada')
            recomendaciones.append('Probable Disfuncion del Nodo Sinusal')
        elif dns_prob >= decision_threshold:
            estado = 'DNS probable'
            recomendaciones.append('Se recomienda evaluacion cardiologica')
            recomendaciones.append('Monitoreo de sintomas clinicos')
        elif dns_prob >= max(10.0, decision_threshold - 20):
            estado = 'Hallazgos inconclusivos'
            recomendaciones.append('ECG de mayor duracion recomendado')
            recomendaciones.append('Observacion clinica')
        else:
            estado = 'Sin indicadores de DNS'
            recomendaciones.append('ECG dentro de parametros normales')
        
        # Hallazgos especificos
        if hr_mean < 40:
            recomendaciones.append('Bradicardia severa detectada (HR: {:.1f} bpm)'.format(hr_mean))
        elif hr_mean < 60:
            recomendaciones.append('Bradicardia leve (HR: {:.1f} bpm)'.format(hr_mean))
        
        if pauses_gt4s > 0:
            recomendaciones.append('Pausas sinusales severas detectadas (>4s): {:.1f}/min'.format(pauses_gt4s))
        elif pauses_gt3s >= 3.0:
            recomendaciones.append('Pausas sinusales prolongadas detectadas (>3s): {:.1f}/min'.format(pauses_gt3s))
        
        # Calidad de senal
        quality = result.get('signal_quality_score', 0)
        if quality < 70:
            recomendaciones.append('Calidad de senal baja ({:.1f}%) - repetir ECG si es posible'.format(quality))
        
        if confidence < 60:
            recomendaciones.append('Baja confianza en prediccion - ECG muy corto o ruidoso')
        
        # Compatibilidad frontend: mantener lista y tambien versiones en texto.
        recomendaciones = [r for r in recomendaciones if str(r).strip()]
        if not recomendaciones:
            recomendaciones = ['Sin recomendaciones adicionales.']

        return {
            'estado': estado,
            'recomendaciones': recomendaciones,
            'recommendations': recomendaciones,
            'recomendacion': recomendaciones[0],
            'recomendaciones_texto': ' | '.join(recomendaciones),
            'dns_probability_pct': dns_prob,
            'decision_threshold_pct': decision_threshold,
            'confidence_score_pct': confidence,
            'model_type': result.get('model_type', 'unknown'),
            'literature_override': bool(result.get('literature_override', False)),
            'model_explanation': result.get('model_explanation'),
            'metricas': {
                'hr_mean_bpm': hr_mean,
                'rr_max_s': result.get('rr_max_s', 0),
                'sdnn_s': result.get('sdnn_s', 0),
                'pauses_gt3s_per_min': pauses_gt3s,
                'pauses_gt4s_per_min': pauses_gt4s
            }
        }
    
    def get_comparative_analysis(self, resultado_actual, resultado_previo):
        """
        Compara resultado actual con examen previo del mismo paciente.
        
        Args:
            resultado_actual: dict del analisis actual
            resultado_previo: dict con metricas del examen previo
            
        Returns:
            dict con analisis comparativo y tendencias
        """
        
        if not resultado_previo or 'error' in resultado_actual:
            return None
        
        dns_actual = resultado_actual.get('dns_probability_pct', 0)
        dns_previo = resultado_previo.get('dns_probability_pct', 0)
        
        cambio_dns = dns_actual - dns_previo
        
        # Determinar tendencia
        if abs(cambio_dns) < 5:
            tendencia = 'estable'
            tendencia_desc = 'Sin cambios significativos'
        elif cambio_dns > 0:
            if cambio_dns >= 20:
                tendencia = 'empeoramiento_severo'
                tendencia_desc = 'Empeoramiento significativo detectado'
            elif cambio_dns >= 10:
                tendencia = 'empeoramiento_moderado'
                tendencia_desc = 'Empeoramiento moderado'
            else:
                tendencia = 'empeoramiento_leve'
                tendencia_desc = 'Leve empeoramiento'
        else:
            if cambio_dns <= -20:
                tendencia = 'mejoria_significativa'
                tendencia_desc = 'Mejoria significativa detectada'
            elif cambio_dns <= -10:
                tendencia = 'mejoria_moderada'
                tendencia_desc = 'Mejoria moderada'
            else:
                tendencia = 'mejoria_leve'
                tendencia_desc = 'Leve mejoria'
        
        # Alertas especificas
        alertas = []
        
        if tendencia.startswith('empeoramiento'):
            alertas.append('⚠️ ALERTA: Deterioro en condicion cardiaca detectado')
            alertas.append('Comparacion con examen previo muestra aumento de {:.1f}% en probabilidad DNS'.format(cambio_dns))
            
            if dns_actual >= 70 and dns_previo < 70:
                alertas.append('🔴 CRITICO: Paciente cruzo umbral de DNS altamente probable')
            elif dns_actual >= 50 and dns_previo < 50:
                alertas.append('🟡 ATENCION: Paciente cruzo umbral de DNS probable')
        
        elif tendencia.startswith('mejoria'):
            alertas.append('✅ POSITIVO: Mejoria en condicion cardiaca detectada')
            alertas.append('Reduccion de {:.1f}% en probabilidad DNS respecto a examen previo'.format(abs(cambio_dns)))
        
        return {
            'tiene_historico': True,
            'tendencia': tendencia,
            'tendencia_descripcion': tendencia_desc,
            'cambio_dns_pct': cambio_dns,
            'dns_actual': dns_actual,
            'dns_previo': dns_previo,
            'fecha_examen_previo': resultado_previo.get('fecha_analisis'),
            'alertas_comparativas': alertas
        }
