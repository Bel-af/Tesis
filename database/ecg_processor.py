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
    from preprocessing import process_ecg_file
    from peak_detection import detect_r_peaks
    from rr_analysis import calculate_rr_intervals, analyze_rr_intervals
    from features import extract_features
    
    # Intentar usar modelo ML si existe, fallback a reglas
    try:
        from model_ml import predict_dns as predict_dns_probability
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
            result = process_ecg_file(csv_filepath)
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
            else:
                dns_probability_pct, confidence_score_pct = prediction
            
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
                'count_rr_gt3s': rr_analysis.get('count_rr_gt3s', 0),
                'count_rr_gt4s': rr_analysis.get('count_rr_gt4s', 0),
                'duration_s': duration_s,
                'signal_quality_score': features.get('signal_quality_score', 0),
                'num_samples': metadata.get('num_samples', 0),
                'fs_hz': fs_hz,
                'derivations': metadata.get('derivations', []),
                'metadata': metadata
            }
            
            self.last_result = resultado
            return resultado
            
        except Exception as e:
            logger.error("Error procesando ECG: {}".format(str(e)), exc_info=True)
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
        hr_mean = result.get('hr_mean_bpm', 0)
        pauses_gt3s = result.get('pauses_gt3s_per_min', 0)
        pauses_gt4s = result.get('pauses_gt4s_per_min', 0)
        
        recomendaciones = []
        
        # Clasificacion segun probabilidad DNS
        if dns_prob >= 70:
            estado = 'DNS altamente probable'
            recomendaciones.append('Consulta cardiologica urgente recomendada')
            recomendaciones.append('Probable Disfuncion del Nodo Sinusal')
        elif dns_prob >= 50:
            estado = 'DNS probable'
            recomendaciones.append('Se recomienda evaluacion cardiologica')
            recomendaciones.append('Monitoreo de sintomas clinicos')
        elif dns_prob >= 30:
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
        
        return {
            'estado': estado,
            'recomendaciones': recomendaciones,
            'dns_probability_pct': dns_prob,
            'confidence_score_pct': confidence,
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
