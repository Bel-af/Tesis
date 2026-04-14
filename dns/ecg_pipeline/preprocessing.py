#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocessing.py - Lectura y preprocesamiento de senales ECG EasyG

Responsabilidades:
- Parsear metadatos de archivos CSV EasyG
- Convertir datos numericos a uV usando LSB
- Normalizar amplitud para procesamiento
- Aplicar filtros (paso-alto para remover drift DC)
- Validar duracion minima del ECG
"""

import re
import csv
import logging
import numpy as np
import os
import codecs

logger = logging.getLogger(__name__)

# ============================================
# Excepciones Custom
# ============================================

class ECGParsingError(Exception):
    """Error al parsear archivo CSV"""
    pass

class ShortECGWarning(Warning):
    """ECG más corto que lo ideal (pero procesable)"""
    pass

class NoValidSignalError(Exception):
    """No se logró extraer señal válida"""
    pass

# ============================================
# Constantes Fisiológicas
# ============================================

MIN_ECG_DURATION_S = 8  # segundos (mínimo exploración)
IDEAL_ECG_DURATION_S = 30  # segundos (ideal)

# ============================================
# Parser CSV EasyG
# ============================================

def parse_easyg_csv(filepath):
    """
    Parsea archivo CSV formato EasyG y retorna matriz ECG + metadatos.
    
    Parámetros:
        filepath (str): Ruta al archivo .csv
        
    Retorna:
        tuple: (ecg_array, metadata)
            - ecg_array: np.ndarray shape (num_samples, num_derivations) en µV
            - metadata: dict con claves {num_samples, lsb_uv, fs_hz, derivations, 
                                         data_type, duration_s}
                                         
    Raises:
        ECGParsingError: Si no se encuentran metadatos críticos
        NoValidSignalError: Si matriz de datos está vacía
    """
    
    metadata = {}
    data_lines = []
    
    try:
        # Intentar multiples encodings (compatible Python 2.7)
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        file_content = None
        
        for enc in encodings:
            try:
                f = codecs.open(filepath, 'r', encoding=enc)
                file_content = f.readlines()
                f.close()
                logger.debug("Archivo parseado con encoding: {}".format(enc))
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if file_content is None:
            raise ECGParsingError("No se pudo decodificar archivo con encodings: {}".format(encodings))
        
        for line in file_content:
            line = line.replace(u'\xa0', ' ').strip()
            
            # Procesar líneas de metadatos (comienzan con %)
            if line.startswith('%'):
                # Cantidad de muestras
                match = re.search(r'Cantidad de muestras\s*=\s*(\d+)', line)
                if match:
                    metadata['num_samples'] = int(match.group(1))
                
                # LSB en µV
                match = re.search(r'LSB\s*\[uV\]\s*=\s*([\d.,]+)', line)
                if match:
                    metadata['lsb_uv'] = float(match.group(1).replace(',', '.'))
                
                # Frecuencia de muestreo
                match = re.search(r'Frecuencia de muestreo efectiva\s*\[Hz\]\s*=\s*(\d+)', line)
                if match:
                    metadata['fs_hz'] = int(match.group(1))
                
                # Derivaciones
                match = re.search(r'Orden\s*=\s*(.+)$', line)
                if match:
                    deriv_str = match.group(1).strip()
                    metadata['derivations'] = [d.strip() for d in deriv_str.split(',') if d.strip()]
                
                # Tipo de dato
                match = re.search(r'Dato tipo:\s*(.+)$', line)
                if match:
                    metadata['data_type'] = match.group(1).strip()
            
            else:
                # Línea de datos (skip vacías)
                if line:
                    data_lines.append(line)
    
    except IOError as e:
        raise ECGParsingError("Archivo no encontrado: {}".format(filepath))
    except Exception as e:
        raise ECGParsingError("Error leyendo {}: {}".format(filepath, str(e)))
    
    # Validar que se extrajeron metadatos críticos
    required_keys = {'num_samples', 'lsb_uv', 'fs_hz', 'derivations'}
    missing = required_keys - set(metadata.keys())
    if missing:
        raise ECGParsingError("Metadatos faltantes: {}".format(missing))

    # Parsear matriz de datos
    try:
        reader = csv.reader(data_lines)
        raw_data = []
        for row_idx, row in enumerate(reader):
            # Compatibilidad con exportaciones que usan ';' como separador
            if len(row) == 1 and ';' in row[0]:
                row = row[0].split(';')

            # Convertir strings a float, ignorar blancos
            row_values = []
            for val in row:
                val_str = val.replace(u'\xa0', ' ').strip()
                if val_str:
                    row_values.append(float(val_str.replace(',', '.')))
            
            if row_values:
                raw_data.append(row_values)
        
        if not raw_data:
            raise NoValidSignalError("No se encontraron datos de senal en CSV")
        
        raw_array = np.array(raw_data, dtype=np.float32)
        
    except ValueError as e:
        raise ECGParsingError("Error parsing valores numericos: {}".format(str(e)))
    
    # Convertir de valores numericos a µV usando LSB
    lsb = metadata['lsb_uv']
    ecg_uv = raw_array.astype(np.float32) * lsb
    
    # Calcular duración real del ECG
    actual_samples = ecg_uv.shape[0]
    duration_s = actual_samples / metadata['fs_hz']
    metadata['duration_s'] = duration_s
    
    logger.info("ECG parseado: {} muestras, duracion: {:.2f}s, {} derivaciones".format(
        actual_samples, duration_s, len(metadata['derivations'])))
    
    return ecg_uv, metadata


# ============================================
# Validación de Metadatos
# ============================================

def validate_metadata(metadata):
    """
    Valida que metadatos sean coherentes y dentro de rangos esperados.
    
    Parámetros:
        metadata (dict): Diccionario de metadatos
        
    Raises:
        ShortECGWarning: Si duración < 30s (pero procesable)
        ValueError: Si parámetro está fuera de rango aceptable
    """
    
    # Duración mínima
    duration = metadata['duration_s']
    if duration < MIN_ECG_DURATION_S:
        raise ValueError("ECG demasiado corto: {:.1f}s < {}s (minimo)".format(duration, MIN_ECG_DURATION_S))
    
    if duration < IDEAL_ECG_DURATION_S:
        logger.warning("ECG corto: {:.1f}s < {}s ideal".format(duration, IDEAL_ECG_DURATION_S))
    
    # LSB (típicamente 0.01-1 µV)
    if not (0.001 < metadata['lsb_uv'] < 10):
        raise ValueError("LSB fuera de rango: {} uV".format(metadata['lsb_uv']))
    
    # Frecuencia de muestreo (típicamente 200-500 Hz para ECG veterinario)
    if not (50 < metadata['fs_hz'] < 1000):
        raise ValueError("Frecuencia de muestreo irreal: {} Hz".format(metadata['fs_hz']))
    
    # Cantidad mínima de derivaciones
    if len(metadata['derivations']) < 1:
        raise ValueError("No hay derivaciones especificadas")
    
    logger.debug("Validacion exitosa: {} muestras @ {}Hz".format(metadata['num_samples'], metadata['fs_hz']))


# ============================================
# Preprocesamiento de Señal
# ============================================

def preprocess_signal(signal_uv, fs_hz, hp_cutoff=0.5):
    """
    Normaliza y filtra la senal ECG.
    
    Parametros:
        signal_uv (np.ndarray): Senal en µV
        fs_hz (int): Frecuencia de muestreo en Hz
        hp_cutoff (float): Frecuencia corte paso-alto en Hz (default 0.5)
        
    Retorna:
        np.ndarray: Senal normalizada a rango [-1, 1]
    """
    
    # Remover media (componente DC)
    signal_centered = signal_uv - np.mean(signal_uv)
    
    # Filtro paso-alto simple sin scipy
    # Usar diferenciacion simple como aproximacion
    signal_filtered = signal_centered.copy()
    
    # Aplicar diferenciacion simple para remover drift DC
    for i in range(1, len(signal_filtered)):
        signal_filtered[i] = signal_filtered[i] - signal_filtered[i-1] * 0.5
    
    # Normalizar a [-1, 1]
    max_val = np.max(np.abs(signal_filtered))
    if max_val > 0:
        signal_normalized = signal_filtered / max_val
    else:
        signal_normalized = signal_filtered
    
    logger.debug("Senal procesada: rango [{:.3f}, {:.3f}]".format(signal_normalized.min(), signal_normalized.max()))
    
    return signal_normalized


# ============================================
# Selección Automática de Mejor Derivación
# ============================================

def select_best_derivation(ecg_uv, derivations, fs_hz):
    """
    Selecciona automaticamente la mejor derivacion para deteccion de picos R.
    Prueba varias derivaciones y retorna la que tenga mayor amplitud de picos.
    
    Parametros:
        ecg_uv (np.ndarray): Matriz (muestras x derivaciones)
        derivations (list): Lista de nombres de derivaciones
        fs_hz (int): Frecuencia de muestreo
        
    Retorna:
        tuple: (signal_best, derivation_name)
    """
    
    # Orden de prioridad: II > I > V1 > III > aVf > aVl > aVr
    priority = ['II', 'I', 'V1', 'III', 'aVf', 'aVl', 'aVr']
    
    best_signal = None
    best_deriv = None
    best_score = -1
    
    for i, deriv_name in enumerate(derivations):
        if i >= ecg_uv.shape[1]:
            continue
        
        signal_raw = ecg_uv[:, i]
        signal_processed = preprocess_signal(signal_raw, fs_hz)
        
        # Calcular score: amplitud pico-a-pico y varianza
        peak_to_peak = signal_processed.max() - signal_processed.min()
        variance = np.var(signal_processed)
        
        # Bonus si es derivacion prioritaria
        priority_bonus = 1.0
        if deriv_name in priority:
            priority_bonus = 1.5 - (priority.index(deriv_name) * 0.1)
        
        score = peak_to_peak * variance * priority_bonus
        
        logger.debug("Derivacion {}: score={:.3f}, p2p={:.3f}, var={:.3f}".format(
            deriv_name, score, peak_to_peak, variance))
        
        if score > best_score:
            best_score = score
            best_signal = signal_processed
            best_deriv = deriv_name
    
    logger.info("Mejor derivacion seleccionada: {} (score={:.3f})".format(
        best_deriv, best_score))
    
    return best_signal, best_deriv


# ============================================
# Pipeline Completo
# ============================================

def process_ecg_file(filepath):
    """
    Pipeline completo: parsea, valida, preprocesa ECG.
    
    Parámetros:
        filepath (str): Ruta a archivo CSV EasyG
        
    Retorna:
        dict: {
            'signal_normalized': np.ndarray,  # Mejor derivación procesada
            'signal_raw_all': np.ndarray,      # Todas derivaciones sin filtrar
            'metadata': dict,
            'duration_s': float,
            'fs_hz': int,
            'selected_derivation': str         # Derivación usada
        }
        
    Raises:
        ECGParsingError, ValueError: Si algún paso falla
    """
    
    logger.info("Procesando {}".format(filepath))
    
    # Parsear
    ecg_uv, metadata = parse_easyg_csv(filepath)
    
    # Validar
    validate_metadata(metadata)
    
    # Seleccionar MEJOR derivacion automaticamente
    signal_norm, selected_deriv = select_best_derivation(
        ecg_uv, metadata['derivations'], metadata['fs_hz']
    )
    
    logger.info("Procesamiento exitoso: {:.2f}s, derivacion: {}".format(
        metadata['duration_s'], selected_deriv))
    
    return {
        'signal_normalized': signal_norm,
        'signal_raw_all': ecg_uv,
        'metadata': metadata,
        'duration_s': metadata['duration_s'],
        'fs_hz': metadata['fs_hz'],
        'selected_derivation': selected_deriv
    }
