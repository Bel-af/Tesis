#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
peak_detection.py - Deteccion de picos R con adaptive thresholding

Responsabilidades:
- Detectar picos R usando ventana movil adaptativa
- Tolerante a ruido e intervalos incompletos
- Retornar indices de picos en segundos
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

from config import PEAK_DETECTION_WINDOW_S, PEAK_THRESHOLD_STD_MULTIPLIER, MIN_PEAK_DISTANCE_S

def detect_r_peaks(signal_norm, fs_hz):
    """
    Detecta picos R usando adaptive thresholding con ventana movil.
    Ahora prueba con señal normal E INVERTIDA para manejar polaridades opuestas.
    
    Parametros:
        signal_norm (np.ndarray): Senal normalizada [-1, 1]
        fs_hz (int): Frecuencia de muestreo en Hz
        
    Retorna:
        np.ndarray: Indices de picos en segundos (shape: (num_peaks,))
    """
    
    if len(signal_norm) == 0:
        logger.warning("Senal vacia")
        return np.array([])
    
    # Intentar deteccion con señal normal
    peaks_normal = _detect_peaks_adaptive(signal_norm, fs_hz)
    
    # Intentar deteccion con señal INVERTIDA
    peaks_inverted = _detect_peaks_adaptive(-signal_norm, fs_hz)
    
    # Usar el resultado que tenga MAS picos detectados
    if len(peaks_inverted) > len(peaks_normal):
        logger.info("Señal invertida detectada - usando {} picos (vs {} en normal)".format(
            len(peaks_inverted), len(peaks_normal)))
        return peaks_inverted
    else:
        return peaks_normal


def _detect_peaks_adaptive(signal_norm, fs_hz):
    """
    Funcion interna: deteccion adaptativa de picos.
    """
    
    # Ventana adaptativa en muestras
    window_samples = int(PEAK_DETECTION_WINDOW_S * fs_hz)
    if window_samples < 3:
        window_samples = 3
    
    # Suavizar senal manualmente (sin scipy.signal.savgol_filter)
    signal_smooth = simple_smooth(signal_norm, window_samples)
    
    # Calcular threshold adaptativo usando ventana movil
    peaks = []
    
    for i in range(1, len(signal_smooth) - 1):
        # Ventana centrada
        window_start = max(0, i - window_samples)
        window_end = min(len(signal_smooth), i + window_samples)
        window_data = signal_smooth[window_start:window_end]
        
        # Umbral local: media + 1.5*std
        threshold = np.mean(window_data) + PEAK_THRESHOLD_STD_MULTIPLIER * np.std(window_data)
        
        # Si el punto actual es maximo local y supera umbral
        if (signal_smooth[i] > threshold and 
            signal_smooth[i] > signal_smooth[i-1] and 
            signal_smooth[i] > signal_smooth[i+1]):
            peaks.append(i)
    
    if not peaks:
        logger.warning("No se detectaron picos R")
        return np.array([])
    
    # Eliminar picos duplicados (distancia minima)
    min_dist_samples = int(MIN_PEAK_DISTANCE_S * fs_hz)
    filtered_peaks = []
    
    for peak_idx in peaks:
        if not filtered_peaks or (peak_idx - filtered_peaks[-1]) >= min_dist_samples:
            filtered_peaks.append(peak_idx)
    
    # Convertir indices a segundos
    peaks_seconds = np.array(filtered_peaks) / float(fs_hz)
    
    logger.info("Picos R detectados: " + str(len(peaks_seconds)))
    logger.debug("Indices (muestras): " + str(filtered_peaks[:5]) + ("..." if len(filtered_peaks) > 5 else ""))
    
    return peaks_seconds


def simple_smooth(signal, window_size):
    """
    Suaviza senal usando media movil simple.
    """
    if window_size <= 1 or len(signal) < window_size:
        return signal
    
    smoothed = np.zeros_like(signal)
    for i in range(len(signal)):
        start = int(max(0, i - window_size / 2))
        end = int(min(len(signal), i + window_size / 2))
        smoothed[i] = np.mean(signal[start:end])
    
    return smoothed
