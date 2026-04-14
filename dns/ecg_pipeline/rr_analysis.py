#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rr_analysis.py - Analisis de intervalos RR y pausas sinusales

Responsabilidades:
- Calcular intervalos RR a partir de indices de picos
- Detectar pausas sinusales (RR > 2s, 3s, 4s)
- Normalizar conteos por duracion de ECG
- Calcular estadisticas (SDNN, RMSSD, media, max)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

from config import (
    RR_RANGE_S, RR_OUTLIER_MIN_S, RR_OUTLIER_MAX_S,
    RR_PAUSE_MINOR_S, RR_PAUSE_MAJOR_S, RR_PAUSE_SEVERE_S
)

def calculate_rr_intervals(peaks_seconds):
    """
    Calcula intervalos RR a partir de indices de picos R.
    
    Parametros:
        peaks_seconds (np.ndarray): Indices de picos en segundos
        
    Retorna:
        tuple: (rr_intervals_s, valid_rr_fraction)
            - rr_intervals_s: Array de intervalos RR validos en segundos
            - valid_rr_fraction: Fraccion de RR validos
    """
    
    if len(peaks_seconds) < 2:
        logger.warning("Insuficientes picos para calcular RR: " + str(len(peaks_seconds)))
        return np.array([]), 0.0
    
    # Calcular RR como diferencia entre picos consecutivos
    rr_raw = np.diff(peaks_seconds)
    
    # Validar y filtrar RR
    rr_valid = []
    for rr in rr_raw:
        if RR_OUTLIER_MIN_S <= rr <= RR_OUTLIER_MAX_S:
            rr_valid.append(rr)
    
    rr_intervals_s = np.array(rr_valid)
    
    if len(rr_intervals_s) == 0:
        logger.warning("No hay intervalos RR validos")
        return np.array([]), 0.0
    
    # Fraccion de RR validos
    valid_fraction = len(rr_intervals_s) / len(rr_raw) if len(rr_raw) > 0 else 0.0
    
    logger.info("RR: " + str(len(rr_intervals_s)) + " validos de " + str(len(rr_raw)) + " total")
    logger.info("  Rango: [" + str(round(rr_intervals_s.min(), 2)) + "s, " + str(round(rr_intervals_s.max(), 2)) + "s]")
    
    return rr_intervals_s, valid_fraction


def analyze_rr_intervals(rr_intervals_s, duration_s):
    """
    Analiza estadisticas de RR y detecta pausas sinusales.
    
    Parametros:
        rr_intervals_s (np.ndarray): Intervalos RR en segundos
        duration_s (float): Duracion total del ECG en segundos
        
    Retorna:
        dict: {
            'rr_mean_s': media RR,
            'rr_max_s': maximo RR,
            'rr_min_s': minimo RR,
            'sdnn_s': desviacion std (SDNN),
            'rmssd_s': sqrt(mean(diff^2)) (RMSSD),
            'count_rr_gt2s': pausas > 2.0s / min,
            'count_rr_gt3s': pausas > 3.0s / min,
            'count_rr_gt4s': pausas > 4.0s / min,
        }
    """
    
    if len(rr_intervals_s) == 0:
        logger.warning("No hay RR para analizar")
        return {}
    
    # Estadisticas basicas
    rr_mean = np.mean(rr_intervals_s)
    rr_max = np.max(rr_intervals_s)
    rr_min = np.min(rr_intervals_s)
    
    # SDNN: desviacion estandar
    sdnn = np.std(rr_intervals_s)
    
    # RMSSD: raiz cuadrada de media de diferencias al cuadrado
    if len(rr_intervals_s) > 1:
        rr_diff = np.diff(rr_intervals_s)
        rmssd = np.sqrt(np.mean(rr_diff ** 2))
    else:
        rmssd = 0.0
    
    # Normalizar pausas por minuto
    duration_min = duration_s / 60.0
    
    count_gt2s = np.sum(rr_intervals_s > RR_PAUSE_MINOR_S)
    pauses_per_min_gt2s = (count_gt2s / duration_min) if duration_min > 0 else 0
    
    count_gt3s = np.sum(rr_intervals_s > RR_PAUSE_MAJOR_S)
    pauses_per_min_gt3s = (count_gt3s / duration_min) if duration_min > 0 else 0
    
    count_gt4s = np.sum(rr_intervals_s > RR_PAUSE_SEVERE_S)
    pauses_per_min_gt4s = (count_gt4s / duration_min) if duration_min > 0 else 0
    
    logger.info("RR estadisticas:")
    logger.info("  Media: " + str(round(rr_mean, 2)) + "s")
    logger.info("  Max: " + str(round(rr_max, 2)) + "s")
    logger.info("  SDNN: " + str(round(sdnn, 3)) + "s")
    logger.info("  RMSSD: " + str(round(rmssd, 3)) + "s")
    logger.info("Pausas detectadas:")
    logger.info("  RR>2s: " + str(count_gt2s) + " (" + str(round(pauses_per_min_gt2s, 1)) + "/min)")
    logger.info("  RR>3s: " + str(count_gt3s) + " (" + str(round(pauses_per_min_gt3s, 1)) + "/min)")
    logger.info("  RR>4s: " + str(count_gt4s) + " (" + str(round(pauses_per_min_gt4s, 1)) + "/min)")
    
    # Calcular HR media
    hr_mean_bpm = (60.0 / rr_mean) if rr_mean > 0 else 0
    
    return {
        'rr_mean_s': rr_mean,
        'rr_max_s': rr_max,
        'rr_min_s': rr_min,
        'sdnn_s': sdnn,
        'rmssd_s': rmssd,
        'hr_mean_bpm': hr_mean_bpm,
        'count_rr_gt2s': count_gt2s,
        'count_rr_gt3s': count_gt3s,
        'count_rr_gt4s': count_gt4s,
        'pauses_per_min_gt2s': pauses_per_min_gt2s,
        'pauses_per_min_gt3s': pauses_per_min_gt3s,
        'pauses_per_min_gt4s': pauses_per_min_gt4s,
    }
