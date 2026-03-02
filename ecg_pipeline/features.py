#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
features.py - Extraccion de caracteristicas para modelo ML

Responsabilidades:
- Calcular HR y zona fisiologica
- Normalizar caracteristicas
- Retornar vector de features para prediccion
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

from config import (
    HR_RANGE_BPM, HR_DNS_ZONE_BPM, HR_INCONCLUSIVE_BPM, HR_HIGH_BPM,
    SDNN_DNS_THRESHOLD_S, RMSSD_DNS_THRESHOLD_S, RR_MAX_DNS_THRESHOLD_S,
    MIN_VALID_RR_FRACTION
)

def extract_features(rr_analysis, valid_rr_fraction, duration_s):
    """
    Extrae caracteristicas para modelo ML.
    
    Parametros:
        rr_analysis (dict): Resultado de analyze_rr_intervals()
        valid_rr_fraction (float): Fraccion de RR validos
        duration_s (float): Duracion ECG en segundos
        
    Retorna:
        dict: {
            'hr_mean_bpm': Frecuencia cardiaca media,
            'hr_zone': 'bajo_dns' | 'inconcluso' | 'alto',
            'sdnn_s': SDNN,
            'rmssd_s': RMSSD,
            'rr_max_s': RR maximo,
            'pauses_per_min_gt3s': Pausas RR>3s por minuto,
            'pauses_per_min_gt4s': Pausas RR>4s por minuto,
            'signal_quality_score': % de RR validos,
            'ecg_duration_s': Duracion ECG,
            'features_normalized': array normalizado para modelo
        }
    """
    
    if not rr_analysis or len(rr_analysis) == 0:
        logger.warning("No hay analisis RR")
        return {}
    
    # Calcular HR
    rr_mean = rr_analysis.get('rr_mean_s', 0)
    if rr_mean > 0:
        hr_mean_bpm = 60.0 / rr_mean
    else:
        hr_mean_bpm = 0
    
    # Clasificar zona HR
    if hr_mean_bpm < HR_DNS_ZONE_BPM[1]:
        hr_zone = 'bajo_dns'
    elif hr_mean_bpm < HR_INCONCLUSIVE_BPM[1]:
        hr_zone = 'inconcluso'
    else:
        hr_zone = 'alto'
    
    # Normalizar HR a [0, 1]
    hr_norm = (hr_mean_bpm - HR_RANGE_BPM[0]) / (HR_RANGE_BPM[1] - HR_RANGE_BPM[0])
    hr_norm = np.clip(hr_norm, 0, 1)
    
    # Extraer features
    sdnn = rr_analysis.get('sdnn_s', 0)
    rmssd = rr_analysis.get('rmssd_s', 0)
    rr_max = rr_analysis.get('rr_max_s', 0)
    pauses_gt3s = rr_analysis.get('pauses_per_min_gt3s', 0)
    pauses_gt4s = rr_analysis.get('pauses_per_min_gt4s', 0)
    
    quality_score = valid_rr_fraction * 100.0
    
    logger.info("Caracteristicas extraidas:")
    logger.info("  HR: " + str(round(hr_mean_bpm, 1)) + " bpm (zona: " + hr_zone + ")")
    logger.info("  SDNN: " + str(round(sdnn, 3)) + "s")
    logger.info("  RMSSD: " + str(round(rmssd, 3)) + "s")
    logger.info("  RR max: " + str(round(rr_max, 2)) + "s")
    logger.info("  Calidad: " + str(round(quality_score, 1)) + "%")
    
    # Verificar calidad minima
    if valid_rr_fraction < MIN_VALID_RR_FRACTION:
        logger.warning("Calidad baja: " + str(round(quality_score, 1)) + "% < " + str(MIN_VALID_RR_FRACTION*100) + "%")
    
    # Vector normalizado para modelo (para uso futuro con ML avanzado)
    features_normalized = np.array([
        hr_norm,                    # [0, 1]
        np.clip(sdnn / 0.70, 0, 1),   # Normalizar a max ~0.70s
        np.clip(rmssd / 0.70, 0, 1),  # Normalizar a max ~0.70s
        np.clip(rr_max / 10.0, 0, 1), # Normalizar a max 10s
        np.clip(pauses_gt3s / 15.0, 0, 1),  # Normalizar a max ~15/min
        np.clip(pauses_gt4s / 5.0, 0, 1),   # Normalizar a max ~5/min
        valid_rr_fraction           # [0, 1]
    ])
    
    return {
        'hr_mean_bpm': hr_mean_bpm,
        'hr_zone': hr_zone,
        'sdnn_s': sdnn,
        'rmssd_s': rmssd,
        'rr_max_s': rr_max,
        'pauses_per_min_gt3s': pauses_gt3s,
        'pauses_per_min_gt4s': pauses_gt4s,
        'signal_quality_score': quality_score,
        'ecg_duration_s': duration_s,
        'features_normalized': features_normalized
    }
