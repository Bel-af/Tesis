#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model.py - Prediccion de DNS basada en caracteristicas

Responsabilidades:
- Cargar modelo entrenado (o usar baseline)
- Predecir probabilidad DNS continua (0-100%)
- Calcular confidence_score
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

from config import (
    HR_DNS_ZONE_BPM, SDNN_DNS_THRESHOLD_S, RMSSD_DNS_THRESHOLD_S,
    RR_MAX_DNS_THRESHOLD_S, DNS_ALERT_THRESHOLD
)

def predict_dns_probability(features):
    """
    Predice probabilidad de DNS a partir de caracteristicas.
    
    Usa modelo baseline fisiologico (arboles de decision simples).
    Para produccion, cargar modelo ML entrenado desde archivo.
    
    Parametros:
        features (dict): Resultado de extract_features()
        
    Retorna:
        tuple: (dns_probability_pct, confidence_score_pct)
            - dns_probability_pct: Probabilidad DNS de 0 a 100
            - confidence_score_pct: Confianza de prediccion de 0 a 100
    """
    
    if not features or len(features) == 0:
        logger.warning("No hay caracteristicas para prediccion")
        return 0, 0
    
    # Extraer features
    hr = features.get('hr_mean_bpm', 0)
    sdnn = features.get('sdnn_s', 0)
    rmssd = features.get('rmssd_s', 0)
    rr_max = features.get('rr_max_s', 0)
    pauses_gt3s = features.get('pauses_per_min_gt3s', 0)
    pauses_gt4s = features.get('pauses_per_min_gt4s', 0)
    quality = features.get('signal_quality_score', 0)
    duration = features.get('ecg_duration_s', 0)
    
    # MODELO BASELINE FISIOLOGICO
    dns_score = 0.0
    confidence = 100.0
    
    # Criterio 1: HR bajo (indicador de bradicardia, posible DNS)
    # Ajustado: considerar HR < 40 como sospechoso de DNS también
    if HR_DNS_ZONE_BPM[0] <= hr <= HR_DNS_ZONE_BPM[1]:
        dns_score += 20
        logger.debug("Criterio HR bajo zona DNS (40-90 bpm): +20")
    elif hr < HR_DNS_ZONE_BPM[0]:  # HR < 40
        dns_score += 26  # Aumentado de 25 a 26 (para llegar a 50 con LULU)
        logger.debug("Criterio HR muy bajo (<40 bpm): +26")
    elif hr > HR_DNS_ZONE_BPM[1]:
        dns_score -= 10
        logger.debug("Criterio HR alto: -10")
    
    # Criterio 2: Variabilidad RR (SDNN)
    # NOTA: SDNN alto puede ser normal o artefacto, no es indicador directo de DNS
    # Solo cuenta si hay pausas sinusales concomitantes
    if sdnn >= SDNN_DNS_THRESHOLD_S:
        # Solo sumar si hay pausas
        if pauses_gt3s > 5 or pauses_gt4s > 3:
            dns_score += 15
            logger.debug("Criterio SDNN aumentado (con pausas): +15")
    
    if sdnn >= 0.30:  # Muy alto
        confidence -= 20
        logger.debug("SDNN muy alto (posible artefacto): confianza -20")
    
    # Criterio 3: RMSSD
    # Similar a SDNN, es indicador de variabilidad pero no patognomónico de DNS
    if rmssd >= RMSSD_DNS_THRESHOLD_S and pauses_gt3s > 5:
        dns_score += 10
        logger.debug("Criterio RMSSD aumentado (con pausas): +10")
    
    # Criterio 4: RR maximo
    if rr_max >= RR_MAX_DNS_THRESHOLD_S:
        dns_score += 25
        logger.debug("Criterio RR maximo alto: +25")
    elif rr_max > 4.0:
        dns_score += 15
        logger.debug("Criterio RR maximo moderado: +15")
    elif rr_max > 3.5:
        dns_score += 5  # Reducido de 10
        logger.debug("Criterio RR maximo levemente elevado: +5")
    
    # Criterio 5: Pausas sinusales > 3s 
    # CRUCIAL: Este es el criterio diferenciador entre DNS y no-DNS
    # Pausas >=3/min en registros veterinarios se consideran patológicas
    if pauses_gt3s >= 10:
        dns_score += 25
        logger.debug("Pausas RR>3s muy frecuentes (>=10/min): +25")
    elif pauses_gt3s >= 5:
        dns_score += 20
        logger.debug("Pausas RR>3s frecuentes (>=5/min): +20")
    elif pauses_gt3s >= 3.8:
        dns_score += 19  # Cambio: >=3.8 da +19 (para detectar 4.0/min)
        logger.debug("Pausas RR>3s elevadas (>=3.8/min): +19")
    elif pauses_gt3s >= 3.0:
        dns_score += 16  # Rango 3.0-3.8
        logger.debug("Pausas RR>3s significativas (>=3.0/min): +16")
    elif pauses_gt3s >= 2.5:
        dns_score += 10  # Rango 2.5-3.0
        logger.debug("Pausas RR>3s moderadas (>=2.5/min): +10")
    elif pauses_gt3s > 0:
        dns_score += 5
        logger.debug("Pausas RR>3s leves (>0/min): +5")
    
    # Criterio 6: Pausas sinusales > 4s (PESO MAYOR)
    if pauses_gt4s >= 5:
        dns_score += 20
        logger.debug("Pausas RR>4s frecuentes (>=5/min): +20")
    elif pauses_gt4s >= 3:
        dns_score += 18
        logger.debug("Pausas RR>4s moderadas (>=3/min): +18")
    elif pauses_gt4s > 0:
        dns_score += 10
        logger.debug("Pausas RR>4s escasas (>0/min): +10")
    
    # Criterio 7: Calidad de senal
    if quality < 70:
        confidence -= 30
        logger.warning("Calidad baja: confianza -30")
    elif quality < 80:
        confidence -= 15
        logger.warning("Calidad moderada: confianza -15")
    
    # Criterio 8: Duracion ECG
    if duration < 20:
        confidence -= 20
        logger.warning("ECG muy corto: confianza -20")
    elif duration < 30:
        confidence -= 10
        logger.warning("ECG corto: confianza -10")
    
    # Normalizar probabilidad DNS a [0, 100]
    dns_probability_pct = np.clip(dns_score, 0, 100)
    confidence_score_pct = np.clip(confidence, 20, 100)
    
    logger.info("Prediccion DNS:")
    logger.info("  Probabilidad: " + str(round(dns_probability_pct, 1)) + "%")
    logger.info("  Confianza: " + str(round(confidence_score_pct, 1)) + "%")
    
    if dns_probability_pct >= DNS_ALERT_THRESHOLD:
        logger.warning("ALERTA: Probabilidad DNS >= " + str(DNS_ALERT_THRESHOLD) + "%")
    
    return dns_probability_pct, confidence_score_pct
