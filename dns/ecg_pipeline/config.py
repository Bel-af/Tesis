#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py - Configuracion centralizada del sistema ECG

Define constantes, rangos fisiologicos y parametros globales.
Todos los agentes IA deben consultar este archivo para validar
que sus valores de referencia sean correctos.
"""

import logging
import os

# ============================================
# Rutas Base
# ============================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
RAW_ECGS_DIR = os.path.join(DATA_DIR, "raw_ecgs")
TEST_ECGS_DIR = os.path.join(DATA_DIR, "test_ecgs")
DB_PATH = os.path.join(DATA_DIR, "ecg_database.db")

# ============================================
# Logging
# ============================================

def setup_logging(level=logging.INFO):
    """Configura logging consistente para todo el proyecto"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(PROJECT_ROOT, 'ecg_analysis.log'))
        ]
    )

# ============================================
# Rangos Fisiológicos - CRITICAL
# No modificar sin revisar clinical references
# ============================================

# Frecuencia Cardíaca (bpm)
HR_RANGE_BPM = (40, 180)           # Rango operativo total
HR_DNS_ZONE_BPM = (40, 90)         # Sugestivo DNS (ponderación mayor)
HR_INCONCLUSIVE_BPM = (90, 140)    # Zona no concluyente
HR_HIGH_BPM = (140, 180)           # Zona alta (menor peso DNS)

# Intervalo RR (segundos)
RR_RANGE_S = (0.3, 10.0)           # Rango operativo total
RR_PAUSE_MINOR_S = 2.0             # Pausa sinusal relevante
RR_PAUSE_MAJOR_S = 3.0             # Pausa altamente sugestiva DNS
RR_PAUSE_SEVERE_S = 4.0            # Pausa severa

# RR Máximo detectado (segundos)
RR_MAX_RANGE_S = (1.0, 10.0)       # Rango esperado amplio
RR_MAX_DNS_THRESHOLD_S = 5.5       # Alta sospecha DNS

# Variabilidad RR - SDNN (segundos)
SDNN_RANGE_S = (0.02, 0.70)        # Rango operativo (sensible a errores)
SDNN_DNS_THRESHOLD_S = 0.15        # Sugestivo de DNS
SDNN_CRITICAL_S = 0.30             # Muy alta variabilidad patológica

# Variabilidad RR - RMSSD (segundos)
RMSSD_RANGE_S = (0.02, 0.70)       # Rango operativo
RMSSD_DNS_THRESHOLD_S = 0.12       # Aumentado (pondera DNS)
RMSSD_CRITICAL_S = 0.30            # Muy inestable

# Pausas Sinusales Normalizadas (conteo / minuto)
RR_GT2S_RANGE = (0, 30)            # Rango 0-30+ / min
RR_GT3S_RANGE = (0, 15)            # Rango 0-15+ / min (presencia = relevante)
RR_GT4S_RANGE = (0, 5)             # Rango 0-5+ / min (alto peso si presente)

# ============================================
# Duración ECG
# ============================================

MIN_ECG_DURATION_S = 8              # Mínimo funcional (exploratorio)
IDEAL_ECG_DURATION_S = 30           # Ideal para análisis

# ============================================
# Calidad de Señal
# ============================================

MIN_VALID_RR_FRACTION = 0.70        # Si <70% RR son válidos: baja confianza
RR_OUTLIER_MIN_S = 0.3             # Rechazar RR menor que esto
RR_OUTLIER_MAX_S = 10.0            # Rechazar RR mayor que esto

# ============================================
# Parámetros de Detección de Picos (adaptive threshold)
# ============================================

# Ventana móvil para cálculo de umbral local
PEAK_DETECTION_WINDOW_S = 0.35     # Ventana móvil en segundos

# Multiplicador de std para umbral adaptativo
PEAK_THRESHOLD_STD_MULTIPLIER = 1.5  # umbral = mean + std * 1.5

# Distancia mínima entre picos (evitar detecciones duplicadas)
MIN_PEAK_DISTANCE_S = 0.2          # 0.2s ≈ 300 bpm máximo

# ============================================
# Parámetros de Filtrado
# ============================================

HP_CUTOFF_HZ = 0.5                 # Filtro paso-alto (remover drift DC)
HP_FILTER_ORDER = 2                # Orden Butterworth

LP_CUTOFF_HZ = 40.0                # Filtro paso-bajo (remover ruido HF)
LP_FILTER_ORDER = 2

# ============================================
# Modelo ML
# ============================================

# Tipos de modelos soportados
MODEL_TYPE_RANDOM_FOREST = "random_forest"
MODEL_TYPE_XGBOOST = "xgboost"
MODEL_TYPE_LINEAR = "linear_regression"

# Configuración default para Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# ============================================
# Reportes
# ============================================

CONFIDENCE_THRESHOLD_HIGH = 85      # Confianza alta (%)
CONFIDENCE_THRESHOLD_MEDIUM = 60    # Confianza media
CONFIDENCE_THRESHOLD_LOW = 40       # Confianza baja

# Umbral de probabilidad DNS para alertar
DNS_ALERT_THRESHOLD = 50            # Si DNS >= 50%, generar alerta

# ============================================
# Utilidades
# ============================================

def get_hr_zone(hr_bpm):
    """
    Clasifica HR en zonas fisiológicas.
    
    Retorna: 'bajo' | 'medio' | 'alto' | 'fuera_rango'
    """
    if hr_bpm < HR_DNS_ZONE_BPM[0] or hr_bpm > HR_RANGE_BPM[1]:
        return 'fuera_rango'
    elif hr_bpm <= HR_DNS_ZONE_BPM[1]:
        return 'bajo_dns'
    elif hr_bpm <= HR_INCONCLUSIVE_BPM[1]:
        return 'inconcluso'
    else:
        return 'alto'


def get_confidence_level(confidence_score):
    """
    Clasifica nivel de confianza.
    
    Retorna: 'baja' | 'media' | 'alta'
    """
    if confidence_score >= CONFIDENCE_THRESHOLD_HIGH:
        return 'alta'
    elif confidence_score >= CONFIDENCE_THRESHOLD_MEDIUM:
        return 'media'
    else:
        return 'baja'
