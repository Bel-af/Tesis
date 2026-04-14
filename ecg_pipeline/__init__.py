#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ecg_pipeline - Pipeline de deteccion DNS para ECG veterinario

Modulos:
- config: Configuracion y constantes fisiologicas
- preprocessing: Lectura y preprocesamiento de CSV EasyG
- peak_detection: Deteccion de picos R
- rr_analysis: Analisis de intervalos RR
- features: Extraccion de caracteristicas
- model: Prediccion de probabilidad DNS
- reporter: Generacion de reportes

Modulos:
- config: Configuracion centralizada y constantes fisiologicas
- preprocessing: Parsing y preprocesamiento de senales CSV EasyG
- peak_detection: Deteccion adaptativa de picos R
- rr_analysis: Calculo intervalos RR y estadisticas
- features: Extraccion de caracteristicas para ML
- model: Entrenamiento y prediccion ML
- reporter: Generación de reportes
"""

__version__ = "0.1.0"
__author__ = "Proyecto ECG Veterinario"

from .config import setup_logging

# Configurar logging al importar paquete
setup_logging()
