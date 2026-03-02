#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reporter.py - Generacion de reportes en texto

Responsabilidades:
- Consolidar resultados del pipeline
- Generar reporte formateado en texto
- Guardar a archivo reports/
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

from config import REPORTS_DIR, DNS_ALERT_THRESHOLD

def generate_report(filename, metadata, rr_analysis, features, dns_pred, confidence):
    """
    Genera reporte ECG con resultados del analisis DNS.
    
    Parametros:
        filename (str): Nombre original del archivo CSV
        metadata (dict): Resultado parse_easyg_csv
        rr_analysis (dict): Resultado analyze_rr_intervals
        features (dict): Resultado extract_features
        dns_pred (float): Probabilidad DNS (0-100)
        confidence (float): Confianza de prediccion (0-100)
        
    Retorna:
        str: Ruta al archivo reporte generado
    """
    
    # Crear directorio reports si no existe
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    
    # Nombre del reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = "report_{}_{}. txt".format(
        os.path.splitext(os.path.basename(filename))[0], timestamp)
    report_path = os.path.join(REPORTS_DIR, report_filename)
    
    # Extractar datos
    duration_s = metadata.get('duration_s', 0)
    fs_hz = metadata.get('fs_hz', 300)
    num_samples = metadata.get('num_samples', 0)
    derivations = metadata.get('derivations', [])
    
    hr_mean = features.get('hr_mean_bpm', 0)
    hr_zone = features.get('hr_zone', 'desconocido')
    sdnn = features.get('sdnn_s', 0)
    rmssd = features.get('rmssd_s', 0)
    rr_max = features.get('rr_max_s', 0)
    pauses_gt2s = features.get('pauses_per_min_gt2s', 0)
    pauses_gt3s = features.get('pauses_per_min_gt3s', 0)
    pauses_gt4s = features.get('pauses_per_min_gt4s', 0)
    quality = features.get('signal_quality_score', 0)
    
    # Generar contenido del reporte
    lines = []
    lines.append("=" * 70)
    lines.append("ECG VETERINARIO - REPORTE DE ANALISIS DNS")
    lines.append("=" * 70)
    lines.append("")
    
    # Seccion 1: Informacion de archivo
    lines.append("INFORMACION DE ARCHIVO")
    lines.append("-" * 70)
    lines.append("Archivo: {}".format(os.path.basename(filename)))
    lines.append("Duracion: {:.2f} segundos".format(duration_s))
    lines.append("Muestras: {} (fs={} Hz)".format(num_samples, fs_hz))
    lines.append("Derivaciones: {}".format(', '.join(derivations)))
    lines.append("Calidad senal: {:.1f}%".format(quality))
    lines.append("Confianza prediccion: {:.1f}%".format(confidence))
    lines.append("")
    
    # Seccion 2: Eventos detectados
    lines.append("EVENTOS DETECTADOS")
    lines.append("-" * 70)
    if pauses_gt2s > 0:
        lines.append("- Pausas sinusales RR > 2s: {:.1f} eventos/min".format(pauses_gt2s))
    if pauses_gt3s > 0:
        lines.append("- Pausas sinusales RR > 3s: {:.1f} eventos/min  <-- CLINICAMENTE RELEVANTE".format(pauses_gt3s))
    if pauses_gt4s > 0:
        lines.append("- Pausas sinusales RR > 4s: {:.1f} eventos/min  <-- ALTO PESO CLINICO".format(pauses_gt4s))
    if pauses_gt2s == 0 and pauses_gt3s == 0 and pauses_gt4s == 0:
        lines.append("- Sin pausas sinusales significativas detectadas")
    lines.append("")
    
    # Seccion 3: Metricas cardiaca
    lines.append("METRICAS CARDIACAS")
    lines.append("-" * 70)
    lines.append("Frecuencia cardiaca media (HR): {:.1f} bpm".format(hr_mean))
    if 40 <= hr_mean <= 90:
        lines.append("  -> Zona sugestiva de DNS")
    elif 90 < hr_mean <= 140:
        lines.append("  -> Zona inconclusa")
    else:
        lines.append("  -> Zona normal (sin DNS sugestiva)")
    
    lines.append("RR maximo: {:.2f} segundos".format(rr_max))
    if rr_max >= 5.5:
        lines.append("  -> Muy elevado, altamente sugestivo de pausa sinusal")
    elif rr_max >= 3.0:
        lines.append("  -> Moderadamente elevado")
    
    lines.append("")
    lines.append("Variabilidad RR (indicadores de desorden del NS):")
    lines.append("  SDNN: {:.3f} s".format(sdnn))
    if sdnn >= 0.15:
        lines.append("    -> Aumentado (sugestivo DNS)")
    
    lines.append("  RMSSD: {:.3f} s".format(rmssd))
    if rmssd >= 0.12:
        lines.append("    -> Aumentado (sugestivo DNS)")
    
    lines.append("")
    
    # Seccion 4: Probabilidad DNS
    lines.append("RESULTADO FINAL")
    lines.append("-" * 70)
    lines.append("PROBABILIDAD DNS: {:.1f}%".format(dns_pred))
    
    if dns_pred >= DNS_ALERT_THRESHOLD:
        lines[-1] = lines[-1] + "  <-- ALERTA CLINICA"
        lines.append("")
        lines.append("INTERPRETACION: Probabilidad CLINICAMENTE RELEVANTE de DNS.")
        lines.append("Se recomienda revision clinica y criterios ACVIM.")
    elif dns_pred >= 50:
        lines.append("")
        lines.append("INTERPRETACION: Probabilidad moderada de DNS.")
        lines.append("Correlacionar con hallazgos clinicos.")
    else:
        lines.append("")
        lines.append("INTERPRETACION: Probabilidad baja de DNS.")
        lines.append("Patron ECG sin hallazgos sugestivos de DNS en este analisis.")
    
    lines.append("")
    
    # Disclaimer
    lines.append("=" * 70)
    lines.append("ACLARACION IMPORTANTE")
    lines.append("=" * 70)
    lines.append("NO reemplaza el diagnostico clinico veterinario.")
    lines.append("El veterinario debe evaluar el ECG completo junto con")
    lines.append("antecedentes clinicos, examen fisico, y otros diagnosticos.")
    lines.append("")
    lines.append("")
    lines.append("Reporte generado: {}".format(datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
    lines.append("=" * 70)
    
    # Escribir a archivo
    try:
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info("Reporte guardado: {}".format(report_path))
        return report_path
    except IOError as e:
        logger.error("Error escribiendo reporte: {}".format(e))
        raise

def print_report_summary(dns_pred, confidence, hr_mean, pauses_gt3s, pauses_gt4s):
    """
    Imprime resumen del reporte a consola.
    """
    print("\n" + "=" * 70)
    print("RESULTADOS DEL ANALISIS")
    print("=" * 70)
    print("Probabilidad DNS: {:.1f}%".format(dns_pred))
    print("Confianza: {:.1f}%".format(confidence))
    print("HR media: {:.1f} bpm".format(hr_mean))
    print("Pausas RR>3s: {:.1f}/min".format(pauses_gt3s))
    print("Pausas RR>4s: {:.1f}/min".format(pauses_gt4s))
    
    if dns_pred >= DNS_ALERT_THRESHOLD:
        print("\n*** ALERTA: Probabilidad DNS >= 50% ***")
    
    print("=" * 70 + "\n")
