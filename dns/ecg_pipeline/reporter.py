#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reporter.py - Generacion de reportes en texto y PDF

Responsabilidades:
- Consolidar resultados del pipeline
- Generar reporte formateado en texto
- Generar informe PDF con todas las metricas clinicas
- Guardar a archivo reports/
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from .config import REPORTS_DIR, DNS_ALERT_THRESHOLD, HR_HIGH_BPM, HR_DNS_ZONE_BPM
except ImportError:
    from config import REPORTS_DIR, DNS_ALERT_THRESHOLD, HR_HIGH_BPM, HR_DNS_ZONE_BPM

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        SimpleDocTemplate,
    )
except Exception:  # pragma: no cover - fallback when dependency is missing
    colors = None


def _ensure_reports_dir():
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)


def _hr_interpretation(hr_mean):
    if hr_mean < 40:
        return 'Bradicardia severa'
    if hr_mean < 60:
        return 'Bradicardia leve'
    if hr_mean <= HR_DNS_ZONE_BPM[1]:
        return 'Zona sugestiva de DNS'
    if hr_mean <= HR_HIGH_BPM[0]:
        return 'Zona inconclusiva'
    return 'Taquicardia'


def _build_metric_rows(metadata, rr_analysis, features, dns_pred, confidence, interpretation=None):
    duration_s = metadata.get('duration_s', 0) if isinstance(metadata, dict) else rr_analysis.get('duration_s', 0)
    fs_hz = metadata.get('fs_hz', 300) if isinstance(metadata, dict) else rr_analysis.get('fs_hz', 300)
    num_samples = metadata.get('num_samples', 0) if isinstance(metadata, dict) else rr_analysis.get('num_samples', 0)
    derivations = metadata.get('derivations', []) if isinstance(metadata, dict) else rr_analysis.get('derivations', [])

    # Obtener características desde features (puede ser un dict con datos completos o solo features)
    if isinstance(features, dict):
        hr_mean = features.get('hr_mean_bpm', 0)
        rr_max = features.get('rr_max_s', 0)
        sdnn = features.get('sdnn_s', 0)
        rmssd = features.get('rmssd_s', 0)
        # Intentar obtener pausas desde features o rr_analysis
        pauses_gt2s = features.get('pauses_gt2s_per_min', 
                                  features.get('pauses_per_min_gt2s', 
                                  rr_analysis.get('pauses_per_min_gt2s', 0)))
        pauses_gt3s = features.get('pauses_gt3s_per_min',
                                  features.get('pauses_per_min_gt3s',
                                  rr_analysis.get('pauses_per_min_gt3s', 0)))
        pauses_gt4s = features.get('pauses_gt4s_per_min',
                                  features.get('pauses_per_min_gt4s',
                                  rr_analysis.get('pauses_per_min_gt4s', 0)))
        quality = features.get('signal_quality_score', 0)
        count_rr_gt3s = features.get('count_rr_gt3s', 0)
        count_rr_gt4s = features.get('count_rr_gt4s', 0)
        rr_mean = features.get('rr_mean_s', 0)
    else:
        hr_mean = rr_max = sdnn = rmssd = pauses_gt2s = pauses_gt3s = pauses_gt4s = quality = 0
        count_rr_gt3s = count_rr_gt4s = rr_mean = 0

    rows = [
        ['Archivo', os.path.basename(metadata.get('source_file', 'ECG')) if isinstance(metadata, dict) else 'ECG'],
        ['Duracion total (s)', f'{duration_s:.2f}'],
        ['Muestras', f'{num_samples}'],
        ['Frecuencia de muestreo (Hz)', f'{fs_hz}'],
        ['Derivaciones', ', '.join(derivations) if derivations else 'No disponible'],
        ['Calidad de señal (%)', f'{quality:.1f}'],
        ['═══════════════════════════', '═════════════════════'],
        ['METRICAS CARDIACAS', ''],
        ['Frecuencia cardiaca media (bpm)', f'{hr_mean:.1f}'],
        ['Estado HR', _hr_interpretation(hr_mean)],
        ['RR promedio (s)', f'{rr_mean:.3f}'],
        ['RR maximo (s)', f'{rr_max:.2f}'],
        ['═══════════════════════════', '═════════════════════'],
        ['VARIABILIDAD RR (HRV)', ''],
        ['SDNN - Desviacion std de RR (ms)', f'{sdnn*1000:.1f}'],
        ['RMSSD - Raiz cuad media RR (ms)', f'{rmssd*1000:.1f}'],
        ['═══════════════════════════', '═════════════════════'],
        ['PAUSAS SINUSALES (eventos/min)', ''],
        ['Pausas RR > 2 s', f'{pauses_gt2s:.2f}'],
        ['Pausas RR > 3 s', f'{pauses_gt3s:.2f}'],
        ['Pausas RR > 4 s', f'{pauses_gt4s:.2f}'],
        ['Cantidad pausas > 3 s (total)', f'{int(count_rr_gt3s)}'],
        ['Cantidad pausas > 4 s (total)', f'{int(count_rr_gt4s)}'],
        ['═══════════════════════════', '═════════════════════'],
        ['PREDICCION DNS', ''],
        ['Probabilidad DNS (%)', f'{dns_pred:.1f}'],
        ['Confianza de prediccion (%)', f'{confidence:.1f}'],
        ['Umbral clinico (%)', f"{float((interpretation or {}).get('decision_threshold_pct', DNS_ALERT_THRESHOLD)):.1f}"],
        ['Tipo de modelo', (interpretation or {}).get('model_type', 'unknown')],
        ['Ajuste por literatura', 'Si' if (interpretation or {}).get('literature_override', False) else 'No'],
    ]

    return rows


def generate_pdf_report(filename, metadata, rr_analysis, features, dns_pred, confidence, interpretation=None, pdf_filename=None):
    """
    Genera un informe PDF con todas las metricas del ECG.

    Retorna:
        str: Ruta al PDF generado
    """

    if colors is None:
        raise ImportError('reportlab no esta instalado')

    _ensure_reports_dir()

    base_name = os.path.splitext(os.path.basename(filename))[0]
    if pdf_filename:
        pdf_filename = os.path.basename(pdf_filename)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"report_{base_name}_{timestamp}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title='Informe ECG DNS',
        author='Sistema DNS Veterinario',
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='TitleCenter',
        parent=styles['Title'],
        alignment=TA_CENTER,
        fontSize=16,
        leading=20,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        textColor=colors.HexColor('#1F3A5F'),
        spaceBefore=10,
        spaceAfter=6,
    ))

    interpretation = interpretation or {}
    metric_rows = _build_metric_rows(metadata, rr_analysis, features, dns_pred, confidence, interpretation)

    story = []
    story.append(Paragraph('INFORME DE ANALISIS ECG VETERINARIO', styles['TitleCenter']))
    story.append(Paragraph(f'Archivo analizado: {os.path.basename(filename)}', styles['BodyText']))
    story.append(Paragraph(
        f"Fecha de generacion: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        styles['BodyText']
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph('Resumen clinico', styles['SectionTitle']))
    resumen = interpretation.get('estado') or _hr_interpretation(features.get('hr_mean_bpm', 0))
    story.append(Paragraph(f'<b>Estado:</b> {resumen}', styles['BodyText']))
    story.append(Paragraph(
        f"<b>Interpretacion:</b> {interpretation.get('model_explanation') or 'Análisis automático de señales ECG'}",
        styles['BodyText']
    ))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph('Metricas del analisis', styles['SectionTitle']))
    table_data = [['Variable', 'Valor']]
    table_data.extend(metric_rows)

    table = Table(table_data, colWidths=[8.5 * cm, 7.0 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph('Hallazgos clave', styles['SectionTitle']))
    hallazgos = []
    
    # Análisis de HR
    hr_mean = features.get('hr_mean_bpm', 0)
    if hr_mean < 40:
        hallazgos.append('🔴 Bradicardia severa detectada (FC < 40 bpm).')
    elif hr_mean < 60:
        hallazgos.append('🟡 Bradicardia leve detectada (FC 40-60 bpm).')
    
    if hr_mean > 140:
        hallazgos.append('🔴 Taquicardia detectada (FC > 140 bpm).')
    elif hr_mean > 100:
        hallazgos.append('🟡 Frecuencia cardiaca elevada (FC 100-140 bpm).')

    # Análisis de pausas sinusales
    pauses_gt2s = features.get('pauses_gt2s_per_min', 
                              features.get('pauses_per_min_gt2s', 0))
    pauses_gt3s = features.get('pauses_gt3s_per_min',
                              features.get('pauses_per_min_gt3s', 0))
    pauses_gt4s = features.get('pauses_gt4s_per_min',
                              features.get('pauses_per_min_gt4s', 0))
    count_rr_gt4s = features.get('count_rr_gt4s', 0)
    
    if pauses_gt4s > 0 or count_rr_gt4s > 0:
        hallazgos.append(f'🔴 Pausas sinusales SEVERAS detectadas (>4 s): {pauses_gt4s:.2f} eventos/min, {int(count_rr_gt4s)} pausas totales.')
    elif pauses_gt3s >= 3:
        hallazgos.append(f'🟡 Pausas sinusales prolongadas detectadas (>3 s): {pauses_gt3s:.2f} eventos/min.')
    elif pauses_gt2s > 0:
        hallazgos.append(f'🟡 Pausas sinusales detectadas (>2 s): {pauses_gt2s:.2f} eventos/min.')

    # Análisis de variabilidad RR
    sdnn = features.get('sdnn_s', 0)
    rmssd = features.get('rmssd_s', 0)
    
    if sdnn < 0.05:
        hallazgos.append('🟡 Baja variabilidad RR (SDNN < 50 ms) - posible arritmia.')
    elif sdnn > 0.2:
        hallazgos.append('🟢 Variabilidad RR normal (SDNN > 200 ms).')
    
    if rmssd < 0.02:
        hallazgos.append('🟡 Baja variabilidad parasimpática (RMSSD < 20 ms).')

    # Análisis de DNS
    dns_prob = features.get('dns_probability_pct', 0)
    decision_threshold = float((interpretation or {}).get('decision_threshold_pct', DNS_ALERT_THRESHOLD))
    
    if dns_prob >= decision_threshold:
        hallazgos.append(f'🔴 DEPRESION DEL SEGMENTO ST (DNS) DETECTADA: Probabilidad {dns_prob:.1f}% (umbral: {decision_threshold:.1f}%).')
    elif dns_prob > decision_threshold - 20:
        hallazgos.append(f'🟡 Riesgo moderado de DNS: Probabilidad {dns_prob:.1f}% (próximo a umbral: {decision_threshold:.1f}%).')
    else:
        hallazgos.append(f'🟢 Sin DNS detectado: Probabilidad {dns_prob:.1f}% (muy por debajo del umbral: {decision_threshold:.1f}%).')

    # Análisis de calidad
    signal_quality = features.get('signal_quality_score', 0)
    if signal_quality < 50:
        hallazgos.append(f'🔴 CALIDAD DE SEÑAL DEFICIENTE ({signal_quality:.1f}%) - Repita el análisis.')
    elif signal_quality < 70:
        hallazgos.append(f'🟡 Calidad de señal reducida ({signal_quality:.1f}%) - Interpretar con cautela.')
    else:
        hallazgos.append(f'🟢 Calidad de señal aceptable ({signal_quality:.1f}%).')

    # Comparación longitudinal
    comparacion = interpretation.get('comparacion') or {}
    if comparacion:
        tendencia = comparacion.get('tendencia_descripcion', 'No disponible')
        cambio = comparacion.get('cambio_dns_pct', 0)
        hallazgos.append(f'📊 Tendencia longitudinal: {tendencia}.')
        if cambio > 0:
            hallazgos.append(f'📈 Aumento en probabilidad DNS: {cambio:+.1f}%.')
        elif cambio < 0:
            hallazgos.append(f'📉 Disminución en probabilidad DNS: {cambio:+.1f}%.')

    if not hallazgos:
        hallazgos.append('🟢 No se identificaron hallazgos mayores en el ECG analizado.')

    for item in hallazgos:
        story.append(Paragraph(f'{item}', styles['BodyText']))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph('Recomendaciones', styles['SectionTitle']))
    recomendaciones = interpretation.get('recomendaciones') or []
    if recomendaciones:
        for item in recomendaciones:
            story.append(Paragraph(f'• {item}', styles['BodyText']))
    else:
        story.append(Paragraph('• Sin recomendaciones adicionales.', styles['BodyText']))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph('Aclaracion', styles['SectionTitle']))
    story.append(Paragraph(
        'Este informe es una ayuda al diagnostico y no sustituye la evaluacion veterinaria completa.',
        styles['BodyText']
    ))

    doc.build(story)
    logger.info("Reporte PDF guardado: %s", pdf_path)
    return pdf_path

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
    report_filename = "report_{}_{}.txt".format(
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
