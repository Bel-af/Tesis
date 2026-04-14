#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_model.py - Validación y análisis del modelo ML

Este script evalúa el rendimiento del modelo ML entrenado:
- Métricas de clasificación
- Curva ROC
- Matriz de confusión
- Análisis de casos mal clasificados

COMO USAR:
  py -3 validate_model.py
"""

import sys
import os
import numpy as np
import joblib

# Agregar paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ecg_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))

from postgres import Postgres
from preprocessing import process_ecg_file
from peak_detection import detect_r_peaks
from rr_analysis import calculate_rr_intervals, analyze_rr_intervals
from features import extract_features

try:
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        roc_auc_score, roc_curve, precision_recall_curve
    )
except ImportError:
    print("❌ sklearn no instalado: pip install scikit-learn")
    sys.exit(1)


MODEL_PATH = 'models/dns_ml_model.pkl'


def procesar_ecg_para_features(csv_path):
    """Procesa un ECG y extrae features"""
    try:
        result = process_ecg_file(csv_path)
        signal_norm = result['signal_normalized']
        duration_s = result['duration_s']
        fs_hz = result['fs_hz']
        
        peaks_seconds = detect_r_peaks(signal_norm, fs_hz)
        
        if len(peaks_seconds) < 3:
            return None
        
        rr_intervals_s, valid_rr_fraction = calculate_rr_intervals(peaks_seconds)
        rr_analysis = analyze_rr_intervals(rr_intervals_s, duration_s)
        features = extract_features(rr_analysis, valid_rr_fraction, duration_s)
        
        return features
        
    except Exception as e:
        return None


def extraer_vector_features(features_dict):
    """Convierte dict a vector numpy"""
    return np.array([
        features_dict.get('hr_mean_bpm', 0),
        features_dict.get('sdnn_s', 0),
        features_dict.get('rmssd_s', 0),
        features_dict.get('rr_max_s', 0),
        features_dict.get('count_rr_gt3s', 0),
        features_dict.get('count_rr_gt4s', 0),
        features_dict.get('pauses_per_min_gt3s', 0),
        features_dict.get('pauses_per_min_gt4s', 0),
        features_dict.get('signal_quality_score', 0),
        features_dict.get('ecg_duration_s', 0)
    ])


FEATURE_NAMES = [
    'hr_mean_bpm', 'sdnn_s', 'rmssd_s', 'rr_max_s',
    'count_rr_gt3s', 'count_rr_gt4s', 'pauses_per_min_gt3s',
    'pauses_per_min_gt4s', 'signal_quality_score', 'ecg_duration_s'
]


def cargar_dataset():
    """Carga todos los ECGs etiquetados desde PostgreSQL"""
    
    print("\n" + "="*70)
    print("CARGANDO DATASET")
    print("="*70)
    
    db = Postgres()
    if db.connection is None:
        db.connect()
    
    cursor = db.connection.cursor()
    cursor.execute("""
        SELECT 
            de.id_examen,
            de.diagnostico_dns_pct,
            p.nombre,
            pdf.ruta_pdf
        FROM datos_examen de
        JOIN pacientes p ON de.id_paciente = p.id_paciente
        LEFT JOIN informacion_examen_pdf pdf ON de.id_examen = pdf.id_examen
        WHERE de.diagnostico_dns_pct IS NOT NULL
    """)
    
    examenes = cursor.fetchall()
    cursor.close()
    
    if len(examenes) == 0:
        print("❌ No hay datos etiquetados")
        return None, None, None
    
    print(f"\n✓ {len(examenes)} ECGs etiquetados encontrados")
    
    X_list = []
    y_list = []
    metadata_list = []
    
    for id_examen, dns_pct, nombre, ruta_pdf in examenes:
        
        # Buscar archivo CSV
        if ruta_pdf and os.path.exists(ruta_pdf):
            csv_path = ruta_pdf
        else:
            posibles = [
                f"C:\\Users\\isabe\\OneDrive\\Documentos\\EasyG - Informes\\{nombre}.csv",
                f"C:\\Users\\isabe\\OneDrive\\Escritorio\\registros\\{nombre}.csv",
            ]
            csv_path = next((p for p in posibles if os.path.exists(p)), None)
        
        if not csv_path:
            continue
        
        features = procesar_ecg_para_features(csv_path)
        if features is None:
            continue
        
        x_vector = extraer_vector_features(features)
        y_label = 1 if dns_pct >= 50 else 0
        
        X_list.append(x_vector)
        y_list.append(y_label)
        metadata_list.append({
            'id_examen': id_examen,
            'nombre': nombre,
            'dns_pct': dns_pct,
            'csv_path': csv_path
        })
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✓ Dataset: {X.shape[0]} muestras procesadas")
    print(f"  - Clase 0 (SANO): {np.sum(y == 0)}")
    print(f"  - Clase 1 (DNS): {np.sum(y == 1)}")
    
    return X, y, metadata_list


def validar_modelo(model, X, y, metadata):
    """Realiza validación completa del modelo"""
    
    print("\n" + "="*70)
    print("VALIDACION DEL MODELO")
    print("="*70)
    
    # 1. Validación cruzada
    print("\n[1/5] Validación Cruzada (5-fold)...")
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
    
    print("\nMétricas (media ± std):")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"  {metric.upper():12s}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # 2. Predicciones
    print("\n[2/5] Generando predicciones...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 3. Matriz de confusión
    print("\n[3/5] Matriz de Confusión:")
    cm = confusion_matrix(y, y_pred)
    print("\n              Predicho")
    print("              SANO  DNS")
    print(f"Real  SANO     {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"      DNS      {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # 4. Reporte de clasificación
    print("\n[4/5] Reporte de Clasificación:")
    print(classification_report(y, y_pred, target_names=['SANO', 'DNS']))
    
    # 5. ROC-AUC
    print("\n[5/5] Curva ROC...")
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, y_pred_proba)
        print(f"  ROC-AUC: {auc:.3f}")
        
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        
        # Encontrar mejor threshold (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        
        print(f"  Mejor threshold: {best_threshold:.3f}")
        print(f"  En ese punto: TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f}")
    
    # Análisis de casos mal clasificados
    print("\n" + "="*70)
    print("ANALISIS DE CASOS MAL CLASIFICADOS")
    print("="*70)
    
    errores = []
    for i in range(len(y)):
        if y[i] != y_pred[i]:
            errores.append({
                'nombre': metadata[i]['nombre'],
                'real': 'DNS' if y[i] == 1 else 'SANO',
                'predicho': 'DNS' if y_pred[i] == 1 else 'SANO',
                'dns_real_pct': metadata[i]['dns_pct'],
                'proba_predicha': y_pred_proba[i] * 100
            })
    
    if len(errores) == 0:
        print("\n✓ ¡Clasificación perfecta! Sin errores.")
    else:
        print(f"\n⚠️ {len(errores)} errores encontrados:")
        for err in errores:
            print(f"\n  {err['nombre']}:")
            print(f"    Real: {err['real']} ({err['dns_real_pct']:.0f}%)")
            print(f"    Predicho: {err['predicho']} ({err['proba_predicha']:.1f}%)")
    
    return cv_results, errores


def analizar_importancia_features(model):
    """Analiza importancia de features del modelo"""
    
    print("\n" + "="*70)
    print("IMPORTANCIA DE FEATURES")
    print("="*70)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nRanking de importancia:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {FEATURE_NAMES[idx]:25s} {importances[idx]:.4f}")


def main():
    
    # Verificar que modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Modelo no encontrado: {MODEL_PATH}")
        print("Entrena el modelo primero: py -3 train_ml_model.py")
        return
    
    # Cargar modelo
    print("\n" + "="*70)
    print("VALIDACION DE MODELO ML - DETECCION DNS")
    print("="*70)
    
    print(f"\nCargando modelo desde: {MODEL_PATH}")
    metadata_model = joblib.load(MODEL_PATH)
    model = metadata_model['model']
    
    print(f"  Tipo: {metadata_model.get('model_type', 'unknown')}")
    print(f"  Entrenado: {metadata_model.get('trained_at', 'unknown')}")
    
    # Cargar dataset
    X, y, metadata = cargar_dataset()
    
    if X is None:
        print("\n❌ No se pudo cargar dataset")
        return
    
    # Validar
    cv_results, errores = validar_modelo(model, X, y, metadata)
    
    # Importancia de features
    analizar_importancia_features(model)
    
    print("\n" + "="*70)
    print("✅ VALIDACION COMPLETADA")
    print("="*70)
    
    # Resumen
    print("\nRESUMEN:")
    print(f"  Dataset: {len(X)} muestras")
    print(f"  Accuracy: {cv_results['test_accuracy'].mean():.3f}")
    print(f"  ROC-AUC: {cv_results['test_roc_auc'].mean():.3f}")
    print(f"  Errores: {len(errores)}/{len(X)}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
