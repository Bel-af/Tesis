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
import glob
import re
import csv

# Agregar paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ecg_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))

from postgres import Postgres
from preprocessing import process_ecg_file, select_best_derivation
from peak_detection import detect_r_peaks
from rr_analysis import calculate_rr_intervals, analyze_rr_intervals
from features import extract_features
from config import DNS_ALERT_THRESHOLD

try:
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
except ImportError:
    print("❌ sklearn no instalado: pip install scikit-learn")
    sys.exit(1)


SYNTHETIC_FS_HZ = 300


def _parse_csv_numeric_matrix_loose(csv_path):
    """Parsea CSV sintético tolerando cabeceras y filas no numéricas."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    content = None

    for enc in encodings:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                content = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        return None

    rows = []
    expected_cols = None
    for line in content:
        line = line.replace('\xa0', ' ').strip()
        if not line or line.startswith('%'):
            continue

        delimiters = [',', ';', '\t']
        parts = [line]
        for d in delimiters:
            if d in line:
                parts = [p.strip() for p in line.split(d)]
                break

        values = []
        valid_row = True
        for p in parts:
            if not p:
                continue
            try:
                values.append(float(p.replace(',', '.')))
            except ValueError:
                valid_row = False
                break

        if not valid_row or not values:
            continue

        if expected_cols is None:
            expected_cols = len(values)

        if len(values) != expected_cols:
            continue

        rows.append(values)

    if not rows:
        return None

    return np.array(rows, dtype=np.float32)


def procesar_ecg_sintetico_para_features(csv_path, fs_hz):
    """Fallback opcional para CSVs sintéticos sin metadatos EasyG."""
    try:
        raw = _parse_csv_numeric_matrix_loose(csv_path)
        if raw is None or raw.shape[0] < 16:
            return None

        derivations = [f"D{i+1}" for i in range(raw.shape[1])]
        signal_norm, _ = select_best_derivation(raw, derivations, fs_hz)
        duration_s = float(raw.shape[0]) / float(fs_hz)

        peaks_seconds = detect_r_peaks(signal_norm, fs_hz)
        if len(peaks_seconds) < 3:
            return None

        rr_intervals_s, valid_rr_fraction = calculate_rr_intervals(peaks_seconds)
        rr_analysis = analyze_rr_intervals(rr_intervals_s, duration_s)
        features = extract_features(rr_analysis, valid_rr_fraction, duration_s)
        if features is not None:
            features['_parsed_from_synthetic'] = True
        return features
    except Exception:
        return None


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
        synthetic_features = procesar_ecg_sintetico_para_features(csv_path, SYNTHETIC_FS_HZ)
        if synthetic_features is not None:
            return synthetic_features
        return None


def _parse_float_maybe(value, default=0.0):
    if value is None:
        return default
    try:
        return float(str(value).replace(',', '.').strip())
    except (TypeError, ValueError):
        return default


def diagnostico_a_dns_pct(raw_value):
    if raw_value is None:
        return None

    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    text = str(raw_value).strip()
    if not text:
        return None

    low = text.lower()
    if 'dns' in low:
        return 100.0
    if 'sano' in low or 'normal' in low:
        return 0.0

    match = re.search(r'(-?\d+(?:[\.,]\d+)?)', text)
    if match:
        return _parse_float_maybe(match.group(1), default=None)
    return None


def obtener_dataset_schema(cursor):
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='datos_examen'
              AND column_name='diagnostico_dns_pct'
        )
        """
    )
    return {'has_dns_pct': cursor.fetchone()[0]}


def procesar_ecg_desde_bd(db, id_examen):
    if db.connection is None:
        return None

    cursor = db.connection.cursor()
    try:
        cursor.execute(
            """
            SELECT frecuencia_muestreo, "LSB", orden
            FROM metadatos_examen
            WHERE id_examen = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (id_examen,)
        )
        metadata_row = cursor.fetchone()
        if metadata_row:
            fs_hz = int(metadata_row[0]) if metadata_row[0] else 300
            lsb_uv = _parse_float_maybe(metadata_row[1], default=1.0)
            deriv_str = metadata_row[2] or ''
            derivations = [d.strip() for d in str(deriv_str).split(',') if d.strip()]
        else:
            fs_hz = 300
            lsb_uv = 1.0
            derivations = []

        cursor.execute(
            """
            SELECT "F1", "F2", "F3", "F4", "F5", "F6", "F7"
            FROM infoexamen_excel
            WHERE id_examen = %s
            ORDER BY id
            """,
            (id_examen,)
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        ecg_raw = np.array([
            [_parse_float_maybe(v, default=0.0) for v in row]
            for row in rows
        ], dtype=np.float32)
        ecg_uv = ecg_raw * float(lsb_uv if lsb_uv else 1.0)

        if not derivations:
            derivations = [f"D{i+1}" for i in range(ecg_uv.shape[1])]

        signal_norm, _ = select_best_derivation(ecg_uv, derivations, fs_hz)
        duration_s = float(ecg_uv.shape[0]) / float(fs_hz)

        peaks_seconds = detect_r_peaks(signal_norm, fs_hz)
        if len(peaks_seconds) < 3:
            return None

        rr_intervals_s, valid_rr_fraction = calculate_rr_intervals(peaks_seconds)
        rr_analysis = analyze_rr_intervals(rr_intervals_s, duration_s)
        return extract_features(rr_analysis, valid_rr_fraction, duration_s)
    except Exception:
        return None
    finally:
        cursor.close()


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


def _is_filtered_path(path):
    return bool(path and re.search(r'(filtrad|filtered|filtro)', path, re.IGNORECASE))


def _build_candidate_paths(base_dir, nombre):
    if not base_dir:
        return []

    candidates = [
        os.path.join(base_dir, f"{nombre}_filtrado.csv"),
        os.path.join(base_dir, f"{nombre}-filtrado.csv"),
        os.path.join(base_dir, f"{nombre} filtrado.csv"),
        os.path.join(base_dir, f"{nombre}_filtered.csv"),
        os.path.join(base_dir, f"{nombre}.csv"),
    ]

    wildcard_patterns = [
        os.path.join(base_dir, f"*{nombre}*filtrad*.csv"),
        os.path.join(base_dir, f"*{nombre}*filter*.csv"),
        os.path.join(base_dir, f"*{nombre}*.csv"),
    ]
    for pattern in wildcard_patterns:
        candidates.extend(glob.glob(pattern))

    return candidates


def resolver_csv_path_prefer_filtrado(nombre):
    candidate_paths = []

    env_dirs = os.getenv('ECG_FILTERED_DIRS', '')
    env_dirs_list = [d.strip() for d in env_dirs.split(';') if d.strip()]
    default_dirs = [
        r"C:\Users\isabe\OneDrive\Escritorio\registros_filtrados",
        r"C:\Users\isabe\OneDrive\Documentos\EasyG - Informes\Filtrados",
        r"C:\Users\isabe\OneDrive\Documentos\EasyG - Informes",
        r"C:\Users\isabe\OneDrive\Escritorio\registros",
    ]

    for directory in env_dirs_list + default_dirs:
        candidate_paths.extend(_build_candidate_paths(directory, nombre))

    existing = []
    seen = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            existing.append(path)

    if not existing:
        return None, False

    existing.sort(key=lambda p: (0 if _is_filtered_path(p) else 1, len(p)))
    selected = existing[0]
    return selected, _is_filtered_path(selected)


def cargar_dataset():
    """Carga todos los ECGs etiquetados desde PostgreSQL"""
    
    print("\n" + "="*70)
    print("CARGANDO DATASET")
    print("="*70)
    
    db = Postgres()
    if db.connection is None:
        db.connect()

    cursor = db.connection.cursor()
    schema = obtener_dataset_schema(cursor)

    if schema['has_dns_pct']:
        cursor.execute(
            """
            SELECT
                de.id_examen,
                de.diagnostico_dns_pct AS dns_raw,
                p.nombre,
                NULL::text AS ruta_ref
            FROM datos_examen de
            JOIN pacientes p ON de.id_paciente = p.id_paciente
            WHERE de.diagnostico_dns_pct IS NOT NULL
            ORDER BY de.id_examen
            """
        )
    else:
        cursor.execute(
            """
            SELECT
                de.id_examen,
                de.diagnostico AS dns_raw,
                p.nombre,
                NULL::text AS ruta_ref
            FROM datos_examen de
            JOIN pacientes p ON de.id_paciente = p.id_paciente
            WHERE de.diagnostico IS NOT NULL
            ORDER BY de.id_examen
            """
        )

    examenes_raw = cursor.fetchall()
    cursor.close()

    examenes = []
    for id_examen, dns_raw, nombre, ruta_ref in examenes_raw:
        dns_pct = diagnostico_a_dns_pct(dns_raw)
        if dns_pct is None:
            continue
        examenes.append((id_examen, dns_pct, nombre, ruta_ref))
    
    if len(examenes) == 0:
        print("❌ No hay datos etiquetados")
        return None, None, None
    
    print(f"\n✓ {len(examenes)} ECGs etiquetados encontrados")
    
    X_list = []
    y_list = []
    metadata_list = []
    
    filtered_count = 0
    from_db_count = 0
    for id_examen, dns_pct, nombre, ruta_ref in examenes:
        csv_path, is_filtered = resolver_csv_path_prefer_filtrado(nombre)

        if csv_path:
            features = procesar_ecg_para_features(csv_path)
        else:
            features = procesar_ecg_desde_bd(db, id_examen)
            from_db_count += 1

        if features is None:
            continue
        
        x_vector = extraer_vector_features(features)
        y_label = 1 if dns_pct >= DNS_ALERT_THRESHOLD else 0
        if is_filtered:
            filtered_count += 1
        
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
    print(f"  - Rutas filtradas usadas: {filtered_count}/{len(X_list)}")
    print(f"  - Señales tomadas directo de BD: {from_db_count}/{len(X_list)}")
    
    return X, y, metadata_list


def validar_modelo(model, X, y, metadata):
    """Realiza validación completa del modelo"""
    
    print("\n" + "="*70)
    print("VALIDACION DEL MODELO")
    print("="*70)
    
    # 1. Validación cruzada Leave-One-Out
    print("\n[1/5] Validación Leave-One-Out...")
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo, method='predict')
    y_pred_proba = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print("\nMétricas LOO:")
    print(f"  ACCURACY    : {accuracy:.3f}")
    print(f"  PRECISION   : {precision:.3f}")
    print(f"  RECALL      : {recall:.3f}")
    print(f"  F1          : {f1:.3f}")

    # 2. Predicciones
    print("\n[2/5] Predicciones LOO completadas")
    
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
    
    cv_results = {
        'test_accuracy': np.array([accuracy], dtype=float),
        'test_precision': np.array([precision], dtype=float),
        'test_recall': np.array([recall], dtype=float),
        'test_f1': np.array([f1], dtype=float),
        'test_roc_auc': np.array([auc], dtype=float) if len(np.unique(y)) > 1 else np.array([np.nan], dtype=float),
    }
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
