#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_ml_model.py - Entrenamiento del modelo ML para prediccion DNS

REQUISITOS MINIMOS:
- 30+ ECGs con DNS confirmado (diagnostico_dns_pct >= 50)
- 30+ ECGs sanos confirmados (diagnostico_dns_pct < 50)
- Total: 60+ ECGs etiquetados en la base de datos

COMO USAR:
1. Asegurate de tener ECGs etiquetados en PostgreSQL
2. Ejecuta: py -3 train_ml_model.py
3. El modelo entrenado se guarda en: models/dns_ml_model.pkl
4. La API automaticamente usara el modelo ML si existe
"""

import sys
import os
import numpy as np
from datetime import datetime
import glob
import re
import csv

# Agregar paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ecg_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))

from preprocessing import process_ecg_file, select_best_derivation
from peak_detection import detect_r_peaks
from rr_analysis import calculate_rr_intervals, analyze_rr_intervals
from features import extract_features
from postgres import Postgres
from config import (
    DNS_ALERT_THRESHOLD,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
)

# Importar sklearn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    import joblib
except ImportError:
    print("\n❌ ERROR: sklearn no instalado")
    print("Instala con: pip install scikit-learn joblib")
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

        # Soporta separadores comunes en exportaciones sintéticas.
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


def procesar_ecg_para_features(csv_path):
    """Procesa un ECG y extrae features para ML"""
    try:
        # Pipeline completo
        result = process_ecg_file(csv_path)
        signal_norm = result['signal_normalized']
        duration_s = result['duration_s']
        fs_hz = result['fs_hz']
        
        # Detectar picos
        peaks_seconds = detect_r_peaks(signal_norm, fs_hz)
        
        if len(peaks_seconds) < 3:
            return None
        
        # Analizar RR
        rr_intervals_s, valid_rr_fraction = calculate_rr_intervals(peaks_seconds)
        rr_analysis = analyze_rr_intervals(rr_intervals_s, duration_s)
        
        # Extraer features
        features = extract_features(rr_analysis, valid_rr_fraction, duration_s)
        
        return features
        
    except Exception as e:
        synthetic_features = procesar_ecg_sintetico_para_features(csv_path, SYNTHETIC_FS_HZ)
        if synthetic_features is not None:
            print(f"  ℹ️ Fallback sintético aplicado en {os.path.basename(csv_path)}")
            return synthetic_features
        print(f"  ⚠️ Error procesando {os.path.basename(csv_path)}: {e}")
        return None


def _parse_float_maybe(value, default=0.0):
    """Convierte valor potencialmente textual/decimal a float."""
    if value is None:
        return default
    try:
        return float(str(value).replace(',', '.').strip())
    except (TypeError, ValueError):
        return default


def diagnostico_a_dns_pct(raw_value):
    """Normaliza diagnóstico textual/numérico a escala 0-100."""
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
    """Detecta esquema disponible en la base para entrenamiento."""
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
    has_dns_pct = cursor.fetchone()[0]

    return {
        'has_dns_pct': has_dns_pct,
    }


def procesar_ecg_desde_bd(db, id_examen):
    """
    Fallback: procesa ECG directamente desde infoexamen_excel/metadatos_examen.
    """
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
            # Fallback robusto para históricos sin metadatos_examen.
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
        features = extract_features(rr_analysis, valid_rr_fraction, duration_s)
        return features
    except Exception:
        return None
    finally:
        cursor.close()


def extraer_vector_features(features_dict):
    """Convierte dict de features a vector numpy para ML"""
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
    'hr_mean_bpm',
    'sdnn_s',
    'rmssd_s',
    'rr_max_s',
    'count_rr_gt3s',
    'count_rr_gt4s',
    'pauses_per_min_gt3s',
    'pauses_per_min_gt4s',
    'signal_quality_score',
    'ecg_duration_s'
]


def _is_filtered_path(path):
    """Heurística para detectar si una ruta apunta a un archivo filtrado."""
    if not path:
        return False
    return bool(re.search(r'(filtrad|filtered|filtro)', path, re.IGNORECASE))


def _build_candidate_paths(base_dir, nombre):
    """Genera candidatos de CSV filtrado y no filtrado por nombre de paciente."""
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
    """
    Resuelve la ruta del ECG priorizando archivos filtrados.

    Busca por nombre de paciente en carpetas de ECG configuradas.
    """
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


def cargar_datos_entrenamiento():
    """
    Carga ECGs etiquetados desde PostgreSQL y procesa features.
    
    Retorna:
        X: matriz de features (n_samples, n_features)
        y: labels (0 = sano, 1 = DNS)
        metadatos: info de cada ECG
    """
    
    print("\n" + "="*70)
    print("CARGANDO DATOS DE ENTRENAMIENTO DESDE POSTGRESQL")
    print("="*70)
    
    db = Postgres()
    
    # Obtener todos los examenes etiquetados
    if db.connection is None:
        db.connect()
    
    if db.connection is None:
        print("❌ Error: No se pudo conectar a PostgreSQL")
        return None, None, None
    
    cursor = db.connection.cursor()
    schema = obtener_dataset_schema(cursor)

    if schema['has_dns_pct']:
        query = f"""
            SELECT
                de.id_examen,
                de.id_paciente,
                de.diagnostico_dns_pct AS dns_raw,
                p.nombre,
                NULL::text AS ruta_ref
            FROM datos_examen de
            JOIN pacientes p ON de.id_paciente = p.id_paciente
            WHERE de.diagnostico_dns_pct IS NOT NULL
            ORDER BY de.id_examen
        """
    else:
        query = f"""
            SELECT
                de.id_examen,
                de.id_paciente,
                de.diagnostico AS dns_raw,
                p.nombre,
                NULL::text AS ruta_ref
            FROM datos_examen de
            JOIN pacientes p ON de.id_paciente = p.id_paciente
            WHERE de.diagnostico IS NOT NULL
            ORDER BY de.id_examen
        """

    cursor.execute(query)
    examenes_raw = cursor.fetchall()
    cursor.close()

    examenes = []
    for row in examenes_raw:
        dns_pct = diagnostico_a_dns_pct(row[2])
        if dns_pct is None:
            continue
        examenes.append((row[0], row[1], dns_pct, row[3], row[4]))
    
    if len(examenes) == 0:
        print("❌ No hay ECGs etiquetados en la base de datos")
        print("\nPara entrenar el modelo necesitas:")
        print("  1. Subir ECGs via API /diagnosticar")
        print("  2. Actualizar diagnostico_dns_pct manualmente:")
        print("     UPDATE datos_examen SET diagnostico_dns_pct = 0 WHERE id_examen = X;  -- Sano")
        print("     UPDATE datos_examen SET diagnostico_dns_pct = 100 WHERE id_examen = Y; -- DNS")
        return None, None, None
    
    print(f"\n✓ Encontrados {len(examenes)} ECGs etiquetados")
    
    # Contar casos
    dns_cases = sum(1 for e in examenes if e[2] >= 50)
    sano_cases = sum(1 for e in examenes if e[2] < 50)
    
    print(f"  - DNS confirmado (>={DNS_ALERT_THRESHOLD}%): {dns_cases}")
    print(f"  - Sanos confirmados (<{DNS_ALERT_THRESHOLD}%): {sano_cases}")
    
    if dns_cases < 10 or sano_cases < 10:
        print("\n⚠️ ADVERTENCIA: Pocos datos para entrenamiento robusto")
        print(f"   Recomendado: 30+ de cada clase")
        print(f"   Mínimo funcional: 10+ de cada clase")
        
        if dns_cases < 5 or sano_cases < 5:
            print("\n❌ INSUFICIENTES DATOS - Se requieren al menos 5 de cada clase")
            return None, None, None
    
    # Procesar cada ECG
    X_list = []
    y_list = []
    metadata_list = []
    filtered_count = 0
    from_db_attempted = 0
    from_db_used = 0
    
    print("\nProcesando ECGs...")
    for idx, (id_examen, id_paciente, dns_pct, nombre, ruta_ref) in enumerate(examenes, 1):
        csv_path, is_filtered = resolver_csv_path_prefer_filtrado(nombre)
        
        if csv_path and os.path.exists(csv_path):
            features = procesar_ecg_para_features(csv_path)
        else:
            from_db_attempted += 1
            features = procesar_ecg_desde_bd(db, id_examen)
            if features is not None:
                from_db_used += 1
        
        if features is None:
            print(f"  [{idx}/{len(examenes)}] ❌ {nombre}: error al procesar")
            continue
        
        # Extraer vector
        x_vector = extraer_vector_features(features)
        
        # Label binario según umbral clínico centralizado.
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
        
        estado = "DNS" if y_label == 1 else "SANO"
        origen_txt = "csv_filtrado" if is_filtered else ("csv" if csv_path else "db_excel")
        print(f"  [{idx}/{len(examenes)}] ✓ {nombre}: {estado} ({dns_pct:.0f}%) [{origen_txt}]")
    
    if len(X_list) == 0:
        print("\n❌ No se pudo procesar ningún ECG")
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✓ Dataset creado: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"  - Clase 0 (SANO): {np.sum(y == 0)}")
    print(f"  - Clase 1 (DNS): {np.sum(y == 1)}")
    print(f"  - Rutas filtradas usadas: {filtered_count}/{len(X_list)}")
    print(f"  - Señales desde BD intentadas: {from_db_attempted}")
    print(f"  - Señales desde BD usadas: {from_db_used}")
    
    return X, y, metadata_list


def entrenar_modelo(X, y):
    """
    Entrena modelo RandomForest y evalua con Leave-One-Out.
    """
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL MODELO ML")
    print("="*70)
    
    print("\n[1/3] Configurando Random Forest con parámetros de literatura...")
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RF_RANDOM_STATE,
        class_weight='balanced'  # Importante si hay desbalance
    )

    # Leave-One-Out: cada ECG se evalúa dejando 1 fuera, ideal para datasets pequeños.
    print("\n[2/3] Validación Leave-One-Out...")
    loo = LeaveOneOut()
    y_pred = cross_val_predict(rf_model, X, y, cv=loo, method='predict')
    y_pred_proba = cross_val_predict(rf_model, X, y, cv=loo, method='predict_proba')[:, 1]

    print("\n" + "-"*70)
    print("EVALUACION LEAVE-ONE-OUT")
    print("-"*70)

    print("\nMatriz de Confusion:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print("\nMetricas:")
    print(classification_report(y, y_pred, target_names=['SANO', 'DNS']))

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision (DNS): {precision:.3f}")
    print(f"  Recall (DNS): {recall:.3f}")
    print(f"  F1 (DNS): {f1:.3f}")

    auc = None
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, y_pred_proba)
        print(f"\nROC-AUC (LOO): {auc:.3f}")

    # Buscar umbral operativo que priorice precision DNS >= 0.85.
    threshold_candidates = np.linspace(0.05, 0.95, 91)
    candidate_rows = []
    for th in threshold_candidates:
        y_hat = (y_pred_proba >= th).astype(int)
        p = precision_score(y, y_hat, zero_division=0)
        r = recall_score(y, y_hat, zero_division=0)
        f1_th = f1_score(y, y_hat, zero_division=0)
        acc_th = accuracy_score(y, y_hat)
        candidate_rows.append((th, p, r, f1_th, acc_th))

    target_precision = 0.85
    eligible = [row for row in candidate_rows if row[1] >= target_precision]
    if eligible:
        best_row = max(eligible, key=lambda row: (row[2], row[3], -row[0]))
    else:
        best_row = max(candidate_rows, key=lambda row: (row[1], row[3], row[4]))

    recommended_threshold = float(best_row[0])
    print("\nUmbral operativo recomendado (precision-prioritario):")
    print(f"  Threshold: {recommended_threshold:.2f}")
    print(f"  Precision: {best_row[1]:.3f}")
    print(f"  Recall: {best_row[2]:.3f}")
    print(f"  F1: {best_row[3]:.3f}")
    print(f"  Accuracy: {best_row[4]:.3f}")

    # Entrenar modelo final con todos los datos para despliegue.
    print("\n[3/3] Entrenando modelo final con dataset completo...")
    rf_model.fit(X, y)
    
    # Importancia de features
    print("\n" + "-"*70)
    print("IMPORTANCIA DE FEATURES")
    print("-"*70)
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i, idx in enumerate(indices[:10], 1):
        print(f"  {i}. {FEATURE_NAMES[idx]}: {importances[idx]:.3f}")
    
    evaluation_metrics = {
        'validation': 'LeaveOneOut',
        'accuracy': float(accuracy),
        'precision_dns': float(precision),
        'recall_dns': float(recall),
        'f1_dns': float(f1),
        'roc_auc': float(auc) if auc is not None else None,
        'n_samples': int(X.shape[0]),
        'target_precision_dns': float(target_precision),
        'recommended_threshold_pct': float(recommended_threshold * 100.0),
        'recommended_threshold_precision_dns': float(best_row[1]),
        'recommended_threshold_recall_dns': float(best_row[2]),
        'recommended_threshold_f1_dns': float(best_row[3]),
        'recommended_threshold_accuracy': float(best_row[4]),
    }

    return rf_model, evaluation_metrics


def guardar_modelo(model, evaluation_metrics, output_path="models/dns_ml_model.pkl"):
    """Guarda modelo entrenado"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    metadata = {
        'model': model,
        'feature_names': FEATURE_NAMES,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'dns_label_threshold_pct': DNS_ALERT_THRESHOLD,
        'decision_threshold_pct': float(evaluation_metrics.get('recommended_threshold_pct', DNS_ALERT_THRESHOLD)),
        'evaluation_metrics': evaluation_metrics,
    }
    
    joblib.dump(metadata, output_path)
    print(f"\n✓ Modelo guardado en: {output_path}")
    print(f"  Tamaño: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    print("\n" + "="*70)
    print("SISTEMA DE ENTRENAMIENTO ML - DETECCION DNS")
    print("="*70)
    
    # Cargar datos
    X, y, _ = cargar_datos_entrenamiento()
    
    if X is None:
        print("\n❌ No se pudieron cargar datos de entrenamiento")
        print("\nVerifica que:")
        print("  1. PostgreSQL esté corriendo")
        print("  2. Haya ECGs etiquetados en datos_examen")
        print("  3. Los archivos CSV existan en el sistema")
        return
    
    # Entrenar
    model, evaluation_metrics = entrenar_modelo(X, y)
    
    # Guardar
    guardar_modelo(model, evaluation_metrics)
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print("\nProximos pasos:")
    print("  1. El modelo ML ya está guardado")
    print("  2. La API lo usará automaticamente si existe")
    print("  3. Para re-entrenar con más datos: py -3 train_ml_model.py")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
