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
import pickle
import numpy as np
from datetime import datetime

# Agregar paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ecg_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))

from preprocessing import process_ecg_file
from peak_detection import detect_r_peaks
from rr_analysis import calculate_rr_intervals, analyze_rr_intervals
from features import extract_features
from postgres import Postgres

# Importar sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import joblib
except ImportError:
    print("\n❌ ERROR: sklearn no instalado")
    print("Instala con: pip install scikit-learn joblib")
    sys.exit(1)


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
        print(f"  ⚠️ Error procesando {os.path.basename(csv_path)}: {e}")
        return None


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
    cursor.execute("""
        SELECT 
            de.id_examen,
            de.id_paciente,
            de.diagnostico_dns_pct,
            p.nombre,
            pdf.ruta_pdf
        FROM datos_examen de
        JOIN pacientes p ON de.id_paciente = p.id_paciente
        LEFT JOIN informacion_examen_pdf pdf ON de.id_examen = pdf.id_examen
        WHERE de.diagnostico_dns_pct IS NOT NULL
        ORDER BY de.diagnostico_dns_pct
    """)
    
    examenes = cursor.fetchall()
    cursor.close()
    
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
    
    print(f"  - DNS confirmado (>=50%): {dns_cases}")
    print(f"  - Sanos confirmados (<50%): {sano_cases}")
    
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
    
    print("\nProcesando ECGs...")
    for idx, (id_examen, id_paciente, dns_pct, nombre, ruta_pdf) in enumerate(examenes, 1):
        
        # Necesitamos la ruta al archivo CSV
        # Si no está en ruta_pdf, buscar en directorio conocido
        if ruta_pdf and os.path.exists(ruta_pdf):
            csv_path = ruta_pdf
        else:
            # Intentar buscar en directorios comunes
            posibles_rutas = [
                f"C:\\Users\\isabe\\OneDrive\\Documentos\\EasyG - Informes\\{nombre}.csv",
                f"C:\\Users\\isabe\\OneDrive\\Escritorio\\registros\\{nombre}.csv",
            ]
            csv_path = None
            for ruta in posibles_rutas:
                if os.path.exists(ruta):
                    csv_path = ruta
                    break
        
        if not csv_path or not os.path.exists(csv_path):
            print(f"  [{idx}/{len(examenes)}] ⚠️ {nombre}: archivo CSV no encontrado")
            continue
        
        # Procesar
        features = procesar_ecg_para_features(csv_path)
        
        if features is None:
            print(f"  [{idx}/{len(examenes)}] ❌ {nombre}: error al procesar")
            continue
        
        # Extraer vector
        x_vector = extraer_vector_features(features)
        
        # Label: 1 si DNS >= 50%, 0 si < 50%
        y_label = 1 if dns_pct >= 50 else 0
        
        X_list.append(x_vector)
        y_list.append(y_label)
        metadata_list.append({
            'id_examen': id_examen,
            'nombre': nombre,
            'dns_pct': dns_pct,
            'csv_path': csv_path
        })
        
        estado = "DNS" if y_label == 1 else "SANO"
        print(f"  [{idx}/{len(examenes)}] ✓ {nombre}: {estado} ({dns_pct:.0f}%)")
    
    if len(X_list) == 0:
        print("\n❌ No se pudo procesar ningún ECG")
        return None, None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✓ Dataset creado: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"  - Clase 0 (SANO): {np.sum(y == 0)}")
    print(f"  - Clase 1 (DNS): {np.sum(y == 1)}")
    
    return X, y, metadata_list


def entrenar_modelo(X, y):
    """
    Entrena modelo RandomForest con validacion cruzada.
    """
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL MODELO ML")
    print("="*70)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba: {X_test.shape[0]} muestras")
    
    # Entrenar RandomForest
    print("\n[1/2] Entrenando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Importante si hay desbalance
    )
    
    rf_model.fit(X_train, y_train)
    
    # Validacion cruzada
    print("\n[2/2] Validacion cruzada (5-fold)...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"  ROC-AUC CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Evaluar en test set
    print("\n" + "-"*70)
    print("EVALUACION EN TEST SET")
    print("-"*70)
    
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    print("\nMatriz de Confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    
    print("\nMetricas:")
    print(classification_report(y_test, y_pred, target_names=['SANO', 'DNS']))
    
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC: {auc:.3f}")
    
    # Importancia de features
    print("\n" + "-"*70)
    print("IMPORTANCIA DE FEATURES")
    print("-"*70)
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i, idx in enumerate(indices[:10], 1):
        print(f"  {i}. {FEATURE_NAMES[idx]}: {importances[idx]:.3f}")
    
    return rf_model


def guardar_modelo(model, output_path="models/dns_ml_model.pkl"):
    """Guarda modelo entrenado"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    metadata = {
        'model': model,
        'feature_names': FEATURE_NAMES,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier'
    }
    
    joblib.dump(metadata, output_path)
    print(f"\n✓ Modelo guardado en: {output_path}")
    print(f"  Tamaño: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    print("\n" + "="*70)
    print("SISTEMA DE ENTRENAMIENTO ML - DETECCION DNS")
    print("="*70)
    
    # Cargar datos
    X, y, metadata = cargar_datos_entrenamiento()
    
    if X is None:
        print("\n❌ No se pudieron cargar datos de entrenamiento")
        print("\nVerifica que:")
        print("  1. PostgreSQL esté corriendo")
        print("  2. Haya ECGs etiquetados en datos_examen")
        print("  3. Los archivos CSV existan en el sistema")
        return
    
    # Entrenar
    model = entrenar_modelo(X, y)
    
    # Guardar
    guardar_modelo(model)
    
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
