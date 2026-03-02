#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_ml.py - Predicción DNS usando Machine Learning

Este módulo usa un modelo entrenado (RandomForest) en lugar de reglas fisiológicas.
Si el modelo ML no existe, fallback al modelo basado en reglas.
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Path al modelo entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'dns_ml_model.pkl')


def cargar_modelo_ml():
    """
    Carga modelo ML si existe.
    
    Returns:
        dict con 'model' y 'feature_names', o None si no existe
    """
    if not os.path.exists(MODEL_PATH):
        logger.info("Modelo ML no encontrado, usando modelo basado en reglas")
        return None
    
    try:
        import joblib
        metadata = joblib.load(MODEL_PATH)
        logger.info(f"✓ Modelo ML cargado: {metadata.get('model_type', 'unknown')}")
        return metadata
    except ImportError:
        logger.warning("joblib no instalado, usando modelo basado en reglas")
        return None
    except Exception as e:
        logger.error(f"Error cargando modelo ML: {e}")
        return None


def extraer_vector_features(features_dict):
    """
    Convierte dict de features a vector numpy para ML.
    
    IMPORTANTE: Orden debe coincidir con train_ml_model.py
    """
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


def predict_dns_ml(features_dict):
    """
    Predice DNS usando modelo ML.
    
    Args:
        features_dict: dict con features extraídas
        
    Returns:
        dict con:
        - dns_probability_pct: probabilidad DNS (0-100)
        - confidence_score: confianza (0-100)
        - model_type: 'ml' o 'rules'
        - explanation: descripción del resultado
    """
    
    # Intentar cargar modelo ML
    model_metadata = cargar_modelo_ml()
    
    if model_metadata is None:
        # Fallback al modelo basado en reglas
        logger.info("Usando modelo basado en reglas (ML no disponible)")
        from model import predict_dns as predict_rules
        return predict_rules(features_dict)
    
    # Usar modelo ML
    model = model_metadata['model']
    
    # Extraer vector de features
    X = extraer_vector_features(features_dict).reshape(1, -1)
    
    # Predicción
    try:
        # Probabilidad de clase DNS (clase 1)
        proba = model.predict_proba(X)[0, 1]
        dns_probability_pct = proba * 100
        
        # Confianza basada en decisión del ensemble
        # En RandomForest: confianza = max(proba) porque hay consenso entre árboles
        confidence_score = max(proba, 1 - proba) * 100
        
        # Explicación
        if dns_probability_pct >= 70:
            explanation = "DNS altamente probable según modelo ML"
        elif dns_probability_pct >= 50:
            explanation = "DNS probable según modelo ML"
        elif dns_probability_pct >= 30:
            explanation = "Hallazgos inconclusivos según modelo ML"
        else:
            explanation = "Sin indicadores significativos de DNS según modelo ML"
        
        logger.info(f"Predicción ML: DNS={dns_probability_pct:.1f}%, Confianza={confidence_score:.1f}%")
        
        return {
            'dns_probability_pct': round(dns_probability_pct, 1),
            'confidence_score': round(confidence_score, 1),
            'model_type': 'ml',
            'explanation': explanation,
            'feature_importance': _get_top_features(model_metadata, X)
        }
        
    except Exception as e:
        logger.error(f"Error en predicción ML: {e}, fallback a reglas")
        from model import predict_dns as predict_rules
        return predict_rules(features_dict)


def _get_top_features(model_metadata, X):
    """
    Retorna las 3 features más importantes para esta predicción.
    """
    try:
        model = model_metadata['model']
        feature_names = model_metadata['feature_names']
        
        # Importancia global del modelo
        importances = model.feature_importances_
        
        # Top 3
        top_indices = np.argsort(importances)[::-1][:3]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'name': feature_names[idx],
                'value': float(X[0, idx]),
                'importance': float(importances[idx])
            })
        
        return top_features
        
    except Exception as e:
        logger.warning(f"No se pudo extraer importancia de features: {e}")
        return []


# Alias para compatibilidad con código existente
def predict_dns(features_dict):
    """
    Wrapper para mantener compatibilidad con código existente.
    Intenta usar ML, fallback a reglas si no existe.
    """
    return predict_dns_ml(features_dict)


if __name__ == '__main__':
    # Test básico
    print("=" * 70)
    print("TEST - MODELO ML")
    print("=" * 70)
    
    # Cargar modelo
    metadata = cargar_modelo_ml()
    
    if metadata:
        print(f"\n✓ Modelo cargado:")
        print(f"  Tipo: {metadata.get('model_type', 'unknown')}")
        print(f"  Entrenado: {metadata.get('trained_at', 'unknown')}")
        print(f"  Features: {len(metadata.get('feature_names', []))}")
        
        # Test con features sintéticas
        test_features = {
            'hr_mean_bpm': 65,
            'sdnn_s': 0.18,
            'rmssd_s': 0.14,
            'rr_max_s': 5.2,
            'count_rr_gt3s': 4,
            'count_rr_gt4s': 1,
            'pauses_per_min_gt3s': 8.0,
            'pauses_per_min_gt4s': 2.0,
            'signal_quality_score': 92,
            'ecg_duration_s': 30.0
        }
        
        print("\n" + "-" * 70)
        print("PREDICCION CON FEATURES DE PRUEBA:")
        print("-" * 70)
        
        result = predict_dns_ml(test_features)
        
        print(f"\nDNS Probabilidad: {result['dns_probability_pct']}%")
        print(f"Confianza: {result['confidence_score']}%")
        print(f"Modelo: {result['model_type']}")
        print(f"Explicación: {result['explanation']}")
        
        if 'feature_importance' in result and result['feature_importance']:
            print("\nTop 3 Features Importantes:")
            for i, feat in enumerate(result['feature_importance'], 1):
                print(f"  {i}. {feat['name']}: {feat['value']:.2f} (importancia: {feat['importance']:.3f})")
        
    else:
        print("\n⚠️ Modelo ML no disponible")
        print("Entrena el modelo con: py -3 train_ml_model.py")
    
    print("\n" + "=" * 70 + "\n")
