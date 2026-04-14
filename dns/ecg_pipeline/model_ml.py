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

from config import (
    SDNN_DNS_THRESHOLD_S,
    RMSSD_DNS_THRESHOLD_S,
    RR_MAX_DNS_THRESHOLD_S,
    HR_DNS_ZONE_BPM,
)

logger = logging.getLogger(__name__)

# Path al modelo entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'dns_ml_model.pkl')


def _apply_literature_safeguard(features_dict, dns_probability_pct, decision_threshold_pct):
    """
    Ajuste clínico de seguridad para no subestimar casos DNS con patrón clásico.

    Regla: si hay pausas sinusales recurrentes y RR prolongado con calidad aceptable,
    asegurar clasificación al menos como DNS probable.
    """
    hr = float(features_dict.get('hr_mean_bpm', 0) or 0)
    rr_max = float(features_dict.get('rr_max_s', 0) or 0)
    sdnn = float(features_dict.get('sdnn_s', 0) or 0)
    rmssd = float(features_dict.get('rmssd_s', 0) or 0)
    pauses_gt3s = float(features_dict.get('pauses_per_min_gt3s', 0) or 0)
    pauses_gt4s = float(features_dict.get('pauses_per_min_gt4s', 0) or 0)
    quality = float(features_dict.get('signal_quality_score', 0) or 0)

    # Evidencia fuerte por literatura en este proyecto.
    has_recurrent_pauses = (pauses_gt4s > 0) or (pauses_gt3s >= 5.0)
    has_rr_prolongation = rr_max >= max(4.0, RR_MAX_DNS_THRESHOLD_S - 1.5)
    compatible_hr = hr <= max(85.0, float(HR_DNS_ZONE_BPM[1]))
    acceptable_quality = quality >= 80.0
    has_rr_variability_support = (sdnn >= SDNN_DNS_THRESHOLD_S) or (rmssd >= RMSSD_DNS_THRESHOLD_S)

    if has_recurrent_pauses and has_rr_prolongation and compatible_hr and acceptable_quality and has_rr_variability_support:
        floor = decision_threshold_pct + 2.0

        # Refuerzo extra si además hay variabilidad elevada asociada a pausas.
        if pauses_gt4s > 0 and sdnn >= SDNN_DNS_THRESHOLD_S and rmssd >= RMSSD_DNS_THRESHOLD_S:
            floor = decision_threshold_pct + 4.0

        adjusted = max(float(dns_probability_pct), float(floor))
        return adjusted, (adjusted > float(dns_probability_pct))

    return float(dns_probability_pct), False


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
        logger.info(f"Modelo ML cargado: {metadata.get('model_type', 'unknown')}")
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


def _estimate_confidence(model, X, dns_probability_pct, decision_threshold_pct, features_dict):
    """
    Estima una confianza separada de la probabilidad.

    Usa consenso entre árboles, distancia al umbral clínico y calidad de señal.
    De esta forma la confianza no replica el mismo valor que la probabilidad.
    """
    quality = float(features_dict.get('signal_quality_score', 0) or 0)
    quality_factor = np.clip(quality, 0, 100)

    margin_pct = abs(float(dns_probability_pct) - float(decision_threshold_pct))
    margin_factor = np.clip(100.0 - (margin_pct * 1.5), 20.0, 100.0)

    consensus_factor = None
    estimators = getattr(model, 'estimators_', None)
    if estimators:
        try:
            votes = []
            for tree in estimators:
                if hasattr(tree, 'predict_proba'):
                    vote = float(tree.predict_proba(X)[0, 1])
                    votes.append(1.0 if vote >= 0.5 else 0.0)
                else:
                    votes.append(float(tree.predict(X)[0]))

            if votes:
                positive_ratio = float(np.mean(votes))
                consensus_factor = max(positive_ratio, 1.0 - positive_ratio) * 100.0
        except Exception:
            consensus_factor = None

    if consensus_factor is None:
        consensus_factor = margin_factor

    confidence_score = (0.50 * consensus_factor) + (0.30 * margin_factor) + (0.20 * quality_factor)
    return float(np.clip(confidence_score, 20.0, 100.0))


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
        from model import predict_dns_probability as predict_rules
        dns_probability_pct, confidence_score_pct = predict_rules(features_dict)
        return {
            'dns_probability_pct': round(float(dns_probability_pct), 1),
            'confidence_score': round(float(confidence_score_pct), 1),
            'model_type': 'rules',
            'explanation': 'Predicción basada en reglas fisiológicas',
            'feature_importance': []
        }
    
    # Usar modelo ML
    model = model_metadata['model']
    decision_threshold_pct = float(model_metadata.get('decision_threshold_pct', 50.0))
    
    # Extraer vector de features
    X = extraer_vector_features(features_dict).reshape(1, -1)
    
    # Predicción
    try:
        # Probabilidad de clase DNS (clase 1)
        proba = model.predict_proba(X)[0, 1]
        dns_probability_pct = proba * 100
        
        # Explicación
        dns_probability_pct, literature_override = _apply_literature_safeguard(
            features_dict,
            dns_probability_pct,
            decision_threshold_pct,
        )

        # Confianza separada de la probabilidad para que no repita el mismo valor.
        confidence_score = _estimate_confidence(
            model,
            X,
            dns_probability_pct,
            decision_threshold_pct,
            features_dict,
        )

        if dns_probability_pct >= decision_threshold_pct + 20:
            explanation = "DNS altamente probable según modelo ML"
        elif dns_probability_pct >= decision_threshold_pct:
            explanation = "DNS probable según modelo ML"
        elif dns_probability_pct >= max(10.0, decision_threshold_pct - 20):
            explanation = "Hallazgos inconclusivos según modelo ML"
        else:
            explanation = "Sin indicadores significativos de DNS según modelo ML"

        if literature_override:
            explanation += " (ajuste por criterios de literatura: pausas y RR prolongado)"
        
        logger.info(f"Predicción ML: DNS={dns_probability_pct:.1f}%, Confianza={confidence_score:.1f}%")
        
        return {
            'dns_probability_pct': round(dns_probability_pct, 1),
            'confidence_score': round(confidence_score, 1),
            'decision_threshold_pct': round(decision_threshold_pct, 1),
            'is_dns_positive': bool(dns_probability_pct >= decision_threshold_pct),
            'literature_override': bool(literature_override),
            'model_type': 'ml',
            'explanation': explanation,
            'feature_importance': _get_top_features(model_metadata, X)
        }
        
    except Exception as e:
        logger.error(f"Error en predicción ML: {e}, fallback a reglas")
        from model import predict_dns_probability as predict_rules
        dns_probability_pct, confidence_score_pct = predict_rules(features_dict)
        return {
            'dns_probability_pct': round(float(dns_probability_pct), 1),
            'confidence_score': round(float(confidence_score_pct), 1),
            'model_type': 'rules',
            'explanation': 'Predicción basada en reglas fisiológicas',
            'feature_importance': []
        }


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
    result = predict_dns_ml(features_dict)
    if isinstance(result, dict):
        dns_probability_pct = float(result.get('dns_probability_pct', 0))
        confidence_score = float(result.get('confidence_score_pct', result.get('confidence_score', 0)))
        return dns_probability_pct, confidence_score
    return result


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
