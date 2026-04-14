# Sistema ML para Detección DNS - README

## 📁 Archivos Creados

### 1. `train_ml_model.py`
**Propósito:** Script principal de entrenamiento del modelo ML

**Funciones principales:**
- `cargar_datos_entrenamiento()`: Carga ECGs etiquetados desde PostgreSQL
- `procesar_ecg_para_features()`: Procesa cada ECG (detección picos, features)
- `entrenar_modelo()`: Entrena RandomForest con validación cruzada
- `guardar_modelo()`: Guarda modelo en `models/dns_ml_model.pkl`

**Como usar:**
```bash
py -3 train_ml_model.py
```

**Requisitos:**
- Mínimo 10 ECGs de cada clase (DNS y sanos) etiquetados en BD
- Archivos CSV accesibles en el sistema
- sklearn instalado: `pip install scikit-learn joblib`

---

### 2. `ecg_pipeline/model_ml.py`
**Propósito:** Módulo de predicción usando ML (con auto-fallback a reglas)

**Funciones principales:**
- `cargar_modelo_ml()`: Carga modelo entrenado si existe
- `predict_dns_ml()`: Predice probabilidad DNS usando ML
- `extraer_vector_features()`: Convierte dict de features a vector numpy
- `_get_top_features()`: Retorna features más importantes para predicción

**Comportamiento:**
- Si `models/dns_ml_model.pkl` existe → usa ML
- Si no existe → fallback automático a modelo basado en reglas
- Sin modificar código, sin errores

**Features usadas (orden crítico):**
1. `hr_mean_bpm`
2. `sdnn_s`
3. `rmssd_s`
4. `rr_max_s`
5. `count_rr_gt3s`
6. `count_rr_gt4s`
7. `pauses_per_min_gt3s`
8. `pauses_per_min_gt4s`
9. `signal_quality_score`
10. `ecg_duration_s`

---

### 3. `validate_model.py`
**Propósito:** Validación completa del modelo entrenado

**Análisis que realiza:**
- Validación cruzada 5-fold
- Matriz de confusión
- Precision, Recall, F1-score, ROC-AUC
- Análisis de casos mal clasificados
- Importancia de features

**Como usar:**
```bash
py -3 validate_model.py
```

**Output esperado:**
```
VALIDACION DEL MODELO
======================================================================

[1/5] Validación Cruzada (5-fold)...

Métricas (media ± std):
  ACCURACY    : 0.923 ± 0.045
  PRECISION   : 0.912 ± 0.058
  RECALL      : 0.943 ± 0.067
  F1          : 0.925 ± 0.048
  ROC_AUC     : 0.967 ± 0.032

[2/5] Generando predicciones...
[3/5] Matriz de Confusión:
...
```

---

### 4. `ML_TRAINING_GUIDE.md`
**Propósito:** Guía completa de uso del sistema ML

**Contenido:**
- Comparación modelo reglas vs ML
- Proceso paso a paso de conversión
- Como etiquetar datos en PostgreSQL
- Como entrenar y validar el modelo
- Interpretación de métricas
- Solución de problemas comunes
- Mejoras y optimización

---

### 5. Modificación en `database/ecg_processor.py`
**Cambio:** Auto-detección de modelo ML vs reglas

**ANTES:**
```python
from model import predict_dns_probability
```

**AHORA:**
```python
try:
    from model_ml import predict_dns as predict_dns_probability
    logger.info("✓ Usando modelo ML")
except (ImportError, FileNotFoundError):
    from model import predict_dns_probability
    logger.info("✓ Usando modelo basado en reglas")
```

**Efecto:**
- Si existe `models/dns_ml_model.pkl` → usa ML automáticamente
- Si no existe → usa reglas sin errores
- No necesitas cambiar código al entrenar el modelo

---

## 🔄 Flujo de Trabajo ML

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ETIQUETAR DATOS EN POSTGRESQL                            │
│    UPDATE datos_examen SET diagnostico_dns_pct = 0/100     │
│    WHERE id_examen = X;                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ENTRENAR MODELO                                          │
│    py -3 train_ml_model.py                                  │
│    → Carga ECGs etiquetados                                 │
│    → Entrena RandomForest                                   │
│    → Guarda models/dns_ml_model.pkl                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. VALIDAR MODELO (opcional pero recomendado)              │
│    py -3 validate_model.py                                  │
│    → Validación cruzada                                     │
│    → Análisis de errores                                    │
│    → Importancia de features                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. REINICIAR FLASK (usa ML automáticamente)                │
│    py -3 application.py                                     │
│    → Detecta models/dns_ml_model.pkl                        │
│    → "✓ Usando modelo ML para predicciones DNS"            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. USAR API NORMALMENTE                                    │
│    POST /diagnosticar con archivo ECG                       │
│    → Ahora usa ML en lugar de reglas                        │
│    → Sin cambios en el endpoint                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. RE-ENTRENAR CON MÁS DATOS (ciclo continuo)              │
│    Agregar más ECGs etiquetados → volver al paso 2         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Estructura del Modelo Guardado

`models/dns_ml_model.pkl` contiene un diccionario:

```python
{
    'model': RandomForestClassifier(...),  # Modelo entrenado
    'feature_names': [                      # Nombres de features
        'hr_mean_bpm',
        'sdnn_s',
        ...
    ],
    'trained_at': '2026-01-30T14:23:45',   # Timestamp
    'model_type': 'RandomForestClassifier'  # Tipo de modelo
}
```

---

## ⚙️ Configuración del Modelo

### RandomForest (default)

```python
RandomForestClassifier(
    n_estimators=100,          # 100 árboles
    max_depth=10,              # Profundidad máxima
    min_samples_split=5,       # Min samples para split
    min_samples_leaf=2,        # Min samples en hoja
    random_state=42,           # Reproducibilidad
    class_weight='balanced'    # Compensa desbalance de clases
)
```

**Ventajas:**
- Robusto al overfitting
- Maneja features correlacionadas
- Proporciona importancia de features
- Rápido en predicción

---

## 🎯 Métricas Objetivo

Para considerar el modelo ML **listo para producción**:

| Métrica | Objetivo | Crítico |
|---------|----------|---------|
| **Recall (DNS)** | >90% | **>85%** |
| **Precision (DNS)** | >80% | >70% |
| **ROC-AUC** | >0.90 | >0.85 |
| **Accuracy** | >85% | >75% |

**Prioridad:** Recall > Precision (no queremos perder casos DNS reales)

---

## 🚨 Alertas y Validaciones

### En `train_ml_model.py`:

```python
# ALERTA si pocos datos
if dns_cases < 10 or sano_cases < 10:
    print("⚠️ ADVERTENCIA: Pocos datos para entrenamiento robusto")
    print("   Recomendado: 30+ de cada clase")

# ERROR si muy pocos
if dns_cases < 5 or sano_cases < 5:
    print("❌ INSUFICIENTES DATOS - Se requieren al menos 5 de cada clase")
    return None
```

### En `model_ml.py`:

```python
# Fallback automático si modelo no existe
if model_metadata is None:
    logger.info("Usando modelo basado en reglas (ML no disponible)")
    from model import predict_dns as predict_rules
    return predict_rules(features_dict)
```

---

## 📝 Ejemplo de Uso Completo

### 1. Etiquetar datos

```sql
-- En PostgreSQL
UPDATE datos_examen SET diagnostico_dns_pct = 100 WHERE id_examen = 1; -- LULU (DNS)
UPDATE datos_examen SET diagnostico_dns_pct = 90 WHERE id_examen = 2;  -- MARCEL (DNS)
UPDATE datos_examen SET diagnostico_dns_pct = 0 WHERE id_examen = 3;   -- Sierra (sano)
...
```

### 2. Entrenar

```bash
PS C:\Users\isabe\OneDrive\Escritorio\tesis> py -3 train_ml_model.py

======================================================================
SISTEMA DE ENTRENAMIENTO ML - DETECCION DNS
======================================================================

======================================================================
CARGANDO DATOS DE ENTRENAMIENTO DESDE POSTGRESQL
======================================================================

✓ Encontrados 65 ECGs etiquetados
  - DNS confirmado (>=50%): 32
  - Sanos confirmados (<50%): 33

Procesando ECGs...
  [1/65] ✓ LULU: DNS (100%)
  [2/65] ✓ Sierra: SANO (0%)
  ...
  [65/65] ✓ MARCEL: DNS (90%)

✓ Dataset creado: 65 muestras, 10 features
  - Clase 0 (SANO): 33
  - Clase 1 (DNS): 32

======================================================================
ENTRENAMIENTO DEL MODELO ML
======================================================================

Datos de entrenamiento: 48 muestras
Datos de prueba: 17 muestras

[1/2] Entrenando Random Forest...
[2/2] Validacion cruzada (5-fold)...
  ROC-AUC CV: 0.945 (+/- 0.038)

----------------------------------------------------------------------
EVALUACION EN TEST SET
----------------------------------------------------------------------

Matriz de Confusion:
  TN=8   FP=1
  FN=0   TP=8

Metricas:
              precision    recall  f1-score   support

       SANO       1.00      0.89      0.94         9
        DNS       0.89      1.00      0.94         8

    accuracy                           0.94        17

ROC-AUC: 0.972

----------------------------------------------------------------------
IMPORTANCIA DE FEATURES
----------------------------------------------------------------------

Ranking de importancia:
  1. pauses_per_min_gt3s: 0.285
  2. sdnn_s: 0.198
  3. hr_mean_bpm: 0.164
  4. rr_max_s: 0.132
  5. count_rr_gt4s: 0.089
  6. rmssd_s: 0.067
  7. count_rr_gt3s: 0.044
  8. pauses_per_min_gt4s: 0.021
  9. signal_quality_score: 0.015
 10. ecg_duration_s: 0.008

✓ Modelo guardado en: models/dns_ml_model.pkl
  Tamaño: 12.3 KB

======================================================================
✅ ENTRENAMIENTO COMPLETADO
======================================================================

Proximos pasos:
  1. El modelo ML ya está guardado
  2. La API lo usará automaticamente si existe
  3. Para re-entrenar con más datos: py -3 train_ml_model.py

======================================================================
```

### 3. Validar (opcional)

```bash
PS C:\Users\isabe\OneDrive\Escritorio\tesis> py -3 validate_model.py
# ... output detallado de validación
```

### 4. Reiniciar Flask

```bash
PS C:\Users\isabe\OneDrive\Escritorio\tesis> py -3 application.py

2026-01-30 14:25:30 - database.ecg_processor - INFO - ✓ Usando modelo ML para predicciones DNS
 * Serving Flask app 'application'
 * Running on http://127.0.0.1:5000
```

### 5. Usar API

```bash
# El endpoint /diagnosticar ahora usa ML automáticamente
POST http://127.0.0.1:5000/diagnosticar
Content-Type: multipart/form-data

file: NUEVO_PACIENTE.csv
nombre: "Nuevo Paciente"
edad: 8
raza: "Golden Retriever"

# Respuesta incluye predicción ML
{
  "dns_probability_pct": 73.2,
  "model_type": "ml",
  "explanation": "DNS probable según modelo ML",
  ...
}
```

---

## 🔧 Personalización

### Cambiar algoritmo a XGBoost

En `train_ml_model.py`, reemplaza líneas 186-195:

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
```

### Agregar más features

1. Modifica `ecg_pipeline/features.py` para extraer nuevas features
2. Actualiza `FEATURE_NAMES` en `train_ml_model.py` y `model_ml.py`
3. Actualiza `extraer_vector_features()` en ambos scripts
4. Re-entrena el modelo

---

## 📚 Documentación Completa

Ver `ML_TRAINING_GUIDE.md` para:
- Guía paso a paso completa
- Interpretación de métricas
- Solución de problemas
- Mejores prácticas
- Optimización y tuning

---

**¡Listo para entrenar cuando tengas los datos etiquetados!** 🚀
