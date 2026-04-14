# Guía de Entrenamiento ML - Sistema DNS

## 📋 Resumen del Sistema ML

Este sistema permite **convertir el algoritmo basado en reglas a Machine Learning** cuando tengas suficientes ECGs etiquetados en la base de datos.

### ¿Cuándo usar ML en lugar de reglas?

| Aspecto | Modelo Reglas | Modelo ML |
|---------|---------------|-----------|
| **Datos requeridos** | 0 (calibrado manualmente) | 60+ ECGs etiquetados |
| **Precisión** | 100% en test set (4 casos) | Depende de datos de entrenamiento |
| **Flexibilidad** | Fija (thresholds manuales) | Aprende de datos reales |
| **Interpretabilidad** | Alta (reglas fisiológicas) | Media (importancia de features) |
| **Mantenimiento** | Manual (ajustar thresholds) | Automático (re-entrenar con nuevos datos) |

**Recomendación:** Usa reglas hasta tener 60+ ECGs, luego cambia a ML.

---

## 🚀 Proceso de Conversión a ML

### **PASO 1: Etiquetar ECGs en la Base de Datos**

Tienes que **clasificar manualmente** los ECGs ya procesados:

```sql
-- Para ECG confirmado SIN DNS (sano)
UPDATE datos_examen 
SET diagnostico_dns_pct = 0 
WHERE id_examen = 1;

-- Para ECG confirmado CON DNS (patológico)
UPDATE datos_examen 
SET diagnostico_dns_pct = 100 
WHERE id_examen = 2;

-- Para casos intermedios (usa el porcentaje que consideres)
UPDATE datos_examen 
SET diagnostico_dns_pct = 65 
WHERE id_examen = 3;
```

**IMPORTANTE:** El campo `diagnostico_dns_pct` debe ser:
- **0-49%**: Casos sanos (sin DNS)
- **50-100%**: Casos con DNS confirmado

El modelo ML aprenderá a distinguir entre estas dos clases.

---

### **PASO 2: Verificar Cantidad de Datos**

Verifica cuántos ECGs tienes etiquetados:

```sql
SELECT 
    COUNT(*) as total_etiquetados,
    SUM(CASE WHEN diagnostico_dns_pct >= 50 THEN 1 ELSE 0 END) as dns_confirmados,
    SUM(CASE WHEN diagnostico_dns_pct < 50 THEN 1 ELSE 0 END) as sanos_confirmados
FROM datos_examen
WHERE diagnostico_dns_pct IS NOT NULL;
```

**Requisitos mínimos:**
- ✅ **Ideal:** 30+ DNS, 30+ sanos (60+ total)
- ⚠️ **Mínimo funcional:** 10+ DNS, 10+ sanos (20+ total)
- ❌ **Insuficiente:** Menos de 5 de cada clase

---

### **PASO 3: Entrenar el Modelo ML**

Una vez tengas suficientes datos etiquetados:

```bash
cd C:\Users\isabe\OneDrive\Escritorio\tesis
py -3 train_ml_model.py
```

**El script hará:**

1. **Cargar ECGs etiquetados** desde PostgreSQL
2. **Procesar cada ECG** (detección de picos, cálculo de features)
3. **Entrenar RandomForest** con validación cruzada
4. **Evaluar métricas:**
   - Matriz de confusión
   - Precision, Recall, F1-score
   - ROC-AUC
   - Importancia de features
5. **Guardar modelo** en `models/dns_ml_model.pkl`

**Output esperado:**

```
======================================================================
CARGANDO DATOS DE ENTRENAMIENTO DESDE POSTGRESQL
======================================================================

✓ Encontrados 65 ECGs etiquetados
  - DNS confirmado (>=50%): 32
  - Sanos confirmados (<50%): 33

Procesando ECGs...
  [1/65] ✓ LULU: DNS (90%)
  [2/65] ✓ Sierra: SANO (0%)
  ...

✓ Dataset creado: 65 muestras, 10 features

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
  ...

✓ Modelo guardado en: models/dns_ml_model.pkl
  Tamaño: 12.3 KB

======================================================================
✅ ENTRENAMIENTO COMPLETADO
======================================================================
```

---

### **PASO 4: Validar el Modelo**

Después de entrenar, valida el rendimiento:

```bash
py -3 validate_model.py
```

Este script:
- Hace validación cruzada
- Muestra casos mal clasificados
- Analiza importancia de features
- Calcula métricas detalladas

---

### **PASO 5: Usar el Modelo ML en la API**

**¡El cambio es automático!**

Una vez que `models/dns_ml_model.pkl` existe, la API **automáticamente** lo usará en lugar del modelo basado en reglas.

```python
# En database/ecg_processor.py (YA CONFIGURADO):

try:
    from model_ml import predict_dns  # Intenta usar ML
    print("✓ Usando modelo ML")
except ImportError:
    from model import predict_dns     # Fallback a reglas
    print("✓ Usando modelo basado en reglas")
```

**Reinicia Flask:**

```bash
py -3 application.py
```

**Verás en el log:**

```
✓ Modelo ML cargado: RandomForestClassifier
✓ Usando modelo ML para predicciones DNS
 * Running on http://127.0.0.1:5000
```

---

## 🔄 Re-entrenar con Nuevos Datos

Cada vez que agregues más ECGs etiquetados:

1. **Actualiza la base de datos** con nuevas etiquetas
2. **Re-ejecuta entrenamiento:**
   ```bash
   py -3 train_ml_model.py
   ```
3. **El modelo se sobrescribe** (automáticamente usa el nuevo)
4. **Opcional:** Valida nuevamente
   ```bash
   py -3 validate_model.py
   ```

**No necesitas cambiar código** - el sistema detecta automáticamente el modelo más reciente.

---

## 📊 Interpretando Resultados

### Matriz de Confusión

```
              Predicho
              SANO  DNS
Real  SANO     TN    FP
      DNS      FN    TP
```

- **TN (True Negative):** Sanos correctamente identificados ✅
- **TP (True Positive):** DNS correctamente identificados ✅
- **FP (False Positive):** Sano clasificado como DNS ❌ (genera alarma innecesaria)
- **FN (False Negative):** DNS clasificado como sano ❌ (peligroso, no detecta patología)

**En veterinaria: Preferimos FP sobre FN** (mejor una falsa alarma que no detectar DNS real).

---

### Métricas Clave

| Métrica | Fórmula | Interpretación | Objetivo |
|---------|---------|----------------|----------|
| **Precision** | TP/(TP+FP) | De los que dijo DNS, ¿cuántos lo son? | >80% |
| **Recall (Sensibilidad)** | TP/(TP+FN) | De los DNS reales, ¿cuántos detectó? | **>90%** (crítico) |
| **F1-Score** | 2×(P×R)/(P+R) | Balance precision/recall | >85% |
| **ROC-AUC** | Área bajo curva ROC | Capacidad de discriminación | >0.90 |

**Prioridad:** En diagnóstico DNS, **Recall > Precision** (no queremos perder casos reales).

---

### Importancia de Features

El modelo ML te dirá **qué features son más importantes**:

```
1. pauses_per_min_gt3s: 0.285  ← La más importante
2. sdnn_s: 0.198
3. hr_mean_bpm: 0.164
...
```

Esto te ayuda a entender **qué está aprendiendo el modelo** y validar que coincida con fisiología veterinaria.

---

## 🛠️ Solución de Problemas

### Error: "No hay ECGs etiquetados"

**Causa:** La tabla `datos_examen` no tiene registros con `diagnostico_dns_pct != NULL`

**Solución:**
```sql
-- Verifica qué examenes existen
SELECT id_examen, id_paciente, diagnostico_dns_pct 
FROM datos_examen;

-- Etiqueta los que conozcas
UPDATE datos_examen SET diagnostico_dns_pct = 0 WHERE id_examen IN (1, 3, 5);
UPDATE datos_examen SET diagnostico_dns_pct = 100 WHERE id_examen IN (2, 4);
```

---

### Error: "INSUFICIENTES DATOS - Se requieren al menos 5 de cada clase"

**Causa:** Muy pocos ECGs etiquetados

**Solución:** Continúa procesando ECGs y etiquetándolos manualmente hasta tener al menos:
- 10+ sanos
- 10+ con DNS

---

### Error: "Archivo CSV no encontrado"

**Causa:** El script busca archivos en rutas predefinidas

**Solución 1:** Asegúrate de que `informacion_examen_pdf.ruta_pdf` apunte al archivo correcto:

```sql
UPDATE informacion_examen_pdf 
SET ruta_pdf = 'C:\Users\isabe\OneDrive\Documentos\EasyG - Informes\LULU.csv'
WHERE id_examen = 1;
```

**Solución 2:** Modifica rutas en `train_ml_model.py` líneas 135-138:

```python
posibles_rutas = [
    f"C:\\TU\\RUTA\\AQUI\\{nombre}.csv",  # ← Agrega tus rutas
    f"C:\\Users\\isabe\\OneDrive\\Documentos\\EasyG - Informes\\{nombre}.csv",
]
```

---

### Modelo ML no se usa (sigue usando reglas)

**Verificar:**

1. Archivo existe:
   ```bash
   dir models\dns_ml_model.pkl
   ```

2. `joblib` instalado:
   ```bash
   pip install joblib
   ```

3. Logs de Flask muestran:
   ```
   ✓ Modelo ML cargado: RandomForestClassifier
   ```

---

## 📈 Mejorando el Modelo

### 1. Agregar más datos

**La forma #1 de mejorar:** Más ECGs etiquetados = mejor modelo

Objetivo: 100+ ECGs (50 DNS, 50 sanos)

---

### 2. Balancear clases

Si tienes desbalance (ej: 50 sanos, 10 DNS):

```python
# Ya configurado en train_ml_model.py línea 193
RandomForestClassifier(
    class_weight='balanced'  # ← Compensa desbalance
)
```

---

### 3. Ajustar hiperparámetros

En `train_ml_model.py`, experimenta con:

```python
rf_model = RandomForestClassifier(
    n_estimators=200,      # ← Prueba 100, 200, 500
    max_depth=15,          # ← Prueba 5, 10, 15, None
    min_samples_split=3,   # ← Prueba 2, 5, 10
    min_samples_leaf=1,    # ← Prueba 1, 2, 5
    random_state=42,
    class_weight='balanced'
)
```

Re-entrena después de cada cambio.

---

### 4. Probar XGBoost

Descomenta en `train_ml_model.py` para usar XGBoost (suele dar mejor rendimiento):

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

---

## ✅ Checklist de Producción

Antes de confiar en el modelo ML:

- [ ] Mínimo 30 ECGs DNS etiquetados
- [ ] Mínimo 30 ECGs sanos etiquetados
- [ ] ROC-AUC > 0.90 en validación cruzada
- [ ] Recall (DNS) > 90% (crítico - no perder casos reales)
- [ ] Precision (DNS) > 80%
- [ ] Analizar casos mal clasificados (entender por qué)
- [ ] Validar con casos nuevos NO vistos en entrenamiento
- [ ] Comparar predicciones ML vs. criterio clínico veterinario

---

## 📝 Archivos del Sistema ML

```
tesis/
├── train_ml_model.py          ← Script de entrenamiento
├── validate_model.py          ← Script de validación
├── ML_TRAINING_GUIDE.md       ← Esta guía
├── models/
│   └── dns_ml_model.pkl       ← Modelo entrenado (se crea al entrenar)
├── ecg_pipeline/
│   ├── model.py               ← Modelo basado en reglas (original)
│   └── model_ml.py            ← Modelo ML (nuevo, auto-fallback)
└── database/
    └── ecg_processor.py       ← Usa model_ml.py si existe
```

---

## 🎯 Próximos Pasos

1. **Etiqueta tus ECGs actuales** en PostgreSQL
   - Marca los que sabes que son DNS
   - Marca los que sabes que son sanos

2. **Cuando tengas 60+, ejecuta:**
   ```bash
   py -3 train_ml_model.py
   ```

3. **Valida el modelo:**
   ```bash
   py -3 validate_model.py
   ```

4. **Reinicia Flask** - el sistema usará ML automáticamente

5. **Continúa agregando datos** y re-entrena cada 20-30 ECGs nuevos

---

**¡Listo!** Cuando tengas los datos etiquetados, solo ejecuta `train_ml_model.py` y el sistema cambiará automáticamente de reglas a ML. 🚀
