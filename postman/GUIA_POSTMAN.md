# Guía rápida Postman (local, sin servicios públicos)

## 1) Importar colección
- En Postman: **Import** → seleccionar `Tesis_API_Local.postman_collection.json`.
- Verifica variable `base_url = http://localhost:5000`.

## 1.1) Importar environment (recomendado)
- En Postman: **Import** → seleccionar `Tesis_API_Local.postman_environment.json`.
- En la esquina superior derecha, seleccionar el environment **Tesis API Local**.
- Variables preconfiguradas: `base_url`, `paciente_id`, `nombre`, `edad`, `raza`.

## 2) Actualizar datos del paciente (PUT)
- Endpoint: `PUT /pacientes/{id_paciente}`
- El **ID va en la URL** (ejemplo: `/pacientes/1`).
- Body: **raw + JSON**

```json
{
  "nombre": "Mocca",
  "edad": 5,
  "raza": "French Bulldog"
}
```

Notas:
- Puedes enviar solo un campo si quieres (ej: solo `edad`).
- Los valores pueden tener espacios (nombre/raza) sin problema.
- No necesitas enviar `id` dentro del JSON.

## 3) Cargar archivo para algoritmo (POST /diagnosticar)
- Endpoint: `POST /diagnosticar`
- Body: **form-data** (NO raw JSON)
- Campos exactos:
  - `file` → tipo **File** (selecciona el CSV)
  - `nombre` → tipo **Text**
  - `edad` → tipo **Text**
  - `raza` → tipo **Text**

Ejemplo de valores:
- `nombre`: `Mocca`
- `edad`: `5`
- `raza`: `French Bulldog`

## 4) Validación recomendada
1. Ejecutar `GET /pacientes/{id}` (ver estado actual).
2. Ejecutar `PUT /pacientes/{id}` (actualizar).
3. Repetir `GET /pacientes/{id}` (confirmar cambio).
4. Ejecutar `POST /diagnosticar` con archivo ECG.
