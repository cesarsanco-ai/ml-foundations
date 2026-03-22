# Semana 14: Despliegue de Modelos (MLOps)

**Autor:** Carlos César Sánchez Coronel

El valor de un modelo se materializa en producción. **MLOps** integra ML, ingeniería de software y DevOps: serialización, APIs, contenedores, versionado y monitoreo frente a degradación (drift).

---

## Logro de la sesión

Llevar un modelo a producción, conocer conceptos básicos de MLOps y monitorear rendimiento en el tiempo.

---

## Ciclo de vida

Flujo típico: **Datos → Entrenamiento → Validación → Despliegue → Monitoreo →** (retroalimentación a datos y reentrenamiento).

---

## Entrenamiento vs inferencia

- **Online:** API REST, baja latencia por petición.
- **Batch:** grandes volúmenes periódicos (p. ej. recomendaciones nocturnas).

Separar pipelines pesados de entrenamiento de servicios ligeros de predicción.

---

## Serialización

### Pickle

```python
import pickle

with open("modelo.pkl", "wb") as f:
    pickle.dump(model, f)
with open("modelo.pkl", "rb") as f:
    model = pickle.load(f)
```

No cargar archivos no confiables; posibles problemas de versión.

### Joblib (recomendado para sklearn)

```python
import joblib

joblib.dump(model, "modelo.joblib")
model = joblib.load("modelo.joblib")
```

### ONNX

Interoperabilidad entre frameworks e inferencia optimizada (requiere `skl2onnx` u herramientas equivalentes).

### MLflow

Registro y versionado de modelos con metadatos y métricas.

```python
import mlflow

mlflow.sklearn.log_model(model, "model")
```

---

## APIs: Flask y FastAPI

### Flask

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("modelo.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    pred = model.predict([features])
    return jsonify({"prediction": pred.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("modelo.joblib")

class Item(BaseModel):
    features: list

@app.post("/predict")
def predict(item: Item):
    pred = model.predict([item.features])
    return {"prediction": pred.tolist()}
```

Documentación interactiva en `/docs`.

---

## Docker (ejemplo)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.joblib .
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
```

```bash
docker build -t modelo-api .
docker run -p 80:80 modelo-api
```

---

## Versionado

### MLflow runs

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("auc", 0.85)
    mlflow.sklearn.log_model(model, "model")
```

### DVC

Versionado de datos y pipelines (`.dvc` + almacenamiento remoto).

---

## Monitoreo

### Data drift y concept drift

- **Data drift:** cambio en distribución de entradas.
- **Concept drift:** cambio en la relación $X \to y$.

### Con etiquetas

Mismas métricas que en entrenamiento sobre ventanas recientes.

### Sin etiquetas

Distribuciones de features y de predicciones.

#### PSI (Population Stability Index)

$$
PSI = \sum_{i=1}^{B} (p_i - q_i) \cdot \ln\left(\frac{p_i}{q_i}\right)
$$

$p_i$: proporción en producción; $q_i$: en referencia (entrenamiento), por bins.

- PSI < 0.1: cambio pequeño.
- 0.1 ≤ PSI < 0.25: revisar.
- PSI ≥ 0.25: considerar reentrenar.

#### Kolmogorov–Smirnov

Compara dos distribuciones continuas; p-valor bajo sugiere drift.

### Herramientas

Evidently AI, WhyLabs/Whylogs, Great Expectations.

---

## Caso integrado: riesgo crediticio

1. Entrenar y `joblib.dump`.
2. FastAPI con esquema Pydantic por solicitante.
3. Imagen Docker.
4. Registro en MLflow.
5. Monitoreo semanal con PSI sobre features y predicciones.

---

## Mapa del curso (14 sesiones)

| Sesión | Tema principal |
| :--- | :--- |
| 1–2 | Introducción al ML; EDA y feature engineering |
| 3 | Regresión lineal y regularización |
| 4–5 | Clasificación (logística, métricas, desbalance; KNN, Naive Bayes, SVM) |
| 6–7 | Árboles, Random Forest; Gradient Boosting |
| 8 | Validación y selección de modelos |
| 9 | No supervisado: PCA, clustering |
| 10 | Series temporales |
| 11 | Modelos complementarios (robustos, SVR, DBSCAN profundo, métricas externas) |
| 12 | Sistemas de recomendación |
| 13 | Interpretabilidad (SHAP, LIME, PDP) |
| 14 | MLOps y despliegue |

---

## Resumen

- Producción requiere separar entrenamiento e inferencia, serializar de forma segura y exponer APIs.
- Docker y MLflow/DVC ayudan a reproducibilidad y versionado.
- Monitorear drift con PSI, KS y herramientas de perfilado.
- MLOps es competencia clave para llevar modelos del notebook al valor de negocio sostenible.
