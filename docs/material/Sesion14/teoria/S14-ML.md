---
layout: default
---
# Semana 14: Despliegue, Observabilidad y FinOps en ML


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


***




> **“Optimización costo–valor en Machine Learning (ML FinOps / Cost-Aware ML)”**

Esto permite cubrir:

* cloud ✅
* on-premise ✅
* serverless (Fabric, Databricks) ✅
* modelos clásicos (tu foco) ✅

---


| Enfoque       | Qué optimiza                                      |
| ------------- | ------------------------------------------------- |
| FinOps        | Infraestructura (CPU, GPU, storage)               |
| MLOps         | Ciclo de vida del modelo                          |
| Hibrido | **Decisión de modelo + costo + valor de negocio** |

---

* Empresas grandes → serverless (ej: Fabric)
* Empresas medianas → cloud con instancias
* Empresas pequeñas → on-premise / PCs


---

## 🎓 Módulo FinOps: **ML Cost & Value Optimization (FinOps aplicado a ML)**

---

## 🔹 1. Introducción: el costo en ML (realidad industrial)

* Por qué accuracy ≠ valor
* Casos reales:

  * modelo simple vs complejo
* Concepto:

  * **ROI en modelos ML**

---

## 🔹 2. Fundamentos de costo en computación

* CPU vs GPU vs RAM
* Tiempo de cómputo
* Batch vs real-time
* On-premise vs cloud vs serverless

---

## 🔹 3. Cost-Performance Tradeoff

* Curva de costo vs performance
* Ley de rendimientos decrecientes
* Cómo encontrar el “sweet spot”

---

## 🔹 4. Estimación de costos en ML (práctico)

* Complejidad de modelos:

  * regresión
  * árboles
  * boosting
* Aproximación de tiempo:

  * n_samples × n_features
* Benchmarking:

  * cómo extrapolar

👉 Mini práctica:

* medir tiempo en dataset pequeño
* escalar estimación

---

## 🔹 5. Total Cost of Ownership (TCO) en ML

* Entrenamiento
* Inferencia
* Retraining
* Monitoreo
* Personas (tiempo equipo)

👉 Caso:

* XGBoost vs logística (costo real)

---

## 🔹 6. Drift y costo oculto

* Concept drift vs data drift
* Frecuencia de retraining
* Impacto en costos

👉 Insight:

> modelos complejos → más costo de mantenimiento

---

## 🔹 7. Estrategias de optimización

* Reducir features
* Sampling
* Early stopping
* Modelos más simples
* Híbridos:

  * simple + complejo

---

## 🔹 8. Arquitecturas según presupuesto

* Small data → local / simple
* Medium → cloud CPU
* Large → distribuido

---

## 🔹 9. Métricas de negocio (clave)

* $ por punto de AUC
* $ por predicción
* $ por cliente impactado

👉 Esto es lo más poderoso del módulo

---

## 🔹 10. Toma de decisiones (nivel gerente)

Cómo presentar:

| Modelo | Accuracy | Tiempo | Costo | Recomendación |
| ------ | -------- | ------ | ----- | ------------- |

👉 storytelling con datos

---

## 🔹 11. Caso práctico final (muy importante)

Escenario:

> “Tienes 10M filas, presupuesto limitado”

Evaluar:

* regresión logística
* random forest
* XGBoost

👉 entregar:

* costo estimado
* performance
* recomendación


---

> **“Capstone: Cómo decidir modelos ML en contexto real (costo, drift, negocio)”**



