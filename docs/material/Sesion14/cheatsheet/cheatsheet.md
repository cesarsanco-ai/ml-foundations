

# Cheatsheet: MLOps y Despliegue
**Autor:** Carlos César Sánchez Coronel  

---

## Ciclo de vida

**Datos → Entrenamiento → Validación → Despliegue → Monitoreo →** reentrenamiento.

---

## Serialización

| Formato | Uso |
| :--- | :--- |
| **Pickle** | Prototipo; riesgo seguridad y versiones |
| **Joblib** | Modelos sklearn + arrays grandes |
| **ONNX** | Interoperabilidad e inferencia optimizada |
| **MLflow** | Registry + metadatos + versiones |

```python
import joblib
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

---

## API REST

* **Flask:** simple.  
* **FastAPI:** tipado, async, docs en `/docs`.  

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class Item(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(item: Item):
    return {"pred": model.predict([item.features]).tolist()}
```

---

## Docker (idea)

* Imagen con `requirements.txt`, código y artefacto del modelo.  
* `CMD` con **uvicorn** / gunicorn para FastAPI.  

---

## Drift

| Tipo | Qué cambia |
| :--- | :--- |
| **Data drift** | $P(X)$ |
| **Concept drift** | $P(y\mid X)$ |

**PSI** por bins: $<0.1$ OK; $0.1$–$0.25$ revisar; $\ge 0.25$ alerta fuerte.  
**KS test** para comparar distribuciones continuas.

---

## Monitoreo práctico

* Métricas de negocio y técnicas en ventanas recientes.  
* Si no hay etiquetas: distribución de **inputs** y de **scores** predichos.  
* Herramientas: Evidently, WhyLogs, Great Expectations.  

---

## Buenas prácticas

* Separar **entrenamiento** (batch pesado) de **inferencia** (servicio ligero).  
* Versionar **código + datos + modelo** (Git + DVC/MLflow).  
* Tests: contrato de API, latencia, carga mínima.  
* Rollback y **shadow mode** antes de cambiar 100% tráfico.  

---

## Puntos críticos

* **Pickle** de fuentes no confiables = riesgo de ejecución arbitraria.  
* Mismas **transformaciones** (scaler, encoders) en train y en producción → **pipeline** serializado junto al modelo cuando sea posible.  
* Privacidad: no loguear PII en claro.  

> *“Un modelo en producción es un servicio de software: SLA, seguridad y observabilidad cuentan.”*
