
## Anexo
# Fundamento Matemático y Computacional del Despliegue de Modelos (MLOps)
#### Autor: Carlos César Sánchez Coronel

*(Alineado con la Semana 14: serialización, inferencia, contenedores, monitoreo y deriva.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Sea $\hat{f}$ un modelo entrenado y $\phi$ el preprocesado (pipeline). En **inferencia** se aplica $g(\mathbf{x}) = \hat{f}(\phi(\mathbf{x}))$ con latencia y throughput acotados.

El **sistema de producción** debe minimizar el riesgo operativo:

$$
R_{\text{ops}} = \mathbb{E}[\text{costo de error}] + \lambda_1 \cdot \text{latencia} + \lambda_2 \cdot \text{recursos}
$$

### 1.2 Notación

- **SLA:** acuerdo de nivel de servicio (p99 latencia, disponibilidad).
- **Drift:** cambio de $P(\mathbf{X})$ o $P(Y\mid\mathbf{X})$ entre entrenamiento y producción.

### 1.3 Supuestos

- Versiones de dependencias compatibles con el artefacto serializado.
- Esquema de features estable o versionado conjuntamente con el modelo.

---

## 2. Fundamento Matemático

### 2.1 Descomposición del error en producción

$$
\boxed{\text{Error}_{\text{prod}} \approx \text{Error}_{\text{test}} + \text{Drift} + \text{Sesgo de pipeline} + \text{Ruido numérico}}
$$

### 2.2 Drift de datos (ejemplo univariado)

Distribución entrenamiento $P$, producción $Q$. **Divergencia KL** (si domina $P$):

$$
D_{\text{KL}}(Q \| P) = \int q(x) \log\frac{q(x)}{p(x)}\,dx
$$

**PSI (Population Stability Index)** — métrica práctica en tabular:

$$
\text{PSI} = \sum_{\text{bins}} (Q_b - E_b)\log\frac{Q_b}{E_b}
$$

$E_b$, $Q_b$: proporciones esperadas vs actuales por bin.

### 2.3 Latencia de inferencia (orden)

Para producto matriz-vector con $\mathbf{W} \in \mathbb{R}^{d \times k}$:

$$
\text{tiempo} \propto O(d \cdot k) \quad \text{en CPU secuencial}
$$

Ensembles/árboles: $O(\text{árboles} \times \text{profundidad})$ aprox.

### 2.4 Throughput y Little (colas)

En régimen estable aproximado: $L = \lambda W$ (relación entre clientes en sistema, tasa de llegada y tiempo medio en sistema). Guía dimensionamiento de réplicas.

### 2.5 Optimización de despliegue

Minimizar coste sujeto a SLA:

$$
\min_{n_{\text{replicas}}} \; n_{\text{replicas}} \cdot \text{coste\_unidad} \quad \text{s.a.} \; P(\text{latencia} > L_{\max}) \le \epsilon
$$

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo servicio de predicción

```
Al arrancar:
  Cargar modelo y artefactos de preprocesado en memoria
Por cada petición:
  Validar esquema de entrada
  x ← preprocesar(datos)
  y ← modelo.predict(x) o predict_proba
  Devolver respuesta + id de versión
```

### 3.2 Serialización

- **Pickle/joblib:** grafo de objetos Python; riesgo de seguridad y compatibilidad de versiones.
- **ONNX:** grafo estático para runtimes optimizados.

### 3.3 Numpy: batching para throughput

```python
import numpy as np

def predict_batch(model_predict_fn, X, batch_size=1024):
    """model_predict_fn acepta matriz 2D."""
    outs = []
    for i in range(0, len(X), batch_size):
        outs.append(model_predict_fn(X[i : i + batch_size]))
    return np.vstack(outs) if outs[0].ndim > 1 else np.concatenate(outs)
```

### 3.4 Escalamiento

- Réplicas horizontales detrás de balanceador; colas (Kafka) para batch.
- GPUs para modelos profundos o batch grande.

---

## 4. Métricas de Evaluación Específicas

| Métrica | Definición / uso |
|---------|------------------|
| Latencia p50/p95/p99 | Percentiles de tiempo de respuesta |
| Throughput | Peticiones/s |
| Error rate HTTP | Salud del servicio |
| PSI / KS | Drift de covariables |
| Métricas de negocio | Mantiene alineación con el modelo offline |

---

## 5. Descomposición Teórica

Separación **entrenamiento / inferencia** reduce varianza operativa: el modelo fijo en prod tiene error de generalización + deriva; el monitoreo estima el segundo término.

---

## 6. Selección de Hiperparámetros (operativos)

- Tamaño de batch vs latencia.
- Auto-scaling basado en CPU/GPU y cola.
- Umbrales de reentrenamiento por PSI o caída de métricas.

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Expresión |
|----------|-----------|
| Error prod | $\approx \text{error test} + \text{drift} + \ldots$ |
| PSI | $\sum (Q_b-E_b)\log(Q_b/E_b)$ |
| Little | $L = \lambda W$ (colas) |

---

## 8. Referencias y Lecturas Complementarias

- Treveil & team — *Introducing MLOps* (O'Reilly).
- Lakshmanan, Robinson, Munn — *Machine Learning Design Patterns*.
- Google — *The ML Test Score* (whitepaper); documentación MLflow / Vertex / equivalentes.
