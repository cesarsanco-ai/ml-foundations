# Semana 12: Sistemas de Recomendación

**Autor:** Carlos César Sánchez Coronel

Los sistemas de recomendación son centrales en plataformas digitales (streaming, e-commerce, redes sociales). Esta sesión cubre fundamentos, filtrado colaborativo, factorización matricial (Funk SVD), métricas y desafíos (cold start, escalabilidad).

---

## Logro de la sesión

Diseñar e implementar recomendadores basados en filtrado colaborativo y factorización matricial, con métricas apropiadas y comprensión de cold start y escalabilidad.

---

## Introducción

- **Datos explícitos:** valoraciones, likes, comentarios.
- **Datos implícitos:** clics, tiempo de visualización, compras, escuchas.

---

## Tipos de sistemas

### Popularidad

Recomendaciones globales (ventas, vistas). Útil con poco historial (cold start del sistema).

### Basados en contenido

Vector de ítem $\mathbf{x}_i$; perfil de usuario como promedio de ítems positivos:

$$
\mathbf{p}_u = \frac{1}{|I_u^+|}\sum_{i \in I_u^+} \mathbf{x}_i
$$

Relevancia con coseno:

$$
\text{relevancia}(u,j) = \frac{\mathbf{p}_u \cdot \mathbf{x}_j}{\|\mathbf{p}_u\| \|\mathbf{x}_j\|}
$$

**Pros:** nuevos ítems con metadata. **Contras:** necesita buenas features; riesgo de falta de diversidad.

### Colaborativo

**User-based:** vecinos por similitud de ratings; predicción ponderada.

Similitud Pearson entre usuarios $a$ y $b$:

$$
\text{sim}(a,b) = \frac{\sum_{i \in I_{ab}} (r_{a,i} - \bar{r}_a)(r_{b,i} - \bar{r}_b)}{\sqrt{\sum_{i \in I_{ab}} (r_{a,i} - \bar{r}_a)^2} \sqrt{\sum_{i \in I_{ab}} (r_{b,i} - \bar{r}_b)^2}}
$$

Predicción (esquema típico):

$$
\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}
$$

**Item-based:** similitud entre ítems; suele ser más estable y precomputable.

### Híbridos

Combinan contenido y colaborativo (p. ej. Netflix).

---

## Factorización matricial

Matriz $\mathbf{R}$ sparse $m \times n$ (usuarios $\times$ ítems). Objetivo:

$$
\mathbf{R} \approx \mathbf{P}\mathbf{Q}^T, \quad \hat{r}_{u,i} = \mathbf{p}_u \cdot \mathbf{q}_i = \sum_{f=1}^k p_{u,f} q_{i,f}
$$

### SVD clásico

Limitaciones: imputar faltantes, coste, no ideal para sparse extremo.

### Funk SVD (SGD)

Minimizar solo sobre pares observados:

$$
\mathcal{L} = \sum_{(u,i) \in \mathcal{K}} (r_{u,i} - \mathbf{p}_u \cdot \mathbf{q}_i)^2 + \lambda (\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)
$$

Actualizaciones típicas por observación $r_{u,i}$:

$$
e_{u,i} = r_{u,i} - \mathbf{p}_u \cdot \mathbf{q}_i
$$

$$
\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta (e_{u,i} \mathbf{q}_i - \lambda \mathbf{p}_u), \quad
\mathbf{q}_i \leftarrow \mathbf{q}_i + \eta (e_{u,i} \mathbf{p}_u - \lambda \mathbf{q}_i)
$$

### Sesgos

$$
\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i
$$

---

## Métricas

### Predicción de rating

$$
\text{RMSE} = \sqrt{\frac{1}{|\mathcal{K}|} \sum_{(u,i) \in \mathcal{K}} (r_{u,i} - \hat{r}_{u,i})^2}, \quad
\text{MAE} = \frac{1}{|\mathcal{K}|} \sum_{(u,i) \in \mathcal{K}} |r_{u,i} - \hat{r}_{u,i}|
$$

### Rankings

- **Precision@k:** relevantes en top-$k$ dividido entre $k$.
- **Recall@k:** relevantes capturados / total relevantes del usuario.
- **F-score@k:** combina precisión y recall.
- **MAP:** promedio de precisión media por usuario.
- **NDCG@k:** calidad del ranking con descuento por posición.
- **Coverage:** diversidad de ítems o usuarios cubiertos.

---

## Desafíos

### Cold start

- **Usuario nuevo:** onboarding, demografía, popular.
- **Ítem nuevo:** contenido, metadata.
- **Sistema nuevo:** transfer learning, encuestas.

### Escalabilidad

LSH, factorización con SGD lineal en número de ratings, modelos basados en grafos.

### Dispersión extrema

Pocos ratings por usuario dificultan la generalización.

### Diversidad y serendipia

Evitar solo recomendar lo más popular o demasiado similar (“filtro burbuja”).

---

## Caso MovieLens 100k

Línea base (sesgos), item-based CF, Funk SVD con factores y regularización; RMSE/MAE y métricas de ranking (p. ej. Precision@10, NDCG@10).

---

## Implementación en Python

### Surprise (SVD)

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

reader = Reader(line_format="user item rating", sep="\t")
data = Dataset.load_from_file("u.data", reader=reader)
trainset, testset = train_test_split(data, test_size=0.2)

algo = SVD(n_factors=20, reg_all=0.1, lr_all=0.005, n_epochs=20)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)
```

### Funk SGD esquemático (NumPy)

```python
import numpy as np

def funk_sgd(R, k=20, lambda_reg=0.1, lr=0.005, epochs=20):
    m, n = R.shape
    P = np.random.normal(0, 0.1, (m, k))
    Q = np.random.normal(0, 0.1, (n, k))
    nonzeros = np.argwhere(~np.isnan(R))
    for _ in range(epochs):
        np.random.shuffle(nonzeros)
        for u, i in nonzeros:
            rui = R[u, i]
            pred = np.dot(P[u], Q[i])
            eui = rui - pred
            P[u] += lr * (eui * Q[i] - lambda_reg * P[u])
            Q[i] += lr * (eui * P[u] - lambda_reg * Q[i])
    return P, Q
```

### Precision@k

```python
def precision_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / k
```

---

## Conexión con el curso

Álgebra lineal (SVD), métricas tipo clasificación extendidas a rankings, datos faltantes (FE), SGD relacionado con redes neuronales.

---

## Resumen

- Popularidad, contenido, colaborativo e híbridos.
- User-based vs item-based; similitud Pearson / coseno.
- Funk SVD con regularización para matrices sparse.
- RMSE/MAE para ratings; Precision@k, Recall@k, MAP, NDCG para listas.
- Cold start y escalabilidad son retos centrales en producción.
