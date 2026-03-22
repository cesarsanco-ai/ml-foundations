---
layout: default
---

# Fundamento Matemático y Computacional de Sistemas de Recomendación
#### Autor: Carlos César Sánchez Coronel

*(Alineado con la Semana 12: colaborativo, factorización matricial, Funk SVD, métricas de ranking.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Matriz de interacciones $\mathbf{R} \in \mathbb{R}^{m \times n}$ (usuarios $\times$ ítems), entradas observadas $r_{u,i}$ (ratings o feedback implícito). Objetivo: predecir $r_{u,i}$ no observados o rankear ítems para cada usuario.

### 1.2 Notación

- $m$: usuarios; $n$: ítems; $\mathcal{K}$: conjunto de pares observados $(u,i)$.
- Factores latentes $\mathbf{p}_u, \mathbf{q}_i \in \mathbb{R}^k$.

### 1.3 Supuestos

- **Bajo rango aproximado:** $\mathbf{R} \approx \mathbf{P}\mathbf{Q}^\top$.
- **User-based CF:** similitud estable entre usuarios con suficiente solapamiento.

---

## 2. Fundamento Matemático

### 2.1 Basado en contenido

Perfil usuario:

$$
\boxed{\mathbf{p}_u = \frac{1}{|I_u^+|}\sum_{i \in I_u^+} \mathbf{x}_i}
$$

Relevancia (coseno):

$$
\text{sim}(u,j) = \frac{\mathbf{p}_u^\top \mathbf{x}_j}{\|\mathbf{p}_u\|\|\mathbf{x}_j\|}
$$

### 2.2 Colaborativo user-based (Pearson)

$$
\text{sim}(a,b) = \frac{\sum_{i \in I_{ab}} (r_{a,i}-\bar{r}_a)(r_{b,i}-\bar{r}_b)}{\sqrt{\sum_{i\in I_{ab}}(r_{a,i}-\bar{r}_a)^2}\sqrt{\sum_{i\in I_{ab}}(r_{b,i}-\bar{r}_b)^2}}
$$

Predicción:

$$
\boxed{\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v)(r_{v,i}-\bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}}
$$

### 2.3 Factorización matricial (Funk SVD)

Modelo:

$$
\boxed{\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i}
$$

**Pérdida regularizada:**

$$
\mathcal{L} = \sum_{(u,i)\in\mathcal{K}} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda \big( \|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2 + b_u^2 + b_i^2 \big)
$$

### 2.4 Gradiente estocástico (un paso)

Error $e_{u,i} = r_{u,i} - \hat{r}_{u,i}$:

$$
\boxed{\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta(e_{u,i}\mathbf{q}_i - \lambda\mathbf{p}_u), \quad
\mathbf{q}_i \leftarrow \mathbf{q}_i + \eta(e_{u,i}\mathbf{p}_u - \lambda\mathbf{q}_i)}
$$

Análogamente $b_u$, $b_i$, $\mu$.

### 2.5 Optimización

SGD, ALS (alternar mínimos cuadrados en $\mathbf{P}$ y $\mathbf{Q}$), BPR para ranking implícito.

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo SGD Funk SVD

```
Inicializar P, Q, b_u, b_i, μ
Para épocas:
  Barajar (u,i) en K
  Para cada (u,i):
    e ← r_{u,i} - (μ + b_u + b_i + p_u^T q_i)
    Actualizar por gradiente con η y λ
```

- **Por época:** $O(|\mathcal{K}| \cdot k)$.
- **Espacio:** $O((m+n)k)$.

### 3.2 Numpy: un paso de actualización

```python
import numpy as np

def svd_step(p_u, q_i, b_u, b_i, mu, r, eta, lam):
    pred = mu + b_u + b_i + p_u @ q_i
    e = r - pred
    p_u_n = p_u + eta * (e * q_i - lam * p_u)
    q_i_n = q_i + eta * (e * p_u - lam * q_i)
    b_u_n = b_u + eta * (e - lam * b_u)
    b_i_n = b_i + eta * (e - lam * b_i)
    mu_n = mu + eta * (e - lam * mu)  # opcional actualizar μ
    return p_u_n, q_i_n, b_u_n, b_i_n, mu_n
```

### 3.3 Escalamiento

- **Grandes matrices:** muestreo negativo, factorización implícita, factorización distribuida.
- **Ítems cold-start:** híbridos con contenido.

---

## 4. Métricas de Evaluación Específicas

**RMSE / MAE** sobre $\mathcal{K}_{\text{test}}$.

**Ranking:**

$$
\text{Precision@}k = \frac{|\text{relevantes en top-}k|}{k}, \quad
\text{Recall@}k = \frac{|\text{relevantes en top-}k|}{|\text{relevantes totales}|}
$$

**NDCG@k** para graded relevance.

---

## 5. Descomposición Teórica

Factorización aproxima rango bajo; el error de aproximación + regularización controla sobreajuste a interacciones observadas.

---

## 6. Selección de Hiperparámetros

- $k$ (latent dim), $\lambda$, $\eta$, número de épocas; validación hold-out por usuario o temporal.
- **Early stopping** sobre conjunto de validación.

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Fórmula |
|----------|---------|
| Predicción MF | $\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^\top\mathbf{q}_i$ |
| Objetivo | $\sum_{(u,i)\in\mathcal{K}}(r_{u,i}-\hat{r}_{u,i})^2 + \lambda\Omega$ |
| Paso SGD | $\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta(e\mathbf{q}_i - \lambda\mathbf{p}_u)$ |

---

## 8. Referencias y Lecturas Complementarias

- Koren, Bell, Volinsky — Matrix Factorization Techniques for Recommender Systems (IEEE Computer, 2009).
- Ricci et al. — *Recommender Systems Handbook*.
- Rendle et al. — BPR (2009).
