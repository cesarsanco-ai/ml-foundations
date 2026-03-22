---
layout: default
---

# Fundamento Matemático y Computacional de la Regresión Logística y el Balanceo de Clases
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-04](../../../sesiones/sesion-04.md)

*(Alineado con la Semana 4: clasificación binaria/multiclase, log-loss, regularización, técnicas de balanceo y métricas.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

**Clasificación binaria:** $\mathcal{Y} = \{0,1\}$. Se modela:

$$
\boxed{P(y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1+e^{-z}}, \quad z = \beta_0 + \mathbf{x}^\top \boldsymbol{\beta}}
$$

**Multiclase (softmax):** $k \in \{1,\ldots,K\}$:

$$
P(y=k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad z_k = \beta_{0k} + \mathbf{x}^\top \boldsymbol{\beta}_k
$$

### 1.2 Notación

- $N$ o $n$: número de observaciones; $p$: número de features.
- $y_i \in \{0,1\}$ o one-hot $y_{ik}$ en multiclase.
- $\hat{p}_i = P(y_i=1 \mid \mathbf{x}_i)$.

### 1.3 Supuestos

- **Logística:** log-odds lineal en $\mathbf{x}$ (linealidad en el espacio de logit).
- **Independencia** condicional entre muestras para verosimilitud.

---

## 2. Fundamento Matemático

### 2.1 Verosimilitud y log-loss

$$
L(\boldsymbol{\beta}) = \prod_{i=1}^n \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}
$$

$$
\boxed{\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \big[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \big]}
$$

**Pérdida a minimizar (media):** $J(\boldsymbol{\beta}) = -\frac{1}{n}\ell(\boldsymbol{\beta})$.

**Multiclase:**

$$
J = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}
$$

### 2.2 Derivación del gradiente (binaria)

$\hat{p}_i = \sigma(z_i)$, $\sigma' = \sigma(1-\sigma)$:

$$
\frac{\partial z_i}{\partial \beta_j} = x_{ij}, \quad
\frac{\partial \hat{p}_i}{\partial \beta_j} = \hat{p}_i(1-\hat{p}_i)x_{ij}
$$

Cadena en $-y_i\log\hat{p}_i$:

$$
\frac{\partial J}{\partial \beta_j} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i) x_{ij}
$$

En forma compacta con $X$ y vector residual:

$$
\boxed{\nabla_{\boldsymbol{\beta}} J = \frac{1}{n} X^\top (\hat{\mathbf{p}} - \mathbf{y})}
$$

### 2.3 Regularización

**Elastic Net:**

$$
J_{\text{reg}} = J(\boldsymbol{\beta}) + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2
$$

(subgradiente para L1).

### 2.4 Optimización

- GD, SGD, L-BFGS, Newton (segunda orden en problemas convexos moderados).

**Convexidad:** $J$ es convexa en $\boldsymbol{\beta}$ para log-loss sin penalización no convexa adicional.

### 2.5 Ponderación de clases (cost-sensitive)

$$
\boxed{J_w = -\frac{1}{n}\sum_{i=1}^n w_{y_i}\big[ y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\big]}
$$

**Balanced** en sklearn binario: $w_c = \frac{n}{K n_c}$ con $K=2$.

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo: SGD logística binaria

```
Inicializar β
Para cada época:
  Para cada minibatch B:
    z_i = β0 + x_i^T β,  p_i = σ(z_i)
    g = (1/|B|) sum_i (p_i - y_i) [1; x_i]
    β ← β - η g
```

- **Tiempo por época (batch completo):** $O(np)$.
- **Espacio:** $O(p)$ parámetros + datos.

### 3.2 Numpy: una pasada de gradiente

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def logistic_gradient(X, y, beta):
    """X: (n,p) con columna de unos; beta: (p,)"""
    z = X @ beta
    p = sigmoid(z)
    return (X.T @ (p - y)) / X.shape[0]

def sgd_logistic(X, y, lr=0.5, epochs=100):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            xi = X[i : i + 1]
            yi = y[i : i + 1]
            pi = sigmoid(xi @ beta)
            g = (pi - yi) * xi
            beta -= lr * g.ravel()
    return beta
```

### 3.3 Escalamiento

- **SMOTE:** interpola en espacio de features — coste $O(n_{\min})$ por sintético; atención a leakage si se aplica antes del split.
- **Grandes $n$:** class weights + subsampling mayoritario.

---

## 4. Métricas de Evaluación Específicas

$$
\text{Precision} = \frac{TP}{TP+FP}, \quad
\text{Recall} = \frac{TP}{TP+FN}, \quad
F_1 = \frac{2PR}{P+R}
$$

**AUC-ROC:** probabilidad de ranking correcto entre par positivo-negativo.

**PR-AUC:** preferible con clases raras.

---

## 5. Descomposición Teórica

- **Desbalance:** el mínimo de $J$ sin pesos puede coincidir con predicción trivial mayoritaria.
- **Calibración:** oversampling puede distorsionar probabilidades; pesos suelen preservar interpretación si el modelo está bien especificado.

---

## 6. Selección de Hiperparámetros

- $\lambda$ (regularización): grid con validación estratificada.
- Umbral de decisión: optimizar F1 o costo esperado en validación (no fijar 0.5).
- **Threshold tuning:** $\hat{y}_i = \mathbb{1}[\hat{p}_i \ge \tau]$.

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Fórmula |
|----------|---------|
| Logit | $\log\frac{p}{1-p} = \beta_0 + \mathbf{x}^\top\boldsymbol{\beta}$ |
| Softmax | $P(y=k)\propto e^{z_k}$ |
| Gradiente binario | $\frac{1}{n}X^\top(\hat{\mathbf{p}}-\mathbf{y})$ |
| F1 | $2PR/(P+R)$ |

---

## 8. Referencias y Lecturas Complementarias

- Hosmer, Lemeshow — *Applied Logistic Regression*.
- Murphy — *Probabilistic Machine Learning*.
- Chawla et al. — SMOTE (paper original).
