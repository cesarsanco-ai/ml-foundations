---
layout: default
---

# Fundamento Matemático y Computacional del Gradient Boosting (XGBoost / LightGBM)
#### Autor: Carlos César Sánchez Coronel

*(Alineado con la Semana 7: boosting aditivo, pseudo-residuos, shrinkage, regularización de árbol.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Dado $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, buscar una función aditiva:

$$
\boxed{F_M(\mathbf{x}) = \sum_{m=0}^M \nu \, h_m(\mathbf{x}; \theta_m)}
$$

donde $h_m$ son **árboles de regresión** (funciones base en regiones constantes), $\nu \in (0,1]$ es **learning rate** (shrinkage), y $F_0$ es constante (ej. media para MSE, log-odds para log-loss).

### 1.2 Objetivo

$$
\min_{F} \sum_{i=1}^n L(y_i, F(\mathbf{x}_i)) + \sum_m \Omega(h_m)
$$

$\Omega$: regularización de complejidad del árbol (XGBoost).

### 1.3 Notación

- $r_{im}$: pseudo-residuo (gradiente negativo) en la iteración $m$.
- $\gamma_{jm}$: valor óptimo en hoja $j$ del árbol $m$.
- $T$: número de hojas; $w$: vector de pesos por hoja.

---

## 2. Fundamento Matemático

### 2.1 Descenso de gradiente en espacio funcional (gradient boosting)

En la iteración $m$, sea $F_{m-1}$ fijo. Aproximación de primer orden: añadir $h_m$ que aproxime el **gradiente negativo** respecto a $F$ evaluado en $F_{m-1}$:

$$
\boxed{r_{im} = -\left.\frac{\partial L(y_i, F)}{\partial F}\right|_{F = F_{m-1}(\mathbf{x}_i)}}
$$

**MSE:** $L = \frac{1}{2}(y_i - F)^2$ → $r_{im} = y_i - F_{m-1}(\mathbf{x}_i)$.

**Log-loss binaria** con $F$ en escala logit: $r_{im} = y_i - \sigma(F_{m-1}(\mathbf{x}_i))$.

### 2.2 Ajuste del árbol y actualización

1. Entrenar $h_m$ minimizando $\sum_i (r_{im} - h_m(\mathbf{x}_i))^2$ (o segundo orden en XGBoost).
2. Por hoja $j$, elegir $\gamma_{jm}$ que minimice $\sum_{i \in R_{jm}} L(y_i, F_{m-1}(\mathbf{x}_i) + \gamma)$.
3. $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \sum_j \gamma_{jm} \mathbb{1}[\mathbf{x} \in R_{jm}]$.

### 2.3 XGBoost — regularización del árbol

Para un árbol con $T$ hojas y pesos $w_j$ en hojas:

$$
\boxed{\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|_2^2 + \alpha \|w\|_1}
$$

**Ganancia aproximada de un split** (segundo orden, con $g_i = \partial_{\hat{y}} L$, $h_i = \partial^2_{\hat{y}} L$):

$$
\mathcal{L}_{\text{split}} \approx \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

con $G = \sum_{i \in \text{nodo}} g_i$, $H = \sum_{i \in \text{nodo}} h_i$.

### 2.4 Optimización

- **Greedy:** splits que maximizan ganancia; profundidad limitada.
- **LightGBM:** histogramas sobre bins; GOSS/EFB para velocidad.

### 2.5 Regularización práctica

- Shrinkage $\nu$, `subsample` (filas), `colsample_bytree`, `max_depth`, `min_child_weight`, $\lambda$, $\alpha$, $\gamma$.

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo Gradient Boosting (MSE)

```
F0 ← mean(y)
Para m = 1 … M:
  r_i ← y_i - F_{m-1}(x_i)
  Entrenar árbol h_m a {(x_i, r_i)}
  F_m ← F_{m-1} + ν * h_m
```

### 3.2 Complejidad (orden de magnitud)

- Por árbol: similar a CART, $O(p \cdot n \log n)$ o menos con histogramas $O(p \cdot n \cdot B_{\text{bins}})$.
- Total: $O(M \cdot \text{coste árbol})$; secuencial en $M$, paralelizable dentro del árbol.

### 3.3 Numpy: un paso de boosting (stump simplificado)

```python
import numpy as np

def stump_fit_residuals(X, y, residual, feature_idx):
    """Un nodo: predicción constante = media del residual en partición."""
    x = X[:, feature_idx]
    best_gain, best_t, best_left, best_right = -np.inf, None, 0.0, 0.0
    for t in np.unique(x)[:-1]:
        left = residual[x <= t]
        right = residual[x > t]
        gain = left.var() * len(left) + right.var() * len(right)  # proxy MSE
        gain = -gain
        if gain > best_gain:
            best_gain, best_t = gain, t
            best_left, best_right = left.mean(), right.mean()
    return feature_idx, best_t, best_left, best_right

def apply_stump(X, stump):
    j, t, pl, pr = stump
    return np.where(X[:, j] <= t, pl, pr)
```

---

## 4. Métricas de Evaluación Específicas

- **Regresión:** RMSE, MAE, quantile loss si se modela.
- **Clasificación:** log-loss, AUC, F1.
- **Early stopping:** monitorizar métrica en validación vs número de árboles.

---

## 5. Descomposición Teórica

Boosting reduce **sesgo** secuencialmente; con shrinkage y regularización controla **varianza**. Riesgo de overfitting si $M$ es grande sin regularizar.

---

## 6. Selección de Hiperparámetros

- `learning_rate` $\times$ `n_estimators`: trade-off; $\nu$ pequeño + muchos árboles suele generalizar mejor.
- Grid/Random/Bayesian search con validación cruzada.
- **Early stopping** con `validation_fraction` o conjunto hold-out.

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Fórmula |
|----------|---------|
| Modelo aditivo | $F_M = \sum_m \nu h_m$ |
| Pseudo-residuo | $r_i = -\partial L/\partial F$ |
| Shrinkage | $F_m = F_{m-1} + \nu h_m$ |
| Penalización XGB | $\gamma T + \frac{\lambda}{2}\|w\|^2 + \alpha\|w\|_1$ |

---

## 8. Referencias y Lecturas Complementarias

- Friedman — *Greedy Function Approximation: A Gradient Boosting Machine* (2001).
- Chen & Guestrin — XGBoost (KDD 2016).
- Ke et al. — LightGBM (NIPS 2017).
