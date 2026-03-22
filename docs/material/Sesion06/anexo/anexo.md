---
layout: default
---

# Fundamento Matemático y Computacional de Árboles de Decisión y Random Forest
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-06](../../../sesiones/sesion-06.md)

*(Alineado con la Semana 6: impureza, poda, bagging, OOB y complejidad.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

**Clasificación:** particionar $\mathcal{X}$ en regiones $\{R_m\}_{m=1}^M$ y asignar clase constante $\hat{y} = c_m$ si $\mathbf{x} \in R_m$.

**Regresión:** $\hat{y} = \bar{y}_m$ (media local) en $R_m$.

El árbol se construye **recursivamente** eligiendo división $(j, t)$ que maximice reducción de impureza o de MSE.

### 1.2 Notación

- $N_m$: número de muestras en nodo $m$.
- $\hat{p}_{mk}$: proporción de clase $k$ en nodo $m$.
- $B$: número de árboles en el bosque.

### 1.3 Supuestos

- Divisiones paralelas a ejes (CART estándar).
- **Random Forest:** bootstrap i.i.d. y submuestreo de features en cada split.

---

## 2. Fundamento Matemático

### 2.1 Impureza Gini (clasificación)

$$
\boxed{G_m = \sum_{k=1}^K \hat{p}_{mk}(1-\hat{p}_{mk}) = 1 - \sum_{k=1}^K \hat{p}_{mk}^2}
$$

### 2.2 Entropía e information gain

$$
H_m = -\sum_{k=1}^K \hat{p}_{mk} \log_2 \hat{p}_{mk}
$$

$$
\Delta H = H_m - \frac{N_L}{N_m} H_L - \frac{N_R}{N_m} H_R
$$

### 2.3 MSE en regresión

En nodo $m$, predicción $\bar{y}_m = \frac{1}{N_m}\sum_{i \in m} y_i$:

$$
\boxed{\text{MSE}_m = \frac{1}{N_m}\sum_{i \in m} (y_i - \bar{y}_m)^2}
$$

### 2.4 Reducción de impureza al dividir

$$
\boxed{\Delta I = I_m - \frac{N_L}{N_m} I_L - \frac{N_R}{N_m} I_R}
$$

Se elige $(j^*, t^*) = \arg\max_{j,t} \Delta I$.

### 2.5 Poda por cost-complexity (CART)

Para subárbol $T$:

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

$|T|$: número de hojas; $\alpha$ controla trade-off complejidad-ajuste.

### 2.6 Random Forest — varianza del promedio

Árboles correlacionados $\rho$, varianza individual $\sigma^2$:

$$
\boxed{\mathbb{V}\left[\frac{1}{B}\sum_{b=1}^B \hat{f}^{(b)}\right] = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2}
$$

### 2.7 Bootstrap — fracción OOB

$$
P(\text{muestra } i \text{ en bootstrap de } n) = 1 - \left(1-\frac{1}{n}\right)^n \xrightarrow{n\to\infty} 1 - e^{-1} \approx 0.632
$$

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo árbol (clasificación)

```
ConstruirNodo(D):
  Si criterio_parada(D): devolver hoja(majority_class(D))
  Buscar mejor (j, t) maximizando ΔI
  Partir D en D_L, D_R según x_j ≤ t
  return Nodo(j, t, ConstruirNodo(D_L), ConstruirNodo(D_R))
```

- **Entrenamiento típico:** $O(p \cdot n \log n)$ promedio (ordenar por feature).
- **Predicción:** $O(\text{profundidad})$ por instancia.

### 3.2 Pseudocódigo Random Forest

```
Para b = 1 … B:
  D_b ← bootstrap de tamaño n desde D
  Entrenar árbol profundo en D_b usando solo √p features aleatorias por split
Predicción ← promedio (regresión) o voto (clasificación)
```

### 3.3 Numpy: impureza Gini vectorizada en un nodo

```python
import numpy as np

def gini(proportions):
    p = np.asarray(proportions)
    return 1.0 - np.sum(p ** 2)

def gini_node(y, classes):
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    p = counts / counts.sum()
    return gini(p)
```

### 3.4 Escalamiento

- Paralelizar por árbol (`n_jobs`).
- Para $n$ enorme: subsampling de filas (`max_samples`), límites de profundidad.

---

## 4. Métricas de Evaluación Específicas

- Clasificación: F1, AUC, log-loss si se promedian probabilidades de hojas.
- Regresión: RMSE, MAE, $R^2$.
- **OOB error:** estimación out-of-bag sin conjunto de validación explícito.

---

## 5. Descomposición Teórica

$$
\mathbb{E}[(y - \hat{f})^2] = \text{Sesgo}^2 + \text{Varianza} + \sigma^2
$$

Árbol profundo: bajo sesgo, alta varianza. **Bagging/RF** reduce varianza manteniendo sesgo similar.

---

## 6. Selección de Hiperparámetros

- `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha`.
- RF: `n_estimators` (crecer hasta plateau), `max_features` (`sqrt`, `log2`).

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Fórmula |
|----------|---------|
| Gini | $G = 1 - \sum_k p_k^2$ |
| Reducción | $\Delta I = I_m - \frac{N_L}{N_m}I_L - \frac{N_R}{N_m}I_R$ |
| Varianza ensemble | $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ |
| MSE nodo | $\frac{1}{N_m}\sum_{i\in m}(y_i-\bar{y}_m)^2$ |

---

## 8. Referencias y Lecturas Complementarias

- Breiman — Random Forests (2001).
- Hastie et al. — *ESL* (árboles y bagging).
- Loh — *Classification and regression trees* (historia CART).
