---
layout: default
---

# Fundamento Matemático y Computacional de Modelos Complementarios (Regresión robusta, SVR, clustering avanzado)
#### Autor: Carlos César Sánchez Coronel

*(Alineado con la Semana 11: polinomios, Huber, RANSAC, SVR, DBSCAN en profundidad, jerárquico.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

**Regresión:** estimar $f$ tal que $y \approx f(\mathbf{x}) + \varepsilon$ con $\varepsilon$ posiblemente con colas pesadas (outliers).

**SVR:** encontrar $f(\mathbf{x}) = \mathbf{w}^\top\phi(\mathbf{x}) + b$ plana en norma $\|\mathbf{w}\|$ tolerando desviaciones $\varepsilon$ y penalizando violaciones.

**Clustering densidad/jerárquico:** extensiones de k-means cuando la geometría o densidad no son homogéneas.

### 1.2 Notación

- Residual $r_i = y_i - f(\mathbf{x}_i)$.
- Slack $\xi_i, \xi_i^*$ en SVR.

---

## 2. Fundamento Matemático

### 2.1 Regresión polinómica

Features $\phi(\mathbf{x}) = (1, x_1, x_1^2, \ldots)$; el modelo sigue **lineal en parámetros**:

$$
\boxed{y = \boldsymbol{\beta}^\top \phi(\mathbf{x}) + \varepsilon}
$$

Mismo formalismo OLS con riesgo de overfitting si el grado es alto.

### 2.2 Pérdida de Huber

$$
\boxed{L_\delta(r) = \begin{cases}
\frac{1}{2} r^2 & |r| \le \delta \\
\delta(|r| - \frac{1}{2}\delta) & |r| > \delta
\end{cases}}
$$

Combina cuadratica cerca de 0 (eficiente bajo ruido gaussiano) y lineal en colas (robusta).

### 2.3 RANSAC (idea de optimización)

Minimizar número de outliers bajo umbral de residual, explorando subconjuntos aleatorios; no convexo, solución aproximada.

### 2.4 SVR $\varepsilon$-insensible

Primal (formulación típica):

$$
\min_{\mathbf{w},b,\boldsymbol{\xi},\boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i (\xi_i + \xi_i^*)
$$

s.a. $y_i - \mathbf{w}^\top\phi(\mathbf{x}_i) - b \le \varepsilon + \xi_i$, $\mathbf{w}^\top\phi(\mathbf{x}_i) + b - y_i \le \varepsilon + \xi_i^*$, $\xi_i,\xi_i^* \ge 0$.

**Dual** con kernel $K$:

$$
f(\mathbf{x}) = \sum_i (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b
$$

con restricciones de caja en $\alpha_i,\alpha_i^*$.

### 2.5 DBSCAN — densidad

Definiciones estándar de **núcleo**, **borde**, **ruido** según $\varepsilon$ y `minPts`.

### 2.6 Enlace de Ward

Al fusionar clusters $A,B$, minimizar incremento de **SSE**:

$$
\Delta W(A,B) = \frac{n_A n_B}{n_A+n_B}\|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2
$$

---

## 3. Algoritmos Computacionales

### 3.1 Huber — IRLS (esquema)

Pesos $w_i$ según magnitud de $r_i$; resolver WLS ponderado hasta convergencia. **Complejidad** similar a mínimos cuadrados por iteración.

### 3.2 Pseudocódigo RANSAC (regresión lineal)

```
mejor_modelo ← None, mejor_inliers ← 0
Repetir N veces:
  Muestra mínima aleatoria S
  Ajustar modelo a S
  Contar inliers con |r_i| < τ
  Si mejora, guardar modelo y reajustar con todos los inliers
```

### 3.3 Numpy: pérdida Huber

```python
import numpy as np

def huber_loss(r, delta=1.0):
    abs_r = np.abs(r)
    quad = 0.5 * r ** 2
    lin = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quad, lin)
```

---

## 4. Métricas de Evaluación Específicas

- Regresión robusta: MAE, Huber loss en validación.
- SVR: $\varepsilon$-insensitive loss, RMSE complementario.
- Clustering: silueta (con precaución en ruido), DBCV para densidad.

---

## 5. Descomposición Teórica

Huber interpola entre eficiencia de MSE y robustez de MAE. SVR controla capacidad vía $\|\mathbf{w}\|$ (margen en feature space).

---

## 6. Selección de Hiperparámetros

- Polinomio: validación cruzada del grado.
- Huber: $\delta$ relacionado con escala de ruido.
- SVR: $C$, $\varepsilon$, $\gamma$ (RBF).
- DBSCAN: k-dist plot para $\varepsilon$.

---

## 7. Ecuaciones Clave (resumen)

| Método | Expresión |
|--------|-----------|
| Huber | $L_\delta(r)$ mixta cuadrática/lineal |
| SVR | $f(\mathbf{x}) = \sum (\alpha_i-\alpha_i^*)K(\mathbf{x}_i,\mathbf{x})+b$ |
| Ward | $\Delta W \propto \|\boldsymbol{\mu}_A-\boldsymbol{\mu}_B\|^2$ |

---

## 8. Referencias y Lecturas Complementarias

- Huber — *Robust Statistics*.
- Vapnik — *Statistical Learning Theory* (SVR).
- Campello, Moulavi, Sander — HDBSCAN.
