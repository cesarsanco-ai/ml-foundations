---
layout: default
---

# Fundamento Matemático y Computacional del EDA y Feature Engineering
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-02](../../../sesiones/sesion-02.md)

*(Alineado con la Semana 2: limpieza, análisis uni/bi/multivariado, ingeniería de características y escalado.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Sea $\mathbf{X} \in \mathbb{R}^{n \times p}$ la matriz de características y $\mathbf{y} \in \mathbb{R}^n$ (o etiquetas discretas) el vector objetivo cuando aplique. El **EDA** trata de estimar propiedades de la distribución conjunta $P(\mathbf{X}, Y)$ o marginales $P(X_j)$, $P(\mathbf{X})$, y detectar desviaciones (valores faltantes, outliers, sesgos de muestreo).

El **feature engineering** construye un mapa $\phi: \mathbb{R}^p \to \mathbb{R}^{p'}$ tal que un modelo $f \circ \phi$ minimice mejor el riesgo empírico:

$$
\min_\theta \frac{1}{n} \sum_{i=1}^n \ell\big( f(\phi(\mathbf{x}_i); \theta), y_i \big) + \Omega(\theta)
$$

### 1.2 Notación

- $x_{ij}$: valor de la variable $j$ en la observación $i$.
- $\mu_j$, $\sigma_j$: media y desviación típica de la columna $j$.
- $\mathbf{x}_i \in \mathbb{R}^p$: vector de características de la observación $i$.

### 1.3 Supuestos

- Para **Pearson**, relación lineal y comportamiento razonable de varianzas (sensible a outliers).
- Para **imputación por media**, MCAR/MAR simplificados; de lo contrario sesgo.
- **PCA** asume que la varianza lineal captura la información relevante.

---

## 2. Fundamento Matemático

### 2.1 Estadísticos univariados

**Media y varianza muestral:**

$$
\bar{x}_j = \frac{1}{n}\sum_{i=1}^n x_{ij}, \quad
s_j^2 = \frac{1}{n-1}\sum_{i=1}^n (x_{ij} - \bar{x}_j)^2
$$

**Asimetría y curtosis** describen colas y forma de la distribución (útil para decidir transformaciones).

### 2.2 Detección de outliers

**IQR:** para la variable $j$, con cuartiles $Q_1, Q_3$ y $IQR = Q_3 - Q_1$, regla típica:

$$
\text{outlier si } x_{ij} < Q_1 - 1.5 \cdot IQR \quad \text{o} \quad x_{ij} > Q_3 + 1.5 \cdot IQR
$$

**Z-score:** $z_{ij} = (x_{ij} - \bar{x}_j)/s_j$; umbral $|z| > 3$ como heurística.

**Z-score robusto (MAD):**

$$
\text{MAD}_j = \text{mediana}_i(|x_{ij} - \text{mediana}_j|), \quad
z^*_ij = \frac{0.6745\,(x_{ij} - \text{mediana}_j)}{\text{MAD}_j}
$$

### 2.3 Correlación

**Pearson** entre variables $a$ y $b$:

$$
\boxed{\rho_{ab} = \frac{\sum_i (x_{ia}-\bar{x}_a)(x_{ib}-\bar{x}_b)}{\sqrt{\sum_i (x_{ia}-\bar{x}_a)^2}\sqrt{\sum_i (x_{ib}-\bar{x}_b)^2}}}
$$

**Spearman:** Pearson aplicado a rangos; mide monotonía, más robusta a outliers.

### 2.4 Distancia de Mahalanobis (outliers multivariantes)

Con matriz de covarianza muestral $\mathbf{S}$:

$$
\boxed{d_M^2(\mathbf{x}_i, \boldsymbol{\mu}) = (\mathbf{x}_i - \boldsymbol{\mu})^\top \mathbf{S}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})}
$$

### 2.5 PCA (reducción lineal)

Datos centrados en matriz $\tilde{\mathbf{X}}$. La matriz de covarianza $\mathbf{S} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}$. Autovalores $\lambda_1 \ge \cdots \ge \lambda_p$ y autovectores $\mathbf{v}_j$:

$$
\boxed{\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{V}_k, \quad \mathbf{V}_k = [\mathbf{v}_1,\ldots,\mathbf{v}_k]}
$$

Varianza explicada por componente $j$: $\lambda_j / \sum_{\ell=1}^p \lambda_\ell$.

### 2.6 Transformaciones

**Box-Cox** (requiere $x > 0$):

$$
x^{(\lambda)} = \frac{x^\lambda - 1}{\lambda} \quad (\lambda \neq 0), \quad \log x \quad (\lambda = 0)
$$

**Yeo-Johnson:** extiende a valores con signo.

**Min-Max a $[0,1]$:** $x' = (x - x_{\min})/(x_{\max} - x_{\min})$.

**Estandarización:** $z = (x - \mu)/\sigma$.

### 2.7 Codificación cíclica (tiempo)

Para período $P$ (ej. hora 24):

$$
\boxed{x_{\sin} = \sin\left(\frac{2\pi h}{P}\right), \quad x_{\cos} = \cos\left(\frac{2\pi h}{P}\right)}
$$

### 2.8 TF-IDF (texto)

Para término $t$ en documento $d$:

$$
\text{tf-idf}(t,d) = f_{t,d} \cdot \log\frac{N}{|\{d': t \in d'\}|}
$$

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo: estandarización column-wise

```
Para j = 1 … p:
  μ_j ← mean(X[:, j])
  σ_j ← std(X[:, j])
  X[:, j] ← (X[:, j] - μ_j) / σ_j
```

Complejidad: **$O(np)$** tiempo, **$O(np)$** espacio.

### 3.2 PCA vía SVD (estable numéricamente)

```python
import numpy as np

def pca_svd(X, k):
    """X: (n, p) centrado por columnas."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :k] * s[:k]  # scores hasta escala; o Xc @ Vt[:k].T
    return Z, Vt[:k].T, (s**2) / (X.shape[0] - 1)
```

### 3.3 Escalamiento

- **$n$ grande:** estadísticos online (Welford) para media/varianza; hashing para features categóricas de alta cardinalidad.
- **$p$ grande:** PCA aproximado (randomized SVD); sparse matrices para TF-IDF.

---

## 4. Métricas de Evaluación Específicas

| Contexto | Métrica | Uso |
|----------|---------|-----|
| Calidad de imputación | RMSE sobre valores simulados | Comparar estrategias |
| Reducción PCA | Varianza acumulada | Elegir $k$ |
| Asociación categórica | $\chi^2$, Cramér's V | EDA bivariado |
| Desbalance | Proporción de clases | Decidir métricas de modelo |

---

## 5. Descomposición Teórica

- **Sesgo en imputación:** sustituir por media reduce varianza aparente y correlaciones.
- **Data leakage:** si $\phi$ usa estadísticos de test, el riesgo empírico está sesgado hacia abajo.

---

## 6. Selección de Hiperparámetros (preprocesado)

- Número de componentes PCA: codo en scree plot o umbral de varianza (ej. 90%).
- Grado polinomial / bins: validación cruzada en pipeline conjunto con el modelo (`sklearn.pipeline`).

---

## 7. Ecuaciones Clave (resumen)

| Tema | Ecuación |
|------|----------|
| Covarianza muestral | $S_{ab} = \frac{1}{n-1}\sum_i (x_{ia}-\bar{x}_a)(x_{ib}-\bar{x}_b)$ |
| Pearson | $\rho_{ab} = S_{ab}/(s_a s_b)$ |
| Mahalanobis | $d_M^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{S}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ |
| Proyección PCA | $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}_k$ |

---

## 8. Referencias y Lecturas Complementarias

- Tukey — *Exploratory Data Analysis*.
- Hastie et al. — *ESL* (cap. sobre splines, wavelets y reducción de dimensión).
- Jolliffe — *Principal Component Analysis*.
