---
layout: default
---

# Fundamento Matemático y Computacional de la Visión General del Machine Learning
#### Autor: Carlos César Sánchez Coronel

*(Alineado con la teoría de la Semana 1: introducción al ML, tipos de aprendizaje y ciclo de vida.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal del problema de aprendizaje supervisado

Dado un conjunto de entrenamiento $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ con $\mathbf{x}_i \in \mathcal{X}$ y $y_i \in \mathcal{Y}$, el objetivo es elegir una función $f \in \mathcal{F}$ (familia de hipótesis) que **minimice el riesgo esperado**:

$$
R(f) = \mathbb{E}_{(\mathbf{x},y) \sim P}\big[ \ell(f(\mathbf{x}), y) \big]
$$

donde $P$ es la distribución conjunta desconocida y $\ell$ es una **función de pérdida** instantánea.

En la práctica solo tenemos $\mathcal{D}$; se minimiza el **riesgo empírico**:

$$
\boxed{\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(\mathbf{x}_i), y_i)}
$$

**Notación:** $n$: tamaño muestral; $p$: dimensión de $\mathbf{x}$ cuando $\mathbf{x} \in \mathbb{R}^p$; $\theta$ o $\boldsymbol{\beta}$: parámetros de modelos paramétricos $f(\mathbf{x}; \theta)$.

### 1.2 Tipos de aprendizaje (formalización breve)

| Tipo | Datos | Objetivo típico |
|------|--------|-----------------|
| Supervisado | $(\mathbf{x}_i, y_i)$ | Minimizar $\hat{R}_n(f)$ con $\ell$ acorde a regresión o clasificación |
| No supervisado | solo $\mathbf{x}_i$ | Minimizar criterio de estructura (inercia, log-verosimilitud, etc.) |
| Refuerzo | $(s,a,r,s')$ | Maximizar retorno esperado (MDP) |

### 1.3 Supuestos habituales

- **Muestra i.i.d.** (idealización): los ejemplos provienen de la misma $P$ e independientemente.
- **Modelos paramétricos:** $\mathcal{F} = \{ f(\cdot; \theta) : \theta \in \Theta \}$ con dimensión fija de $\theta$.
- **Optimización:** $\ell$ convexa en $\theta$ en modelos lineales clásicos; no convexa en redes profundas.

---

## 2. Fundamento Matemático

### 2.1 Formulación de modelos representativos (catálogo)

| Tarea | Predicción $\hat{y}$ o score | Parámetros |
|--------|------------------------------|------------|
| Regresión lineal | $\hat{y} = \mathbf{x}^\top \boldsymbol{\beta}$ (con bias en $\mathbf{x}$) | $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$ |
| Regresión logística | $\hat{p} = \sigma(\mathbf{x}^\top \boldsymbol{\beta})$ | $\boldsymbol{\beta}$ |
| SVM | $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ (+ kernel) | $\mathbf{w}, b, \alpha_i$ (dual) |

### 2.2 Funciones de pérdida / objetivo (referencia unificada)

**Regresión — MSE:**

$$
\boxed{J(\theta) = \frac{1}{n} \sum_{i=1}^n (y_i - f(\mathbf{x}_i; \theta))^2}
$$

**Clasificación binaria — log-loss (entropía cruzada):**

$$
\boxed{J(\theta) = -\frac{1}{n} \sum_{i=1}^n \big[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \big]}
$$

**SVM — hinge + regularización:**

$$
J(\mathbf{w},b) = \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \max\big(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\big), \quad y_i \in \{-1,1\}
$$

**Clustering k-means — inercia:**

$$
J = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

### 2.3 Derivación ejemplo: gradiente para MSE lineal

Modelo $\hat{y}_i = \mathbf{x}_i^\top \boldsymbol{\beta}$, matriz de diseño $X$ filas $\mathbf{x}_i^\top$:

$$
J(\boldsymbol{\beta}) = \frac{1}{n}\| \mathbf{y} - X\boldsymbol{\beta} \|_2^2, \quad
\boxed{\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = -\frac{2}{n} X^\top (\mathbf{y} - X\boldsymbol{\beta})}
$$

### 2.4 Optimización

- **Descenso de gradiente (batch):** $\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)$.
- **SGD / mini-batch:** estimación ruidosa del gradiente; $\alpha$ puede programarse.
- **Solución cerrada (OLS):** $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}$ si $X^\top X$ es invertible.

**Convergencia (convexo, Lipschitz):** con paso adecuado, descenso de gradiente converge a mínimo global para funciones convexas suaves.

### 2.5 Regularización

**Ridge (L2):** $J + \lambda \|\theta\|_2^2$ — encoge coeficientes, estable numéricamente.

**Lasso (L1):** $J + \lambda \|\theta\|_1$ — favorece esparcidad (selección de variables).

**Interpretación geométrica:** intersección de contornos de pérdida con bola L2 o cruce L1 en el óptimo.

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo: descenso de gradiente para MSE lineal

```
1. Inicializar β (ceros o pequeño aleatorio)
2. Para t = 1 … T_max:
3.   g ← -(2/n) X^T (y - X β)
4.   β ← β - α g
5.   Si ||g|| < ε, parar
```

- **Tiempo por iteración:** $O(np)$ para el producto $X^\top(\mathbf{y}-X\boldsymbol{\beta})$.
- **Espacio:** $O(np)$ almacenando $X$.

### 3.2 Implementación vectorizada (numpy)

```python
import numpy as np

def gradient_descent_linear_mse(X, y, lr=0.1, n_iter=1000, tol=1e-6):
    """X: (n, p) con columna de unos si hay intercepto; y: (n,)"""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(n_iter):
        residual = y - X @ beta
        g = -(2.0 / n) * (X.T @ residual)
        beta = beta - lr * g
        if np.linalg.norm(g) < tol:
            break
    return beta

# Versión cerrada OLS (pequeña/mediana p)
def ols_closed_form(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]
```

### 3.3 Escalamiento

- **$n$ grande:** mini-batch SGD; algoritmos online; feature hashing.
- **$p$ grande:** regularización; reducción de dimensionalidad; métodos sparse.
- **Paralelización:** productos matriz-vector por bloques; entrenamiento distribuido en modelos que lo permitan.

---

## 4. Métricas de Evaluación Específicas (por paradigma)

| Paradigma | Métricas | Notas |
|-----------|----------|--------|
| Regresión | MAE, RMSE, $R^2$ | RMSE penaliza outliers más que MAE |
| Clasificación | Accuracy, Precision, Recall, F1, AUC-ROC | Accuracy engaña con clases desbalanceadas |
| Clustering | Silueta, Davies–Bouldin, inercia | Sin etiquetas de verdad, interpretación cautelosa |

---

## 5. Descomposición Teórica

**Sesgo–varianza (regresión cuadrática):**

$$
\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f^*(x)\big)^2}_{\text{sesgo}^2} + \underbrace{\mathbb{V}(\hat{f}(x))}_{\text{varianza}} + \sigma^2
$$

Modelos simples → más sesgo, menos varianza; modelos complejos → lo contrario.

**Interpretación probabilística:** minimizar log-loss equivale a maximizar verosimilitud en modelos Bernoulli.

---

## 6. Selección de Hiperparámetros

- **Validación cruzada $k$-fold** sobre train para estimar riesgo de generalización.
- **Conjunto de validación** separado si $n$ es muy grande.
- **Criterios de información (AIC/BIC)** en modelos estadísticos paramétricos.

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Ecuación |
|----------|----------|
| Riesgo empírico | $\hat{R}_n(f)=\frac{1}{n}\sum_i \ell(f(\mathbf{x}_i),y_i)$ |
| Gradiente MSE lineal | $\nabla J = -\frac{2}{n}X^\top(\mathbf{y}-X\boldsymbol{\beta})$ |
| Paso GD | $\theta \leftarrow \theta - \alpha \nabla J$ |
| Ridge | $J + \lambda\|\theta\|_2^2$ |

---

## 8. Referencias y Lecturas Complementarias

- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*.
- Bishop — *Pattern Recognition and Machine Learning*.
- Goodfellow, Bengio, Courville — *Deep Learning* (cap. de ML clásico y optimización).
