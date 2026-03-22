
## Anexo: 
# Fundamento Matemático y Computacional de la Regresión Lineal y Regularización
#### Autor: Carlos César Sánchez Coronel

## 1. Planteamiento General del Problema de Regresión

### 1.1 Definición del problema

Dado un conjunto de datos $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ donde:
- $\mathbf{x}_i \in \mathbb{R}^p$ son las variables predictoras (características)
- $y_i \in \mathbb{R}$ es la variable objetivo (respuesta continua)

El objetivo es encontrar una función $f: \mathbb{R}^p \rightarrow \mathbb{R}$ que mejor aproxime la relación entre $\mathbf{x}_i$ e $y_i$.

### 1.2 Modelo de regresión lineal múltiple

El modelo lineal asume que la relación es lineal en los parámetros:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \varepsilon_i, \quad i = 1,\dots,n
$$

donde:
- $\beta_0$ es el intercepto (término constante)
- $\beta_j$ son los coeficientes de regresión
- $\varepsilon_i$ es el error aleatorio (ruido) con $E[\varepsilon_i] = 0$ y $Var(\varepsilon_i) = \sigma^2$

### 1.3 Representación matricial

Definiendo:

$$
\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}, \quad 
X = \begin{bmatrix} 
1 & x_{11} & x_{12} & \dots & x_{1p} \\
1 & x_{21} & x_{22} & \dots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \dots & x_{np}
\end{bmatrix}, \quad
\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{bmatrix}, \quad
\boldsymbol{\varepsilon} = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}
$$

El modelo se expresa como:

$$
\boxed{\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}}
$$

---

## 2. Función de Pérdida y Optimización

### 2.1 Función de pérdida: Error Cuadrático Medio (MSE)

La función de pérdida a minimizar es el Error Cuadrático Medio:

$$
J(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2
$$

donde $\hat{y}_i = \mathbf{x}_i^T \boldsymbol{\beta}$ es la predicción para la observación $i$.

### 2.2 Forma matricial de la función de pérdida

Expandiendo en notación matricial:

$$
J(\boldsymbol{\beta}) = \frac{1}{n} (\mathbf{y} - X\boldsymbol{\beta})^T (\mathbf{y} - X\boldsymbol{\beta})
$$

Desarrollando:

$$
J(\boldsymbol{\beta}) = \frac{1}{n} \left( \mathbf{y}^T\mathbf{y} - \mathbf{y}^T X\boldsymbol{\beta} - \boldsymbol{\beta}^T X^T \mathbf{y} + \boldsymbol{\beta}^T X^T X \boldsymbol{\beta} \right)
$$

Dado que $\mathbf{y}^T X\boldsymbol{\beta}$ es un escalar, es igual a su transpuesta $\boldsymbol{\beta}^T X^T \mathbf{y}$, por lo tanto:

$$
J(\boldsymbol{\beta}) = \frac{1}{n} \left( \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T X^T \mathbf{y} + \boldsymbol{\beta}^T X^T X \boldsymbol{\beta} \right)
$$

### 2.3 Derivación del gradiente

Para minimizar $J(\boldsymbol{\beta})$, calculamos su gradiente con respecto a $\boldsymbol{\beta}$.

Recordando las reglas de derivación matricial:
- $\frac{\partial}{\partial \boldsymbol{\beta}} (\boldsymbol{\beta}^T A) = A^T$
- $\frac{\partial}{\partial \boldsymbol{\beta}} (\boldsymbol{\beta}^T A \boldsymbol{\beta}) = (A + A^T)\boldsymbol{\beta}$

Aplicando:

$$
\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = \frac{1}{n} \left( -2 X^T \mathbf{y} + 2 X^T X \boldsymbol{\beta} \right)
$$

Simplificando:

$$
\boxed{\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = -\frac{2}{n} X^T (\mathbf{y} - X\boldsymbol{\beta})}
$$

### 2.4 Solución analítica (Mínimos Cuadrados Ordinarios - OLS)

Igualando el gradiente a cero para encontrar el punto óptimo:

$$
-\frac{2}{n} X^T (\mathbf{y} - X\boldsymbol{\beta}) = 0
$$

$$
X^T (\mathbf{y} - X\boldsymbol{\beta}) = 0
$$

$$
X^T \mathbf{y} = X^T X \boldsymbol{\beta}
$$

Despejando:

$$
\boxed{\hat{\boldsymbol{\beta}}_{OLS} = (X^T X)^{-1} X^T \mathbf{y}}
$$

Esta es la **ecuación normal** o solución de mínimos cuadrados.

### 2.5 Condiciones de existencia

Para que $(X^T X)^{-1}$ exista:
1. Las columnas de $X$ deben ser linealmente independientes
2. $n \geq p+1$ (más observaciones que parámetros)
3. $X$ debe tener rango completo $p+1$

Cuando estas condiciones no se cumplen, la solución no es única (problema de multicolinealidad).

### 2.6 Interpretación geométrica

La solución OLS corresponde a la proyección ortogonal de $\mathbf{y}$ sobre el espacio columna de $X$:

$$
\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}} = X(X^T X)^{-1} X^T \mathbf{y} = H\mathbf{y}
$$

donde $H = X(X^T X)^{-1} X^T$ es la **matriz de proyección** (matriz sombrero).

---

## 3. Regularización: Fundamentos Matemáticos

### 3.1 Motivación

Cuando hay multicolinealidad ($X^T X$ es casi singular) o $p > n$, la solución OLS:
- Tiene varianza muy alta
- Sufre de overfitting
- Coeficientes son inestables

La regularización introduce un sesgo controlado para reducir la varianza.

### 3.2 Ridge Regression (Regularización L2)

#### 3.2.1 Función objetivo

$$
J_{ridge}(\boldsymbol{\beta}) = \underbrace{\frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2}_{\text{MSE}} + \underbrace{\lambda \sum_{j=1}^p \beta_j^2}_{\text{Penalización L2}}
$$

En forma matricial:

$$
J_{ridge}(\boldsymbol{\beta}) = \frac{1}{n} (\mathbf{y} - X\boldsymbol{\beta})^T (\mathbf{y} - X\boldsymbol{\beta}) + \lambda \boldsymbol{\beta}^T \boldsymbol{\beta}
$$

**Nota:** El intercepto $\beta_0$ típicamente no se regulariza.

#### 3.2.2 Derivación del gradiente

Derivando término a término:

$$
\nabla_{\boldsymbol{\beta}} \left[ \frac{1}{n} (\mathbf{y} - X\boldsymbol{\beta})^T (\mathbf{y} - X\boldsymbol{\beta}) \right] = -\frac{2}{n} X^T (\mathbf{y} - X\boldsymbol{\beta})
$$

$$
\nabla_{\boldsymbol{\beta}} \left[ \lambda \boldsymbol{\beta}^T \boldsymbol{\beta} \right] = 2\lambda \boldsymbol{\beta}
$$

Por lo tanto:

$$
\nabla_{\boldsymbol{\beta}} J_{ridge}(\boldsymbol{\beta}) = -\frac{2}{n} X^T (\mathbf{y} - X\boldsymbol{\beta}) + 2\lambda \boldsymbol{\beta}
$$

#### 3.2.3 Solución analítica de Ridge

Igualando a cero:

$$
-\frac{2}{n} X^T (\mathbf{y} - X\boldsymbol{\beta}) + 2\lambda \boldsymbol{\beta} = 0
$$

Multiplicando por $n/2$:

$$
- X^T \mathbf{y} + X^T X \boldsymbol{\beta} + n\lambda \boldsymbol{\beta} = 0
$$

$$
(X^T X + n\lambda I) \boldsymbol{\beta} = X^T \mathbf{y}
$$

Finalmente:

$$
\boxed{\hat{\boldsymbol{\beta}}_{ridge} = (X^T X + n\lambda I)^{-1} X^T \mathbf{y}}
$$

#### 3.2.4 Propiedades de Ridge

- La matriz $X^T X + n\lambda I$ es siempre invertible (definida positiva)
- Los coeficientes se "encogen" hacia cero, pero nunca son exactamente cero
- El parámetro $\lambda \geq 0$ controla la intensidad de la regularización:
  - $\lambda = 0$: recupera OLS
  - $\lambda \rightarrow \infty$: $\hat{\boldsymbol{\beta}} \rightarrow 0$

### 3.3 Lasso Regression (Regularización L1)

#### 3.3.1 Función objetivo

$$
J_{lasso}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p |\beta_j|
$$

#### 3.3.2 No diferenciabilidad y solución

La función $|\beta_j|$ no es diferenciable en $\beta_j = 0$, por lo que no existe una solución analítica cerrada.

Se utiliza **optimización subgradiente** o **coordinate descent**.

#### 3.3.3 Coordinate Descent para Lasso

Para cada coeficiente $\beta_j$, manteniendo los demás fijos:

$$
\beta_j^{(t+1)} = \frac{S\left( \frac{1}{n} \sum_{i=1}^n x_{ij}(y_i - \hat{y}_i^{(-j)}), \frac{\lambda}{2} \right)}{ \frac{1}{n} \sum_{i=1}^n x_{ij}^2 }
$$

donde:
- $\hat{y}_i^{(-j)} = \sum_{k \neq j} x_{ik} \beta_k$
- $S(z, \gamma) = \text{sign}(z) \cdot \max(|z| - \gamma, 0)$ es la función de **soft-thresholding**

#### 3.3.4 Soft-thresholding explicado

$$
S(z, \gamma) = \begin{cases} 
z - \gamma & \text{si } z > \gamma \\
0 & \text{si } |z| \leq \gamma \\
z + \gamma & \text{si } z < -\gamma
\end{cases}
$$

Esta función es la responsable de que Lasso pueda **anular coeficientes**, produciendo soluciones sparse.

### 3.4 Elastic Net

#### 3.4.1 Función objetivo

$$
J_{EN}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2
$$

Alternativamente, se suele parametrizar como:

$$
J_{EN}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \left( \alpha \sum_{j=1}^p |\beta_j| + (1-\alpha) \sum_{j=1}^p \beta_j^2 \right)
$$

donde:
- $\alpha \in [0,1]$: controla la mezcla L1/L2
  - $\alpha = 1$: Lasso
  - $\alpha = 0$: Ridge
  - $0 < \alpha < 1$: Elastic Net

#### 3.4.2 Propiedades

- Combina selección de variables (L1) con estabilidad de grupos correlacionados (L2)
- Supera la limitación de Lasso que solo selecciona una variable de un grupo correlacionado
- Ideal para $p > n$ o datos con alta correlación

---

## 4. Algoritmos de Optimización Computacional

### 4.1 Gradiente Descendente (Gradient Descent)

Para problemas donde la solución cerrada es computacionalmente costosa ($O(p^3)$ para invertir $X^T X$), se usa optimización iterativa.

**Algoritmo:**
1. Inicializar $\boldsymbol{\beta}^{(0)}$
2. Para $t = 0, 1, \dots$ hasta convergencia:
   $$
   \boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla J(\boldsymbol{\beta}^{(t)})
   $$
   donde $\eta > 0$ es la tasa de aprendizaje

**Gradiente para OLS:**
$$
\nabla J(\boldsymbol{\beta}^{(t)}) = -\frac{2}{n} X^T (\mathbf{y} - X\boldsymbol{\beta}^{(t)})
$$

**Actualización:**
$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + \frac{2\eta}{n} X^T (\mathbf{y} - X\boldsymbol{\beta}^{(t)})
$$

### 4.2 Gradiente Descendente Estocástico (SGD)

Para datasets muy grandes ($n \gg 1$):

1. Aleatorizar el orden de los datos
2. Para cada época $e$:
   - Para cada mini-batch $\mathcal{B}$ de tamaño $m$:
     $$
     \boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \eta \cdot \frac{1}{m} \sum_{i \in \mathcal{B}} \nabla J_i(\boldsymbol{\beta})
     $$

### 4.3 Coordinate Descent (para Lasso y Elastic Net)

Ideal para problemas con regularización L1:

1. Inicializar $\boldsymbol{\beta}$
2. Hasta convergencia:
   - Para $j = 1, \dots, p$:
     - Calcular el residual parcial
     - Actualizar $\beta_j$ usando soft-thresholding

### 4.4 Complejidad Algorítmica

| Método | Complejidad | Ventaja | Desventaja |
|--------|-------------|---------|------------|
| OLS cerrada | $O(p^2 n + p^3)$ | Exacta, rápida para $p$ pequeño | Costosa si $p$ grande |
| Gradiente Descendente | $O(k \cdot n \cdot p)$ | Escala bien con $n$ | Requiere tuning de $\eta$ |
| SGD | $O(k \cdot m \cdot p)$ | Muy eficiente para $n$ enorme | Convergencia estocástica |
| Coordinate Descent | $O(k \cdot n \cdot p)$ | Excelente para Lasso | Menos eficiente para $n \gg p$ |

---

## 5. Métricas de Evaluación: Fundamentos

### 5.1 Error Cuadrático Medio (MSE)

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

**Propiedades:**
- Penaliza cuadráticamente los errores grandes
- Derivable, adecuada para optimización convexa
- Misma unidad que $y^2$

### 5.2 Raíz del Error Cuadrático Medio (RMSE)

$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

**Propiedades:**
- Misma unidad que $y$ (interpretable)
- Sensible a outliers

### 5.3 Error Absoluto Medio (MAE)

$$
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

**Propiedades:**
- Robusto frente a outliers
- Menos sensible a errores grandes
- No es derivable en cero

### 5.4 Coeficiente de Determinación ($R^2$)

$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}
$$

donde:
- $SS_{res}$: suma de cuadrados residuales (error)
- $SS_{tot}$: suma de cuadrados totales (varianza explicable)

**Propiedades:**
- $R^2 \in (-\infty, 1]$
- $R^2 = 1$: ajuste perfecto
- $R^2 = 0$: modelo solo predice la media
- $R^2 < 0$: modelo peor que predecir la media

### 5.5 $R^2$ Ajustado

Para comparar modelos con diferente número de predictores:

$$
R^2_{adj} = 1 - (1 - R^2) \frac{n-1}{n-p-1}
$$

Penaliza la inclusión de variables irrelevantes.

---

## 6. Descomposición Sesgo-Varianza

Para entender el efecto de la regularización:

$$
E[(y - \hat{f}(x))^2] = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Sesgo}^2} + \underbrace{E[(\hat{f}(x) - E[\hat{f}(x)])^2]}_{\text{Varianza}} + \underbrace{\sigma^2}_{\text{Ruido irreducible}}
$$

- **OLS**: bajo sesgo, alta varianza
- **Ridge/Lasso**: introduce sesgo, reduce varianza
- **Objetivo**: minimizar el error total mediante $\lambda$ óptimo (validación cruzada)

---

## 7. Selección de Hiperparámetros: Validación Cruzada

### 7.1 K-Fold Cross Validation

1. Dividir los datos en $K$ pliegues de igual tamaño
2. Para cada $\lambda$ candidato:
   - Para $k = 1$ a $K$:
     - Entrenar en los $K-1$ pliegues restantes
     - Evaluar en el pliegue $k$
   - Calcular el error promedio

### 7.2 Criterio de selección

$$
\lambda^* = \arg\min_{\lambda} CV(\lambda) = \arg\min_{\lambda} \frac{1}{K} \sum_{k=1}^K MSE_k(\lambda)
$$

---

## 8. Implementación Computacional Vectorizada

### 8.1 Solución OLS sin dependencias

```python
import numpy as np

def ols_fit(X, y):
    """
    X: matriz de diseño (n x p+1) - incluye columna de unos
    y: vector objetivo (n,)
    """
    # Solución de la ecuación normal
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Resolver sistema lineal (más estable que inversa directa)
    beta = np.linalg.solve(XTX, XTy)
    
    return beta

def ols_predict(X, beta):
    return X @ beta
```

### 8.2 Gradiente Descendente Vectorizado

```python
def gradient_descent(X, y, eta=0.01, epochs=1000, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    
    for epoch in range(epochs):
        # Predicciones
        y_pred = X @ beta
        
        # Gradiente vectorizado
        gradient = - (2/n) * X.T @ (y - y_pred)
        
        # Actualización
        beta_new = beta - eta * gradient
        
        # Verificar convergencia
        if np.linalg.norm(beta_new - beta) < tol:
            break
            
        beta = beta_new
    
    return beta
```

### 8.3 Ridge Regression con SVD

Para mejor estabilidad numérica:

```python
def ridge_fit_svd(X, y, lambda_):
    """
    Usa descomposición SVD: X = U @ S @ V^T
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Transformación
    Ut_y = U.T @ y
    
    # Coeficientes en espacio singular
    d = s / (s**2 + lambda_)
    beta_svd = Vt.T @ (d * Ut_y)
    
    return beta_svd
```

---

## 9. Regularización en Práctica: Interpretación de $\lambda$

### 9.1 Efecto de $\lambda$ en Ridge

- $\lambda$ pequeño: cerca de OLS, alta varianza
- $\lambda$ grande: fuerte encogimiento, alto sesgo
- $\lambda$ óptimo: balance sesgo-varianza

### 9.2 Efecto de $\lambda$ en Lasso

- Aumenta la sparsity a medida que $\lambda$ crece
- Secuencia de solución: path de regularización

---

## 10. Resumen de Ecuaciones Clave

| Concepto | Ecuación |
|----------|----------|
| Modelo | $\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ |
| Función pérdida (MSE) | $J(\boldsymbol{\beta}) = \frac{1}{n} \|\mathbf{y} - X\boldsymbol{\beta}\|^2$ |
| Gradiente MSE | $\nabla J = -\frac{2}{n} X^T (\mathbf{y} - X\boldsymbol{\beta})$ |
| Solución OLS | $\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{y}$ |
| Ridge objetivo | $J_{ridge} = MSE + \lambda \|\boldsymbol{\beta}\|^2$ |
| Solución Ridge | $\hat{\boldsymbol{\beta}}_{ridge} = (X^T X + n\lambda I)^{-1} X^T \mathbf{y}$ |
| Lasso objetivo | $J_{lasso} = MSE + \lambda \|\boldsymbol{\beta}\|_1$ |
| Soft-thresholding | $S(z, \gamma) = \text{sign}(z) \cdot \max(|z| - \gamma, 0)$ |
| Elastic Net | $J_{EN} = MSE + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|^2$ |
| RMSE | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ |
| MAE | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ |
| $R^2$ | $1 - \frac{SS_{res}}{SS_{tot}}$ |

---

