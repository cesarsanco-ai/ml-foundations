---
layout: default
---
# Sesión 3: Regresión Lineal y Regularización

### 1. Logro de la sesión

Construir e interpretar modelos de **regresión lineal múltiple** en el marco de **mínimos cuadrados** y de **regularización** (Ridge, Lasso, Elastic Net), relacionando **supuestos estadísticos**, **trade-offs sesgo–varianza** y **métricas de error** con decisiones prácticas en Python (`scikit-learn`).

---

### 2. Historia y línea temporal (contexto)

| Periodo | Hito | Comentario |
|---------|------|------------|
| **1805–1809** | Legendre y Gauss plantean **mínimos cuadrados** para ajustar órbitas y datos astronómicos | El problema original era **determinístico** (ajuste); la inferencia estadística llega después. |
| **Finales s. XIX** | **Galton** y correlación/regresión hacia la media | Origen del término “regresión”; enfoque biológico-social. |
| **Principios s. XX** | **Pearson**, **Yule**: regresión múltiple formal | Base de econometría y estadística aplicada. |
| **1930s** | **Fisher**: diseño experimental, ANOVA relacionado | Conexión con tests e inferencia. |
| **1950–1970** | Econometría clásica; **Hoerl & Kennard (1970)** introducen **Ridge** ante **multicolinealidad** | La regularización L2 se populariza cuando $X^\top X$ está mal condicionada. |
| **1996** | **Tibshirani**: **Lasso** (L1) con selección de variables | Puente entre predicción y **sparse** models. |
| **2005** | **Zou & Hastie**: **Elastic Net** | Mezcla L1+L2 para correlaciones fuertes entre grupos de variables. |
| **Actualidad** | Penalización en **alta dimensión** ($p \gg n$), pipelines en ML | sklearn, glmnet (R), teoría “oracle” y consistencia en sparse (p.ej. Wainwright, Bühlmann & van de Geer). |

**Lectura:** la regresión lineal no es “solo” un algoritmo: es el **punto de partida conceptual** para entender pérdidas cuadráticas, regularización y extensiones (GLM, kernel ridge, redes poco profundas como composición de capas lineales + no linealidades).

---

### 3. Marco teórico: modelo y estimación MCO

#### 3.1 Formulación

Para $n$ observaciones y $p$ predictores (más intercepto si se incluye):

$$ y_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij} + \varepsilon_i, \quad i=1,\ldots,n $$

En forma matricial, con $\mathbf{X}$ de dimensión $n \times (p+1)$ si se añade columna de unos:

$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

El **modelo lineal** puede referirse a **linealidad en parámetros**; las entradas pueden ser transformaciones no lineales de variables crudas (log, splines, interacciones), tema estrechamente ligado al *feature engineering* (Sesión 2).

#### 3.2 Estimador de mínimos cuadrados (OLS)

Se define como el $\hat{\boldsymbol{\beta}}$ que minimiza la suma de errores cuadráticos:

$$ \mathrm{RSS}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 $$

Si $\mathbf{X}$ tiene rango completo (columnas linealmente independientes), la solución cerrada es:

$$ \hat{\boldsymbol{\beta}}_{\mathrm{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} $$

**Interpretación geométrica:** $\mathbf{X}\hat{\boldsymbol{\beta}}$ es la **proyección ortogonal** de $\mathbf{y}$ sobre el espacio columna de $\mathbf{X}$ (ver Hastie et al., cap. 3).

#### 3.3 Teorema de Gauss–Markov (condiciones y mensaje)

Bajo:

1. Linealidad correcta del modelo en $\boldsymbol{\beta}$.  
2. $\mathbb{E}[\boldsymbol{\varepsilon}\mid \mathbf{X}] = \mathbf{0}$.  
3. $\mathrm{Var}(\boldsymbol{\varepsilon}\mid \mathbf{X}) = \sigma^2 \mathbf{I}$ (homocedasticidad y no correlación).  
4. $\mathbf{X}$ fija o exógena en el sentido clásico.

Entonces $\hat{\boldsymbol{\beta}}_{\mathrm{OLS}}$ es el estimador **lineal insesgado de mínima varianza** (BLUE) entre los estimadores lineales insesgados de $\boldsymbol{\beta}$.

**Matices para ML:** en predicción pura, el “mejor” estimador puede ser **sesgado** pero de menor error de generalización → ahí entra **regularización** y modelos más flexibles.

---

### 4. Interpretación de coeficientes y diagnóstico

#### 4.1 Coeficientes *ceteris paribus*

$\hat{\beta}_j$ indica el cambio esperado en $y$ al aumentar una unidad $x_j$, **manteniendo constantes** las demás variables incluidas en el modelo. Si hay **multicolinealidad fuerte**, los coeficientes individuales tienen **alta varianza** e interpretación frágil aunque la predicción conjunta sea buena.

#### 4.2 Supuestos y qué revisar en la práctica

| Supuesto | Implicación si falla | Herramientas típicas |
|----------|----------------------|----------------------|
| **Linealidad** | Sesgo sistemático; predicciones sesgadas por regiones | Residuos vs $\hat{y}$, transformaciones, términos adicionales |
| **Independencia** (residuos) | IC y tests mal calibrados; OLS puede seguir siendo útil para ajuste | ACF en series; modelos para datos correlacionados |
| **Homocedasticidad** | Inferencia estándar incorrecta; MCO sigue siendo razonable en predicción en algunos casos | Gráfico de residuos; errores robustos (HC) en econometría |
| **Normalidad de residuos** | Importa más para intervalos pequeños-$n$ | QQ-plot; en $n$ grande, TCL ayuda para medias |

**Referencia clásica:** Hastie et al. (*ESL*); para enfoque pedagógico James et al. (*ISL*).

---

### 5. Regularización: Ridge, Lasso y Elastic Net

**Marco teórico unificado:** en lugar de solo minimizar RSS, se minimiza RSS **más una penalización** en $\boldsymbol{\beta}$ (usualmente sin penalizar $\beta_0$ o tras centrar $y$ y $X$):

$$ \min_{\boldsymbol{\beta}} \ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \cdot \mathrm{Pen}(\boldsymbol{\beta}) $$

El parámetro $\lambda \geq 0$ controla el **compromiso** entre ajuste a datos y magnitud de coeficientes (Sesión 8: elección por validación cruzada).

#### 5.1 Ridge (penalización L2)

**Objetivo:**

$$ \min_{\boldsymbol{\beta}} \ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{j=1}^{p} \beta_j^2 $$

(Souvent se excluye $\beta_0$ de la penalización; en sklearn puede controlarse con `fit_intercept`.)

**Ventajas:**

- **Estabiliza** la solución cuando $X^\top X$ es casi singular (multicolinealidad): equivale a añadir “información previa” de coeficientes pequeños.  
- **No pone coeficientes exactamente a cero** → mantiene todos los predictores en el modelo (útil si todas las variables tienen algo de señal).  
- Solución cerrada: $\hat{\boldsymbol{\beta}}_{\mathrm{Ridge}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top \mathbf{y}$ (bajo formulación estándar).

**Limitaciones:**

- No realiza **selección de variables** (todos los coeficientes son típicamente no nulos).  
- Requiere **escalado** de variables comparables para que la penalización sea equitativa entre columnas.

**Lectura bayesiana (opcional):** Ridge corresponde a un prior **Gaussiano** sobre $\boldsymbol{\beta}$ (MAP bajo ruido Gaussiano).

**Referencia:** Hoerl & Kennard (1970).

#### 5.2 Lasso (penalización L1)

**Objetivo:**

$$ \min_{\boldsymbol{\beta}} \ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{j=1}^{p} |\beta_j| $$

**Ventajas:**

- Produce soluciones **sparse**: muchos $\beta_j$ exactamente **cero** → **selección automática** de variables en sentido práctico.  
- Útil cuando se sospecha que solo un subconjunto de predictores es relevante (**sparse ground truth** aproximado).

**Limitaciones:**

- Con predictores **fuertemente correlacionados**, tiende a elegir **uno** y anular otros (inestabilidad de selección).  
- La solución no tiene forma cerrada tan simple como OLS/Ridge; se calcula con programación cuadrática / *coordinate descent* (p.ej. `sklearn`, `glmnet`).

**Referencia:** Tibshirani (1996).

#### 5.3 Elastic Net

**Objetivo (forma habitual):**

$$ \min_{\boldsymbol{\beta}} \ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2 $$

o parametrización con mezcla $\alpha \in [0,1]$ frente a un solo $\lambda$ (como en `ElasticNet` de sklearn: `l1_ratio` + `alpha`).

**Ventajas:**

- Combina **encogimiento grupal** (L2) con **selección** (L1): a menudo mejor que Lasso puro cuando hay **grupos de variables correlacionadas** (p.ej. dummies de la misma categoría expandida).  
- Referencia: Zou & Hastie (2005).

**Limitaciones:**

- Más **hiperparámetros** que Ridge o Lasso solo → coste de tuning mayor.

#### 5.4 ¿Cuándo usar cada método? (guía práctica)

| Situación | Preferencia típica |
|-----------|-------------------|
| Multicolinealidad fuerte, todas las variables potencialmente útiles | **Ridge** o **Elastic Net** con `l1_ratio` moderado |
| Se sospecha que **pocas** variables son realmente relevantes ($p$ grande, señal sparse) | **Lasso** o **Elastic Net** |
| Necesidad de **interpretar** un subconjunto pequeño de coeficientes no nulos | **Lasso** / Elastic Net |
| Predicción con $p \approx n$ o $p > n$ | Ridge/Lasso/EN con CV; OLS puede no ser identificable sin regularización |
| Variables en escalas muy distintas | **Estandarizar** (`StandardScaler`) antes de ajustar modelos penalizados |

---

### 6. Métricas de evaluación (regresión)

**Marco teórico:** la elección de métrica debe alinearse con el **coste asimétrico** del error en el negocio (p.ej. subestimar vs sobreestimar precio).

| Métrica | Definición | Ventajas | Inconvenientes |
|---------|------------|----------|-----------------|
| **MSE** | $\frac{1}{n}\sum (y_i-\hat{y}_i)^2$ | Diferenciable; conecta con Gaussiano bajo máxima verosimilitud | Muy sensible a **outliers** |
| **RMSE** | $\sqrt{\mathrm{MSE}}$ | Misma unidad que $y$ | Sigue penalizando fuerte outliers |
| **MAE** | $\frac{1}{n}\sum \|y_i-\hat{y}_i\|$ | Robusta, interpretable | No tan suave para optimización analítica |
| **$R^2$** | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Escalar “varianza explicada” | Puede inflarse con muchos predictores; usar $R^2$ ajustado en comparaciones con distinto $p$ |

**Nota:** en modelos regularizados, comparar métricas en **validación** con el mismo protocolo (Sesión 8).

#### 6.1 $R^2$ ajustado y número de parámetros

Al añadir predictores, el $R^2$ **ordinario** no penaliza la complejidad: casi siempre sube al meter más columnas aunque sean ruido. El **$R^2$ ajustado** corrige por grados de libertad:

$$ \bar{R}^2 = 1 - \frac{\mathrm{RSS}/(n-p-1)}{\mathrm{TSS}/(n-1)} $$

Úsese para comparar modelos lineales con **distinto número de términos** en el mismo conjunto de entrenamiento (con las salvedades habituales de inferencia).

#### 6.2 Conexión con máxima verosimilitud (error Gaussiano)

Si $y_i = \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i$ con $\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$ i.i.d., maximizar la log-verosimilitud equivale a **minimizar la RSS** (MCO). Por eso el **MSE** es el objetivo natural bajo ruido gaussiano; bajo otros ruidos, otros estimadores pueden ser preferibles (p.ej. pérdidas robustas), tema avanzado.

#### 6.3 Sesgo, varianza y regularización (puente a Sesión 8)

En términos de error de predicción esperado, modelos más complejos (muchos coeficientes sin restricción) reducen sesgo pero aumentan varianza muestral. **Ridge/Lasso/EN** introducen **sesgo deliberado** sobre $\boldsymbol{\beta}$ para **reducir varianza** y mejorar error fuera de muestra cuando el problema es mal condicionado o $p$ es grande.

---

### 7. Plantilla base en Python (`scikit-learn`)

La idea es mostrar el **patrón típico**: importaciones, objetos estimador, `fit` / `predict`, métricas y, cuando aplique, **`Pipeline`** con escalado + modelo (recomendado para Ridge/Lasso/EN).

#### 7.1 Imports habituales

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

#### 7.2 División train / test

```python
# X: DataFrame o ndarray de shape (n_samples, n_features)
# y: Series o ndarray de shape (n_samples,)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

`random_state` fija la reproducibilidad del split.

#### 7.3 Regresión lineal ordinaria (sin regularización)

```python
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred = ols.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # sklearn >= 1.0
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE={mse:.4f} RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
```

**Objetos clave:** `LinearRegression` expone `coef_` (pendientes) e `intercept_`.

#### 7.4 Ridge, Lasso y Elastic Net con escalado (patrón recomendado)

La penalización es sensible a la **escala** de cada columna; se encapsula todo en un `Pipeline`:

```python
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0, random_state=42)),
])
ridge_pipe.fit(X_train, y_train)
y_pred_ridge = ridge_pipe.predict(X_test)
```

```python
lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(alpha=0.1, max_iter=10_000, random_state=42)),
])
lasso_pipe.fit(X_train, y_train)
```

```python
enet_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000, random_state=42)),
])
enet_pipe.fit(X_train, y_train)
```

**Parámetros (`sklearn`):**

- `alpha`: intensidad global de la penalización (análogo a $\lambda$ en la literatura, convenciones pueden variar).  
- `l1_ratio` (solo ElasticNet): proporción de L1 en la mezcla (0 = Ridge puro en la implementación, 1 = Lasso puro).

**Acceso a coeficientes tras `Pipeline`:**

```python
coefs = ridge_pipe.named_steps["model"].coef_
intercept = ridge_pipe.named_steps["model"].intercept_
```

#### 7.5 Comparación mínima entre modelos (mismo test)

```python
models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10_000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000),
}

for name, est in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(name, "RMSE:", mean_squared_error(y_test, pred, squared=False))
```

*(Ajustar `alpha` y `l1_ratio` con validación cruzada en proyectos reales — Sesión 8.)*

#### 7.6 Ajuste de $\lambda$ con `RidgeCV` / `LassoCV` / `ElasticNetCV` (opcional pero recomendado)

En lugar de fijar `alpha` a mano, sklearn ofrece clases que exploran una **rejilla** de valores con **leave-one-out** eficiente (en Ridge) o **CV** estándar:

```python
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-4, 4, 50)
ridge_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RidgeCV(alphas=alphas, cv=5)),
])
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.named_steps["model"].alpha_
```

`LassoCV` y `ElasticNetCV` siguen la misma filosofía; revisar la documentación para `cv`, `max_iter` y convergencia.

#### 7.7 Buenas prácticas rápidas

- **Siempre** fijar `random_state` en splits y en algoritmos estocásticos (si aplica).  
- Documentar **unidades** de $y$ y de cada $x_j$ al interpretar coeficientes.  
- Para producción, serializar el **`Pipeline` completo** (escalador + modelo), no solo el estimador final.

#### 7.8 Errores frecuentes al principiar

| Error | Por qué importa | Qué hacer |
|-------|-----------------|-----------|
| Ajustar Ridge/Lasso **sin escalar** | La penalización mezcla magnitudes arbitrarias; un coeficiente en “metros” no es comparable a uno en “dólares” | `StandardScaler` dentro de `Pipeline` |
| Interpretar coeficientes con **variables fuertemente colineales** | Varianza alta; signos pueden invertirse al añadir/quitar columnas | Regularizar, PCA (Sesión 9), o redefinir features |
| Medir rendimiento en **train** únicamente | Sobreajuste invisible | `train_test_split` o CV (Sesión 8) |
| Usar $R^2$ como única métrica | No refleja error en unidades de negocio | Reportar MAE/RMSE junto a $R^2$ |

---

### 8. Interpretación geométrica y optimización convexa

#### 8.1 Ridge como proyección con penalización

Minimizar $\|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_2^2$ es equivalente a buscar el compromiso entre ajuste y norma euclídea de $\boldsymbol{\beta}$. Geométricamente, para $\lambda>0$, la solución **no coincide** con la proyección OLS salvo $\lambda=0$; se “encoge” hacia el origen en el espacio de parámetros.

#### 8.2 Lasso y politopos

La restricción $\|\boldsymbol{\beta}\|_1 \le t$ define un **diamante** (*cross-polytope*) en $\mathbb{R}^p$. Las soluciones suelen aparecer en **vértices** del politopo → coeficientes exactamente nulos. Por eso Lasso **selecciona variables** cuando la señal es sparse.

#### 8.3 Condición del problema y estabilidad numérica

El número de condición $\kappa(\mathbf{X}^\top\mathbf{X})$ grande implica que pequeños cambios en $\mathbf{y}$ producen grandes cambios en $\hat{\boldsymbol{\beta}}_{\mathrm{OLS}}$. Ridge **añade** $\lambda \mathbf{I}$ al Gram matrix → mejora el condicionamiento y estabiliza la solución (Hoerl & Kennard).

#### 8.4 Relación bayesiana (MAP)

- Ridge ≈ prior **Gaussiano** sobre $\boldsymbol{\beta}$.  
- Lasso ≈ prior **Laplace** (doble exponencial) → colas pesadas y masa en cero.

Esto conecta regularización frecuentista con **inferencia bayesiana** (ver Murphy, *Machine Learning: A Probabilistic Perspective*).

---

### 9. Laboratorio (según sílabo)

- **NTB 1 —** Regresión múltiple frente a Ridge, Lasso y Elastic Net: métricas y comparación — *dataset 1*.  
- **NTB 2 —** Regresión múltiple frente a Ridge, Lasso y Elastic Net: métricas y comparación — *dataset 2*.

---

## Referencias bibliográficas principales

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.  
2. James, G., et al. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.  
3. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.  
4. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.  
5. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301–320.  
6. Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.  
