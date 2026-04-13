---
layout: default
---
# Sesión 8: Bias–Variance, Validación, Optimización de Hiperparámetros y Selección de Modelos

### 1. Logro de la sesión

Dominar el **dilema sesgo–varianza**, diseñar protocolos **train/validation/test** y **validación cruzada** sin *data leakage*, aplicar **búsqueda de hiperparámetros** (grid, random, bayesiana introductoria) y **seleccionar modelos** con métricas alineadas al negocio, usando de forma fluida `sklearn.model_selection`.

---

### 2. Problemática de negocio (por qué esta sesión es central)

| Fenómeno | Síntoma en producción | Causa típica |
|----------|------------------------|--------------|
| Métricas excelentes en train, pobres en cliente real | **Overfitting** | Modelo demasiado complejo o validación incorrecta |
| Métricas mediocres en todo | **Underfitting** | Modelo demasiado simple o features débiles |
| Líder en validación, perdedor en test | **Optimización accidental del test** o leakage | Reutilizar test para elegir modelo |
| Estimación de rendimiento demasiado optimista | **Leakage** en preprocesamiento | Estadísticas calculadas con train+test |

**Mensaje:** la **generalización** es el objetivo; el ajuste a la muestra es solo un medio.

---

### 3. Descomposición sesgo–varianza (intuición y uso)

#### 3.1 Forma clásica (regresión cuadrática)

Para un punto fijo $x$, bajo supuestos:

$$ \mathbb{E}\bigl[(\hat{f}(x) - f(x))^2\bigr] = \underbrace{\bigl(\mathbb{E}[\hat{f}(x)] - f(x)\bigr)^2}_{\text{sesgo}^2} + \underbrace{\mathbb{E}\bigl[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\bigr]}_{\text{varianza}} + \sigma^2 $$

- **Sesgo alto:** el modelo no puede aproximar la verdadera función (familia demasiado rígida).  
- **Varianza alta:** el modelo es muy sensible a la muestra concreta.

#### 3.2 Complejidad del modelo

| Complejidad | Sesgo | Varianza |
|-------------|-------|----------|
| Baja (modelo simple) | Alto | Baja |
| Alta (modelo flexible) | Bajo | Alta |

**Regularización**, **más datos** y **menos features ruidosas** suelen **bajar varianza** a costa de un poco de sesgo (bias–variance tradeoff).

#### 3.3 Curvas de aprendizaje

`learning_curve` en sklearn grafica error vs **tamaño de entrenamiento**:

- Train y validation **altos y cercanos** → **underfitting** (más complejidad o mejores features).  
- Train bajo, validation alto y separados → **overfitting** (regularizar, más datos, menos complejidad).

---

### 4. Partición de datos: train / validation / test

#### 4.1 Roles

1. **Training:** estimar parámetros del modelo (pesos del árbol, coeficientes, etc.).  
2. **Validation:** elegir **hiperparámetros** y comparar modelos.  
3. **Test:** **una sola vez** al final para estimar rendimiento futuro.

#### 4.2 Validación cruzada (CV)

**K-fold:** dividir en $K$ partes; por ronda, $K-1$ para entrenar y 1 para medir. La métrica es el **promedio** (y desviación) sobre folds.

**StratifiedKFold:** mantiene proporción de clases en cada fold → **clasificación** con desbalance moderado.

**Leave-One-Out (LOO):** $K=n$; bajo sesgo de estimación, **alta varianza** y coste computacional $O(n^2)$ en muchos modelos.

#### 4.3 Validación anidada

- **Bucle externo:** estima rendimiento del **proceso completo** (incluida selección de $\lambda$).  
- **Bucle interno:** selecciona hiperparámetros en cada fold externo.

Evita que la elección de hiperparámetros **inflación** el error reportado.

---

### 5. Data leakage: ejemplos y prevención

| Escenario incorrecto | Correcto |
|----------------------|----------|
| `StandardScaler().fit(X_train + X_test)` | `fit` solo en train; `transform` en test dentro de `Pipeline` |
| Incluir variable **derivada del target** por error | Auditar que solo información *a priori* entre en $X$ |
| CV con grupos mezclados (mismo paciente en train y test) | `GroupKFold` cuando existan dependencias |

**Regla de oro:** cualquier transformación que **aprenda estadísticas** de los datos debe aprenderlas **solo del material de entrenamiento** de esa iteración.

---

### 6. Búsqueda de hiperparámetros

#### 6.1 Grid Search

Explora **todas** las combinaciones en una rejilla discreta.

**Ventaja:** exhaustivo en el conjunto dado.  
**Desventaja:** **curse of dimensionality** en muchos hiperparámetros.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
}
```

#### 6.2 Random Search

Muestrea aleatoriamente $N$ combinaciones (Bergstra & Bengio, 2012). A menudo encuentra buenos valores con **menos evaluaciones** que grid completo cuando hay dimensiones irrelevantes.

#### 6.3 Optimización Bayesiana (introducción)

Modela $f(\lambda) = $ métrica como función costosa y elige el siguiente $\lambda$ con criterio de adquisición (expected improvement). Implementaciones: **Optuna**, **scikit-optimize**, **Hyperopt**.

---

### 7. Plantillas Python (`sklearn.model_selection`)

#### 7.1 `train_test_split`

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### 7.2 `Pipeline` + `GridSearchCV` (evita leakage)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
])

param_grid = {"clf__C": [0.01, 0.1, 1, 10]}

search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("CV AUC:", search.best_score_)
```

#### 7.3 `RandomizedSearchCV`

Igual interfaz que `GridSearchCV` pero `param_distributions` y `n_iter`.

#### 7.4 `cross_val_score` y `cross_validate`

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
print(scores.mean(), scores.std())
```

`cross_validate` permite múltiples métricas y tiempos.

#### 7.5 `learning_curve` y `validation_curve`

```python
from sklearn.model_selection import learning_curve, validation_curve
```

Útiles para diagnosticar bias–variance y sensibilidad a un hiperparámetro.

---

### 8. Métricas en CV

Reportar **media ± desviación** sobre folds; no comparar modelos por un solo split aleatorio si $n$ es pequeño.

---

### 9. Profundización: validación anidada, curvas y anti-*leakage*

#### 9.1 Esquema de doble CV

**Externo (p.ej. 5 folds):** estima rendimiento del **proceso completo** de modelado.  
**Interno (p.ej. 3 folds dentro de cada train externo):** selecciona hiperparámetros.

Así el error reportado no está contaminado por haber “visto” los datos de validación interna al ajustar el modelo final.

#### 9.2 `GroupKFold` y datos dependientes

Si hay **múltiples filas por mismo paciente, usuario o sensor**, el azar fila a fila filtra información. Usar:

```python
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups=group_ids):
    ...
```

#### 9.3 Curva de aprendizaje completa (código)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring="roc_auc", n_jobs=-1,
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
plt.plot(train_sizes, train_mean, label="train")
plt.plot(train_sizes, val_mean, label="val")
plt.legend(); plt.xlabel("Training examples"); plt.ylabel("ROC-AUC")
```

Interpretación: si **ambas** curvas convergen a un valor bajo → subajuste. Si **train** alto y **val** bajo con brecha → sobreajuste.

#### 9.4 `validation_curve` para un hiperparámetro

```python
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 3, 15)
train_scores, val_scores = validation_curve(
    pipe, X_train, y_train, param_name="clf__C",
    param_range=param_range, cv=5, scoring="roc_auc", n_jobs=-1,
)
```

#### 9.5 RandomizedSearchCV con distribuciones

```python
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {"clf__C": loguniform(1e-3, 1e3)}
search = RandomizedSearchCV(
    pipe, param_distributions, n_iter=40, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
```

#### 9.6 Errores frecuentes (tabla)

| Error | Síntoma |
|-------|---------|
| Usar el mismo test para muchas corridas | Optimismo extremo |
| Feature selection en todo el dataset antes del split | Leakage masivo |
| No fijar `random_state` en CV | Resultados irreproducibles |

---

### 10. Laboratorio (según sílabo)

- **NTB 1 —** Flujo completo de clasificación: validación cruzada, búsqueda de hiperparámetros y selección de modelo.  
- **NTB 2 —** Flujo completo de regresión: curvas de aprendizaje, bias–variance y comparación de modelos.

---

## Referencias bibliográficas principales

1. Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*.  
2. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281–305.  
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (cap. 7). Springer.  
4. Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.  
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.  
