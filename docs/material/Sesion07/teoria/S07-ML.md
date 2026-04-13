---
layout: default
---
# Sesión 7: Gradient Boosting

### 1. Logro de la sesión

Comprender el **principio de boosting** y su instanciación en **Gradient Boosting**, dominar las particularidades de **XGBoost**, **LightGBM** y **CatBoost**, y relacionar **hiperparámetros**, **funciones de pérdida** y **estrategias anti-overfitting** con prácticas de entrenamiento en Python.

---

### 2. Historia y línea temporal

| Periodo | Hito |
|---------|------|
| **1990** | **AdaBoost** (Freund & Schapire): repondera ejemplos difíciles |
| **1999–2001** | **Gradient Boosting** (Friedman): boosting como descenso funcional en espacio de funciones |
| **2010s** | **XGBoost** (Chen & Guestrin, 2016): regularización de hojas, sistema eficiente |
| **2017** | **LightGBM** (Ke et al.): histogramas, GOSS, leaf-wise |
| **2018** | **CatBoost** (Prokhorenkova et al.): categorías, ordered boosting |
| **Actualidad** | Dominio en datos tabulares competitivos; integración con Optuna, MLflow |

---

### 3. Marco teórico: de AdaBoost a Gradient Boosting

#### 3.1 Idea de boosting

A diferencia del **bagging** (promedio de modelos **independientes**), el boosting entrena modelos **secuencialmente**; cada etapa intenta corregir los errores de la combinación anterior.

#### 3.2 Gradient Boosting (Friedman, 2001)

Sea $\hat{F}^{(m-1)}$ el modelo acumulado hasta la iteración $m-1$. Se añade un **árbol débil** $h_m$ escalado por $\eta$ (learning rate):

$$ \hat{F}^{(m)}(\mathbf{x}) = \hat{F}^{(m-1)}(\mathbf{x}) + \eta\, h_m(\mathbf{x}) $$

donde $h_m$ aproxima el **gradiente negativo** de la pérdida $L\bigl(y, F(\mathbf{x})\bigr)$ respecto a $F$ evaluado en los datos (pseudoresiduos).

**Pérdidas típicas:**

- Regresión: **MSE** → pseudoresiduos $y_i - \hat{F}^{(m-1)}(x_i)$.  
- Clasificación binaria: **log-loss** → pseudoresiduos relacionados con probabilidades.

#### 3.3 Bagging vs boosting (contraste)

| Aspecto | Bagging (RF) | Boosting |
|---------|--------------|----------|
| Entrenamiento | Paralelo | Secuencial |
| Objetivo principal | Reducir **varianza** | Reducir **sesgo** (y luego regularizar varianza) |
| Riesgo típico | Subajuste si árboles muy simples | **Overfitting** si demasiadas iteraciones sin control |

---

### 4. XGBoost: elementos distintivos

**Publicación:** Chen & Guestrin, KDD 2016.

**Características:**

1. **Regularización** en la función objetivo: penalización L1/L2 sobre pesos de hojas (según formulación) además de la calidad del split.  
2. **Segunda derivada** (aproximación de Taylor) para elegir splits de forma eficiente.  
3. **Manejo de nulos:** aprende **dirección por defecto** en splits (muy útil en datos reales).  
4. **Paralelización** a nivel de columnas y bloques.

**Parámetros frecuentes (concepto):** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`.

**Plantilla mínima:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=50,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
```

*(Sintaxis exacta de `early_stopping` puede variar según versión de xgboost; consultar documentación vigente.)*

---

### 5. LightGBM

**Ideas clave (Ke et al., 2017):**

- **Histogram-based:** discretiza features en bins → menos memoria y más velocidad.  
- **GOSS:** muestrea gradientes grandes y una parte aleatoria de pequeños para acelerar sin perder demasiada señal.  
- **Leaf-wise** (crecimiento por hoja) en lugar de nivel por nivel → puede lograr menor error con **riesgo de overfitting** si no se controla `num_leaves`.

**Ventaja:** datasets **grandes** y muchas columnas.

```python
import lightgbm as lgb

clf = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
clf.fit(X_train, y_train)
```

---

### 6. CatBoost

**Prokhorenkova et al., 2018.**

**Fortalezas:**

- Tratamiento nativo de **variables categóricas** (codificación ordenada con *ordered target statistics* para reducir *target leakage* interno).  
- **Ordered boosting** para mitigar sesgo de predicción en pequeñas muestras por categoría.

**Cuándo brilla:** muchas columnas categóricas de alta cardinalidad sin one-hot masivo manual.

```python
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=False,
)
cat.fit(X_train, y_train, cat_features=cat_cols)
```

---

### 7. Comparativa práctica (sin dogmas)

| Librería | Fortalezas típicas | Puntos de atención |
|----------|---------------------|---------------------|
| XGBoost | Madurez, ecosistema, regularización explícita | Sintonización de muchos knobs |
| LightGBM | Velocidad y escala | Controlar leaf-wise |
| CatBoost | Categorías, defaults razonables | Tiempo y memoria en algunos regímenes |

**Regla:** validar con **misma métrica y mismos folds** (Sesión 8); no extrapolar resultados de benchmarks ajenos.

---

### 8. Profundización: pseudocódigo, pérdidas y *early stopping*

#### 8.1 Esquema iterativo (gradient tree boosting)

En la iteración $m$:

1. Calcular pseudoresiduos $r_{im} = -\frac{\partial L(y_i, F)}{\partial F}\Big|_{F=\hat{F}^{(m-1)}(x_i)}$.  
2. Ajustar un árbol de regresión $h_m$ a los $\{r_{im}\}$ (con límites de profundidad).  
3. Line search para $\gamma_m$ óptimo o absorber escala en $\eta$.  
4. Actualizar $\hat{F}^{(m)} = \hat{F}^{(m-1)} + \eta\, \gamma_m h_m$.

Para **log-loss** binaria, los pseudoresiduos coinciden con $y_i - p_i^{(m-1)}$ donde $p_i$ es la probabilidad estimada en la iteración previa — puente intuitivo con “residuos de clasificación”.

#### 8.2 *Shrinkage* ($\eta$ pequeño)

Friedman recomienda **learning rates** bajos (p.ej. 0.01–0.1) y muchos árboles: reduce overfitting al impedir que un solo árbol corrija demasiado de una vez.

#### 8.3 *Early stopping*

Se monitoriza la métrica en validación tras cada iteración (o cada $k$ iteraciones). Si no mejora durante `patience` rondas, se restaura el mejor modelo. **Crítico:** el conjunto de validación debe ser **honesto** (no el test final).

#### 8.4 Comparativa rápida en Python (tres APIs)

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

X, y = make_classification(n_samples=8000, n_features=30, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, eval_metric="logloss", random_state=42,
)
xgb.fit(X_tr, y_tr)
print("XGB AUC:", roc_auc_score(y_te, xgb.predict_proba(X_te)[:, 1]))

lgbm = LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
)
lgbm.fit(X_tr, y_tr)
print("LGBM AUC:", roc_auc_score(y_te, lgbm.predict_proba(X_te)[:, 1]))

cat = CatBoostClassifier(
    iterations=400, learning_rate=0.05, depth=4, verbose=False, random_state=42,
)
cat.fit(X_tr, y_tr)
print("CAT AUC:", roc_auc_score(y_te, cat.predict_proba(X_te)[:, 1]))
```

---

### 9. Hiperparámetros, métricas y laboratorio

#### 9.1 Tabla de hiperparámetros (temario)

| Parámetro | Rol conceptual |
|-----------|----------------|
| `n_estimators` | Número de árboles / rondas de boosting |
| `learning_rate` ($\eta$) | Paso de cada contribución; menor $\eta$ suele requerir más árboles |
| `max_depth` / `num_leaves` | Complejidad de cada árbol débil |
| `subsample`, `colsample_bytree` | Bagging de filas/columnas → regularización |
| `reg_lambda`, `reg_alpha` (XGBoost) | Penalización L2/L1 en hojas/pesos |

#### 9.2 Anti-overfitting

- `early_stopping` con conjunto de validación.  
- Limitar `max_depth` / `num_leaves`.  
- Aumentar `min_child_samples` (LightGBM) o equivalentes.  
- Más datos reales siempre que sea posible.

#### 9.3 Métricas

- Clasificación: **log-loss**, AUC-ROC, F1.  
- Regresión: RMSE, MAE, $R^2$.

#### 9.4 Laboratorio (según sílabo)

- **NTB 1 —** Clasificación con XGBoost, LightGBM y CatBoost (hiperparámetros y validación).  
- **NTB 2 —** Regresión con XGBoost, LightGBM y CatBoost.

---

## Referencias bibliográficas principales

1. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.  
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.  
3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.  
4. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.  
5. Natekin, A., & Knoll, A. (2013). Gradient boosting machines, a tutorial. *Frontiers in Neurorobotics*, 7, 21.  
