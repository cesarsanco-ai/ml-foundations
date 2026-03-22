---
layout: default
---

# Cheatsheet: Regresión Lineal y Regularización
**Autor:** Carlos César Sánchez Coronel  

---

## Modelo

* **Predicción:** $\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j$  
* **OLS:** minimizar $\sum_i (y_i - \hat{y}_i)^2$  

---

## Supuestos (interpretación clásica)

1. Linealidad aproximada  
2. Errores independientes  
3. Homocedasticidad  
4. Normalidad de residuos (sobre todo para intervalos)  

---

## Regularización

| Método | Penalización | Efecto |
| :--- | :--- | :--- |
| **Ridge (L2)** | $\lambda \sum \beta_j^2$ | Encoge coeficientes; multicolinealidad |
| **Lasso (L1)** | $\lambda \sum |\beta_j|$ | Selección de variables (sparse) |
| **Elastic Net** | L1 + L2 | Compromiso; muchas correladas |

Elegir $\lambda$ con **validación cruzada** (`RidgeCV`, `LassoCV`, `ElasticNetCV`).

---

## Métricas de regresión

| Métrica | Fórmula / idea | Nota |
| :--- | :--- | :--- |
| **MSE / RMSE** | Error cuadrático; RMSE en unidades de $y$ | Penaliza grandes errores |
| **MAE** | $\frac{1}{n}\sum \|y_i-\hat{y}_i\|$ | Robusto a outliers |
| **$R^2$** | Varianza explicada | No sustituye error en unidades |

---

## Código base

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)

rmse = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE lin:", rmse(y_test, lin.predict(X_test)))
```

---

## Cuándo usar qué

| Situación | Elección |
| :--- | :--- |
| Muchas features correlacionadas | Ridge o Elastic Net |
| Selección automática de variables | Lasso / Elastic Net |
| Baseline interpretable | Regresión lineal sin penalizar |
| Relación claramente no lineal | Otros modelos (árboles, etc.) |

---

## Puntos críticos

* Estandarizar suele ayudar a comparar magnitudes de $\beta$ con Ridge/Lasso.  
* **Outliers** inflan MSE; considerar MAE o modelos robustos.  
* $R^2$ alto con overfitting es engañoso: mirar **validación**.  

> *“Lineal + regularización sigue siendo un baseline fuerte en tabular.”*
