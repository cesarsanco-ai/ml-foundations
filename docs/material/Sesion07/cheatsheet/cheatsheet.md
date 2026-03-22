---
layout: default
---

# Cheatsheet: Gradient Boosting (XGBoost, LightGBM)
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-07](../../../sesiones/sesion-07.md)

---

## Idea central

Secuencia de árboles débiles; cada uno corrige los **residuos** (gradiente de la pérdida) del conjunto actual.

* **Bagging (RF):** paralelo, reduce varianza.  
* **Boosting:** secuencial, baja sesgo y refina errores.  

---

## Hiperparámetros esenciales

| Parámetro | Rol |
| :--- | :--- |
| `n_estimators` | Número de árboles; subir con `learning_rate` bajo |
| `learning_rate` ($\eta$) | Paso del boosting; más bajo → más árboles |
| `max_depth` | Profundidad típica 3–10 en tabular |
| `subsample` | Fracción de filas por árbol (stochastic boosting) |
| `colsample_bytree` | Fracción de columnas por árbol |
| `reg_lambda` / `reg_alpha` | Regularización L2 / L1 (XGBoost) |

---

## XGBoost (clasificación)

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

---

## LightGBM (rápido en grandes datos)

* Hojas **best-first** (leaf-wise); cuidado con overfitting → `num_leaves`, `min_child_samples`.  
* Manejo eficiente de categóricas y datos grandes.  

```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "metric": "binary_logloss",
}
model = lgb.train(params, train_data, num_boost_round=500)
```

---

## Early stopping y validación

* Usar **conjunto de validación** real; no mirar test para cortar rounds.  
* **StratifiedKFold** en clasificación; **TimeSeriesSplit** si hay tiempo.  

---

## Puntos críticos

* Boosting puede **sobreajustar** con `learning_rate` alto y demasiados árboles sin early stopping.  
* **Orden de filas** no importa salvo splits temporales.  
* Para interpretabilidad: SHAP (sesión 13) o importancia nativa con cautela.  

> *“En tabular competitivo, boosting suele estar en el podio.”*
