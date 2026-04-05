---
layout: default
---

# Plantillas base en Python
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-07](../../../sesiones/sesion-07.md)


---

## Uso en clasificación y regresión completo

#### Introduccion: clasificación binaria con LightGBM
```python
import lightgbm as lgb                                      # Librería principal
from sklearn.model_selection import train_test_split        # División train/test
from sklearn.metrics import roc_auc_score, log_loss         # Métricas

# Asumiendo X (DataFrame o numpy) e y (serie 0/1) ya cargados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)                                                           # División estratificada

# Crear dataset en formato LightGBM (opcional pero acelera)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parámetros base
params = {
    'objective': 'binary',          # Clasificación binaria
    'metric': 'auc',                # Métrica de evaluación
    'boosting_type': 'gbdt',        # Gradient Boosting tradicional
    'num_leaves': 31,               # Máximo hojas por árbol (leaf-wise)
    'learning_rate': 0.05,          # Contracción (eta)
    'feature_fraction': 0.8,        # Submuestreo de columnas
    'bagging_fraction': 0.8,        # Submuestreo de filas
    'bagging_freq': 5,              # Frecuencia de bagging
    'verbose': -1                   # Silencioso
}

# Entrenamiento
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,            # Número de árboles (M)
    valid_sets=[test_data],         # Para early stopping
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)                                   # Early stopping con paciencia 10

# Predicción de probabilidades
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)

# Evaluación
auc = roc_auc_score(y_test, y_pred_proba)   # Área bajo la curva ROC
logloss = log_loss(y_test, y_pred_proba)    # Pérdida logarítmica

print(f"AUC: {auc:.4f} | LogLoss: {logloss:.4f}")
```

#### XGBoost

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score, 
                             mean_squared_error, mean_absolute_error)

# Asumir X (DataFrame/array) e y ya cargados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
    stratify=y if y.dtype.kind in 'iuf' else None   # estratificar solo clasificación
)

# --- Configuración según tarea ---
task = 'regression'   # Cambiar a 'binary' o 'multiclass'

if task == 'binary':
    objective = 'binary:logistic'
    metric = 'auc'
    num_classes = 1
    eval_metric_list = ['auc', 'logloss']
    predict_func = lambda m, X: m.predict(xgb.DMatrix(X))  # probabilidades
    # Métricas finales
    y_pred = predict_func(model, X_test)
    score1 = roc_auc_score(y_test, y_pred)
    score2 = log_loss(y_test, y_pred)
    print(f"AUC: {score1:.4f} | LogLoss: {score2:.4f}")

elif task == 'multiclass':
    objective = 'multi:softprob'
    metric = 'mlogloss'
    num_classes = len(set(y_train))
    eval_metric_list = ['mlogloss']
    predict_func = lambda m, X: m.predict(xgb.DMatrix(X))  # matriz (n, clases)
    y_pred_proba = predict_func(model, X_test)
    y_pred_class = y_pred_proba.argmax(axis=1)
    score1 = log_loss(y_test, y_pred_proba)
    score2 = accuracy_score(y_test, y_pred_class)
    print(f"LogLoss: {score1:.4f} | Accuracy: {score2:.4f}")

else:  # regression
    objective = 'reg:squarederror'
    metric = 'rmse'
    num_classes = 1
    eval_metric_list = ['rmse', 'mae']
    predict_func = lambda m, X: m.predict(xgb.DMatrix(X))  # valores continuos
    y_pred = predict_func(model, X_test)
    score1 = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    score2 = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {score1:.4f} | MAE: {score2:.4f}")

# Parámetros base (comunes)
params = {
    'objective': objective,
    'metric': metric,
    'booster': 'gbtree',
    'eta': 0.05,                # learning rate
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbosity': 0
}
if task == 'multiclass':
    params['num_class'] = num_classes

# Crear DMatrix (eficiente para XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Entrenamiento con early stopping
evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params, dtrain,
    num_boost_round=100,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=False
)

# Predicción (usar la función definida según tarea)
# (Las líneas de predicción y métricas ya están dentro de los condicionales)
```

---

#### LightGBM

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if y.dtype.kind in 'iuf' else None
)

# --- Selección de tarea ---
task = 'binary'   # 'binary', 'multiclass', 'regression'

if task == 'binary':
    objective = 'binary'
    metric = 'auc'
    eval_metric_list = ['auc', 'binary_logloss']
    predict_func = lambda m, X: m.predict(X)  # probabilidades
    # Evaluación
    y_pred = predict_func(model, X_test)
    score1 = roc_auc_score(y_test, y_pred)
    score2 = log_loss(y_test, y_pred)
    print(f"AUC: {score1:.4f} | LogLoss: {score2:.4f}")

elif task == 'multiclass':
    objective = 'multiclass'
    metric = 'multi_logloss'
    num_classes = len(set(y_train))
    eval_metric_list = ['multi_logloss']
    predict_func = lambda m, X: m.predict(X)  # matriz (n, clases)
    y_pred_proba = predict_func(model, X_test)
    y_pred_class = y_pred_proba.argmax(axis=1)
    score1 = log_loss(y_test, y_pred_proba)
    score2 = accuracy_score(y_test, y_pred_class)
    print(f"LogLoss: {score1:.4f} | Accuracy: {score2:.4f}")

else:  # regression
    objective = 'regression'
    metric = 'rmse'
    eval_metric_list = ['rmse', 'mae']
    predict_func = lambda m, X: m.predict(X)  # valores continuos
    y_pred = predict_func(model, X_test)
    score1 = mean_squared_error(y_test, y_pred, squared=False)
    score2 = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {score1:.4f} | MAE: {score2:.4f}")

# Parámetros base
params = {
    'objective': objective,
    'metric': metric,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
if task == 'multiclass':
    params['num_class'] = num_classes

# Dataset LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Entrenamiento
model = lgb.train(
    params, train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

# Predicción y métricas (ejecutan dentro del condicional superior)
```

---

#### CatBoost

```python
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if y.dtype.kind in 'iuf' else None
)

# --- Selección de tarea ---
task = 'regression'   # 'binary', 'multiclass', 'regression'

# CatBoost maneja automáticamente features categóricas con cat_features=[]
# Asumiendo que X tiene columnas numéricas; si hay categóricas, pasar índices o nombres.
cat_features = []  # Ejemplo: [0,2,5] o ['col1','col3']

if task == 'binary':
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        cat_features=cat_features,
        verbose=False,
        early_stopping_rounds=10
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    y_pred = model.predict_proba(X_test)[:, 1]  # probabilidad clase 1
    score1 = roc_auc_score(y_test, y_pred)
    score2 = log_loss(y_test, y_pred)
    print(f"AUC: {score1:.4f} | LogLoss: {score2:.4f}")

elif task == 'multiclass':
    num_classes = len(set(y_train))
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        cat_features=cat_features,
        verbose=False,
        early_stopping_rounds=10
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    y_pred_proba = model.predict_proba(X_test)  # matriz (n, clases)
    y_pred_class = model.predict(X_test)
    score1 = log_loss(y_test, y_pred_proba)
    score2 = accuracy_score(y_test, y_pred_class)
    print(f"LogLoss: {score1:.4f} | Accuracy: {score2:.4f}")

else:  # regression
    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        cat_features=cat_features,
        verbose=False,
        early_stopping_rounds=10
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    y_pred = model.predict(X_test)
    score1 = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    score2 = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {score1:.4f} | MAE: {score2:.4f}")
```

**Notas importantes**:
- En **XGBoost** y **LightGBM**, la predicción para regresión es directa (`predict` devuelve el valor estimado); para clasificación binaria devuelve probabilidad (si `objective` termina en `logistic` o `binary`); para multiclase devuelve matriz de probabilidades.
- **CatBoost** usa clases separadas (`CatBoostClassifier` y `CatBoostRegressor`), lo que simplifica la configuración.
- En **multiclase**, recuerda que `y` debe estar etiquetada como enteros consecutivos desde 0.
- El **early stopping** y la validación cruzada interna se muestran en cada plantilla. Ajusta `num_boost_round`/`iterations` según tamaño de datos.
- Para **datos con muchas categorías**, CatBoost es el único que no requiere preprocesamiento (`cat_features`).