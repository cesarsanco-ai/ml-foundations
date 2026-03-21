# Semana 8: Evaluación y Validación de Modelos

## Logro de la sesión

Evaluar, validar y seleccionar modelos de machine learning de forma robusta, asegurando su capacidad de generalización mediante técnicas de validación cruzada y optimización de hiperparámetros.

---

## Problemática de negocio

La brecha entre el rendimiento en entrenamiento y el rendimiento en producción es una de las principales causas de fracaso en proyectos de machine learning. Esta sesión aborda los desafíos fundamentales:

- **Modelos que funcionan bien en entrenamiento pero fallan en producción:** Ocurre cuando no se ha validado adecuadamente la capacidad de generalización.
- **Selección del mejor modelo entre múltiples alternativas:** ¿Cómo elegir objetivamente entre decenas de configuraciones posibles?
- **Riesgo de sobreajuste en datasets pequeños o complejos:** Especialmente crítico en dominios con datos limitados (ej. diagnóstico de enfermedades raras).
- **Necesidad de estimar performance real antes de desplegar:** Los stakeholders necesitan confianza en que el modelo rendirá como se espera.

**Impacto en el negocio:**

| Problema | Consecuencia | Costo |
|---------|-------------|-------|
| Overfitting no detectado | Modelo falla en producción | Pérdida de confianza, costos operativos |
| Validación inadecuada | Selección de modelo subóptimo | Oportunidades perdidas |
| Data leakage | Estimados optimistas irreales | Decisiones erróneas basadas en falsas expectativas |
| Mala estimación de performance | Incumplimiento de SLAs | Multas, clientes insatisfechos |

---

## Aplicaciones y casos típicos

### Selección de modelos en proyectos reales

**Caso 1: Competencia Kaggle**
- Múltiples modelos (XGBoost, LightGBM, Random Forest, redes neuronales)
- Validación cruzada para seleccionar el mejor
- Importancia de no contaminarse con el leaderboard público

**Caso 2: Desarrollo de modelo de scoring crediticio**
- Validación temporal (datos de diferentes años)
- Asegurar que el modelo funciona en diferentes ciclos económicos
- Validación cruzada con estratificación por nivel de riesgo

**Caso 3: Sistema de recomendación**
- Validación por usuario (no mezclar usuarios en train/test)
- Simular el comportamiento en producción

### Validación antes de despliegue en producción

**Checklist de validación pre-producción:**
1. **Reproducibilidad:** ¿Obtenemos los mismos resultados con el mismo seed?
2. **Estabilidad:** ¿Varía mucho la performance con diferentes splits?
3. **Generalización:** ¿Se mantiene en datos fuera de tiempo?
4. **Robustez:** ¿Funciona con datos ruidosos o faltantes?
5. **Comparación con baseline:** ¿Supera al modelo actual o a reglas simples?

---

## Modelado y validación

### Underfitting y Overfitting

#### Identificación y Diagnóstico

**Underfitting (subajuste):**
- El modelo no captura la estructura subyacente de los datos
- Alto error en entrenamiento y en prueba
- El modelo es demasiado simple para la complejidad del problema

**Señales de underfitting:**
- Error de entrenamiento cercano al error de prueba (ambos altos)
- Curvas de aprendizaje convergen a error alto
- El modelo no mejora con más datos

**Overfitting (sobreajuste):**
- El modelo memoriza el ruido en lugar de aprender la señal
- Error de entrenamiento muy bajo, error de prueba alto
- El modelo es demasiado complejo para la cantidad de datos

**Señales de overfitting:**
- Gran brecha entre error de entrenamiento y prueba
- Curvas de aprendizaje divergentes
- Pequeños cambios en datos producen grandes cambios en el modelo

**Visualización conceptual:**

```
Error
  ↑
  │                                   Overfitting
  │                          (train bajo, test alto)
  │                    ⋰
  │                 ⋰     ⋱
  │              ⋰           ⋱
  │           ⋰                ⋱
  │        ⋰                     ⋱ (test)
  │     ⋰                          ⋱
  │  ⋰ (train)                      ⋱
  │⋰                                 ⋱
  │                                   ⋱
  +--------------------------------------→ Complejidad
            ↑
      Underfitting
  (train y test altos)
```

#### Estrategias de Mitigación

| Problema | Estrategias |
|----------|------------|
| **Underfitting** | • Aumentar complejidad del modelo<br>• Reducir regularización<br>• Ingeniería de características<br>• Disminuir restricciones |
| **Overfitting** | • Más datos de entrenamiento<br>• Reducir complejidad del modelo<br>• Aumentar regularización<br>• Early stopping<br>• Simplificar características |

### Bias-Variance Tradeoff

#### Interpretación Práctica

El error de generalización puede descomponerse en tres componentes:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Sesgo}^2}_{\text{Bias}} + \underbrace{\text{Varianza}}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Ruido irreducible}}$$

| Componente | Interpretación | Relación con complejidad |
|-----------|---------------|-------------------------|
| **Sesgo (Bias)** | Error por asumciones erróneas del modelo | Disminuye al aumentar complejidad |
| **Varianza** | Error por sensibilidad a fluctuaciones en datos | Aumenta al aumentar complejidad |
| **Ruido** | Error irreducible inherente a los datos | Constante |

**Analogía en tiro al blanco:**

```
Bajo sesgo, baja varianza   Bajo sesgo, alta varianza
        ◎                           ○  ○  ○
      ○   ○                       ○       ○
    ○   ◎   ○                     ○  ◎  ○
      ○   ○                         ○ ○
        ○                           ○

Alto sesgo, baja varianza     Alto sesgo, alta varianza
        ○                           ○   ○
      ○   ○                         ○ ○ ○
      ○   ○                         ○ ○ ○
        ○                           ○ ○
```

#### Relación con Complejidad del Modelo

| Modelo | Sesgo | Varianza | Tendencia |
|--------|-------|----------|-----------|
| Regresión lineal | Alto | Bajo | Underfitting |
| Árbol poco profundo | Medio | Medio | Balance |
| Árbol profundo | Bajo | Alto | Overfitting |
| Random Forest | Bajo | Bajo (promediado) | Robusto |
| XGBoost (bien tunneado) | Bajo | Medio-bajo | Óptimo |

**Punto óptimo:** La complejidad que minimiza el error total en datos no vistos.

### Validación de Modelos

#### Train / Validation / Test Split

**División estándar:**
- **Training (60-80%):** Ajuste de parámetros del modelo
- **Validation (10-20%):** Ajuste de hiperparámetros y selección de modelos
- **Test (10-20%):** Evaluación final de generalización

**Proceso correcto:**
1. Dividir en train + validation y test (hold-out)
2. Usar validación cruzada en train+validation para tuning
3. Evaluar una sola vez en test al final

**Error común:** Usar test múltiples veces (contaminación).

```python
# División correcta
from sklearn.model_selection import train_test_split

# Primero separar test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Luego separar train y validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")
```

#### Validación Cruzada (Cross-Validation)

**K-Fold Cross-Validation:**

Divide los datos en $k$ folds de tamaño similar. Entrena en $k-1$ folds, valida en el fold restante. Repite $k$ veces.

```
Fold 1: [====TRAIN====][====TRAIN====][====TRAIN====][==VAL==]
Fold 2: [====TRAIN====][====TRAIN====][==VAL==][====TRAIN====]
Fold 3: [====TRAIN====][==VAL==][====TRAIN====][====TRAIN====]
Fold 4: [==VAL==][====TRAIN====][====TRAIN====][====TRAIN====]
```

**Ventajas:**
- Más estable que un solo split
- Utiliza todos los datos para entrenamiento y validación
- Mejor estimación del error de generalización

**Stratified K-Fold:**

Mantiene la proporción de clases en cada fold. Esencial para clasificación con clases desbalanceadas.

```python
from sklearn.model_selection import StratifiedKFold, KFold

# Para clasificación
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Para regresión
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"Fold {fold+1}: Train {len(train_idx)}, Val {len(val_idx)}")
```

**Leave-One-Out (LOO):**

Caso extremo de K-Fold con $k = n$ (un fold por observación). Computacionalmente costoso pero útil para datasets muy pequeños.

```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
# n modelos entrenados, uno por observación
```

**Comparación de métodos:**

| Método | Sesgo estimación | Varianza estimación | Costo computacional | Cuándo usar |
|--------|-----------------|---------------------|---------------------|-------------|
| Hold-out simple | Alto | Alta | Bajo | Datos grandes (>1M) |
| K-Fold (k=5-10) | Bajo | Media | Medio | Estándar |
| Stratified K-Fold | Bajo | Media | Medio | Clasificación desbalanceada |
| Leave-One-Out | Muy bajo | Muy alta | Muy alto | Datos muy pequeños (<100) |
| Repeated K-Fold | Muy bajo | Muy baja | Alto | Máxima precisión |

### Curvas de Aprendizaje

#### Diagnóstico de Underfitting vs Overfitting

Las curvas de aprendizaje muestran el error en función del tamaño del conjunto de entrenamiento.

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Calcular curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calcular medias y desviaciones
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Graficar
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento')
plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Validación')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Error')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.grid(True)
plt.show()
```

#### Interpretación de Curvas

**Caso 1: Underfitting**

```
Error
  ↑
  │    Train ────
  │            ╱
  │    ───────╱     Validation
  │   ╱      ╱
  │  ╱      ╱
  │ ╱      ╱
  │─────────────────→ Tamaño dataset
```

**Diagnóstico:** Train y validation convergen a error alto. No mejora con más datos.

**Acción:** Aumentar complejidad del modelo.

**Caso 2: Overfitting**

```
Error
  ↑
  │                    Validation
  │                 ╱
  │               ╱
  │             ╱
  │    Train  ╱
  │   ╱      ╱
  │  ╱      ╱
  │ ╱      ╱
  │─────────────────→ Tamaño dataset
```

**Diagnóstico:** Train error bajo, validation error alto. Brecha no se cierra con más datos.

**Acción:** Reducir complejidad, aumentar regularización.

**Caso 3: Balance ideal**

```
Error
  ↑
  │                    Validation
  │                 ╱
  │               ╱
  │             ╱
  │    Train  ╱
  │   ╱      ╱
  │  ╱      ╱
  │ ╱      ╱
  │─────────────────→ Tamaño dataset
```

**Diagnóstico:** Train y validation convergen a error bajo. Brecha pequeña.

**Acción:** Modelo listo, considerar si más datos ayudarían.

---

## Optimización de modelos

### Búsqueda de Hiperparámetros

#### Grid Search

Busca exhaustivamente todas las combinaciones de hiperparámetros en una grilla definida.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Definir grilla
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search con validación cruzada
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Mejores parámetros:", grid_search.best_params_)
print("Mejor score CV:", grid_search.best_score_)
print("Score en test:", grid_search.score(X_test, y_test))
```

**Ventajas:** Exhaustivo, encuentra óptimo dentro de la grilla.

**Desventajas:** Explosión combinatoria (número de combinaciones = producto de opciones).

**Cuándo usar:** Espacio de búsqueda pequeño (<100 combinaciones).

#### Randomized Search

Muestrea aleatoriamente combinaciones del espacio de búsqueda.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Definir distribuciones
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Randomized search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,  # 100 combinaciones aleatorias
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Mejores parámetros:", random_search.best_params_)
print("Mejor score CV:", random_search.best_score_)
```

**Ventajas:** Más eficiente que grid search en espacios grandes, puede explorar más valores.

**Desventajas:** No garantiza encontrar el óptimo, pero probabilísticamente cercano.

**Cuándo usar:** Espacio de búsqueda grande (>100 combinaciones).

#### Introducción a Bayesian Optimization

Modela la función de performance como un proceso Gaussiano y selecciona inteligentemente las siguientes combinaciones a probar.

```python
# Usando scikit-optimize (ejemplo conceptual)
from skopt import BayesSearchCV
from skopt.space import Real, Integer

param_space = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 50),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0)
}

bayes_search = BayesSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_space,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)
```

**Ventajas:** Más eficiente que random search, encuentra óptimos con menos iteraciones.

**Desventajas:** Más complejo, overhead computacional por iteración.

**Cuándo usar:** Optimización de modelos complejos, espacio de búsqueda grande, presupuesto limitado de iteraciones.

**Comparación de métodos:**

| Método | Eficiencia | Exhaustividad | Complejidad | Uso recomendado |
|--------|-----------|---------------|-------------|-----------------|
| Grid Search | Baja | Alta | Baja | Espacio pequeño (<100 combos) |
| Random Search | Media | Media | Baja | Espacio mediano (100-1000 combos) |
| Bayesian Opt | Alta | N/A | Media-alta | Espacio grande, modelos costosos |

### Buenas Prácticas

#### Evitar Data Leakage

**Data leakage:** Cuando información del futuro o del conjunto de prueba "filtra" al entrenamiento.

**Fuentes comunes de leakage:**

| Tipo | Ejemplo | Prevención |
|------|---------|------------|
| **Escalado** | Ajustar scaler en todo el dataset antes de split | Ajustar scaler solo en train, transformar test |
| **Imputación** | Calcular medias con todo el dataset | Calcular en train, aplicar en test |
| **Feature engineering** | Usar estadísticas globales (ej. frecuencia de categorías) | Calcular en train, aplicar en test |
| **Selección de características** | Seleccionar features usando todo el dataset | Hacer selección dentro de CV |
| **Validación temporal** | Usar datos futuros para predecir pasado | Respetar orden temporal |

```python
# MAL - Data leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leakage!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# BIEN - Sin leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Uso de Pipelines

Los pipelines encapsulan todo el preprocesamiento y modelo, asegurando que las transformaciones se apliquen correctamente en cada fold de CV.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Crear pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid search sobre el pipeline
param_grid = {
    'pca__n_components': [0.85, 0.9, 0.95],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)  # Todo correcto: scaler y PCA se ajustan en cada fold

# Evaluación final
y_pred = grid_search.predict(X_test)  # Transformaciones aplicadas consistentemente
```

**Ventajas de pipelines:**
1. Previene data leakage
2. Código más limpio y mantenible
3. Facilita la validación cruzada
4. Reproducibilidad

#### Separación Correcta de Datos

**Reglas de oro:**

1. **Test set se usa UNA SOLA VEZ** al final del proyecto
2. **Validation set** o CV se usa para tuning y selección
3. **Train set** se usa para ajustar parámetros

**Flujo correcto:**

```python
# 1. Separar test set (hold-out)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. En el resto, hacer validación cruzada para tuning
pipeline = Pipeline([...])
param_grid = {...}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_temp, y_temp)  # Usa validación cruzada internamente

# 3. Evaluar UNA VEZ en test
final_score = grid_search.score(X_test, y_test)
```

---

## Métricas

### Uso de Métricas en Validación Cruzada

La elección de la métrica debe alinearse con el objetivo de negocio y las características del problema.

```python
# Diferentes métricas para scoring en GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

# Para clasificación binaria
grid_search_auc = GridSearchCV(
    model, param_grid, cv=5, scoring='roc_auc'  # AUC-ROC
)

grid_search_f1 = GridSearchCV(
    model, param_grid, cv=5, scoring='f1'  # F1-score
)

# Métrica personalizada
def custom_metric(y_true, y_pred):
    # Implementar métrica específica de negocio
    return np.mean((y_true == y_pred) & (y_true == 1))  # Ejemplo

custom_scorer = make_scorer(custom_metric)
grid_search_custom = GridSearchCV(model, param_grid, cv=5, scoring=custom_scorer)
```

### Selección de Métricas alineadas al Objetivo de Negocio

| Objetivo de Negocio | Métrica Principal | Justificación |
|---------------------|-------------------|---------------|
| **Minimizar falsos negativos** | Recall | Cada FN tiene alto costo |
| **Minimizar falsos positivos** | Precision | Cada FP genera molestia/costo |
| **Balance general** | F1-score | Media armónica de P y R |
| **Ranking de probabilidades** | AUC-ROC | Capacidad discriminativa |
| **Clases muy desbalanceadas** | PR-AUC | Más sensible a minoría |
| **Predicción continua** | RMSE | Penaliza grandes errores |
| **Interpretabilidad** | Coeficientes / importancia | Explicabilidad para stakeholders |

### Comparación de Modelos basada en Métricas Promedio

**Comparación robusta con validación cruzada repetida:**

```python
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

# Validación cruzada repetida para comparación robusta
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# Evaluar múltiples modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': xgb.XGBClassifier(n_estimators=100)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    results[name] = scores
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Comparación visual
plt.figure(figsize=(10, 6))
plt.boxplot(results.values(), labels=results.keys())
plt.ylabel('AUC-ROC')
plt.title('Comparación de Modelos - Validación Cruzada Repetida (5x10)')
plt.grid(True)
plt.show()
```

**Test estadístico para comparación:**
```python
from scipy import stats

# Comparar dos modelos
t_stat, p_value = stats.ttest_rel(results['Random Forest'], results['XGBoost'])
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Diferencia estadísticamente significativa")
```

---

## Plantilla Base de Python

### Funciones Principales de `sklearn.model_selection`

```python
# ============================================
# EVALUACIÓN Y VALIDACIÓN - PLANTILLA COMPLETA
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    KFold, StratifiedKFold, RepeatedStratifiedKFold,
    learning_curve, validation_curve,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. CREAR DATOS DE EJEMPLO
# -------------------------
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42, weights=[0.8, 0.2]
)

# 2. DIVISIÓN TRAIN/VAL/TEST
# --------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. PIPELINE CON PREPROCESAMIENTO
# ---------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 4. VALIDACIÓN CRUZADA BÁSICA
# ----------------------------
cv_scores = cross_val_score(
    pipeline, X_train_val, y_train_val, 
    cv=5, scoring='roc_auc', n_jobs=-1
)
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 5. VALIDACIÓN CRUZADA CON MÚLTIPLES MÉTRICAS
# ---------------------------------------------
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(
    pipeline, X_train_val, y_train_val,
    cv=5, scoring=scoring, n_jobs=-1,
    return_train_score=True
)

for metric in scoring:
    train_score = cv_results[f'train_{metric}'].mean()
    test_score = cv_results[f'test_{metric}'].mean()
    print(f"{metric}: Train={train_score:.4f}, Test={test_score:.4f}")

# 6. CURVAS DE APRENDIZAJE
# -------------------------
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train_val, y_train_val,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='roc_auc', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Train')
plt.plot(train_sizes, val_mean, 'o-', label='Validation')
plt.xlabel('Tamaño del entrenamiento')
plt.ylabel('AUC-ROC')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.grid(True)
plt.show()

# 7. GRID SEARCH CON VALIDACIÓN CRUZADA
# -------------------------------------
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train_val, y_train_val)

print("Mejores parámetros:", grid_search.best_params_)
print("Mejor score CV:", grid_search.best_score_)

# 8. RANDOMIZED SEARCH
# --------------------
from scipy.stats import randint, uniform

param_dist = {
    'classifier__n_estimators': randint(50, 300),
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=30, cv=5,
    scoring='roc_auc', n_jobs=-1, random_state=42
)
random_search.fit(X_train_val, y_train_val)

print("Mejores parámetros (Random):", random_search.best_params_)

# 9. EVALUACIÓN FINAL EN TEST
# ---------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("EVALUACIÓN EN TEST SET")
print("="*50)
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 10. VALIDACIÓN TEMPORAL (para series temporales)
# ------------------------------------------------
# Ejemplo: entrenar con datos antiguos, validar con recientes
# n_train = int(0.7 * len(X))
# X_train_time = X[:n_train]
# y_train_time = y[:n_train]
# X_val_time = X[n_train:n_train + int(0.15 * len(X))]
# y_val_time = y[n_train:n_train + int(0.15 * len(X))]
# X_test_time = X[n_train + int(0.15 * len(X)):]
# y_test_time = y[n_train + int(0.15 * len(X)):]
```

### Funciones Útiles Adicionales

```python
# Validación cruzada repetida para estimación robusta
def robust_cv_evaluation(model, X, y, n_splits=5, n_repeats=10, scoring='roc_auc'):
    """Evalúa modelo con validación cruzada repetida"""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    result = {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'ci_95': 1.96 * scores.std() / np.sqrt(len(scores))
    }
    return result, scores

# Curva de validación para un hiperparámetro
def plot_validation_curve(model, X, y, param_name, param_range, scoring='roc_auc'):
    """Grafica curva de validación para un hiperparámetro"""
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring=scoring, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', label='Train')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(param_range, val_mean, 'o-', label='Validation')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.title(f'Curva de Validación - {param_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

## Comunicación

### Explicación de Performance Esperada en Producción

**Ejemplo de reporte ejecutivo:**

> **Estimación de Performance en Producción**
>
> Hemos evaluado el modelo mediante validación cruzada estratificada de 5 folds repetida 10 veces (50 evaluaciones en total) para obtener una estimación robusta de su rendimiento esperado.
>
> **Resultados:**
> - **AUC-ROC esperado:** 0.87 ± 0.02 (intervalo de confianza 95%)
> - **F1-score esperado:** 0.72 ± 0.03
> - **Recall esperado:** 0.78 ± 0.04
>
> **Interpretación para negocio:**
> - Podemos esperar que el modelo detecte entre 74% y 82% de los fraudes reales (Recall)
> - La precisión se mantendrá entre 65% y 71%, generando entre X y Y falsas alarmas mensuales
> - Estos rangos consideran la variabilidad natural en los datos; el rendimiento real debería estar dentro de estos límites con 95% de confianza

### Comparación Clara entre Modelos

**Tabla comparativa: Baseline vs Optimizado**

| Métrica | Modelo Baseline | Modelo Optimizado | Mejora |
|---------|-----------------|-------------------|--------|
| AUC-ROC (CV) | 0.82 ± 0.03 | 0.89 ± 0.02 | +8.5% |
| F1-score (CV) | 0.64 ± 0.04 | 0.73 ± 0.03 | +14.1% |
| Tiempo entrenamiento | 2.3 min | 45 min | +1,856% |
| Tiempo predicción | 0.8 ms | 1.2 ms | +50% |

**Explicación del trade-off:**

> "El modelo optimizado requiere 20 veces más tiempo de entrenamiento pero mejora el AUC en 8.5% y el F1 en 14%. En producción, la latencia de predicción aumenta solo 0.4ms, lo cual es aceptable para nuestro sistema. Dado el volumen de transacciones (1M mensuales), la mejora en detección de fraude justifica ampliamente el costo computacional adicional."

### Justificación de Selección de Modelo

**Template de justificación:**

> **Modelo seleccionado:** [Nombre del modelo con hiperparámetros]
>
> **Proceso de selección:**
> 1. Evaluamos 5 familias de modelos con validación cruzada de 5 folds
> 2. Los 3 mejores fueron sometidos a optimización de hiperparámetros (Random Search + Grid Search)
> 3. El modelo final se seleccionó basado en [métrica principal] en validación cruzada
> 4. Se evaluó UNA VEZ en test set para confirmar generalización
>
> **Resultados comparativos:**
> [Incluir tabla comparativa]
>
> **Razones de la selección:**
> - [Razón 1: mejor métrica principal]
> - [Razón 2: estabilidad (menor desviación en CV)]
> - [Razón 3: trade-off aceptable entre performance y complejidad]
> - [Razón 4: cumplimiento de requisitos de latencia]

### Comunicación de Riesgos

**Sobre variabilidad y overfitting:**

> "Es importante considerar que el rendimiento estimado tiene una variabilidad inherente. Nuestras validaciones muestran que el AUC-ROC puede variar ±0.02 dependiendo de la composición específica de los datos. Esto significa que en producción podríamos ver valores entre 0.87 y 0.91. Monitorearemos continuamente para detectar desviaciones fuera de este rango."

**Sobre data leakage:**

> "Hemos implementado pipelines que garantizan que no hay data leakage en nuestro proceso de validación. Cada transformación se aprende exclusivamente en los datos de entrenamiento de cada fold, replicando fielmente las condiciones de producción donde el modelo solo verá datos nuevos."

**Sobre necesidad de reentrenamiento:**

> "Las curvas de aprendizaje indican que el modelo aún se beneficia de más datos. Recomendamos reentrenar mensualmente a medida que acumulemos nuevas transacciones, lo que podría mejorar aún más el rendimiento."

---




## Reto: 1 punto

Implementar validación cruzada y búsqueda de hiperparámetros en un modelo (por ejemplo, Random Forest o XGBoost), comparando resultados antes y después del tuning.

---

## Laboratorio

Pipeline completo de validación:
- Train/test split
- Validación cruzada
- Entrenamiento de modelo base
- Búsqueda de hiperparámetros
- Evaluación final en test

Comparación de múltiples modelos (logística, árboles, boosting)

---

## Anexo: Fundamento Matemático y Computacional

### A1. Bias-Variance Decomposition

#### A1.1. Derivación Formal

La descomposición sesgo-varianza es un resultado fundamental que explica el error de generalización de un modelo predictivo. Para un punto fijo $x_0$, consideramos:

- $y_0 = f(x_0) + \epsilon$: valor real, donde $\epsilon$ es ruido con media 0 y varianza $\sigma^2$
- $\hat{f}(x_0)$: predicción del modelo entrenado con una muestra aleatoria $D$

El error cuadrático esperado para este punto es:

$$\mathbb{E}_D[(y_0 - \hat{f}(x_0; D))^2]$$

Desarrollando:

$$\mathbb{E}[(y_0 - \hat{f})^2] = \mathbb{E}[(f(x_0) + \epsilon - \hat{f})^2]$$

$$= \mathbb{E}[(f(x_0) - \hat{f})^2] + \mathbb{E}[\epsilon^2] + 2\mathbb{E}[(f(x_0) - \hat{f})\epsilon]$$

Como $\epsilon$ es independiente de $D$ y tiene media 0, el término cruzado se anula:

$$\mathbb{E}[(f(x_0) - \hat{f})\epsilon] = \mathbb{E}[f(x_0) - \hat{f}] \cdot \mathbb{E}[\epsilon] = 0$$

Por lo tanto:

$$\mathbb{E}[(y_0 - \hat{f})^2] = \mathbb{E}[(f(x_0) - \hat{f})^2] + \sigma^2$$

#### A1.2. Descomposición del Término $(f(x_0) - \hat{f})^2$

Sumamos y restamos $\mathbb{E}[\hat{f}]$ dentro del cuadrado:

$$\mathbb{E}[(f - \hat{f})^2] = \mathbb{E}[(f - \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}] - \hat{f})^2]$$

$$= \mathbb{E}[(f - \mathbb{E}[\hat{f}])^2] + \mathbb{E}[(\mathbb{E}[\hat{f}] - \hat{f})^2] + 2\mathbb{E}[(f - \mathbb{E}[\hat{f}])(\mathbb{E}[\hat{f}] - \hat{f})]$$

El término cruzado se anula porque $\mathbb{E}[\mathbb{E}[\hat{f}] - \hat{f}] = 0$:

$$\mathbb{E}[(f - \mathbb{E}[\hat{f}])(\mathbb{E}[\hat{f}] - \hat{f})] = (f - \mathbb{E}[\hat{f}])\mathbb{E}[\mathbb{E}[\hat{f}] - \hat{f}] = 0$$

#### A1.3. Resultado Final

$$\mathbb{E}[(y_0 - \hat{f}(x_0))^2] = \underbrace{(f(x_0) - \mathbb{E}[\hat{f}(x_0)])^2}_{\text{Sesgo}^2} + \underbrace{\mathbb{E}[(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)])^2]}_{\text{Varianza}} + \underbrace{\sigma^2}_{\text{Ruido}}$$

**Interpretación:**

- **Sesgo:** Error por asumciones erróneas del modelo. Mide cuánto se aleja la predicción promedio del valor real.
- **Varianza:** Error por sensibilidad a fluctuaciones en los datos de entrenamiento.
- **Ruido:** Error irreducible inherente al problema.

#### A1.4. Relación con Complejidad del Modelo

| Modelo | Sesgo | Varianza | Error total |
|--------|-------|----------|-------------|
| Muy simple (ej. media constante) | Alto | Bajo | Sesgo domina |
| Complejidad óptima | Balance | Balance | Mínimo |
| Muy complejo (ej. árbol profundo) | Bajo | Alto | Varianza domina |

**Visualización matemática:**

Para un modelo paramétrico con $d$ parámetros, típicamente:

- Sesgo $\propto \frac{1}{d}$ (decrece con complejidad)
- Varianza $\propto d$ (crece con complejidad)

El error total tiene un mínimo en:

$$\frac{\partial}{\partial d}[\text{Sesgo}^2(d) + \text{Varianza}(d)] = 0$$

---

### A2. Fundamento de Validación Cruzada

#### A2.1. Estimación del Error de Generalización

Sea $D = \{(x_i, y_i)\}_{i=1}^n$ una muestra i.i.d. de una distribución desconocida $P$. Un modelo $\hat{f}$ entrenado en $D$ tiene error esperado:

$$\text{Err} = \mathbb{E}_{(x,y) \sim P}[L(y, \hat{f}(x))]$$

Queremos estimar este error sin tener acceso a $P$.

#### A2.2. Sesgo del Error de Entrenamiento

El error de entrenamiento (resustitución) es un estimador sesgado:

$$\widehat{\text{Err}}_{\text{train}} = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{f}(x_i))$$

**Sesgo:** $\mathbb{E}[\widehat{\text{Err}}_{\text{train}}] < \text{Err}$ porque $\hat{f}$ se ha optimizado precisamente para minimizar este error.

#### A2.3. Validación Cruzada K-Fold

Dividimos los datos en $K$ folds de tamaño aproximadamente igual: $D_1, D_2, \ldots, D_K$.

Para cada fold $k$:
1. Entrenamos $\hat{f}^{(-k)}$ en todos los datos excepto $D_k$
2. Evaluamos en $D_k$: $\text{Err}_k = \frac{1}{|D_k|}\sum_{i \in D_k} L(y_i, \hat{f}^{(-k)}(x_i))$

El estimador CV es:

$$\widehat{\text{Err}}_{\text{CV}} = \frac{1}{K}\sum_{k=1}^K \text{Err}_k$$

**Propiedades:**

- **Insesgamiento:** $\mathbb{E}[\widehat{\text{Err}}_{\text{CV}}] \approx \text{Err}$ (aproximadamente insesgado)
- **Varianza:** Depende de $K$ y de la estabilidad del modelo

#### A2.4. Sesgo y Varianza en CV

**Efecto de $K$:**

| K | Sesgo | Varianza | Caso |
|---|-------|----------|------|
| Pequeño (ej. 5) | Mayor sesgo (menos datos para entrenar) | Menor varianza (folds más grandes) | Datos grandes (>100k) |
| Grande (ej. 10-20) | Menor sesgo | Mayor varianza | Estándar |
| Leave-One-Out ($K=n$) | Mínimo sesgo | Máxima varianza | Datos pequeños (<100) |

**Justificación matemática para LOO:**
- Entrenamiento con $n-1$ observaciones, casi todo el dataset
- Pero los folds están altamente correlacionados
- Varianza puede ser alta

#### A2.5. Validación Cruzada Estratificada

Para clasificación, el error de estimación puede sesgarse si las proporciones de clase varían entre folds. La validación estratificada asegura que cada fold tenga la misma proporción de clases que el dataset original.

Sea $p_c = \frac{n_c}{n}$ la proporción de clase $c$. En cada fold $k$, aseguramos:

$$\frac{n_{c,k}}{n_k} \approx p_c \quad \forall c, k$$

Esto minimiza la varianza adicional por desbalanceo.

#### A2.6. Intervalos de Confianza para CV

Bajo supuestos de normalidad, podemos construir intervalos de confianza:

$$\widehat{\text{Err}}_{\text{CV}} \pm t_{\alpha/2, K-1} \cdot \frac{s}{\sqrt{K}}$$

donde $s$ es la desviación estándar de los errores por fold.

Sin embargo, los folds no son independientes, por lo que estos intervalos son aproximados. Una alternativa más robusta es la **validación cruzada repetida** (repeated CV):

$$\widehat{\text{Err}}_{\text{CV-repeat}} = \frac{1}{R}\sum_{r=1}^R \widehat{\text{Err}}_{\text{CV}}^{(r)}$$

---

### A3. Complejidad Computacional del Tuning

#### A3.1. Grid Search

Para $p$ hiperparámetros, cada uno con $v_i$ valores posibles:

$$\text{Combinaciones} = \prod_{i=1}^p v_i$$

Para cada combinación, realizamos validación cruzada con $K$ folds:

$$\text{Costo total} = \left(\prod_{i=1}^p v_i\right) \cdot K \cdot C_{\text{entrenamiento}}$$

donde $C_{\text{entrenamiento}}$ es el costo de entrenar el modelo una vez.

**Ejemplo:**
- $p=3$ hiperparámetros con $v=[5, 4, 6]$
- Combinaciones: $5 \times 4 \times 6 = 120$
- $K=5$ folds
- Total: $120 \times 5 = 600$ entrenamientos

#### A3.2. Randomized Search

En lugar de probar todas las combinaciones, muestreamos $N$ combinaciones aleatorias:

$$\text{Costo total} = N \cdot K \cdot C_{\text{entrenamiento}}$$

**Probabilidad de encontrar el óptimo:**

Si el óptimo ocupa una fracción $q$ del espacio de búsqueda, la probabilidad de no encontrarlo después de $N$ intentos es $(1-q)^N$.

Para $q=0.05$, necesitamos $N \approx 60$ para tener 95% de probabilidad de incluir el óptimo (vs 120 de grid search).

#### A3.3. Escalamiento con Validación Cruzada

**Complejidad computacional total:**

$$O(\text{combinaciones} \times K \times T(n,p))$$

donde $T(n,p)$ es la complejidad del algoritmo base.

**Ejemplos de $T(n,p)$:**
- Regresión logística: $O(p^2 n)$
- Random Forest: $O(B \cdot p \cdot n \log n)$
- XGBoost: $O(M \cdot p \cdot n \log n)$

#### A3.4. Optimizaciones Prácticas

**1. Validación cruzada anidada (nested CV):**

Para evitar overfitting en la selección de modelos:

```
Capa externa (estimación de error):
├── Fold 1 (test externo)
│   Capa interna (selección de modelo):
│   ├── Grid search con CV interna en datos de entrenamiento
│   └── Mejor modelo evaluado en test externo
├── Fold 2 (test externo)
│   Capa interna: (repetir)
└── ...
```

**Costo:** multiplicativo $K_{\text{externa}} \times K_{\text{interna}} \times \text{combinaciones}$

**2. Early stopping en CV:**

Si la validación cruzada interna muestra que una combinación es claramente inferior, detener su evaluación tempranamente.

**3. Aproximaciones sucesivas:**

Comenzar con búsqueda gruesa, refinar alrededor de las mejores regiones.

---

### A4. Riesgo de Overfitting en Selección de Modelos

#### A4.1. El Problema de la Selección Múltiple

Cuando probamos múltiples modelos/configuraciones y seleccionamos la mejor basada en validación cruzada, estamos realizando **selección de modelos**. Esto introduce un sesgo de optimismo.

**Ejemplo numérico:**

Supongamos que probamos $M=100$ modelos diferentes, todos con error real $\text{Err}=0.20$. Por azar, algunos tendrán errores estimados menores:

- Error estimado mínimo esperado ≈ $\text{Err} - \sigma \cdot \Phi^{-1}(1/M)$
- Para $\sigma=0.02$ (desviación estándar de la estimación), el mínimo esperado ≈ $0.20 - 0.02 \cdot \Phi^{-1}(0.99) \approx 0.153$

El error real del modelo seleccionado será mayor que el estimado.

#### A4.2. Sesgo de Optimismo en CV

Sea $\widehat{\text{Err}}_j$ la estimación CV para el modelo $j$. Seleccionamos:

$$\hat{j} = \arg\min_j \widehat{\text{Err}}_j$$

El optimismo es:

$$\text{Optimismo} = \mathbb{E}[\widehat{\text{Err}}_{\hat{j}} - \text{Err}_{\hat{j}}] < 0$$

Este optimismo aumenta con:
- Número de modelos probados ($M$)
- Varianza de las estimaciones CV
- Correlación entre modelos

#### A4.3. Corrección con Validación Cruzada Anidada

La validación cruzada anidada proporciona una estimación (casi) insesgada del error del proceso de selección:

1. En cada fold externo, se realiza selección de modelos (con CV interna) usando solo los datos de entrenamiento
2. El modelo seleccionado se evalúa en el fold externo
3. Se promedia sobre folds externos

**Resultado:** $\widehat{\text{Err}}_{\text{nested}}$ es un estimador (casi) insesgado del error de generalización del proceso de selección + entrenamiento.

#### A4.4. Teorema de Estabilidad para CV

Para algoritmos estables (pequeños cambios en datos producen pequeños cambios en el modelo), la validación cruzada tiene propiedades deseables:

Si el algoritmo es $\beta$-estable en el sentido de que:

$$\sup_{x,y} |L(y, \hat{f}_D(x)) - L(y, \hat{f}_{D^{(-i)}}(x))| \leq \beta$$

entonces el sesgo de CV es $O(\beta)$ y la varianza es $O(1/(nK))$.

#### A4.5. Pruebas de Significancia Estadística

Para comparar dos modelos después de selección, necesitamos pruebas que consideren la selección múltiple:

**Test de McNemar (para clasificación):**

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

donde $n_{01}$ = número de casos donde modelo A falla y modelo B acierta, y viceversa.

**Test de Diebold-Mariano (para regresión):**

$$DM = \frac{\bar{d}}{\sqrt{\widehat{Var}(d)/n}}$$

donde $d_i = L(y_i, \hat{f}_A(x_i)) - L(y_i, \hat{f}_B(x_i))$

#### A4.6. Corrección por Múltiples Comparaciones

Cuando comparamos múltiples modelos, debemos ajustar los niveles de significancia:

**Bonferroni:** $\alpha_{\text{ajustado}} = \alpha / M$

**Holm:** Paso a paso, más potente que Bonferroni

**FDR (False Discovery Rate):** Controla proporción de falsos positivos

---

### A5. Resumen Matemático

| Concepto | Expresión Matemática |
|----------|---------------------|
| Bias-Variance decomposition | $\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2[\hat{f}] + \text{Var}[\hat{f}] + \sigma^2$ |
| Error de entrenamiento | $\widehat{\text{Err}}_{\text{train}} = \frac{1}{n}\sum_i L(y_i, \hat{f}(x_i))$ |
| CV K-Fold | $\widehat{\text{Err}}_{\text{CV}} = \frac{1}{K}\sum_k \frac{1}{|D_k|}\sum_{i \in D_k} L(y_i, \hat{f}^{(-k)}(x_i))$ |
| Grid Search complexity | $O\left(\prod_i v_i \cdot K \cdot T(n,p)\right)$ |
| Randomized Search complexity | $O(N \cdot K \cdot T(n,p))$ |
| Optimismo en selección | $\mathbb{E}[\min_j \widehat{\text{Err}}_j] < \min_j \mathbb{E}[\widehat{\text{Err}}_j]$ |
| Corrección Bonferroni | $\alpha_{\text{adj}} = \alpha/M$ |

---
