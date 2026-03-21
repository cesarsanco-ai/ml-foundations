# Semana 7: Gradient Boosting (XGBoost, LightGBM)

## Logro de la sesión

Construir, optimizar e interpretar modelos de Gradient Boosting (XGBoost y LightGBM), comprendiendo su funcionamiento secuencial y ajustando hiperparámetros para maximizar el rendimiento en problemas de clasificación y regresión.

---

## Problemática de negocio

Los modelos de Gradient Boosting representan el estado del arte en datos tabulares estructurados, dominando competencias como Kaggle y siendo ampliamente adoptados en la industria. Responden a necesidades críticas:

- **Necesidad de modelos de alto rendimiento:** En entornos competitivos, pequeñas mejoras en precisión se traducen en ventajas significativas (ej. 0.1% de mejora en CTR puede significar millones en ingresos).
- **Mejora incremental de predicciones:** Gradient Boosting construye modelos secuencialmente, cada uno corrigiendo los errores del anterior, lo que permite alcanzar rendimientos superiores.
- **Manejo de relaciones complejas:** Capturan interacciones no lineales y patrones sutiles que otros modelos no detectan.
- **Trade-off estratégico:** Performance, tiempo de entrenamiento e interpretabilidad deben balancearse según el contexto de negocio.

**Ejemplos de aplicación por industria:**

| Industria | Caso de uso | Impacto de negocio |
|-----------|-------------|-------------------|
| **Publicidad digital** | Predicción de CTR (click-through rate) | Optimización de pujas en tiempo real, aumento de ingresos por clics |
| **Banca** | Scoring crediticio avanzado | Reducción de impagos, optimización de aprobaciones |
| **Fintech** | Detección de fraude en tiempo real | Pérdidas evitadas, cumplimiento regulatorio |
| **E-commerce** | Sistemas de recomendación | Incremento de conversión y venta cruzada |
| **Seguros** | Predicción de siniestralidad | Tarificación precisa, selección de riesgos |
| **Telecom** | Predicción de churn | Retención de clientes de alto valor |
| **Energía** | Predicción de demanda | Optimización de generación y distribución |

**Limitaciones de modelos anteriores que resuelve Gradient Boosting:**

| Modelo | Limitación | Cómo lo resuelve Gradient Boosting |
|--------|------------|-----------------------------------|
| Regresión lineal | No captura no linealidades | Árboles como base aprenden relaciones complejas |
| Random Forest | Puede ser subóptimo en ciertos problemas | Optimización directa de la función de pérdida |
| Árbol simple | Alto sesgo o varianza | Ensamble secuencial reduce ambos |
| SVM | Escalamiento cuadrático con n | Escala linealmente con datos grandes |
| Redes neuronales | Requiere muchos datos y tuning | Excelente con datos tabulares medianos |

---

## Modelado

### Boosting: Fundamentos

#### Ensamble Secuencial de Modelos Débiles

A diferencia de bagging (Random Forest) donde los modelos se construyen en paralelo e independientemente, **boosting** construye modelos de manera **secuencial**, donde cada nuevo modelo se enfoca en corregir los errores del conjunto previo.

**Analogía:** Un grupo de estudiantes aprende un tema. El primer estudiante comete errores. El segundo estudia específicamente los errores del primero. El tercero se enfoca en los errores residuales, y así sucesivamente. El conocimiento combinado supera al de cualquier individuo.

**Evolución de algoritmos boosting:**
- **AdaBoost (1995):** Asigna pesos a las observaciones, aumentando el peso de las mal clasificadas.
- **Gradient Boosting (1999):** Generalización usando gradientes de cualquier función de pérdida diferenciable.
- **XGBoost (2014):** Optimizaciones computacionales, regularización, manejo de missing values.
- **LightGBM (2016):** Muestreo basado en gradiente, histogramas, entrenamiento ultra-rápido.
- **CatBoost (2017):** Manejo óptimo de variables categóricas.

#### Diferencia Clave con Bagging (Random Forest)

| Aspecto | Bagging (Random Forest) | Boosting (Gradient Boosting) |
|---------|------------------------|------------------------------|
| **Construcción** | Paralela (independiente) | Secuencial (dependiente) |
| **Objetivo** | Reducir varianza | Reducir sesgo + varianza |
| **Modelos base** | Profundos (bajo sesgo) | Débiles (shallow trees) |
| **Peso de datos** | Bootstrap uniforme | Pesos basados en error/gradiente |
| **Enfoque** | Promediar errores | Corregir errores |
| **Overfitting** | Robusto por diseño | Riesgo si learning rate alto |
| **Tiempo entrenamiento** | Paralelizable | Secuencial (más lento) |

**Visualización conceptual:**

```
Random Forest (Bagging):
    Datos → Árbol₁ → Pred₁ \
    Datos → Árbol₂ → Pred₂  → Promedio → Predicción final
    Datos → Árbol₃ → Pred₃ /

Gradient Boosting:
    Datos → Árbol₁ → Pred₁ → Error₁
    Error₁ → Árbol₂ → Pred₂ → Error₂
    Error₂ → Árbol₃ → Pred₃ → Error₃
    Predicción final = Pred₁ + η·Pred₂ + η·Pred₃ + ...
```

---

### Gradient Boosting

#### Optimización de una Función de Pérdida

Gradient Boosting puede verse como un problema de optimización numérica donde buscamos la función $F(x)$ que minimiza una pérdida esperada $L(y, F(x))$.

**Formulación matemática:**

Dado un conjunto de datos $\{(x_i, y_i)\}_{i=1}^n$, queremos encontrar:

$$\hat{F} = \arg\min_{F} \sum_{i=1}^n L(y_i, F(x_i))$$

En lugar de buscar directamente en el espacio de funciones, construimos una expansión aditiva:

$$F_M(x) = \sum_{m=0}^M \beta_m h_m(x; a_m)$$

donde $h_m(x; a_m)$ son funciones base (típicamente árboles pequeños).

**Algoritmo paso a paso:**

1. Inicializar $F_0(x) = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, \gamma)$
   
   Para regresión con MSE: $F_0(x) = \bar{y}$
   Para clasificación binaria con log-loss: $F_0(x) = \log(p/(1-p))$

2. Para $m = 1$ hasta $M$:
   
   a. Calcular **pseudo-residuos** (gradiente negativo):
      $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x)=F_{m-1}(x)}$$
      
      Para MSE: $r_{im} = y_i - F_{m-1}(x_i)$
      Para log-loss: $r_{im} = y_i - p_{m-1}(x_i)$
   
   b. Ajustar un árbol de regresión $h_m(x)$ a los pseudo-residuos $\{(x_i, r_{im})\}_{i=1}^n$
   
   c. Para cada hoja $j$ del árbol, calcular el valor óptimo $\gamma_{jm}$:
      $$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$
   
   d. Actualizar el modelo:
      $$F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$
      donde $\nu$ es el learning rate (shrinkage).

#### Uso de Árboles Débiles (Shallow Trees)

A diferencia de Random Forest que usa árboles profundos, Gradient Boosting típicamente usa árboles pequeños:

- **Profundidad típica:** 3-8 niveles
- **Número de hojas:** 8-256
- **Justificación:** Árboles débiles tienen bajo sesgo pero alta varianza controlada; el boosting secuencial reduce el sesgo gradualmente.

#### Learning Rate (Shrinkage) y Número de Estimadores

**Learning Rate ($\nu$):** Escala la contribución de cada árbol (típicamente 0.01-0.3).

- $\nu$ pequeño → más árboles necesarios, mejor generalización
- $\nu$ grande → menos árboles, riesgo de overfitting

**Relación fundamental:**
$$\text{Performance} \approx f\left(\frac{\text{n\_estimators} \times \text{learning\_rate}}{\text{complejidad árbol}}\right)$$

**Trade-off:**
- **Early stopping:** Si learning rate es pequeño, se necesitan más árboles pero el modelo es más robusto.
- **Regla práctica:** Reducir learning rate a la mitad y duplicar n_estimators suele mejorar performance.

#### Riesgo de Overfitting

Gradient Boosting puede sobreajustar si:
- Demasiados árboles (`n_estimators` muy alto)
- Árboles muy complejos (`max_depth` alto)
- Learning rate muy alto
- Pocos datos

**Señales de overfitting:**
- Train loss sigue disminuyendo mientras validation loss aumenta
- Diferencia creciente entre métricas train y test

**Mecanismos de control:**
- Early stopping
- Regularización
- Submuestreo
- Árboles débiles

---

### XGBoost (eXtreme Gradient Boosting)

XGBoost es una implementación optimizada de Gradient Boosting que se ha convertido en el estándar de facto en competiciones de machine learning.

#### Requisitos del Modelo

- **Datos:** Numéricos o categóricos (requiere encoding previo)
- **Formato:** Matriz densa o sparse (XGBoost maneja eficientemente datos sparse)
- **Escalado:** No necesario (árboles son invariantes a escala)
- **Valores nulos:** Manejo interno (aprende dirección por defecto)

#### Regularización (L1, L2)

XGBoost incorpora términos de regularización en la función objetivo para controlar la complejidad:

$$\mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)$$

donde:
$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2 + \alpha \|w\|_1$$

- $T$: número de hojas
- $w$: pesos de las hojas
- $\lambda$: regularización L2 (Ridge) sobre pesos
- $\alpha$: regularización L1 (Lasso) sobre pesos
- $\gamma$: penalización por complejidad (mínima reducción de pérdida para dividir)

**Efecto de la regularización:**
- **$\lambda$ alto:** pesos más pequeños, árboles más conservadores
- **$\alpha$ alto:** pesos dispersos (algunas hojas con peso cero)
- **$\gamma$ alto:** árboles menos profundos (menos divisiones)

#### Manejo de Valores Nulos

XGBoost aprende automáticamente la dirección óptima para valores faltantes:

1. Durante entrenamiento, para cada división, prueba ambas direcciones para missing values
2. Selecciona la dirección que maximiza la ganancia
3. Almacena la dirección por defecto para cada nodo

**Ventaja:** No requiere imputación previa; el modelo decide cómo manejar missing values según los datos.

#### Early Stopping

Detiene el entrenamiento cuando la métrica de validación deja de mejorar, evitando overfitting y ahorrando tiempo.

```python
# Configuración de early stopping
model = xgb.XGBClassifier(
    n_estimators=1000,  # Alto, pero se detendrá antes
    early_stopping_rounds=50,  # Detener si no mejora en 50 rondas
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

#### Paralelización Eficiente

XGBoost aprovecha múltiples núcleos:
- **Construcción de histogramas:** paralelizada por característica
- **Búsqueda de divisiones:** paralelizada por nodo
- **Cross-validation:** paralelizada por fold

#### Plantilla Base en Python

```python
# ============================================
# XGBOOST - PLANTILLA BASE
# ============================================

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear conjunto de validación para early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Modelo base
model = xgb.XGBClassifier(
    # Parámetros fundamentales
    n_estimators=1000,           # Número máximo de árboles
    learning_rate=0.05,           # Tamaño del paso (shrinkage)
    max_depth=6,                  # Profundidad de árboles
    subsample=0.8,                # Submuestreo de filas
    colsample_bytree=0.8,         # Submuestreo de columnas
    
    # Regularización
    reg_lambda=1.0,               # Regularización L2
    reg_alpha=0.0,                 # Regularización L1
    
    # Objetivo y métricas
    objective='binary:logistic',   # Cambiar según problema
    eval_metric='logloss',         # Métrica de validación
    
    # Optimización
    n_jobs=-1,                     # Usar todos los núcleos
    random_state=42
)

# Entrenamiento con early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100
)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluación
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Mejor iteración: {model.best_iteration}")

# Búsqueda de hiperparámetros
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_lambda': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(n_estimators=500, objective='binary:logistic', n_jobs=-1),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Mejores parámetros:", grid_search.best_params_)
```

---

### LightGBM

LightGBM es una implementación de Gradient Boosting diseñada para ser **más rápida y eficiente** que XGBoost, especialmente en datasets grandes.

#### Requisitos del Modelo

- **Datos:** Numéricos, con soporte nativo para categóricas
- **Formato:** Dataset específico de LightGBM (`lgb.Dataset`)
- **Escalado:** No necesario
- **Valores nulos:** Manejo interno

#### Gradient-based One-Side Sampling (GOSS)

GOSS es una técnica de muestreo que retiene observaciones con grandes gradientes (mal predichas) y muestrea aleatoriamente las de gradiente pequeño.

**Motivación:** Las observaciones con gradiente grande contribuyen más a la ganancia de información.

**Algoritmo GOSS:**
1. Ordenar observaciones por valor absoluto del gradiente
2. Seleccionar top $a \times 100\%$ con gradiente grande
3. Muestrear aleatoriamente $b \times 100\%$ de las restantes
4. Amplificar los gradientes de las muestreadas por un factor $(1-a)/b$

**Efecto:** Reduce el número de observaciones consideradas sin perder precisión significativa.

#### Histogram-based Splitting

A diferencia de XGBoost que considera todos los valores posibles para puntos de corte, LightGBM usa histogramas:

1. **Discretización:** Agrupa valores continuos en bins (ej. 256 bins)
2. **Construcción de histograma:** Acumula gradientes y cuenta por bin
3. **Búsqueda:** Encuentra mejor división basada en histogramas

**Ventajas:**
- Reducción de complejidad de $O(\text{datos})$ a $O(\text{bins})$
- Menor uso de memoria
- Más rápido, especialmente en datasets grandes

**Comparación de enfoques:**

| Aspecto | XGBoost (exacto) | LightGBM (histograma) |
|---------|------------------|----------------------|
| Búsqueda | Todos los valores únicos | Bins discretos |
| Complejidad | $O(n \log n)$ por feature | $O(\text{bins})$ por feature |
| Precisión | Máxima | Ligeramente menor (discretización) |
| Velocidad | Más lento | 5-10x más rápido |

#### Manejo Eficiente de Variables Categóricas

LightGBM maneja variables categóricas directamente, sin necesidad de one-hot encoding:

1. Calcula estadísticas por categoría (suma de gradientes, conteo)
2. Ordena categorías por estas estadísticas
3. Trata como variable ordinal para encontrar divisiones

**Ventajas:**
- Evita explosión de dimensionalidad
- Captura relaciones entre categorías
- Más rápido que one-hot encoding

#### Entrenamiento Más Rápido en Grandes Volúmenes

LightGBM está optimizado para escalar a millones de observaciones:

- **Leaf-wise growth:** A diferencia de level-wise de XGBoost, LightGBM crece el árbol expandiendo la hoja con mayor pérdida, resultando en árboles más profundos pero mejor ajuste con mismo número de hojas.

```
Level-wise (XGBoost):        Leaf-wise (LightGBM):
      Nivel 1                        Nivel 1
     /      \                        |
    N2       N3                      N2 (mejor ganancia)
   /  \     /  \                      |
  N4  N5   N6  N7                     N3
                                      |
                                      N4
```

**Trade-off:** Leaf-wise puede causar overfitting en datasets pequeños; controlar con `num_leaves` y `min_data_in_leaf`.

#### Plantilla Base en Python

```python
# ============================================
# LIGHTGBM - PLANTILLA BASE
# ============================================

import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear conjunto de validación
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Crear datasets de LightGBM (opcional pero recomendado)
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Parámetros del modelo
params = {
    # Fundamentales
    'boosting_type': 'gbdt',
    'objective': 'binary',           # binary, multiclass, regression
    'metric': 'binary_logloss',       # métrica de evaluación
    
    # Árboles
    'num_leaves': 31,                  # número máximo de hojas
    'max_depth': -1,                    # -1 = sin límite
    
    # Boosting
    'learning_rate': 0.05,              # shrinkage
    'n_estimators': 1000,                # número de árboles
    
    # Submuestreo
    'feature_fraction': 0.8,             # colsample_bytree
    'bagging_fraction': 0.8,             # subsample
    'bagging_freq': 1,                    # frecuencia de bagging
    
    # Regularización
    'lambda_l1': 0.0,                     # reg_alpha
    'lambda_l2': 1.0,                     # reg_lambda
    'min_gain_to_split': 0.0,              # gamma
    'min_child_samples': 20,                # min_samples_leaf
    
    # Optimización
    'num_threads': -1,                      # n_jobs
    'seed': 42,
    'verbosity': -1                          # -1 = sin logs
}

# Entrenamiento con early stopping
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Predicciones
y_pred = (model.predict(X_test) > 0.5).astype(int)
y_proba = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# Usando interfaz scikit-learn (más familiar)
model_sk = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

model_sk.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)

# Búsqueda de hiperparámetros
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_lambda': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    lgb.LGBMClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Mejores parámetros:", grid_search.best_params_)
```

---

### Comparación de Algoritmos

#### Random Forest vs Gradient Boosting

| Aspecto | Random Forest | Gradient Boosting |
|---------|--------------|-------------------|
| **Construcción** | Paralela | Secuencial |
| **Objetivo** | Reducir varianza | Reducir sesgo + varianza |
| **Modelos base** | Profundos (bajo sesgo) | Débiles (shallow trees) |
| **Overfitting** | Robusto por diseño | Requiere control (lr, regularization) |
| **Interpretabilidad** | Importancia de variables | Importancia de variables |
| **Tiempo entrenamiento** | Rápido (paralelo) | Lento (secuencial) |
| **Performance típica** | Buena baseline | Superior en la mayoría de casos |

**Cuándo usar cada uno:**

- **Random Forest:** Baseline rápido, datasets pequeños, cuando la robustez es prioritaria
- **Gradient Boosting:** Cuando se necesita máxima performance, hay tiempo para tuning, datasets grandes

#### XGBoost vs LightGBM

| Aspecto | XGBoost | LightGBM |
|---------|---------|----------|
| **Velocidad entrenamiento** | Rápida | Muy rápida (2-10x más rápido) |
| **Uso de memoria** | Moderado | Bajo |
| **Manejo de categóricas** | Requiere encoding | Nativo (eficiente) |
| **Manejo de missing values** | Automático | Automático |
| **Regularización** | L1, L2, γ | L1, L2, min_gain |
| **Crecimiento árbol** | Level-wise | Leaf-wise |
| **Dataset pequeño** | Excelente | Puede overfittear |
| **Dataset grande** | Bueno | Excelente |
| **GPU soporte** | Sí | Sí |

**Recomendaciones prácticas:**

| Escenario | Algoritmo recomendado |
|-----------|----------------------|
| Dataset pequeño (<10k) | XGBoost (menos propenso a overfitting) |
| Dataset mediano (10k-100k) | Ambos, probar ambos |
| Dataset grande (>100k) | LightGBM (mucho más rápido) |
| Muchas variables categóricas | LightGBM (manejo nativo) |
| Tiempo de entrenamiento crítico | LightGBM |
| Interpretabilidad requerida | Ambos (importancia de variables) |
| Competencias Kaggle | XGBoost + LightGBM ensemble |

#### Interpretabilidad vs Performance

```
Performance
    ↑
10  |                         LightGBM
    |                      XGBoost
8   |                 Gradient Boosting
    |            Random Forest
6   |       Árbol simple
    |  Regresión Logística
4   |
    +--------------------------------→ Interpretabilidad
    4   5   6   7   8   9   10
```

**Trade-off:**
- **Alta interpretabilidad:** Árboles simples, regresión logística
- **Balance:** Random Forest (importancia de variables confiable)
- **Máxima performance:** XGBoost/LightGBM (caja negra, pero con importancia de variables)

---

## Métricas

### Clasificación

| Métrica | Fórmula | Interpretación en Boosting |
|---------|---------|---------------------------|
| **Log-loss** | $-\frac{1}{n}\sum [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$ | Mide calidad de probabilidades; boosting la minimiza directamente |
| **AUC-ROC** | Área bajo curva ROC | Capacidad discriminativa; robusta a desbalanceo |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P+R}$ | Balance precisión-recall; útil en clases desbalanceadas |
| **Precision@k** | Precisión en top k predicciones | Relevante en ranking y recomendación |

### Regresión

| Métrica | Fórmula | Interpretación |
|---------|---------|---------------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Error en unidades originales, sensible a outliers |
| **MAE** | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Robusta a outliers |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proporción de varianza explicada |

### Evaluación en Validación Cruzada

En Gradient Boosting, la validación cruzada requiere atención especial:

```python
from sklearn.model_selection import cross_val_score

# Validación cruzada estándar
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"AUC promedio: {scores.mean():.4f} ± {scores.std():.4f}")

# Para early stopping, usar conjuntos de validación separados
# o implementar validación cruzada con early stopping personalizado
```

### Selección de Métricas según Objetivo de Negocio

| Objetivo | Métrica Principal | Razón |
|----------|-------------------|-------|
| Predicción de CTR | Log-loss o AUC | Log-loss para calibración, AUC para ranking |
| Detección de fraude | PR-AUC o Recall | Clase minoritaria crítica |
| Scoring crediticio | AUC-ROC o KS | Capacidad discriminativa |
| Predicción de demanda | RMSE | Penaliza grandes errores |
| Churn prediction | F1 o Lift@k | Balance y relevancia en top clientes |

---

## Hiperparámetros Clave

### Número de Estimadores (`n_estimators`)

**Descripción:** Número de árboles en el ensamble.

**Efecto:**
- Muy bajo → underfitting (no suficiente capacidad)
- Muy alto → overfitting (si no hay early stopping)

**Tuning:** Usar early stopping con conjunto de validación para encontrar el óptimo automáticamente.

### Learning Rate (`learning_rate`)

**Descripción:** Factor de contracción (shrinkage) que escala la contribución de cada árbol.

**Rango típico:** 0.01 - 0.3

**Relación con n_estimators:**
$$\text{impacto} \approx \text{n\_estimators} \times \text{learning\_rate}$$

**Guía:**
- Valores pequeños (0.01-0.05): más árboles, mejor generalización
- Valores grandes (0.1-0.3): menos árboles, riesgo de overfitting

### Profundidad y Complejidad

| Parámetro | XGBoost | LightGBM | Efecto |
|-----------|---------|----------|--------|
| **max_depth** | `max_depth` | `max_depth` o `num_leaves` | Controla complejidad de árboles individuales |
| **num_leaves** | - | `num_leaves` | Número máximo de hojas (alternativa a profundidad) |
| **min_child_weight** | `min_child_weight` | `min_sum_hessian_in_leaf` | Mínimo peso en hoja para seguir dividiendo |

**Relación max_depth vs num_leaves:**
- Árbol depth=d puede tener hasta 2^d hojas
- `num_leaves` debe ser ≤ 2^(max_depth)

### Submuestreo para Control de Overfitting

| Parámetro | XGBoost | LightGBM | Descripción |
|-----------|---------|----------|-------------|
| **subsample** | `subsample` | `bagging_fraction` | Fracción de filas usadas por árbol |
| **colsample_bytree** | `colsample_bytree` | `feature_fraction` | Fracción de columnas por árbol |
| **colsample_bylevel** | `colsample_bylevel` | - | Fracción de columnas por nivel |
| **colsample_bynode** | `colsample_bynode` | - | Fracción de columnas por nodo |

**Valores típicos:** 0.6 - 0.9

### Regularización

| Parámetro | XGBoost | LightGBM | Efecto |
|-----------|---------|----------|--------|
| **reg_lambda** (L2) | `reg_lambda` | `lambda_l2` | Penaliza pesos grandes (más común) |
| **reg_alpha** (L1) | `reg_alpha` | `lambda_l1` | Genera pesos dispersos |
| **gamma** | `gamma` | `min_gain_to_split` | Pérdida mínima para dividir |

### Early Stopping

**Configuración:**
```python
# XGBoost
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)

# LightGBM
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
```

### Guía de Tuning por Orden de Importancia

1. **Primero:** `learning_rate` + `n_estimators` (con early stopping)
2. **Segundo:** Complejidad del árbol (`max_depth`, `num_leaves`)
3. **Tercero:** Submuestreo (`subsample`, `colsample_bytree`)
4. **Cuarto:** Regularización (`reg_lambda`, `reg_alpha`)
5. **Quinto:** Mínimos en hojas (`min_child_weight`, `min_child_samples`)

---

## Comunicación de Resultados

### Explicación de Mejoras de Performance

**Ejemplo de reporte ejecutivo:**

> **Resumen: Modelo de Predicción de Churn**
>
> Hemos implementado un modelo XGBoost que supera significativamente al modelo actual (regresión logística):
>
> | Métrica | Modelo Actual | XGBoost | Mejora |
> |---------|--------------|---------|--------|
> | AUC-ROC | 0.76 | 0.89 | +17% |
> | Recall (top 20%) | 0.45 | 0.68 | +51% |
> | Precisión | 0.62 | 0.74 | +19% |
>
> **Impacto de negocio:**
> - Identificaremos 51% más clientes en riesgo de abandono
> - Campañas de retención más efectivas: 74% de los contactados realmente estaban en riesgo
> - Reducción estimada de churn: 3.2 puntos porcentuales
> - Ingresos retenidos: $2.8M anuales adicionales

### Interpretación de Importancia de Variables

**Visualización para stakeholders:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Obtener importancia
importance_df = pd.DataFrame({
    'variable': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

# Gráfico
plt.figure(figsize=(10, 6))
plt.barh(importance_df['variable'], importance_df['importance'])
plt.xlabel('Importancia')
plt.title('Top 10 Variables más Importantes - XGBoost')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Interpretación para negocio:**

> "Las tres variables que más impactan en la predicción de fraude son:
> 1. **Monto de transacción:** Responsable del 32% del poder predictivo. Transacciones inusualmente altas son señales de alerta.
> 2. **País de origen:** 18% de importancia. Ciertos países tienen mayor incidencia de fraude.
> 3. **Hora de la transacción:** 15% de importancia. Horarios atípicos (3-5 AM) tienen mayor riesgo.
>
> Esto sugiere enfocar controles en transacciones de alto monto en horarios nocturnos desde países de riesgo, sin necesidad de revisar todas las transacciones."

### Justificación del Uso de Modelos Complejos

**Template para justificación:**

> **¿Por qué usar Gradient Boosting en lugar de un modelo más simple?**
>
> 1. **Complejidad inherente de los datos:** Nuestro problema presenta relaciones no lineales e interacciones que modelos simples (regresión logística) no pueden capturar.
>
> 2. **Requisito de precisión:** Una mejora del X% en AUC se traduce en Y beneficio económico. El costo computacional adicional se amortiza en meses.
>
> 3. **Volumen de datos:** Con n observaciones, podemos entrenar modelos complejos sin sobreajuste gracias a regularización y validación.
>
> 4. **Infraestructura:** Contamos con capacidad computacional suficiente (X cores, Y memoria) para entrenar y servir estos modelos en producción.
>
> 5. **Interpretabilidad compensada:** Aunque el modelo es complejo, podemos extraer importancia de variables y SHAP values para explicar predicciones individuales.

### Balance entre Precisión y Explicabilidad

**Matriz de decisión según contexto:**

| Contexto | Enfoque Recomendado |
|----------|---------------------|
| **Alta regulación (bancos, salud)** | Usar modelos interpretables (árboles) o complementar con SHAP |
| **Optimización de campañas** | Priorizar precisión; LightGBM/XGBoost con análisis de importancia |
| **Detección de fraude** | Máxima precisión; explicabilidad secundaria |
| **Rechazo de crédito** | Requiere explicación individual (usar SHAP o árboles) |
| **Predicción de demanda** | Precisión como objetivo principal |

**Ejemplo de comunicación:**

> "Hemos optado por XGBoost porque maximiza la precisión, crítica para nuestro sistema de detección de fraude. Para mantener la explicabilidad requerida por auditoría, complementamos con:
> - **Importancia global:** Identificamos las variables clave que guían el modelo.
> - **SHAP values:** Para cada transacción rechazada, explicamos qué variables contribuyeron a la decisión.
> - **Reglas simplificadas:** Extraemos patrones generales del modelo (ej. 'transacciones > $10,000 de país X tienen 85% probabilidad de fraude')."

---

## Reto: 1 punto

**Objetivo:** Comparar Random Forest vs XGBoost vs LightGBM en un dataset de Kaggle, optimizando hiperparámetros y evaluando diferencias en performance y tiempo de entrenamiento.




---


## Laboratorio


**Pipeline del laboratorio:**

```python
# Estructura del laboratorio (pseudocódigo conceptual)

# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ------------------------------------
# - Análisis exploratorio (EDA)
# - Ingeniería de características
# - División en train/validation/test
# - Manejo de valores nulos y categóricas

# 2. XGBOOST - MODELO BASE
# ------------------------
# - Entrenamiento con parámetros por defecto
# - Evaluación en validación
# - Early stopping para encontrar número óptimo de árboles

# 3. XGBOOST - AJUSTE DE HIPERPARÁMETROS
# --------------------------------------
# - GridSearchCV o RandomizedSearchCV
# - Optimización de learning_rate, max_depth, subsample, etc.
# - Evaluación del modelo optimizado

# 4. LIGHTGBM - MODELO BASE
# -------------------------
# - Entrenamiento con parámetros por defecto
# - Evaluación en validación
# - Early stopping

# 5. LIGHTGBM - AJUSTE DE HIPERPARÁMETROS
# ---------------------------------------
# - GridSearchCV o RandomizedSearchCV
# - Optimización de num_leaves, learning_rate, feature_fraction, etc.

# 6. COMPARACIÓN DE MODELOS
# -------------------------
# - Métricas en test (AUC-ROC, F1, log-loss)
# - Tiempos de entrenamiento
# - Importancia de variables
# - Curvas ROC y PR

# 7. CONCLUSIONES
# ---------------
# - Selección del mejor modelo
# - Justificación para implementación en producción
```

**Detalles específicos que los estudiantes implementarán:**

- **Preprocesamiento:** Escalamiento (opcional), encoding de categóricas, manejo de missing values
- **Feature engineering:** Creación de interacciones, transformaciones, variables agregadas
- **Validación:** Uso de conjuntos de validación separados para early stopping
- **Tuning:** Búsqueda sistemática de hiperparámetros con validación cruzada
- **Evaluación:** Comparación justa usando las mismas métricas y conjunto de test
- **Interpretación:** Análisis de importancia de variables y SHAP values

---

## Anexo: Fundamento Matemático y Computacional

### A1. Gradient Boosting como Optimización de Funciones

#### A1.1. Formulación General

Gradient Boosting puede entenderse como un algoritmo de optimización numérica en el espacio de funciones. Buscamos una función $F(x)$ que minimice el riesgo esperado:

$$F^*(x) = \arg\min_{F(x)} \mathbb{E}_{y,x}[L(y, F(x))]$$

Restringimos la búsqueda a expansiones de la forma:

$$F_M(x) = \sum_{m=0}^M \beta_m h(x; a_m)$$

donde $h(x; a_m)$ son funciones base (típicamente árboles pequeños) y $\beta_m$ son coeficientes.

#### A1.2. Algoritmo de Gradiente Descendente Funcional

Consideramos $F(x)$ como un vector de valores $\{F(x_1), \ldots, F(x_n)\}$. Queremos minimizar:

$$J(F) = \sum_{i=1}^n L(y_i, F(x_i))$$

El gradiente de $J$ con respecto a $F(x_i)$ es:

$$g_i = \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

El descenso de gradiente en el espacio de funciones actualizaría:

$$F(x_i) \leftarrow F(x_i) - \rho g_i$$

Pero queremos una función que pueda generalizar a nuevos puntos, no solo actualizar los valores observados.

**Idea clave:** Ajustar una función base $h(x; a)$ que se aproxime al gradiente negativo:

$$h_m = \arg\min_{h} \sum_{i=1}^n [(-g_i) - h(x_i)]^2$$

Luego actualizamos:

$$F_m(x) = F_{m-1}(x) + \rho_m h_m(x)$$

#### A1.3. Funciones de Pérdida y sus Gradientes

| Problema | Función de Pérdida $L(y, F)$ | Gradiente Negativo |
|----------|------------------------------|-------------------|
| Regresión (MSE) | $\frac{1}{2}(y - F)^2$ | $y - F$ |
| Regresión (MAE) | $\|y - F\|$ | $\text{sign}(y - F)$ |
| Clasificación Binaria | $\log(1 + e^{-2yF})$ con $y \in \{-1,1\}$ | $\frac{2y}{1 + e^{2yF}}$ |
| Clasificación Binaria (log-loss) | $-y\log(p) - (1-y)\log(1-p)$ con $p = \frac{1}{1+e^{-F}}$ | $y - p$ |

#### A1.4. Algoritmo General de Gradient Boosting

```
Algoritmo: Gradient Boosting
--------------------------------------------------------------------------------
1. Inicializar F₀(x) = arg min_γ Σ L(y_i, γ)
2. Para m = 1 hasta M:
   a. Calcular pseudo-residuos: r_im = -[∂L(y_i, F(x_i))/∂F(x_i)] para F=F_{m-1}
   b. Ajustar un árbol de regresión a los pares (x_i, r_im) produciendo regiones R_jm
   c. Para cada hoja j, calcular γ_jm = arg min_γ Σ_{x_i∈R_jm} L(y_i, F_{m-1}(x_i) + γ)
   d. Actualizar F_m(x) = F_{m-1}(x) + ν Σ γ_jm 1(x ∈ R_jm)
3. Retornar F_M(x)
--------------------------------------------------------------------------------
```

donde $\nu$ es el learning rate (shrinkage) que controla la velocidad de aprendizaje.

---

### A2. Expansión Funcional (Additive Models)

#### A2.1. Modelos Aditivos Generalizados

Los modelos de boosting pertenecen a la familia de **modelos aditivos**:

$$F(x) = \sum_{m=1}^M f_m(x)$$

Cada $f_m(x)$ es una "base" que contribuye aditivamente a la predicción final. En Gradient Boosting, cada $f_m$ es un árbol que se ajusta a los residuos del modelo actual.

#### A2.2. Forward Stagewise Additive Modeling

El algoritmo **forward stagewise** construye el modelo secuencialmente:

1. Comenzar con $F_0(x) = 0$
2. Para $m = 1$ hasta $M$:
   $$(\beta_m, a_m) = \arg\min_{\beta, a} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \beta h(x_i; a))$$
   $$F_m(x) = F_{m-1}(x) + \beta_m h(x; a_m)$$

Gradient Boosting es una implementación específica donde $h(x; a)$ son árboles y la optimización se realiza mediante aproximación de gradiente.

#### A2.3. Representación como Expansión en Funciones Base

Cada árbol $T_m(x)$ puede expresarse como:

$$T_m(x) = \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$

donde $R_{jm}$ son las regiones (hojas) del árbol y $\gamma_{jm}$ es el valor predicho en cada hoja.

El modelo final es:

$$F_M(x) = \sum_{m=1}^M \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$

Esta representación muestra que el modelo es una suma ponderada de indicadores de regiones.

#### A2.4. Interpretación Geométrica

Podemos ver el proceso como una búsqueda en el espacio de funciones:

- El espacio de funciones es un espacio de Hilbert con producto interno $\langle f, g \rangle = \sum_i f(x_i)g(x_i)$
- El gradiente negativo apunta en la dirección de máximo descenso
- Cada árbol es una proyección de ese gradiente en el espacio de funciones representables por árboles
- El learning rate controla el tamaño del paso

---

### A3. Regularización en Boosting

La regularización es fundamental para controlar el overfitting en modelos de boosting, que pueden seguir mejorando en train indefinidamente.

#### A3.1. Shrinkage (Learning Rate)

El parámetro de contracción $\nu$ (learning rate) escala la contribución de cada árbol:

$$F_m(x) = F_{m-1}(x) + \nu \cdot T_m(x)$$

**Efecto matemático:** Reduce la varianza de cada paso, permitiendo más iteraciones y mejor generalización.

**Relación con número de árboles:** Para un número fijo de iteraciones, la capacidad efectiva del modelo es aproximadamente proporcional a $\nu \times M$.

#### A3.2. Submuestreo (Stochastic Gradient Boosting)

Introducir aleatoriedad muestreando una fracción de los datos en cada iteración:

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right] \cdot I(i \in S_m)$$

donde $S_m$ es una muestra aleatoria de tamaño $\eta \cdot n$.

**Efecto:**
- Reduce correlación entre árboles
- Disminuye varianza
- Aumenta sesgo ligeramente
- Mejora generalización

#### A3.3. Regularización en XGBoost

XGBoost introduce una función objetivo regularizada:

$$\mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)$$

donde para cada árbol $f_k$:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2 + \alpha \|w\|_1$$

**Derivación completa:**

Para un árbol con $T$ hojas y vector de pesos $w \in \mathbb{R}^T$, la función objetivo en iteración $t$ es:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

Usando expansión de Taylor de segundo orden:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$

donde $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ y $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$.

Eliminando constantes y agrupando por hojas:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T$$

Para $I_j$ conjunto de índices en hoja $j$.

El peso óptimo para la hoja $j$ es:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Y la ganancia al dividir un nodo es:

$$\text{Ganancia} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

Esta formulación muestra cómo:
- $\lambda$ suaviza los pesos (regularización L2)
- $\gamma$ penaliza añadir nuevas hojas (control de complejidad)

#### A3.4. Early Stopping como Regularización

Early stopping detiene el entrenamiento cuando la métrica de validación deja de mejorar:

$$m^* = \arg\min_{m \leq M} \text{Error}_{val}(F_m)$$

**Interpretación:** Es una forma de regularización que controla la complejidad efectiva del modelo limitando el número de iteraciones.

---

### A4. Complejidad Computacional y Escalabilidad

#### A4.1. Complejidad de Gradient Boosting Clásico

Para $M$ árboles, $n$ observaciones, $p$ características y profundidad máxima $d$:

**Entrenamiento (sin optimizaciones):**
- Cada árbol requiere $O(p \cdot n \log n)$ para ordenar y encontrar divisiones
- Total: $O(M \cdot p \cdot n \log n)$

**Predicción:**
- Por observación: $O(M \cdot d)$
- Total: $O(n_{test} \cdot M \cdot d)$

#### A4.2. Optimizaciones de XGBoost

XGBoost implementa varias optimizaciones:

**1. Bloques de datos comprimidos (compressed columns):**
- Pre-ordena cada característica y almacena en bloques
- Reduce complejidad de búsqueda de divisiones
- Complejidad: $O(p \cdot n \log n)$ una vez, luego $O(p \cdot n)$ por árbol

**2. Algoritmo aproximado con percentiles:**
- En lugar de considerar todos los valores, considera percentiles
- Complejidad: $O(p \cdot \text{bins} \cdot n)$

**3. Parallel learning:**
- Construcción de histogramas paralela por característica
- Escalamiento lineal con número de núcleos

**Complejidad efectiva XGBoost:**
$$O(M \cdot p \cdot n \cdot \text{factor\_paralelización})$$

#### A4.3. Optimizaciones de LightGBM

LightGBM introduce innovaciones que reducen drásticamente la complejidad:

**GOSS (Gradient-based One-Side Sampling):**
- Selecciona top $a \times 100\%$ observaciones con mayor gradiente
- Muestrea $b \times 100\%$ de las restantes
- Reduce $n$ efectivo a $n' \approx n \cdot (a + b(1-a))$

**Complejidad con GOSS:**
$$O(M \cdot p \cdot n' \log n')$$

**Histogram-based algorithm:**
- Discretiza características continuas en bins (típicamente 256)
- Complejidad por división: $O(\text{bins})$ en lugar de $O(n)$

**Leaf-wise growth:**
- En lugar de crecer nivel por nivel, crece expandiendo la hoja con mayor ganancia
- Más eficiente pero requiere control de profundidad

**Complejidad efectiva LightGBM:**
$$O(M \cdot p \cdot \text{bins} \cdot n' \cdot \log(\text{num\_leaves}))$$

#### A4.4. Comparación de Complejidad Asintótica

| Algoritmo | Entrenamiento | Predicción | Memoria |
|-----------|--------------|------------|---------|
| Gradient Boosting clásico | $O(M p n \log n)$ | $O(M d)$ | $O(M n)$ |
| XGBoost (exacto) | $O(M p n \log n)$ | $O(M d)$ | $O(p n)$ + $O(M n_{nodos})$ |
| XGBoost (aproximado) | $O(M p \cdot \text{bins} \cdot n)$ | $O(M d)$ | $O(p \cdot \text{bins})$ + $O(M n_{nodos})$ |
| LightGBM (GOSS + histograma) | $O(M p \cdot \text{bins} \cdot n' \cdot \log L)$ | $O(M d)$ | $O(p \cdot \text{bins})$ + $O(M n_{nodos})$ |

donde $n' < n$, $\text{bins} \ll n$, $L = \text{num\_leaves}$, $d = \text{max\_depth}$

#### A4.5. Escalabilidad con el Volumen de Datos

**Régimen de datos pequeños ($n < 10^4$):**
- Todos los algoritmos son rápidos
- XGBoost exacto puede ser preferible por precisión
- LightGBM puede ser innecesariamente complejo

**Régimen de datos medianos ($10^4 \leq n < 10^6$):**
- XGBoost aproximado o LightGBM
- LightGBM típicamente 5-10x más rápido
- Paralelización importante

**Régimen de datos grandes ($n \geq 10^6$):**
- LightGBM es la única opción práctica
- GOSS reduce drásticamente el costo
- Histogramas esenciales para memoria

#### A4.6. Escalabilidad con Número de Características

**Características moderadas ($p < 1000$):**
- Ambos algoritmos escalan bien
- XGBoost: $O(p)$ por nodo
- LightGBM: $O(p)$ pero con histogramas más eficientes

**Alta dimensionalidad ($p \geq 10000$):**
- XGBoost puede ser lento (búsqueda en todas las características)
- LightGBM con `feature_fraction` selecciona subconjuntos
- Considerar reducción de dimensionalidad previa

#### A4.7. Estrategias para Escalamiento en Producción

**Entrenamiento:**
- Usar muestreo estratificado para reducir $n$
- Paralelizar con `n_jobs = -1`
- Limitar profundidad de árboles
- Usar early stopping para encontrar $M$ óptimo

```python
# Estrategias de escalamiento
xgb_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',        # Algoritmo aproximado
    'n_jobs': -1                   # Paralelización
}

lgb_params = {
    'n_estimators': 1000,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'num_threads': -1
}
```

**Predicción en tiempo real:**
- Modelos ligeros (menos árboles, menor profundidad)
- Compilar a formatos optimizados (PMML, ONNX)
- Usar caché de predicciones para consultas frecuentes

---

### A5. Resumen Matemático

| Concepto | Expresión Matemática |
|----------|---------------------|
| Modelo aditivo | $F_M(x) = \sum_{m=1}^M f_m(x)$ |
| Gradiente | $g_i = \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$ |
| Actualización boosting | $F_m(x) = F_{m-1}(x) + \nu \cdot T_m(x)$ |
| Objetivo XGBoost | $\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k (\gamma T_k + \frac{1}{2}\lambda\|w_k\|^2 + \alpha\|w_k\|_1)$ |
| Peso óptimo hoja | $w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$ |
| Ganancia de división | $\text{Ganancia} = \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}] - \gamma$ |
| Complejidad XGBoost | $O(M \cdot p \cdot n \cdot \log n)$ |
| Complejidad LightGBM | $O(M \cdot p \cdot \text{bins} \cdot n' \cdot \log L)$ |

---

