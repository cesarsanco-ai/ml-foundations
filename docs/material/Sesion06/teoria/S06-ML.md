# Semana 6: Árboles de Decisión y Random Forest

## Logro de la sesión

Construir, interpretar y comparar modelos basados en árboles de decisión y ensambles tipo Random Forest, comprendiendo su capacidad para capturar relaciones no lineales y su aplicación en problemas reales de clasificación y regresión.

---

## Problemática de negocio

Los modelos basados en árboles responden a necesidades críticas en el mundo empresarial:

- **Modelos interpretables vs modelos de alto performance:** ¿Cómo equilibrar la necesidad de explicar las predicciones (requisitos regulatorios) con la precisión requerida para competir?
- **Manejo de relaciones no lineales:** Los datos del mundo real rara vez siguen relaciones lineales simples; los árboles capturan naturalmente interacciones complejas.
- **Variables categóricas:** A diferencia de regresión lineal, los árboles manejan variables categóricas sin necesidad de one-hot encoding extensivo.
- **Reducción de overfitting:** Especialmente con Random Forest, que mediante ensambles controla la varianza excesiva.
- **Interpretabilidad:** Necesidad de entender qué variables impactan más en la predicción para tomar decisiones de negocio informadas.

**Ejemplos de aplicación por industria:**

| Industria | Caso de uso | Objetivo |
|-----------|-------------|----------|
| **Banca** | Scoring crediticio | Evaluar riesgo de impago con reglas interpretables |
| **Fintech** | Detección de fraude | Identificar transacciones sospechosas en tiempo real |
| **Seguros** | Predicción de siniestralidad | Tarificar pólizas según perfil de riesgo |
| **Marketing** | Segmentación de clientes | Identificar perfiles de alto valor |
| **Salud** | Diagnóstico asistido | Apoyar decisiones clínicas con reglas claras |
| **RRHH** | Predicción de rotación | Identificar empleados en riesgo de abandonar |
| **E-commerce** | Recomendación de productos | Personalizar ofertas según comportamiento |
| **Telecom** | Predicción de churn | Retener clientes con mayor probabilidad de fuga |

**Limitaciones de modelos anteriores que resuelven los árboles:**

| Modelo | Limitación | Cómo lo resuelve el árbol |
|--------|------------|---------------------------|
| Regresión lineal | Asume linealidad | Captura relaciones no lineales naturalmente |
| Regresión logística | Requiere transformaciones manuales | Las divisiones sucesivas modelan interacciones |
| KNN | Lento en predicción, sensible a escala | Rápido en predicción, robusto a escala |
| SVM | Caja negra, tuning complejo | Interpretable, hiperparámetros intuitivos |
| Naive Bayes | Asume independencia | Captura dependencias entre variables |

---

## Modelado

### Árboles de Decisión

#### Estructura del Árbol

Un árbol de decisión es una estructura jerárquica que particiona recursivamente el espacio de características:

```
                    [Nodo Raíz]
                   (Todas las datos)
                   /            \
          [Rama] Sí             No [Rama]
                 /                  \
          [Nodo Interno]         [Nodo Interno]
           (Feature j)            (Feature k)
           /        \              /        \
        ...        ...           ...        ...
        /            \           /            \
  [Hoja: Clase A] [Hoja: Clase B] [Hoja: Clase A] [Hoja: Clase C]
```

**Componentes:**
- **Nodo raíz:** Contiene todos los datos, primera división.
- **Nodos internos:** Puntos de decisión donde se evalúa una característica.
- **Ramas:** Resultados de las evaluaciones (Sí/No o valores discretos).
- **Hojas (nodos terminales):** Predicción final (clase o valor).

#### Criterios de División

**Para clasificación:**

| Criterio | Fórmula | Interpretación |
|----------|---------|----------------|
| **Gini Impurity** | $G = 1 - \sum_{i=1}^{c} p_i^2$ | Probabilidad de clasificación errónea si se etiqueta aleatoriamente según distribución. Valor 0 = nodo puro. |
| **Entropía** | $H = -\sum_{i=1}^{c} p_i \log_2(p_i)$ | Medida de desorden/incertidumbre. Valor 0 = nodo puro. |
| **Information Gain** | $IG = H(padre) - \sum_{j} \frac{n_j}{n} H(hijo_j)$ | Reducción de entropía lograda por la división. Se maximiza. |

**Comparación Gini vs Entropía:**

| Aspecto | Gini | Entropía |
|---------|------|----------|
| **Cálculo** | Más rápido (sin logaritmos) | Más lento (requiere log) |
| **Sensibilidad** | Similar en práctica | Similar en práctica |
| **Diferencia** | Máximo en 0.5 (clases balanceadas) | Máximo en 1.0 (clases balanceadas) |
| **Recomendación** | Por defecto en scikit-learn | Cuando se desea máxima pureza |

**Para regresión:**

| Criterio | Fórmula | Interpretación |
|----------|---------|----------------|
| **MSE (Mean Squared Error)** | $MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \bar{y})^2$ | Varianza dentro del nodo. Se minimiza. |
| **MAE (Mean Absolute Error)** | $MAE = \frac{1}{n}\sum_{i=1}^{n} \|y_i - \bar{y}\|$ | Desviación absoluta media. Más robusto a outliers. |

#### Sobreajuste (Overfitting) y Control de Complejidad

Los árboles de decisión son propensos al sobreajuste porque pueden crecer hasta que cada hoja contenga una sola observación. Hiperparámetros para controlar la complejidad:

| Hiperparámetro | Descripción | Efecto | Valor típico |
|----------------|-------------|--------|--------------|
| `max_depth` | Profundidad máxima del árbol | Limita el número de divisiones | 3-10 |
| `min_samples_split` | Mínimo de muestras para dividir un nodo | Evita divisiones con pocos datos | 20-100 |
| `min_samples_leaf` | Mínimo de muestras en una hoja | Suaviza predicciones | 10-50 |
| `max_features` | Número máximo de features a considerar | Reduce correlación entre árboles | `sqrt(n_features)` |
| `ccp_alpha` | Complejidad de poda (cost-complexity pruning) | Poda el árbol después de crecer | Validación cruzada |

**Trade-off Bias-Varianza en árboles:**

| Configuración | Sesgo | Varianza | Riesgo |
|---------------|-------|----------|--------|
| Árbol profundo (`max_depth` alto) | Bajo | Alta | Overfitting |
| Árbol poco profundo (`max_depth` bajo) | Alto | Baja | Underfitting |
| `min_samples_split` bajo | Bajo | Alta | Overfitting |
| `min_samples_split` alto | Alto | Baja | Underfitting |

#### Interpretación de Decisiones

Los árboles ofrecen múltiples formas de interpretación:

1. **Reglas de decisión:** Cada camino del árbol puede expresarse como una regla IF-THEN.
   
   ```
   IF (ingreso > 50000) AND (edad >= 30) AND (historial_crediticio = 'bueno')
   THEN riesgo = 'bajo'
   ```

2. **Visualización del árbol:** Representación gráfica de las divisiones.

3. **Importancia de variables:** Basada en la reducción de impureza ponderada por las muestras.

#### Plantilla Base en Python (Clasificación)

```python
# Importaciones necesarias
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definición del modelo base
model = DecisionTreeClassifier(random_state=42)

# Búsqueda de hiperparámetros (opcional)
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [10, 20, 50, 100],
    'min_samples_leaf': [5, 10, 20, 50],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Entrenamiento (si no se usa grid search)
# best_model = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
# best_model.fit(X_train, y_train)

# Predicciones
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# Visualización del árbol
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=feature_names, 
          class_names=target_names, filled=True, rounded=True)
plt.show()
```

#### Plantilla Base en Python (Regresión)

```python
# Importaciones necesarias
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Definición del modelo
model = DecisionTreeRegressor(random_state=42)

# Búsqueda de hiperparámetros
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [10, 20, 50, 100],
    'min_samples_leaf': [5, 10, 20, 50],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
}

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Predicciones
y_pred = best_model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
```

---

### Random Forest (Bagging)

#### Concepto de Ensamble

Random Forest pertenece a la familia de métodos de **ensamble**, específicamente **bagging** (Bootstrap Aggregating). La idea fundamental: **combinar múltiples modelos débiles para crear un modelo fuerte**.

**Fundamento matemático:** Si tenemos $B$ árboles con varianza $\sigma^2$ y correlación $\rho$, la varianza del promedio es:

$$Var(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

A medida que $B$ aumenta, el segundo término desaparece, pero el primero (correlación) permanece. Random Forest reduce la correlación mediante la aleatorización de características.

#### Bootstrap Sampling (Muestreo con Reemplazo)

**Proceso:**
1. Del dataset original con $n$ muestras, se crean $B$ muestras bootstrap.
2. Cada muestra bootstrap se obtiene muestreando $n$ veces con reemplazo.
3. Aproximadamente 63.2% de las muestras originales aparecen en cada bootstrap (el resto son duplicados).

**Muestras Out-of-Bag (OOB):** Las muestras no seleccionadas en un bootstrap (~36.8%) constituyen el conjunto OOB, que sirve como validación interna sin necesidad de un conjunto de validación separado.

#### Selección Aleatoria de Variables (Feature Randomness)

En cada división del árbol, solo se considera un subconjunto aleatorio de características:

- **Clasificación:** típicamente `max_features = sqrt(n_features)`
- **Regresión:** típicamente `max_features = n_features/3`

Esto descorrelaciona los árboles, haciendo que el promedio sea más estable.

#### Reducción de Varianza vs Árbol Individual

| Aspecto | Árbol Simple | Random Forest |
|---------|--------------|---------------|
| **Sesgo** | Similar | Ligeramente mayor (por aleatoriedad) |
| **Varianza** | Alta | Mucho menor |
| **Overfitting** | Propenso | Robusto (promedia errores) |
| **Estabilidad** | Inestable (cambia con datos) | Estable |
| **Generalización** | Limitada | Excelente |

#### Out-of-Bag (OOB) Error

El error OOB es una estimación del error de generalización sin necesidad de validación cruzada:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"Test Score: {rf.score(X_test, y_test):.4f}")
```

**Interpretación:** Si OOB score y test score son similares, el modelo generaliza bien. Si OOB es mucho menor, hay sobreajuste.

#### Hiperparámetros Clave en Random Forest

| Hiperparámetro | Descripción | Rango típico | Impacto |
|----------------|-------------|--------------|---------|
| `n_estimators` | Número de árboles | 100-1000 | A mayor, mejor (con ley de rendimientos decrecientes) |
| `max_depth` | Profundidad máxima | 10-50 o None | Controla complejidad individual |
| `min_samples_split` | Mínimo para dividir | 2-20 | Evita divisiones espurias |
| `min_samples_leaf` | Mínimo en hoja | 1-10 | Suaviza predicciones |
| `max_features` | Features por división | `sqrt`, `log2`, fracción | Controla correlación entre árboles |
| `bootstrap` | Muestreo con reemplazo | True/False | True por defecto |
| `oob_score` | Calcular error OOB | True/False | Útil para validación |

#### Plantilla Base en Python (Clasificación)

```python
# Importaciones necesarias
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definición del modelo base
rf = RandomForestClassifier(random_state=42, oob_score=True)

# Búsqueda de hiperparámetros (RandomizedSearchCV es más eficiente)
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=50, cv=5, scoring='f1', 
    n_jobs=-1, random_state=42, verbose=1
)
random_search.fit(X_train, y_train)

# Mejor modelo
best_rf = random_search.best_estimator_

print("Mejores parámetros:", random_search.best_params_)
print("Mejor score CV:", random_search.best_score_)
print("OOB Score:", best_rf.oob_score_)

# Predicciones
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)

# Evaluación
print(classification_report(y_test, y_pred))

# Importancia de variables
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Importancia de Variables - Random Forest")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

#### Plantilla Base en Python (Regresión)

```python
# Importaciones necesarias
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Definición del modelo
rf_reg = RandomForestRegressor(random_state=42, oob_score=True)

# Búsqueda de hiperparámetros
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [1.0, 'sqrt', 'log2', 0.3, 0.5],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    rf_reg, param_dist, n_iter=50, cv=5, 
    scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

# Mejor modelo
best_rf = random_search.best_estimator_

# Predicciones
y_pred = best_rf.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"OOB Score: {best_rf.oob_score_:.4f}")
```

---

### Comparación: Árbol Simple vs Random Forest

| Aspecto | Árbol de Decisión | Random Forest |
|---------|-------------------|---------------|
| **Estructura** | Un solo árbol | Conjunto de árboles |
| **Varianza** | Alta | Baja (promedia) |
| **Sesgo** | Bajo (puede sobreajustar) | Similar o ligeramente mayor |
| **Interpretabilidad** | Muy alta (visualizable) | Baja (promedio de muchos árboles) |
| **Importancia variables** | Inestable | Estable y confiable |
| **Tiempo entrenamiento** | Rápido | Lento (muchos árboles) |
| **Tiempo predicción** | Rápido | Moderado (promediar) |
| **Overfitting** | Propenso | Robusto (con suficientes árboles) |
| **Manejo de ruido** | Sensible | Robusto |
| **Outliers** | Pueden crear ramas espurias | Efecto promediado |

**Trade-off Interpretabilidad vs Performance:**

```
Interpretabilidad
       ↑
  10   | Árbol simple
       |   |
  8    |   |   Reglas extraídas
       |   |   |
  6    |   |   |   Árbol podado
       |   |   |   |
  4    |   |   |   |   Random Forest (pocos árboles)
       |   |   |   |   |
  2    |   |   |   |   |   Random Forest (óptimo)
       |   |   |   |   |   |
  0    +---+---+---+---+---+--→ Performance
       0   2   4   6   8   10
```

**Cuándo usar cada uno:**

- **Árbol simple:** Cuando la interpretabilidad es crítica (regulaciones, auditorías, explicaciones a clientes).
- **Random Forest:** Cuando la prioridad es la precisión y se puede sacrificar interpretabilidad.

---

## Métricas

### Métricas para Clasificación

| Métrica | Fórmula | Interpretación | Uso en árboles |
|---------|---------|----------------|----------------|
| **Precisión** | $TP/(TP+FP)$ | Confiabilidad de predicciones positivas | Evaluar falsos positivos |
| **Recall** | $TP/(TP+FN)$ | Capacidad de detectar positivos | Evaluar falsos negativos |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P+R}$ | Balance general | Comparación de modelos |
| **AUC-ROC** | Área bajo curva ROC | Capacidad discriminativa | Independiente del umbral |
| **PR-AUC** | Área bajo curva PR | Mejor para clases desbalanceadas | Crítico en fraude, churn |

### Métricas para Regresión

| Métrica | Fórmula | Interpretación | Rango |
|---------|---------|----------------|-------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Error cuadrático medio | [0, ∞) |
| **RMSE** | $\sqrt{MSE}$ | Error en unidades originales | [0, ∞) |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Error absoluto medio | [0, ∞) |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proporción de varianza explicada | (-∞, 1] |

### Importancia de Variables

#### Gini Importance (Feature Importance)

**Cálculo:** Para cada variable, se suma la reducción de impureza (Gini o MSE) ponderada por la proporción de muestras en cada nodo donde se usa esa variable.

**Interpretación:** Mide cuánto contribuye cada variable a reducir la impureza en las divisiones.

**Ventajas:**
- Integrada en scikit-learn (`model.feature_importances_`)
- Rápida de calcular
- Escala a [0, 1] (suma = 1)

**Desventajas:**
- Sesgo hacia variables numéricas con muchos valores
- Puede sobreestimar variables correlacionadas

#### Permutation Importance

**Cálculo:** 
1. Entrenar el modelo y medir performance base.
2. Para cada variable, permutar aleatoriamente sus valores y medir la caída en performance.
3. Mayor caída → mayor importancia.

**Ventajas:**
- Modelo-agnóstico (funciona con cualquier modelo)
- Más confiable que Gini importance
- Detecta correlaciones

**Desventajas:**
- Computacionalmente costoso
- Puede ser inestable con pocos datos

**Implementación:**

```python
from sklearn.inspection import permutation_importance

# Calcular permutation importance
result = permutation_importance(
    best_rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Visualizar
importances = result.importances_mean
std = result.importances_std

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances, yerr=std)
plt.xticks(range(len(importances)), feature_names, rotation=45)
plt.title("Permutation Importance - Random Forest")
plt.tight_layout()
plt.show()
```

### Selección de Métricas según Objetivo de Negocio

| Objetivo de Negocio | Métrica Principal | Métricas Secundarias |
|---------------------|-------------------|---------------------|
| **Minimizar fraudes no detectados** | Recall | F1, PR-AUC |
| **Minimizar falsas alarmas** | Precision | F1, Especificidad |
| **Maximizar retención de clientes** | F1 o Recall | AUC-ROC, Lift |
| **Predicción de ventas** | RMSE | MAE, R² |
| **Segmentación de clientes** | Interpretabilidad | Reglas del árbol |
| **Scoring crediticio** | AUC-ROC | F1, KS statistic |
| **Optimización de inventario** | MAE (robusto a outliers) | RMSE, MAPE |
| **Explicabilidad regulatoria** | Reglas del árbol | Feature importance |

---

## Comunicación de Resultados

### Explicación de Decisiones mediante Reglas del Árbol

**Ejemplo para negocio (scoring crediticio):**

```python
from sklearn.tree import export_text

# Exportar reglas del árbol
tree_rules = export_text(best_tree, feature_names=feature_names)
print(tree_rules)
```

**Salida interpretable para stakeholders:**

```
|--- ingreso <= 50000.00
|   |--- edad <= 30.00
|   |   |--- historial_crediticio = malo
|   |   |   |--- clase: ALTO RIESGO (84% de impagos)
|   |   |--- historial_crediticio = bueno
|   |       |--- clase: RIESGO MEDIO (32% de impagos)
|   |--- edad > 30.00
|       |--- deuda_ingreso <= 0.40
|           |--- clase: BAJO RIESGO (5% de impagos)
```

**Traducción a lenguaje de negocio:**

> "El perfil de mayor riesgo son clientes con ingresos menores a 50,000, menores de 30 años y con mal historial crediticio. Este segmento tiene una probabilidad de impago del 84%. En contraste, clientes con ingresos superiores a 50,000 o mayores de 30 años con baja relación deuda/ingreso presentan riesgo mínimo (5% de impago)."

### Uso de Importancia de Variables para Generar Insights

**Visualización para stakeholders:**

```python
# Gráfico de importancia
import pandas as pd

importance_df = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': best_rf.feature_importances_
}).sort_values('Importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='Importancia', y='Variable')
plt.title('Top 10 Variables más Importantes - Random Forest')
plt.tight_layout()
plt.show()
```

**Interpretación para negocio:**

> "Las tres variables que más impactan en la predicción de fraude son: **monto de la transacción**, **hora del día** y **país de origen**. Esto sugiere que debemos enfocar nuestros controles en transacciones de alto monto en horarios atípicos y ciertos países de riesgo. Las variables demográficas del cliente tienen menor impacto relativo."

### Comparación Clara entre Modelos Simples y Ensambles

**Tabla comparativa para reporte ejecutivo:**

| Métrica | Árbol Simple | Random Forest | Mejora |
|---------|--------------|---------------|--------|
| Accuracy | 0.82 | 0.89 | +8.5% |
| Precision | 0.68 | 0.75 | +10.3% |
| Recall | 0.71 | 0.84 | +18.3% |
| F1-Score | 0.69 | 0.79 | +14.5% |
| AUC-ROC | 0.85 | 0.93 | +9.4% |

**Explicación del trade-off:**

> "El árbol simple nos permite explicar cada decisión con reglas claras (por ejemplo, 'si ingreso < 50,000 y edad < 30, entonces riesgo alto'). Sin embargo, su precisión es limitada (F1=0.69). Random Forest mejora significativamente todas las métricas (+14.5% en F1), pero perdemos la capacidad de visualizar el modelo completo. Podemos compensar usando importancia de variables y extrayendo reglas de árboles individuales."

### Traducción de Resultados a Impacto de Negocio

**Ejemplo 1: Detección de Fraude**

> **Resultado técnico:** Random Forest con recall=0.85, precisión=0.75
>
> **Impacto de negocio:**
> - De 2,000 fraudes mensuales, detectaremos 1,700 (vs 1,420 del modelo anterior)
> - Pérdidas evitadas: $850,000 adicionales al año
> - Falsas alarmas: 750 mensuales (vs 1,200 anteriores), reduciendo trabajo operativo en 37%
> - ROI estimado: 320% en primer año

**Ejemplo 2: Predicción de Churn**

> **Resultado técnico:** Random Forest con F1=0.72, AUC-ROC=0.88
>
> **Impacto de negocio:**
> - Identificamos 85% de clientes en riesgo de abandono (vs 60% anterior)
> - Campañas de retención focalizadas en 5,000 clientes de alto valor
> - Reducción de churn del 15% al 9% en segmento objetivo
> - Incremento de ingresos retenidos: $2.3M anuales

**Ejemplo 3: Scoring Crediticio**

> **Resultado técnico:** Árbol de decisión (por interpretabilidad) con AUC-ROC=0.82
>
> **Impacto de negocio:**
> - Reducción de impagos: 25% menos que modelo anterior
> - Aprobaciones: 15% más en segmentos de bajo riesgo
> - Cumplimiento regulatorio: modelo completamente explicable a auditores
> - Reglas claras para agentes de crédito en sucursales


---

## Laboratorio

### Experimentación: Implementación de Árboles y Random Forest en Scikit-learn

#### Pipeline Completo

```python
# ============================================
# LABORATORIO: ÁRBOLES Y RANDOM FOREST
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# 1. CARGA Y EXPLORACIÓN DE DATOS
# --------------------------------
# Cargar dataset (ejemplo con datos de fraude o churn)
# df = pd.read_csv('datos.csv')

# Para este ejemplo, usaremos un dataset sintético
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=15, 
    n_redundant=5, random_state=42, weights=[0.9, 0.1]  # Desbalanceo 90-10
)

feature_names = [f'feature_{i}' for i in range(20)]
target_names = ['Clase 0', 'Clase 1']

# 2. PREPROCESAMIENTO
# -------------------
# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar (opcional para árboles, pero puede ayudar en interpretación)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. ÁRBOL DE DECISIÓN - BÚSQUEDA DE HIPERPARÁMETROS
# ---------------------------------------------------
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [10, 20, 50, 100],
    'min_samples_leaf': [5, 10, 20, 50],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt, dt_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
dt_grid.fit(X_train_scaled, y_train)

print("Mejores parámetros Árbol:", dt_grid.best_params_)
print("Mejor F1 CV Árbol:", dt_grid.best_score_)

# 4. RANDOM FOREST - BÚSQUEDA DE HIPERPARÁMETROS
# -----------------------------------------------
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
rf_grid.fit(X_train_scaled, y_train)

print("Mejores parámetros Random Forest:", rf_grid.best_params_)
print("Mejor F1 CV Random Forest:", rf_grid.best_score_)

# 5. EVALUACIÓN EN TEST
# ---------------------
# Árbol
dt_best = dt_grid.best_estimator_
y_pred_dt = dt_best.predict(X_test_scaled)
y_proba_dt = dt_best.predict_proba(X_test_scaled)[:, 1]

# Random Forest
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test_scaled)
y_proba_rf = rf_best.predict_proba(X_test_scaled)[:, 1]

# Métricas
print("\n" + "="*50)
print("ÁRBOL DE DECISIÓN - REPORTE EN TEST")
print("="*50)
print(classification_report(y_test, y_pred_dt, target_names=target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_dt):.4f}")

print("\n" + "="*50)
print("RANDOM FOREST - REPORTE EN TEST")
print("="*50)
print(classification_report(y_test, y_pred_rf, target_names=target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# 6. ANÁLISIS DE OVERFITTING
# --------------------------
train_score_dt = dt_best.score(X_train_scaled, y_train)
test_score_dt = dt_best.score(X_test_scaled, y_test)
train_score_rf = rf_best.score(X_train_scaled, y_train)
test_score_rf = rf_best.score(X_test_scaled, y_test)

print("\n" + "="*50)
print("ANÁLISIS DE OVERFITTING")
print("="*50)
print(f"Árbol - Train Accuracy: {train_score_dt:.4f}, Test Accuracy: {test_score_dt:.4f}, Diferencia: {train_score_dt - test_score_dt:.4f}")
print(f"Random Forest - Train Accuracy: {train_score_rf:.4f}, Test Accuracy: {test_score_rf:.4f}, Diferencia: {train_score_rf - test_score_rf:.4f}")

# 7. IMPORTANCIA DE VARIABLES
# ---------------------------
# Gini Importance
dt_importance = dt_best.feature_importances_
rf_importance = rf_best.feature_importances_

# Permutation Importance (más robusto)
perm_importance_rf = permutation_importance(
    rf_best, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Árbol
axes[0].barh(range(len(dt_importance)), dt_importance)
axes[0].set_yticks(range(len(dt_importance)))
axes[0].set_yticklabels([feature_names[i] for i in range(len(dt_importance))])
axes[0].set_title('Importancia - Árbol')
axes[0].invert_yaxis()

# Random Forest
axes[1].barh(range(len(rf_importance)), rf_importance)
axes[1].set_yticks(range(len(rf_importance)))
axes[1].set_yticklabels([feature_names[i] for i in range(len(rf_importance))])
axes[1].set_title('Importancia - Random Forest')
axes[1].invert_yaxis()

# Permutation Importance
axes[2].barh(range(len(perm_importance_rf.importances_mean)), 
             perm_importance_rf.importances_mean,
             xerr=perm_importance_rf.importances_std)
axes[2].set_yticks(range(len(perm_importance_rf.importances_mean)))
axes[2].set_yticklabels([feature_names[i] for i in range(len(perm_importance_rf.importances_mean))])
axes[2].set_title('Permutation Importance')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()

# 8. VISUALIZACIÓN DEL ÁRBOL (si no es muy profundo)
# ---------------------------------------------------
if dt_best.tree_.max_depth <= 5:
    plt.figure(figsize=(20, 10))
    plot_tree(dt_best, feature_names=feature_names, 
              class_names=target_names, filled=True, rounded=True)
    plt.show()
else:
    # Exportar texto del árbol
    tree_rules = export_text(dt_best, feature_names=feature_names)
    print("\nReglas del árbol (primeras 20 líneas):")
    print("\n".join(tree_rules.split("\n")[:20]))

# 9. CURVAS ROC Y PR
# ------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Curva ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

axes[0].plot(fpr_dt, tpr_dt, label=f'Árbol (AUC={roc_auc_score(y_test, y_proba_dt):.3f})')
axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', label='Aleatorio')
axes[0].set_xlabel('Tasa Falsos Positivos (FPR)')
axes[0].set_ylabel('Tasa Verdaderos Positivos (TPR)')
axes[0].set_title('Curva ROC')
axes[0].legend()

# Curva Precision-Recall
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_proba_dt)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
ap_dt = average_precision_score(y_test, y_proba_dt)
ap_rf = average_precision_score(y_test, y_proba_rf)

axes[1].plot(recall_dt, precision_dt, label=f'Árbol (AP={ap_dt:.3f})')
axes[1].plot(recall_rf, precision_rf, label=f'Random Forest (AP={ap_rf:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precisión')
axes[1].set_title('Curva Precisión-Recall')
axes[1].legend()

plt.tight_layout()
plt.show()

# 10. CONCLUSIONES
# ----------------
print("\n" + "="*50)
print("CONCLUSIONES")
print("="*50)
print(f"""
1. Rendimiento:
   - Árbol: F1={classification_report(y_test, y_pred_dt, output_dict=True)['weighted avg']['f1-score']:.3f}
   - Random Forest: F1={classification_report(y_test, y_pred_rf, output_dict=True)['weighted avg']['f1-score']:.3f}

2. Overfitting:
   - Árbol: diferencia train-test = {train_score_dt - test_score_dt:.3f}
   - Random Forest: diferencia train-test = {train_score_rf - test_score_rf:.3f}
   
3. Variables más importantes (Random Forest):
   - {feature_names[np.argmax(rf_importance)]}: {rf_importance.max():.3f}
   - {feature_names[np.argsort(rf_importance)[-2]]}: {rf_importance[np.argsort(rf_importance)[-2]]:.3f}

4. Recomendación:
   {'Random Forest' if rf_grid.best_score_ > dt_grid.best_score_ else 'Árbol'} es el modelo recomendado por:
   - Mayor F1 en validación cruzada
   - Menor overfitting
   - Mejor AUC-ROC y PR-AUC
""")
```



