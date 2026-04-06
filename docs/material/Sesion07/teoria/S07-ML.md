---
layout: default
---
# Gradient Boosting Machines (XGBoost, LightGBM, CatBoost)

## 1. Introducción y Contextualización

### a) Definición del problema
Es un problema de **aprendizaje supervisado** (tanto regresión como clasificación binaria/multiclase). Se dispone de un conjunto de entrenamiento $\{(x_i, y_i)\}_{i=1}^N$ con $x_i \in \mathbb{R}^d$ e $y_i$ continua (regresión) o categórica (clasificación). El objetivo es aproximar una función $F^*(x)$ que minimice el riesgo esperado bajo una función de pérdida $L(y, F(x))$.

### b) Objetivo del modelo
Construir un **conjunto (ensemble) de árboles de decisión** de forma secuencial, donde cada nuevo árbol corrige los errores residuales del conjunto anterior. La búsqueda se realiza bajo restricciones de **eficiencia computacional** (gran escala: millones de filas y miles de columnas), **robustez frente a datos sucios** (valores nulos, outliers, características categóricas) y **alta precisión** sin overfitting excesivo.

## 2. Antecedentes (Estado del Arte)

### a) Evolución histórica
| Año       | Hito                                 | Descripción                                                                                                                                                             |
| --------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1988-1990 | El Origen Teórico                    | Michael Kearns, Leslie Valiant y Robert Schapire demostraron matemáticamente que un conjunto de "aprendices débiles" podía combinarse para formar un "aprendiz fuerte". |
| 1999      | AdaBoost (Freund & Schapire)         | Primer boosting con pesos adaptativos, árboles como débiles.                                                                                                            |
| 2001      | Gradient Boosting Machine (Friedman) | Generalización a cualquier pérdida diferenciable usando gradientes.                                                                                                     |
| 2014 | XGBoost (Chen & Guestrin) | Optimización con regularización, manejo de nulos, paralelización. |
| 2016 | LightGBM (Microsoft) | Basado en histogramas y crecimiento *leaf-wise*; enorme velocidad. |
| 2017 | CatBoost (Yandex) | Manejo nativo de categóricas con *ordered boosting* y simetría. |

### b) Comparativa de paradigmas
| Paradigma | Ventaja clave | Desventaja | Relevancia actual |
|-----------|---------------|------------|-------------------|
| Random Forest | Paralelizable, robusto a overfitting | Menor precisión en tabulares densos | Baja en alta dimensionalidad |
| Redes Neuronales | Alta capacidad en no estructurados (imagen, texto) | Requiere mucho dato y tuning | Media en tabulares |
| **GBM modernos** | Precisión SOTA en tabulares + velocidad | Sensible a hiperparámetros | **Muy alta** (Kaggle, industria) |

## 3. Fundamentos Técnicos (El Algoritmo)

Sea $F_0(x)$ un modelo base (usualmente la media o log-odds). Para cada iteración $m=1,\dots,M$:

1. Calcular los **pseudo-residuos** (gradiente negativo de la pérdida):
   $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x)=F_{m-1}(x)}$$
2. Ajustar un árbol débil $h_m(x)$ a los residuos $\{(x_i, r_{im})\}$.
3. Actualizar $F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m h_m(x)$, con $\eta$ *learning rate* y $\gamma_m$ obtenido por *line search*.

**Arquitectura específica de cada implementación**:
- **XGBoost**: usa *gradient statistics* y un árbol de estructura exacta (o aproximada por percentiles). Regularización $L_1$/$L_2$ en pesos de hojas.
- **LightGBM**: discretiza features en histogramas (acelera); crecimiento *leaf-wise* (expande la hoja con mayor pérdida) en vez de *level-wise*.
- **CatBoost**: construye árboles simétricos (oblivious); maneja categóricas con estadísticas ordenadas (target encoding con prior) y *ordered boosting* para evitar overfitting.

**Función de pérdida común**: Log-loss para clasificación:
$$L(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]$$
donde $\hat{y} = \sigma(F(x))$ y $\sigma$ es la función sigmoide.

## 4. Problemática y Justificación de Negocio

### a) Drivers de decisión
Usar estos modelos cuando:
- Los datos son **tabulares** con mezcla de numéricas y categóricas.
- El volumen supera los 100k registros (LightGBM/CatBoost excelentes >1M).
- Se necesita **explicabilidad** parcial (importancia de features, SHAP).
- El tiempo de entrenamiento es crítico (modelos baseline rápidos).

### b) Casos de uso industriales
1. **Banca**: scoring crediticio (clasificación de morosidad) – CatBoost por categóricas.
2. **Retail**: predicción de churn (clientes que abandonan) – LightGBM por velocidad.
3. **Seguros**: estimación de siniestralidad (regresión Poisson) – XGBoost con pérdida personalizada.
4. **Logística**: tiempo de entrega (regresión) – cualquier GBM con features de calendario.
5. **Fraude**: detección de transacciones fraudulentas – XGBoost por robustez a desbalanceo.

### c) Trade-offs
| Aspecto | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Velocidad (entrenamiento) | Media | **Muy alta** | Media (con GPU: alta) |
| Precisión baseline | Alta | Alta | **Muy alta** (especialmente con categóricas) |
| Memoria RAM | Media | **Baja** (histogramas) | Media-alta |
| Manejo de categóricas | Requiere encoding manual | Requiere encoding manual | **Nativo** |
| Tolerancia a nulos | Sí (dirección optimizada) | Sí | Sí |

**Decisión**: si importa sobre todo rapidez → LightGBM; si hay muchas categóricas → CatBoost; si se busca madurez y documentación → XGBoost.

## 5. Guía de Implementación y Ecosistema

### a) Ecosistema de despliegue
- **Librerías**: `xgboost`, `lightgbm`, `catboost` (Python); también API `scikit-learn` (wrappers).
- **Plataformas**: CPU/GPU (todas soportan CUDA); Spark (XGBoost4J-Spark, SynapseML); Cloud (AWS SageMaker, GCP Vertex AI).
- **Optimización**: Integración con `Optuna`, `Hyperopt` para búsqueda de hiperparámetros.

### b) Plantilla base en Python (clasificación binaria con LightGBM como ejemplo)
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

## 6. Evaluación de Rendimiento (Métricas)

### a) Formulación Matemática
**AUC-ROC** (clasificación):
$$AUC = \frac{1}{n_+ n_-} \sum_{i=1}^{n_+} \sum_{j=1}^{n_-} \mathbb{1}[s_i > s_j]$$
donde $s_i$ son las predicciones para positivos, $s_j$ para negativos. Interpretación: probabilidad de que un positivo aleatorio tenga score mayor que un negativo.

**Log-Loss**:
$$LogLoss = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

**RMSE** (regresión):
$$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

### b) Rangos y Umbrales
| Métrica | Rango | Óptimo | Umbral de negocio típico |
|---------|-------|--------|---------------------------|
| AUC | [0.5, 1] | 1 | >0.75 (aceptable), >0.85 (bueno) |
| LogLoss | [0, ∞) | 0 | <0.5 (clases balanceadas) |
| RMSE | [0, ∞) | 0 | Depende de escala; se compara con baseline (media) |

### c) Interpretación de Negocio
Un **Falso Positivo** (FP) en detección de fraude: aprobar una transacción fraudulenta → costo de monto perdido. Un **Falso Negativo** (FN): rechazar una transacción legítima → costo de oportunidad + mala experiencia. El umbral de decisión se elige minimizando el costo esperado:
$$C = C_{FP} \cdot FP + C_{FN} \cdot FN$$

### d) Criterio de Selección
1. **AUC** para comparar poder discriminatorio (independiente del umbral).
2. **LogLoss** para calibrar probabilidades (importante en scoring).
3. **Tiempo de entrenamiento** y **memoria** en producción.
4. **Estabilidad** de la importancia de variables entre folds (baja varianza).

## 7. Interpretabilidad y Explicabilidad

### a) Transparencia del modelo
- **Caja gris**: los GBM no son caja negra total porque se pueden extraer reglas de los árboles. Sin embargo, con cientos de árboles, la complejidad es humana-incomprensible.  
- Las implementaciones modernas ofrecen **importancia global** (ganancia, cobertura, frecuencia) y **explicaciones locales** (SHAP, LIME).

### b) Herramientas de diagnóstico
- **Importancia de características**: `model.feature_importances_` en XGBoost/LightGBM.
- **SHAP** (SHapley Additive exPlanations): descompone la predicción como suma de contribuciones de cada feature.
  $$f(x) = \phi_0 + \sum_{j=1}^d \phi_j(x)$$
  donde $\phi_j$ es el valor Shapley (efecto marginal medio).  
  Se implementa con `shap.Explainer(model, X_train)`.
- **LIME** (Local Interpretable Model-agnostic Explanations): ajusta un modelo lineal local alrededor de la instancia.
- **Gráficos de dependencia parcial (PDP)**: muestra el efecto marginal de una feature en la predicción promediando el resto.

## 8. Anexo: Demostración Matemática

**Gradiente de la Log-loss para clasificación**:

Sea $F(x)$ la salida bruta (log-odds), $\hat{y} = \sigma(F) = \frac{1}{1+e^{-F}}$.  
La pérdida por muestra: $L(y, F) = -[y \ln(\hat{y}) + (1-y) \ln(1-\hat{y})]$.

Derivada con respecto a $F$:
$$\frac{\partial L}{\partial F} = -\left[ y \frac{1}{\hat{y}} \frac{\partial \hat{y}}{\partial F} + (1-y) \frac{1}{1-\hat{y}} \left(-\frac{\partial \hat{y}}{\partial F}\right) \right]$$
Sabemos que $\frac{\partial \hat{y}}{\partial F} = \hat{y}(1-\hat{y})$. Sustituyendo:
$$\frac{\partial L}{\partial F} = -\left[ y \frac{1}{\hat{y}} \cdot \hat{y}(1-\hat{y}) - (1-y) \frac{1}{1-\hat{y}} \cdot \hat{y}(1-\hat{y}) \right] = -\left[ y(1-\hat{y}) - (1-y)\hat{y} \right]$$
Simplificando: $= -[y - y\hat{y} - \hat{y} + y\hat{y}] = -[y - \hat{y}] = \hat{y} - y$.

Por tanto, el pseudo-residuo (gradiente negativo) es:
$$r_i = -\frac{\partial L}{\partial F} = y_i - \hat{y}_i$$
Es decir, el **residuo real** entre la etiqueta y la probabilidad estimada. Cada nuevo árbol se ajusta a estos residuos, acercando iterativamente las predicciones a las etiquetas verdaderas.

**Optimización mediante aproximación de Newton (XGBoost)**: XGBoost usa una expansión de Taylor de segundo orden:
$$L^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
con $g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})$, $h_i$ la segunda derivada, y $\Omega$ regularización sobre la estructura del árbol (número de hojas y pesos). Esto permite obtener la ganancia de una división de forma cerrada y acelerar la convergencia.

---