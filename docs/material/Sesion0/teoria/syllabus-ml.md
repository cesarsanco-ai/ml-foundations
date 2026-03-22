# Sílabo del Curso

# **MACHINE LEARNING**
## *para Inteligencia Artificial*

**Fundamentos, algoritmos y aplicaciones prácticas**

*Por*

**Carlos César Sánchez Coronel**

2026




---

## Presentación del Curso

Este curso está diseñado para futuros ingenieros e científicos de datos que deseen dominar los fundamentos del machine learning desde una perspectiva práctica y orientada a la industria. A lo largo de 14 semanas, exploraremos los algoritmos más importantes, técnicas de preprocesamiento, evaluación, interpretabilidad y despliegue, con un enfoque en las preguntas y habilidades que se esperan en entrevistas técnicas y en el trabajo diario.

Cada Modelo de ML contiene:
- **Problematica de negocio**: Problematica de negocio.
- **Modelado de ML**: Selección de modelo
- **Fundamento matematico y computacional**: Matematica y computación que incluye el modelo.
- **Pipeline de datos**: Obtención de datos, preprocesamiento, feature engineering, evaluación.
- **Análisis de metricas**: Interpretacion de metricas del modelo.
- **Comunicación**: Storytelling, elevator pitch a equipo tecnico y no tecnico.

---

## Semana 1: Introducción al ML

### Logro de la sesión
Comprender el flujo completo de un proyecto de machine learning, identificar los tipos de aprendizaje y reconocer casos de uso en la industria.

#### Conceptos
- Definición de Machine Learning, Deep Learning y su relación con la IA.
- Problemas de negocios y Tipos de aprendizaje: supervisado, no supervisado, por refuerzo.
- Metodos parametricos y no parametricos en ML.
- Historia y auge de los Modelos de ML.
- Ciclo de vida y metodologia CRISP-DM.
- Diferencias: Data engineer, Data Scientist, ML Engineer, MLOps.

#### Aplicaciones y casos típicos
- Sistemas de recomendación (Netflix, Amazon).
- Detección de fraude en transacciones bancarias.
- Mantenimiento predictivo en maquinaria industrial.
- Clasificación de imágenes (diagnóstico médico, vehículos autónomos).
- Predicción de series temporales (ventas, demanda energética).

#### Fundamentos matematicos y computacionales
- Matemáticas detrás de los ML clasico: Funciones de costos, optimización, generalizado y especifico por modelos de regresion, clasificacion y clustering.
- Complejidad algoritmica que incluyen los modelos.
- Hardware: CPU, GPU, TPU.

#### Métricas
- Tipos de metricas en ML e importancia de la métrica en el negocio.

#### Laboratorio: Ver Colab
- Repaso de Numpy
- Repaso de Pandas
- Repaso de Matplotlib
- Repaso de Seaborn

#### Reto: 1 punto
- Escoger una empresa tech famosa, y detectar la mayor cantidad de modelos de ML que puede tener. Ejm. en Uber, Netflix, Spotify, TikTok, etc.

---

## Semana 2: Análisis Exploratorio y Feature Engineering

### Logro de la sesión
Realizar un análisis exploratorio y aplicar técnicas de feature engineering para preparar datos de calidad.

#### Conceptos
- Importancia del EDA
- Casos famosos de exito y fracaso en negocios
- Pipeline de EDA
- Limpieza de datos (valores faltantes, valores duplicados, errores de tipo de dato, tratamiento basicode outliers, inconsistencias y errores logicos)
- Analisis univariado (Variables numericas: medidas de tendencia central, medidas de dispersion, medidas de forma, tipos de visualizaciones; Variables categoricas: frecuencias, moda, proporciones, tipos de visualizaciones; Variables de Fecha: Componentes de fechas, tipos de visualizaciones)
- Analisis bivariado (Variables numericas vs numericas, categoricas vs categoricas, numericas vs categoricas)
- Analisis multivariado (Matrices de correlacion, Pairplots, Heatmaps)

- Importancia del Feature engineering
- Pipeline de feature engineering
- Creación de nuevas características (a partir de numericas, categoricas, fechas, agregaciones, texto,etc)
- Codificación de variables categoricas (One-Hot Encoding, Label Encoding, Target Encoding, Binary Encoding, Hashing Encoding)
- Escalado y Normalizacion (Z-score, Min-max, RobustScaler, Normalizer)
- Manejo de datos desbalanceados (Oversampling, Undersampling, SMOTE)
- Imputación de datos (Mean, Median, Mode, KNNImputer, IterativeImputer)
- Manejo de outliers (variable indicadora de outlier, winsorizacion, transformaciones para reducir impacto)
- Selección de características (Filtros, Wrappers, Embebidos)
- Variables derivadas del negocio (ratios, agregaciones temporales, interacciones)

#### Comunicacion
- Storytelling e insights con data dummy
- Elevator pitch a equipo tecnico y no tecnico

#### Laboratorio: Ver Colab
- EDA
- Introducción a scikit-learn
- Feature engineering

#### Reto: 1 punto
- Buscar 3 casos de exito del EDA y/o feature engineering en empresas big tech.

#### Anexo: Fundamentos matematicos y computacionales
- Estadistica descriptiva
- Probabilidad
- Estadistica inferencial
- Optimizacion computacional

---

## Semana 3: Regresión Lineal y Regularización

### Logro de la sesión
Construir e interpretar modelos de regresión lineal, aplicando regularización y sus metricas.

#### Problematica de negocio
- Tipos y ejemplos de problemas de regresion
- Solucion: regresión simple y multiple

#### Modelado
- Requisitos del modelo
- Regresión lineal simple y múltiple: formulación, interpretación de coeficientes.
- Supuestos del modelo: linealidad, independencia, homocedasticidad, normalidad de residuos.
- Regularización: Ridge (L2), Lasso (L1) y Elastic Net.
- Interpretación de la regularización y selección de hiperparámetros.
- Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

#### Métricas
- Error Cuadrático Medio (MSE), Raíz del Error Cuadrático Medio (RMSE), Error Absoluto Medio (MAE), Coeficiente de determinación \(R^2\).
- Interpretacion y selección de metricas

#### Comunicacion
- Storytelling e insights con data dummy
- Elevator pitch a equipo tecnico y no tecnico

#### Reto: 1 punto
- Investigar el computo necesario para entrenar un modelo de regresión lineal en un dataset de 10 millones de filas y 1000 columnas en Databricks.

#### Laboratorio: Ver Colab
- Scikit-learn: Regresión lineal, Regularización, Métricas
- Pipeline datos: EDA + Feature engineering + Regresión multiple + Regularizacion + metricas

#### Anexo: Fundamento matematico y computacional
- Demostracion de la regresión lineal
- Demostracion de la regularización
- Optimizacion computacional
- Complejidad algoritmica

---

## Semana 4: Clasificación I - Regresión Logística y Balance de Datos

### Logro de la sesión
Construir, interpretar y evaluar modelos de clasificación binaria y multiclase utilizando regresión logística, incorporando técnicas de balanceo de datos y selección adecuada de métricas.

#### Problematica de negocio
- Tipos de problemas de clasificación: binaria y multiclase
- Impacto del desbalance de clases en problemas reales
- Ejemplos clasicos: detección de fraude, churn de clientes, diagnóstico médico, clasificación de productos y clientes, score crediticio
- Diferencias clave entre regresión y clasificación (output continuo vs categórico)

#### Modelado de clasificación
- Requisitos del modelo
- Regresión logística binaria: función sigmoide
- Extensión a regresión logística multiclase (One-vs-Rest, Softmax)
- Interpretación probabilística de las predicciones
- Interpretación de coeficientes (log-odds) e impacto de variables
- Límite de decisión (threshold) y trade-off entre métricas
- Regularización: L1 (Lasso), L2 (Ridge) y Elastic Net
- Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

#### Balance de datos
- Problema de clases desbalanceadas
- Técnicas de balanceo:
    - Oversampling (SMOTE)
    - Undersampling
    - Ajuste de pesos (class weights)
- Cuándo usar cada técnica
- Impacto en el modelo y en las métricas

#### Métricas
- Matriz de confusión e interpretacion: TP, FP, TN, FN
- Metricas e interpretacion: Exactitud, Precisión, recall (sensibilidad), especificidad
- F1-score y su interpretación
- Curva ROC y AUC; PR-AUC
- Métricas en escenarios desbalanceados
- Selección de métricas según contexto de negocio (agua potable, phising, covid)

#### Comunicacion
- Storytelling e insights con data dummy
- Interpretación de errores: costo de falsos positivos vs falsos negativos
- Ajuste del threshold según objetivos de negocio
- Elevator pitch a equipo tecnico y no tecnico

#### Reto: 1 punto
- Analizar un dataset desbalanceado, comparar métricas antes y después de aplicar técnicas de balanceo, y justificar cuál estrategia es más adecuada según el caso de negocio.

#### Laboratorio: Ver Colab
- Scikit-learn: regresión logística (binaria y multiclase)
- Pipeline completo:
    - EDA
    - Feature engineering
    - Balanceo de datos
    - Entrenamiento de modelo
    - Evaluación con métricas
    - Ajuste de threshold

#### Anexo: Fundamento matematico y computacional
- Derivación de la regresión logística
- Función de costo: log-loss (cross-entropy)
- Optimización: gradiente descendente
- Extensión a multiclase (softmax)
- Complejidad algorítmica

---

## Semana 5: Clasificación II - Algoritmos Clásicos

### Logro de la sesión
Construir, comparar e interpretar modelos de clasificación utilizando algoritmos clásicos (KNN, Naive Bayes y SVM), comprendiendo sus supuestos, ventajas y limitaciones en distintos contextos de negocio.

#### Problematica de negocio
- Selección de algoritmos de clasificación según el tipo de datos
- Comparación de desempeño entre modelos
- Casos donde la regresión logística no es suficiente (no linealidad, alta dimensionalidad)
- Impacto del tamaño de datos y escalamiento en el rendimiento
- KNN: sistemas de recomendación simples, detección de similitud
- Naive Bayes: clasificación de texto (spam, sentimiento)
- SVM: problemas con fronteras no lineales (visión por computadora, bioinformática)

#### Modelado
- K-Nearest Neighbors (KNN):
    - Requisitos del modelo
    - Concepto de distancia (euclidiana, manhattan)
    - Elección de \(k\) y trade-off bias-varianza
    - Sensibilidad a la escala de variables
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Naive Bayes:
    - Requisitos del modelo
    - Teorema de Bayes
    - Supuesto de independencia condicional
    - Variantes: Gaussian, Multinomial, Bernoulli
    - Interpretación probabilística
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Support Vector Machines (SVM):
    - Requisitos del modelo
    - Margen máximo y vectores de soporte
    - Kernel lineal vs no lineal (RBF)
    - Hiperparámetros: \(C\) y \(\gamma\)
    - Manejo de no linealidad
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Comparación conceptual entre algoritmos:
    - Paramétricos vs no paramétricos
    - Interpretabilidad vs performance
    - Escalabilidad computacional

#### Métricas
- Matriz de confusión: TP, FP, TN, FN
- Exactitud, Precisión, recall, F1-score
- Curva ROC y AUC
- Comparación de modelos usando métricas consistentes

#### Comunicacion
- Comparación de modelos para toma de decisiones
- Explicación de trade-offs: interpretabilidad vs precisión
- Justificación del modelo seleccionado según contexto de negocio

#### Reto: 1 punto
- Comparar KNN, Naive Bayes y SVM en un mismo dataset, evaluando métricas y tiempos de entrenamiento, y justificar cuál modelo es más adecuado según el caso.

#### Laboratorio
- Implementación de KNN, Naive Bayes y SVM en Scikit-learn
- Pipeline:
    - Datos preprocesados (EDA + feature engineering ya realizado)
    - Escalamiento de variables (clave para KNN y SVM)
    - Entrenamiento de múltiples modelos
    - Evaluación con métricas
    - Comparación de resultados

#### Anexo: Fundamento matematico y computacional
- Distancias y espacios métricos (KNN)
- Teorema de Bayes y probabilidades condicionales
- Optimización en SVM (margen máximo)
- Complejidad computacional de cada algoritmo

---

## Semana 6: Árboles de Decisión y Random Forest

### Logro de la sesión
Construir, interpretar y comparar modelos basados en árboles de decisión y ensambles tipo Random Forest, comprendiendo su capacidad para capturar relaciones no lineales y su aplicación en problemas reales de clasificación y regresión.

#### Problematica de negocio
- Modelos interpretables vs modelos de alto performance
- Manejo de relaciones no lineales y variables categóricas
- Reducción de overfitting en modelos complejos
- Necesidad de entender qué variables impactan más en la predicción
- Detección de fraude en tarjetas de crédito
- Scoring crediticio y evaluación de riesgo
- Modelos con alta dimensionalidad (texto, genómica)
- Problemas donde la interpretabilidad es clave (reglas de decisión)

#### Modelado
- Árboles de decisión (clasificación y regresión):
    - Estructura de árbol: nodos, ramas y hojas
    - Criterios de división:
        - Clasificación: Gini, entropía
        - Regresión: MSE
    - Sobreajuste (overfitting) y control de complejidad:
        - `max_depth`, `min_samples_split`, `min_samples_leaf`
    - Interpretación de decisiones del modelo
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Random Forest (Bagging):
    - Concepto de ensamble: combinación de múltiples árboles
    - Bootstrap sampling (muestreo con reemplazo)
    - Selección aleatoria de variables (feature randomness)
    - Reducción de varianza vs árbol individual
    - Out-of-Bag (OOB) error como validación interna
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Comparación:
    - Árbol simple vs Random Forest
    - Interpretabilidad vs performance
    - Bias vs varianza

#### Métricas
- Clasificación: precisión, recall, F1-score, AUC-ROC
- Regresión: MSE, RMSE, MAE, \(R^2\)
- Importancia de variables:
    - Gini importance (feature importance)
    - Permutation importance
- Selección de métricas según objetivo de negocio

#### Comunicacion
- Explicación de decisiones del modelo mediante reglas del árbol
- Uso de importancia de variables para generar insights
- Comparación clara entre modelos simples y ensambles
- Traducción de resultados a impacto de negocio

#### Reto: 1 punto
- Comparar un árbol de decisión y un Random Forest en el mismo dataset, analizando métricas, overfitting y la importancia de variables.

#### Laboratorio
- Experimentacion: Implementación de árboles y Random Forest en Scikit-learn
- Pipeline:
    - Datos preprocesados (EDA + feature engineering)
    - Entrenamiento de árbol de decisión
    - Control de overfitting (tuning de hiperparámetros)
    - Entrenamiento de Random Forest
    - Evaluación y comparación de modelos
    - Análisis de importancia de variables

#### Anexo: Fundamento matematico y computacional
- Criterios de impureza: Gini, entropía, MSE
- Reducción de varianza en bagging
- Bias-variance trade-off
- Complejidad computacional de árboles y ensambles

---

## Semana 7: Gradient Boosting (XGBoost, LightGBM)

### Logro de la sesión
Construir, optimizar e interpretar modelos de Gradient Boosting (XGBoost y LightGBM), comprendiendo su funcionamiento secuencial y ajustando hiperparámetros para maximizar el rendimiento en problemas de clasificación y regresión.

#### Problematica de negocio
- Necesidad de modelos de alto rendimiento en datos tabulares
- Mejora incremental de predicciones (reducción de error)
- Manejo de relaciones no lineales y variables complejas
- Trade-off entre performance, tiempo de entrenamiento e interpretabilidad
- Predicción de clics (CTR) en publicidad online
- Scoring crediticio y detección de fraude
- Modelado de datos tabulares estructurados

#### Modelado
- Boosting:
    - Ensamble secuencial de modelos débiles
    - Corrección iterativa de errores
    - Diferencia clave con bagging (Random Forest)

- Gradient Boosting:
    - Optimización de una función de pérdida (loss function)
    - Uso de árboles débiles (shallow trees)
    - Learning rate (shrinkage) y número de estimadores
    - Riesgo de overfitting

- XGBoost:
    - Requisitos del modelo
    - Regularización (L1, L2)
    - Manejo de valores nulos
    - Early stopping
    - Paralelización eficiente
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- LightGBM:
    - Requisitos del modelo
    - Gradient-based One-Side Sampling (GOSS)
    - Histogram-based splitting
    - Manejo eficiente de variables categóricas
    - Entrenamiento más rápido en grandes volúmenes
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Comparación:
    - Random Forest vs Gradient Boosting (bagging vs boosting)
    - XGBoost vs LightGBM
    - Interpretabilidad vs performance

#### Métricas
- Clasificación: log-loss, AUC-ROC, F1-score
- Regresión: RMSE, MAE, \(R^2\)
- Evaluación en validación cruzada
- Selección de métricas según el objetivo del negocio

#### Hiperparámetros clave
- `n_estimators`: número de árboles
- `learning_rate`: tamaño del paso en cada iteración
- `max_depth`: complejidad de los árboles
- `subsample`, `colsample_bytree`: control de overfitting
- Regularización: `lambda`, `alpha` (XGBoost)

#### Comunicacion
- Explicación de mejoras de performance frente a modelos base
- Interpretación de importancia de variables
- Justificación del uso de modelos complejos
- Balance entre precisión y explicabilidad

#### Reto: 1 punto
- Comparar Random Forest vs XGBoost vs LightGBM en un dataset de kaggle, optimizando hiperparámetros y evaluando diferencias en performance y tiempo de entrenamiento.

#### Laboratorio
- Implementación de XGBoost y LightGBM
- Pipeline:
    - Datos preprocesados (EDA + feature engineering)
    - Entrenamiento de modelo base
    - Ajuste de hiperparámetros
    - Uso de early stopping
    - Evaluación y comparación de modelos
    - Análisis de importancia de variables

#### Anexo: Fundamento matematico y computacional
- Gradient Boosting como optimización de funciones
- Expansión funcional (additive models)
- Regularización en boosting
- Complejidad computacional y escalabilidad

---

## Semana 8: Evaluación y Validación de Modelos

### Logro de la sesión
Evaluar, validar y seleccionar modelos de machine learning de forma robusta, asegurando su capacidad de generalización mediante técnicas de validación cruzada y optimización de hiperparámetros.

#### Problematica de negocio
- Modelos que funcionan bien en entrenamiento pero fallan en producción
- Selección del mejor modelo entre múltiples alternativas
- Riesgo de sobreajuste en datasets pequeños o complejos
- Necesidad de estimar performance real antes de desplegar

#### Aplicaciones y casos típicos
- Selección de modelos en proyectos reales
- Validación antes de despliegue en producción

#### Modelado y validación
- Underfitting y overfitting:
    - Identificación y diagnóstico
    - Estrategias de mitigación

- Bias-Variance tradeoff:
    - Interpretación práctica
    - Relación con complejidad del modelo

- Validación de modelos:
    - Train / Validation / Test split
    - Validación cruzada:
        - K-Fold
        - Stratified K-Fold (clasificación)
        - Leave-One-Out

- Curvas de aprendizaje:
    - Diagnóstico de underfitting vs overfitting
    - Interpretación de curvas

#### Optimización de modelos
- Búsqueda de hiperparámetros:
    - Grid Search
    - Randomized Search
    - Introducción a Bayesian Optimization

- Buenas prácticas:
    - Evitar data leakage
    - Uso de pipelines
    - Separación correcta de datos

#### Métricas
- Uso de métricas en validación cruzada
- Clasificación: AUC, F1-score, log-loss
- Regresión: RMSE, MAE, \(R^2\)
- Selección de métricas alineadas al objetivo de negocio
- Comparación de modelos basada en métricas promedio

#### Plantilla base de Python
- `sklearn.model_selection`:
    - `train_test_split`
    - `cross_val_score`
    - `StratifiedKFold`
    - `learning_curve`
    - `GridSearchCV`
    - `RandomizedSearchCV`

#### Comunicacion
- Explicación de performance esperada en producción
- Comparación clara entre modelos (baseline vs optimizado)
- Justificación de selección de modelo
- Comunicación de riesgos (overfitting, variabilidad)

#### Reto: 1 punto
- Implementar validación cruzada y búsqueda de hiperparámetros en un modelo (por ejemplo, Random Forest o XGBoost), comparando resultados antes y después del tuning.

#### Laboratorio
- Pipeline completo de validación:
    - Train/test split
    - Validación cruzada
    - Entrenamiento de modelo base
    - Búsqueda de hiperparámetros
    - Evaluación final en test
- Comparación de múltiples modelos (logística, árboles, boosting)

#### Anexo: Fundamento matematico y computacional
- Bias-variance decomposition
- Fundamento de validación cruzada
- Complejidad computacional del tuning
- Riesgo de overfitting en selección de modelos

---

## Semana 9: Aprendizaje No Supervisado - Clustering y PCA

### Logro de la sesión
Explorar y extraer valor de datos no etiquetados mediante técnicas de clustering y reducción de dimensionalidad (PCA), identificando patrones, segmentos y estructuras ocultas en los datos.

#### Problematica de negocio
- Falta de etiquetas en los datos (no hay variable objetivo)
- Necesidad de segmentar clientes o comportamientos
- Reducción de complejidad en datasets de alta dimensionalidad
- Identificación de patrones ocultos y anomalías

#### Aplicaciones y casos típicos
- Segmentación de clientes (marketing)
- Reducción de dimensionalidad para visualización
- Compresión de datos
- Detección de anomalías (fraude, comportamiento atípico)

#### Modelado
- PCA (Análisis de Componentes Principales):
    - Reducción de dimensionalidad basada en varianza
    - Componentes principales y su interpretación
    - Explained variance ratio
    - Visualización (2D/3D) para exploración

- K-Means:
    - Requisitos del modelo
    - Algoritmo basado en centroides
    - Inicialización y convergencia
    - Elección de \(k\):
        - Método del codo
        - Coeficiente de silueta
    - Sensibilidad a escala y outliers

- DBSCAN:
    - Requisitos del modelo
    - Clustering basado en densidad
    - Parámetros: \(\epsilon\) (eps) y `min_samples`
    - Detección de ruido (outliers)
    - Ventajas frente a K-Means (no requiere \(k\))

- Clustering jerárquico:
    - Requisitos del modelo
    - Enfoque aglomerativo
    - Dendrogramas e interpretación
    - Selección del número de clusters

- Comparación de métodos:
    - K-Means vs DBSCAN vs Jerárquico
    - Escenarios de uso según tipo de datos

#### Métricas
- Coeficiente de silueta
- Índice de Davies-Bouldin
- Inercia (K-Means)
- Limitaciones de métricas sin ground truth

#### Python
- `sklearn.decomposition.PCA`
- `sklearn.cluster.KMeans`, `DBSCAN`, `AgglomerativeClustering`
- `sklearn.metrics.silhouette_score`
- Buenas prácticas:
    - Escalamiento de datos antes de clustering
    - Uso de PCA para visualización

#### Comunicacion
- Interpretación de clusters (perfilamiento)
- Traducción de segmentos a decisiones de negocio
- Visualización de resultados (scatter plots, PCA)
- Limitaciones del análisis no supervisado

#### Reto: 1 punto
- Aplicar K-Means y DBSCAN a un dataset real, comparar resultados y construir perfiles de los clusters identificados.

#### Laboratorio
- Pipeline de análisis no supervisado:
    - Datos preprocesados (EDA + feature engineering)
    - Escalamiento de variables
    - Reducción de dimensionalidad (PCA)
    - Clustering (K-Means, DBSCAN)
    - Evaluación con métricas
    - Visualización e interpretación de clusters

#### Anexo: Fundamento matematico y computacional
- Álgebra lineal en PCA (autovalores y autovectores)
- Optimización en K-Means
- Concepto de densidad en DBSCAN
- Complejidad computacional de algoritmos de clustering

---

## Semana 10: Series de Tiempo

### Logro de la sesión
Modelar y pronosticar datos temporales utilizando técnicas estadísticas clásicas, machine learning y librerías modernas, extrayendo patrones de tendencia, estacionalidad y ciclos.

#### Problematica de negocio
- Pronóstico de demanda y ventas para planificación
- Detección de patrones estacionales y tendencias
- Análisis de comportamiento temporal para optimización de recursos
- Integración de modelos clásicos y machine learning para predicción

#### Aplicaciones y casos típicos
- Pronóstico de ventas en retail
- Predicción de demanda energética o consumo de servicios
- Análisis y predicción de tráfico web o logs de sistemas
- Forecast financiero (ingresos, precios)

#### Modelado
- Componentes de series temporales:
    - Tendencia, estacionalidad, ciclos y ruido
    - Descomposición aditiva y multiplicativa

- Feature engineering temporal:
    - Rezagos (lags)
    - Ventanas móviles (rolling statistics)
    - Diferencias (differencing)
    - Variables de calendario (día de semana, mes, festivos)

- Modelos clásicos:
    - ARIMA (AutoRegressive Integrated Moving Average)
    - SARIMA (Seasonal ARIMA)
    - Requisitos del modelo
    - Evaluación de residuales, estacionariedad y autocorrelación

- Modelos modernos:
    - Prophet (Facebook) para tendencias y estacionalidad flexibles
    - Requisitos del modelo
    - Modelos basados en machine learning:
        - Regresión con características temporales (lags, rolling features)
        - XGBoost o LightGBM con características de tiempo

- Comparación:
    - Modelos estadísticos vs modelos ML
    - Precisión vs interpretabilidad
    - Escenarios de uso según tipo de serie y granularidad

#### Métricas
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- MASE (Mean Absolute Scaled Error)
- Selección de métricas según negocio (por ejemplo, errores relativos vs absolutos)

#### Python
- `pandas` para manejo de fechas, creación de lags y rolling windows
- `statsmodels` para ARIMA, SARIMA y descomposición de series
- `fbprophet` (Prophet) para modelado flexible de series
- Machine learning: `sklearn`, `xgboost`, `lightgbm` aplicados a series con features temporales
- Evaluación de modelos con métricas: MAE, MAPE, RMSE, MASE

#### Comunicacion
- Visualización de tendencias y estacionalidades
- Comparación de pronósticos vs datos reales
- Explicación del impacto de patrones estacionales y ciclos en decisiones de negocio
- Comunicación de incertidumbre y límites del modelo

#### Reto: 1 punto
- Construir un pronóstico de ventas con ARIMA y Prophet, comparar métricas de error y explicar la elección de features temporales.

#### Laboratorio
- Pipeline de series temporales:
    - Preparación de datos y creación de lags
    - Descomposición de series para análisis de tendencia y estacionalidad
    - Entrenamiento de modelos ARIMA/SARIMA
    - Entrenamiento de Prophet y/o modelo ML con features temporales
    - Evaluación de predicciones con métricas
    - Visualización de pronósticos vs valores reales

#### Anexo: Fundamento matematico y computacional
- Autocorrelación y estacionariedad
- Descomposición aditiva y multiplicativa
- Optimización de parámetros ARIMA/SARIMA (p,d,q)
- Concepto de errores y métricas escaladas
- Complejidad computacional en series temporales y modelos ML

---

## Semana 11: Modelos Complementarios

### Logro de la sesión
Conocer y aplicar algoritmos adicionales de regresión y clustering que son útiles para problemas específicos o entrevistas técnicas, comprendiendo cuándo y por qué utilizarlos.

#### Problematica de negocio
- Manejo de datos con outliers o distribuciones no normales
- Modelado de relaciones no lineales (regresión polinómica)
- Segmentación de clientes o elementos con estructuras complejas o jerárquicas
- Detección de patrones arbitrarios en datasets no etiquetados

#### Aplicaciones y casos típicos
- Regresión robusta: predicción de precios de vivienda con outliers
- SVR: problemas de regresión no lineales con pocos datos
- DBSCAN: detección de anomalías, clusters irregulares
- Clustering jerárquico: segmentación de clientes, análisis filogenético

#### Modelado
- Regresión polinómica:
    - Extensión de la regresión lineal
    - Introducción de términos polinómicos
    - Riesgo de overfitting y regularización

- Regresión robusta:
    - HuberRegressor: combina MSE y MAE para outliers
    - RANSACRegressor: ajuste iterativo ignorando outliers
    - Aplicaciones en precios extremos o datos ruidosos

- SVM para regresión (SVR):
    - Función de pérdida epsilon-insensitive
    - Kernel lineal y no lineal
    - Regularización y parámetro C

- Clustering avanzado:
    - DBSCAN: clustering basado en densidad, detección de ruido
    - Clustering jerárquico: dendrogramas, aglomerativo vs divisivo
    - Elección de parámetros y análisis de estructura jerárquica

- Métricas adicionales:
    - Clustering: Adjusted Rand Index, información mutua
    - Interpretación de silueta y Davies-Bouldin para clusters complejos

#### Métricas
- Regresión robusta: MAE, RMSE, \(R^2\)
- Clustering: Silhouette, Davies-Bouldin, Adjusted Rand Index, Mutual Information

#### Python
- Regresión robusta: `sklearn.linear_model.HuberRegressor`, `RANSACRegressor`
- SVR: `sklearn.svm.SVR