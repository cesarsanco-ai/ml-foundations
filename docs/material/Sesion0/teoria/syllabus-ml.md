# Sílabo del Curso de **MACHINE LEARNING**

## Presentación del Curso

Este curso está diseñado para futuros ingenieros y científicos donde se muestra la teoria necesaria para acompañarlo con python y sus librerias, brindando teoria necesaria para comprender el funcionamiento de los algoritmos y sus aplicaciones en la industria con Python.

## Semana 1: Introducción al ML

### Logro de la sesión
Comprender el flujo completo de un proyecto de machine learning, identificar los tipos de aprendizaje y reconocer casos de uso en la industria.

#### Conceptos
- Definición de Machine Learning, Deep Learning y su relación con la IA.
- Problemas de negocios y Tipos de aprendizaje: supervisado, no supervisado, por refuerzo.
- Metodos parametricos y no parametricos en ML.
- Historia y auge de los Modelos de ML.
- Ciclo de vida y metodologia CRISP-DM.
- Diferencias: Data engineer, Data Scientist, ML Engineer, MLOps engineer.
- Tipos de metricas en ML e interpretación de la métrica.

#### Aplicaciones de ejemplos
- Sistemas de recomendación (Netflix, Amazon).
- Detección de fraude en transacciones bancarias.
- Mantenimiento predictivo en maquinaria industrial.
- Clasificación de imágenes (diagnóstico médico, vehículos autónomos).
- Predicción de series temporales (ventas, demanda energética).

---

## Semana 2: Análisis Exploratorio y Feature Engineering

### Logro de la sesión
Realizar un análisis exploratorio y aplicar técnicas de feature engineering para preparar datos de calidad.

#### Conceptos
- Importancia del EDA
- Casos famosos de exito y fracaso en negocios
- Pipeline de EDA
- Limpieza de datos (valores faltantes, tipos de imputacion de faltantes, valores duplicados, errores de tipo de dato, tratamiento basicode outliers, inconsistencias y errores logicos)
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
- Feature Store


#### Laboratorio
- EDA
- Feature engineering


---

## Semana 3: Regresión Lineal y Regularización

### Logro de la sesión
Construir e interpretar modelos de regresión lineal múltiple, aplicando regularización (L1,L2,L1+L2) y sus metricas.

#### Modelado
- Historia
- Requisitos del modelo
- Regresión lineal simple y múltiple: formulación, interpretación de coeficientes.
- Supuestos del modelo: linealidad, independencia, homocedasticidad, normalidad de residuos.
- Regularización: Ridge (L2), Lasso (L1) y Elastic Net.
- Interpretación de la regularización y sus hiperparámetros.
- Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

#### Métricas
- Error Cuadrático Medio (MSE), Raíz del Error Cuadrático Medio (RMSE), Error Absoluto Medio (MAE), Coeficiente de determinación \(R^2\).
- Interpretacion y selección de metricas


#### Laboratorio: Ver Colab
- NTB1: Comparación de metricas con regresion multiple como baseline, regulacion L1, L2, L1+L2 (Dataset1)
- NTB2: Comparación de metricas con regresion multiple como baseline, regulacion L1, L2, L1+L2 (Dataset2)


---

## Semana 4: Regresión Logística y Balance de Datos

### Logro de la sesión
Construir, interpretar y evaluar modelos de clasificación binaria y multiclase utilizando regresión logística, incorporando técnicas de balanceo de datos, regularización y selección adecuada de métricas.

#### Modelado de clasificación
- Historia
- Requisitos del modelo
- Ejemplos clasicos de aplicacion: detección de fraude, churn de clientes, diagnóstico médico, clasificación de productos y clientes, score crediticio
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


#### Laboratorio: Ver Colab
- NTB 1 : Balanceo de datos
- NTB 2 : Regresión logística


---

## Semana 5: kNN, Naive Bayes y SVM

### Logro de la sesión
Construir, comparar e interpretar modelos de regresión y clasificación, usando los algoritmos kNN, Naive Bayes y SVM, comprendiendo sus supuestos, ventajas y limitaciones en distintos contextos de negocio.

#### Problematica de negocio
- Historia
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
    - Explicacion del modelo en clasificacion
    - Concepto de tipos de distancias
    - Elección de \(k\)
    - Sensibilidad a la escala de variables
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Naive Bayes:
    - Requisitos del modelo
    - Teorema de Bayes
    - Explicacion del modelo en regresion y clasificacion
    - Supuesto de independencia condicional
    - Variantes: Gaussian, Multinomial, Bernoulli
    - Interpretación probabilística
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Support Vector Machines (SVM):
    - Requisitos del modelo
    - Explicacion del modelo en regresion y clasificacion
    - Tipos de Kernel: lineal, polinomial, RBF
    - Hiperparámetros: \(C\) y \(\gamma\)
    - Manejo de no linealidad
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion


#### Métricas
- Métricas en clasificacion / regresion

#### Laboratorio
- NTB1: Clasificacion con KNN, Naive Bayes y SVM
- NTB2: Regresion con Naive Bayes y SVM

---

## Semana 6: Árboles de Decisión y Random Forest

### Logro de la sesión
Construir, interpretar y comparar modelos basados en árboles de decisión y ensambles tipo Random Forest, comprendiendo su capacidad para capturar relaciones no lineales y su aplicación en problemas reales de clasificación y regresión.

#### Problematica de negocio
- Historia
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

- Random Forest (Bagging en clasificación y regresión):
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

#### Laboratorio
- NTB1: Clasificacion con arboles y random forest
- NTB2: Regresion con arboles y random forest

---

## Semana 7: Gradient Boosting 

### Logro de la sesión
Construir, optimizar e interpretar modelos de Gradient Boosting (XGBoost, LightGBM y Catboost), comprendiendo su funcionamiento secuencial y ajustando hiperparámetros para maximizar el rendimiento en problemas de clasificación y regresión.

#### Problematica de negocio
- Historia
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

- Catboost:
    - Requisitos del modelo
    - Manejo automático de variables categóricas
    - Ordered Boosting
    - Plantilla base de Python: Imports, Split de datos, Modelo, Entrenamiento, Predicciones, Evaluacion

- Comparativas

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

#### Laboratorio
- NTB1: Clasificacion con XGBoost, LightGBM y Catboost
- NTB2: Regresion con XGBoost, LightGBM y Catboost

---

## Semana 8: Bias-Variance, Modelado y Validación, Optimización de hiperparámetros y Selección de Modelos

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
- Bias-Variance tradeoff:
    - Que es Underfitting y overfitting
    - Identificación, diagnóstico y Interpretación de curvas
    - Estrategias de mitigación
    - Relación con complejidad del modelo
    - Ejemplos de la vida real e interpretación

- Validación de modelos:
    - Train / Validation / Test split
    - Validación cruzada:
        - K-Fold
        - Stratified K-Fold (clasificación)
        - Leave-One-Out

#### Optimización de modelos
- Búsqueda de hiperparámetros:
    - Grid Search
    - Random Search
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


#### Laboratorio
- NTB1: Flujo completo de Clasificación
- NTB2: Flujo completo de Regresión

---

## Semana 9: Tecnicas de Clustering, PCA y t-SNE

### Logro de la sesión
Explorar y extraer valor de datos no etiquetados mediante técnicas de clustering y reducción de dimensionalidad (PCA) y t-SNE, identificando patrones, segmentos y estructuras ocultas en los datos.

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

#### Laboratorio
- NTB1: Tecnicas de PCA, t-SNE
- NTB2: Tecnicas de Clustering

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

#### Laboratorio
- NTB1: Series temporales con Prophet y modelos ML
- NTB2: Series temporales con ARIMA y SARIMA

---

## 📘 Semana 11: Sistemas de Recomendación

Diseñar e implementar sistemas de recomendación capaces de sugerir productos, contenidos o servicios personalizados, utilizando enfoques basados en popularidad, filtrado colaborativo y contenido.

---

## 💼 Problemática de negocio

* Sobrecarga de información (demasiadas opciones para el usuario)
* Necesidad de personalización
* Baja retención o engagement
* Dificultad para descubrir nuevos productos/contenidos

---

## 📊 Aplicaciones y casos típicos

* Recomendación de productos (e-commerce)
* Recomendación de contenido (streaming, noticias)
* Sistemas de ranking (top items)
* Cross-selling y up-selling
* Personalización de feeds

---

## 🧠 Modelado

### 🔹 Enfoques básicos

* **Popularidad**

  * Top-N más consumidos
  * Basado en frecuencia o rating promedio
  * Benchmark inicial

---

### 🔹 Filtrado basado en contenido

* Uso de características de ítems
* Similaridad (coseno, distancia)
* Recomendación basada en perfil del usuario
* Limitaciones:

  * No descubre nuevos intereses
  * Dependencia de features

---

### 🔹 Filtrado colaborativo

#### Basado en memoria

* User-based
* Item-based
* Matriz usuario–ítem
* Medidas de similitud (coseno, Pearson)

---

#### Basado en modelos

* Factorización de matrices (SVD)
* Representación latente de usuarios e ítems
* Predicción de ratings

---

### 🔹 Problemas comunes

* Cold start (usuarios o ítems nuevos)
* Sparsity (matrices dispersas)
* Escalabilidad

---

### 🔹 Enfoques híbridos

* Combinación de:

  * contenido + colaborativo
  * reglas de negocio

---

## 📏 Métricas

### Offline

* RMSE / MAE (predicción de ratings)
* Precision@K
* Recall@K
* MAP (Mean Average Precision)
* NDCG (ranking)

---

### Limitaciones

* No capturan comportamiento real del usuario
* Diferencia entre métricas offline vs impacto real

---

## 🐍 Python

* Librerías:

  * `surprise` (SVD, KNN para recomendadores)
  * `sklearn.metrics.pairwise` (similitud)
  * `pandas` (matriz usuario–ítem)

* Buenas prácticas:

  * Manejo de matrices dispersas
  * Normalización de ratings
  * Evaluación con train/test temporal

---

## 🧪 Laboratorio

### Pipeline de sistema de recomendación:

* Construcción de dataset:

  * Interacciones usuario–ítem
  * Ratings o eventos implícitos

* Preprocesamiento:

  * Filtrado de usuarios/ítems poco frecuentes
  * Construcción de matriz usuario–ítem

* Modelado:

  * Baseline (popularidad)
  * Filtrado colaborativo (user/item)
  * Factorización de matrices (SVD)

* Evaluación:

  * Métricas de ranking (Precision@K, Recall@K)

* Generación de recomendaciones:

  * Top-N por usuario

* Interpretación:

  * Análisis de resultados
  * Casos de éxito y errores



## 📘 Semana 12: Graph ML y Geodata


Modelar y analizar datos estructurados como grafos y datos geoespaciales, aplicando técnicas de Machine Learning para extraer relaciones, patrones espaciales y estructuras complejas en los datos.

---

## 💼 Problemática de negocio

* Relaciones complejas entre entidades (usuarios, productos, transacciones)
* Dificultad para modelar interacciones (redes)
* Necesidad de explotar información geográfica (ubicación, proximidad)
* Detección de patrones no evidentes en datos no tabulares

---

## 📊 Aplicaciones y casos típicos

### 🔹 Graph ML

* Sistemas de recomendación (user–item graph)
* Detección de fraude (redes de transacciones)
* Redes sociales (influencia, comunidades)
* Análisis de conexiones (network analysis)

---

### 🔹 Geodata

* Segmentación geográfica (clientes por ubicación)
* Optimización de rutas (logística)
* Análisis de zonas (hotspots)
* Ubicación óptima de negocios

---

## 🧠 Modelado

---

## 🔹 Representación de grafos

* Nodos (entidades) y aristas (relaciones)
* Grafos dirigidos vs no dirigidos
* Grafos ponderados
* Matriz de adyacencia

---

## 🔹 Feature Engineering en grafos

* Grado de nodo (degree)
* Centralidad (básico)
* Vecindad (neighbors)
* Conteo de conexiones

---

## 🔹 Enfoques de ML sobre grafos (sin GNN profundo)

* Transformar grafos → features tabulares
* Link prediction (conceptual)
* Node classification (conceptual)

---

## 🔹 Geodata: fundamentos

* Coordenadas geográficas (latitud, longitud)
* Distancias:

  * Euclidiana vs Haversine
* Escalamiento espacial

---

## 🔹 Técnicas sobre geodata

* Clustering geoespacial:

  * DBSCAN con coordenadas
* Feature engineering espacial:

  * Distancia a puntos de interés
  * Densidad de eventos
* Mapas de calor (heatmaps)

---

## 🔹 Problemas comunes

* Alta dimensionalidad en grafos
* Escalabilidad
* Datos dispersos
* Ruido en datos geográficos

---

## 📏 Métricas

### Graph ML (básico)

* Accuracy / F1 (clasificación de nodos)
* AUC (link prediction, conceptual)

---

### Geodata

* Métricas de clustering (silhouette, DB index)
* Evaluación visual (mapas)

---

## 🐍 Python

* Librerías:

  * `networkx` (grafos)
  * `geopandas` (datos geoespaciales)
  * `sklearn` (clustering)
  * `haversine` (distancias)

* Buenas prácticas:

  * Conversión de grafos a features
  * Escalamiento antes de clustering
  * Visualización geoespacial

---

## 🧪 Laboratorio

### Pipeline: datos relacionales y geoespaciales

---

### 🔹 Parte 1: Graph ML

* Construcción de grafo (usuarios–productos o transacciones)
* Extracción de features:

  * grado
  * número de vecinos
* Modelo simple:

  * clasificación o scoring

---

### 🔹 Parte 2: Geodata

* Dataset con coordenadas
* Clustering geoespacial (DBSCAN)
* Identificación de zonas densas

---

### 🔹 Parte 3: Integración

* Combinar features:

  * relacionales + espaciales
* Análisis de patrones

---

### 🔹 Interpretación

* Qué nodos son importantes
* Qué zonas concentran actividad
* Insights de negocio



## 📘 Semana 13: Interpretabilidad de Modelos


Comprender, analizar e interpretar el comportamiento de modelos de Machine Learning, explicando sus predicciones a nivel global y local para generar confianza, detectar sesgos y extraer insights accionables.

---

## 💼 Problemática de negocio

* Modelos tipo “caja negra” difíciles de explicar
* Falta de confianza en decisiones automatizadas
* Necesidad de justificar predicciones (regulación, negocio)
* Dificultad para identificar errores o sesgos del modelo

---

## 📊 Aplicaciones y casos típicos

* Explicación de decisiones (créditos, riesgo)
* Interpretación de variables importantes
* Auditoría de modelos
* Detección de sesgos
* Comunicación con stakeholders no técnicos

---

## 🧠 Modelado

---

## 🔹 Tipos de interpretabilidad

* Global:

  * Entender el modelo completo
* Local:

  * Explicar predicciones individuales

---

## 🔹 Modelos interpretables vs no interpretables

* Interpretables:

  * Regresión lineal
  * Árboles de decisión
* Menos interpretables:

  * Random Forest
  * Gradient Boosting

---

## 🔹 Importancia de variables

* Feature importance (model-based)
* Limitaciones:

  * Correlación entre variables
  * Interpretación incorrecta

---

## 🔹 Técnicas de interpretabilidad

### 🔸 Permutation Importance

* Medir impacto al permutar variables
* Independiente del modelo

---

### 🔸 SHAP (conceptual)

* Contribución de cada feature
* Explicaciones locales y globales
* Interpretación de valores positivos/negativos

---

### 🔸 LIME (conceptual)

* Aproximación local del modelo
* Explicación en torno a una predicción

---

## 🔹 Visualización de interpretabilidad

* Feature importance plots
* SHAP summary plots
* Dependence plots

---

## 🔹 Problemas comunes

* Interpretaciones erróneas
* Correlación vs causalidad
* Exceso de confianza en explicaciones
* Coste computacional

---

## 📏 Métricas

*(Nota: la interpretabilidad no tiene métricas estándar universales)*

Pero se pueden evaluar:

* Estabilidad de explicaciones
* Consistencia entre métodos
* Simplicidad del modelo

---

## 🐍 Python

* Librerías:

  * `sklearn.inspection` (permutation importance)
  * `shap` (SHAP values)
  * `lime` (explicaciones locales)

* Buenas prácticas:

  * Comparar múltiples métodos
  * Usar interpretabilidad post-entrenamiento
  * Validar resultados con conocimiento del dominio

---

## 🧪 Laboratorio

### Pipeline de interpretabilidad:

---

### 🔹 Entrenamiento

* Modelo supervisado (ej: Random Forest o Gradient Boosting)

---

### 🔹 Interpretabilidad global

* Feature importance
* Permutation importance
* SHAP summary plot

---

### 🔹 Interpretabilidad local

* Explicación de predicciones individuales
* Casos:

  * predicción correcta
  * predicción errónea

---

### 🔹 Análisis

* Qué variables influyen más
* Cómo cambian las predicciones
* Detección de posibles sesgos



## 📘 Semana 14: Evaluación en Producción y Ciclo de Vida del ML


Comprender cómo se evalúan y gestionan los modelos de Machine Learning en entornos reales, utilizando experimentación (A/B testing), monitoreo y estrategias de mantenimiento para asegurar su desempeño a lo largo del tiempo.

---

## 💼 Problemática de negocio

* Modelos que funcionan bien offline pero fallan en producción
* Necesidad de medir impacto real en métricas de negocio
* Cambios en los datos a lo largo del tiempo
* Degradación del rendimiento del modelo
* Falta de monitoreo y control del modelo

---

## 📊 Aplicaciones y casos típicos

* Comparación de modelos en producción (A/B testing)
* Sistemas de recomendación en plataformas reales
* Modelos de scoring (fraude, crédito)
* Personalización de productos y contenido
* Optimización continua de modelos

---

## 🧠 Modelado (conceptual)

---

## 🔹 Evaluación offline vs online

* Offline:

  * Métricas en dataset (accuracy, RMSE)
  * Limitaciones
* Online:

  * Métricas de negocio (CTR, conversión, revenue)
  * Diferencias clave

---

## 🔹 A/B Testing

* Qué es un experimento A/B
* División de usuarios (grupos A vs B)
* Comparación de modelos en producción
* Métricas online:

  * CTR (Click Through Rate)
  * Conversion rate
* Consideraciones:

  * tamaño de muestra
  * duración del experimento
  * significancia estadística (conceptual)

---

## 🔹 Tipos de experimentación

* A/B testing clásico
* Multivariate testing (mención)
* Bandits (conceptual, opcional)

---

## 🔹 Drift en Machine Learning

### 🔸 Data Drift

* Cambio en la distribución de los datos de entrada

### 🔸 Concept Drift

* Cambio en la relación entre variables y target

---

## 🔹 Monitoreo de modelos

* Seguimiento de métricas en producción
* Detección de degradación
* Alertas

---

## 🔹 Ciclo de vida del modelo

* Entrenamiento → validación → despliegue (conceptual)
* Retraining periódico
* Versionado de modelos (conceptual)
* Iteración continua

---

## 🔹 Problemas comunes

* Desalineación entre métricas offline y online
* Experimentos mal diseñados
* Falta de datos en producción
* Retraining ineficiente

---

## 📏 Métricas

### Online

* CTR
* Conversion rate
* Retención
* Revenue (según caso)

---

### Evaluación del experimento

* Diferencia entre grupos (A vs B)
* Intervalos de confianza (conceptual)

---

## 🐍 Python (ligero, conceptual)

* Simulación de A/B testing:

  * `numpy`, `pandas`

* Visualización de resultados:

  * `matplotlib`, `seaborn`

* Buenas prácticas:

  * Separar evaluación offline vs online
  * No confiar solo en métricas tradicionales

---

## 🧪 Laboratorio

### Simulación de evaluación en producción

---

### 🔹 Paso 1: Modelo base

* Modelo entrenado previamente (ej: clasificación o recomendador)

---

### 🔹 Paso 2: Simulación A/B

* Dividir datos en grupo A y B
* Simular comportamiento de usuarios
* Comparar métricas (ej: tasa de conversión)

---

### 🔹 Paso 3: Análisis

* Evaluar si el modelo B mejora al A
* Interpretar resultados

---

### 🔹 Paso 4: Simulación de drift

* Introducir cambio en los datos
* Evaluar degradación del modelo

---

### 🔹 Paso 5: Decisiones

* ¿Reentrenar?
* ¿Mantener modelo?
* ¿Cambiar estrategia?

