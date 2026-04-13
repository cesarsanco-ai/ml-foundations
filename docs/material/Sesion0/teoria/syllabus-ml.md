# Sílabo del Curso de **MACHINE LEARNING**

## Presentación del Curso

Este curso está diseñado para futuros ingenieros y científicos donde se muestra la **teoría** necesaria para acompañarlo con Python y sus **librerías**, brindando fundamentos para comprender el funcionamiento de los algoritmos y sus aplicaciones en la industria.

El programa consta de **14 sesiones (semanas)**. Cada sesión sigue la misma estructura: **logro de la sesión**, bloques de contenido (`####`) y, al final, **laboratorio** con dos notebooks (**NTB 1** y **NTB 2**) en Google Colab salvo indicación distinta en clase.

La guía teórica unificada (estructura tipo **Sesión $n$**, *Marco teórico*, tablas y referencias) vive en `5_ML/docs/material/SesionNN/teoria/SNN-ML.md` para $n=1,\ldots,14$, alineada a las secciones de este sílabo.

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

### Laboratorio
- **NTB 1 —** repaso de numpy y pandas
- **NTB 2 —** repaso de matplotlib y seaborn

---

## Semana 2: Análisis Exploratorio y Feature Engineering

### Logro de la sesión
Realizar un análisis exploratorio y aplicar técnicas de feature engineering para preparar datos de calidad.

#### Conceptos
- Importancia del EDA
- Casos famosos de exito y fracaso en negocios
- Pipeline de EDA
- Limpieza de datos (valores faltantes; criterios de imputación: media, mediana, moda, constante/indicador, por grupos, series temporales; KNN y encadenada/MICE; valores duplicados; errores de tipo de dato; tratamiento básico de outliers; inconsistencias y errores lógicos)
- Analisis univariado (Variables numericas: medidas de tendencia central, medidas de dispersion, medidas de forma, tipos de visualizaciones; Variables categoricas: frecuencias, moda, proporciones, tipos de visualizaciones; Variables de Fecha: Componentes de fechas, tipos de visualizaciones)
- Analisis bivariado (Variables numericas vs numericas, categoricas vs categoricas, numericas vs categoricas)
- Analisis multivariado (Matrices de correlacion, Pairplots, Heatmaps)
- Importancia del Feature engineering
- Pipeline de feature engineering
- Creación de nuevas características (a partir de numericas, categoricas, fechas, agregaciones, texto,etc)
- Codificación de variables categoricas (One-Hot Encoding, Label Encoding, Target Encoding, Binary Encoding, Hashing Encoding)
- Escalado y Normalizacion (Z-score, Min-max, RobustScaler, Normalizer)
- Manejo de datos desbalanceados (Oversampling, Undersampling, SMOTE)
- Imputación de datos (cuándo usar media, mediana, moda; indicadores de ausencia; KNNImputer, IterativeImputer)
- Manejo de outliers (variable indicadora de outlier, winsorizacion, transformaciones para reducir impacto)
- Selección de características (Filtros, Wrappers, Embebidos)
- Variables derivadas del negocio (ratios, agregaciones temporales, interacciones)
- Feature Store (consistencia train/serving, reutilización, punto en el tiempo; visión general de herramientas)

### Laboratorio
- **NTB 1 —** Análisis exploratorio de datos (EDA): univariado, bivariado y calidad de datos.
- **NTB 2 —** Feature engineering: codificación, escalado, desbalanceo e imputación.

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

### Laboratorio
- **NTB 1 —** Regresión múltiple frente a Ridge, Lasso y Elastic Net: métricas y comparación — *dataset 1*.
- **NTB 2 —** Regresión múltiple frente a Ridge, Lasso y Elastic Net: métricas y comparación — *dataset 2*.

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

### Laboratorio
- **NTB 1 —** Balanceo de clases (oversampling, undersampling, pesos) y efecto en métricas.
- **NTB 2 —** Regresión logística: entrenamiento, umbral, regularización y evaluación.

---

## Semana 5: kNN, Naive Bayes y SVM

### Logro de la sesión
Construir, comparar e interpretar modelos de regresión y clasificación, usando los algoritmos kNN, Naive Bayes y SVM, comprendiendo sus supuestos, ventajas y limitaciones en distintos contextos de negocio.

#### Problemática de negocio
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

### Laboratorio
- **NTB 1 —** Clasificación con kNN, Naive Bayes y SVM (comparación y ajuste).
- **NTB 2 —** Regresión con Naive Bayes y SVM.

---

## Semana 6: Árboles de Decisión y Random Forest

### Logro de la sesión
Construir, interpretar y comparar modelos basados en árboles de decisión y ensambles tipo Random Forest, comprendiendo su capacidad para capturar relaciones no lineales y su aplicación en problemas reales de clasificación y regresión.

#### Problemática de negocio
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

### Laboratorio
- **NTB 1 —** Clasificación con árboles de decisión y Random Forest.
- **NTB 2 —** Regresión con árboles de decisión y Random Forest (importancia de variables).

---

## Semana 7: Gradient Boosting

### Logro de la sesión
Construir, optimizar e interpretar modelos de Gradient Boosting (XGBoost, LightGBM y Catboost), comprendiendo su funcionamiento secuencial y ajustando hiperparámetros para maximizar el rendimiento en problemas de clasificación y regresión.

#### Problemática de negocio
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

### Laboratorio
- **NTB 1 —** Clasificación con XGBoost, LightGBM y CatBoost (hiperparámetros y validación).
- **NTB 2 —** Regresión con XGBoost, LightGBM y CatBoost.

---

## Semana 8: Bias-Variance, Modelado y Validación, Optimización de hiperparámetros y Selección de Modelos

### Logro de la sesión
Evaluar, validar y seleccionar modelos de machine learning de forma robusta, asegurando su capacidad de generalización mediante técnicas de validación cruzada y optimización de hiperparámetros.

#### Problemática de negocio
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

### Laboratorio
- **NTB 1 —** Flujo completo de clasificación: validación cruzada, búsqueda de hiperparámetros y selección de modelo.
- **NTB 2 —** Flujo completo de regresión: curvas de aprendizaje, bias-variance y comparación de modelos.

---

## Semana 9: Técnicas de Clustering, PCA y t-SNE

### Logro de la sesión
Explorar y extraer valor de datos no etiquetados mediante técnicas de clustering y reducción de dimensionalidad (PCA) y t-SNE, identificando patrones, segmentos y estructuras ocultas en los datos.

#### Problemática de negocio
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

### Laboratorio
- **NTB 1 —** PCA y t-SNE: reducción de dimensionalidad y visualización de estructura.
- **NTB 2 —** Clustering: K-Means, DBSCAN y/o jerárquico con métricas de cohesión y silueta.

---

## Semana 10: Series de Tiempo

### Logro de la sesión
Modelar y pronosticar datos temporales utilizando técnicas estadísticas clásicas, machine learning y librerías modernas, extrayendo patrones de tendencia, estacionalidad y ciclos.

#### Problemática de negocio
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

### Laboratorio
- **NTB 1 —** Series temporales con Prophet y features + modelos ML (XGBoost/LightGBM).
- **NTB 2 —** Series temporales con ARIMA/SARIMA y diagnóstico de residuos.

---

## Semana 11: Sistemas de recomendación

### Logro de la sesión
Diseñar e implementar sistemas de recomendación capaces de sugerir productos, contenidos o servicios personalizados, utilizando enfoques basados en popularidad, filtrado colaborativo y contenido.

#### Problemática de negocio
- Sobrecarga de información (demasiadas opciones para el usuario).
- Necesidad de personalización.
- Baja retención o engagement.
- Dificultad para descubrir nuevos productos o contenidos.

#### Aplicaciones y casos típicos
- Recomendación de productos (e-commerce) y de contenido (streaming, noticias).
- Sistemas de ranking (top items), cross-selling y up-selling.
- Personalización de feeds.

#### Modelado
- **Enfoques básicos:** popularidad (top-N, frecuencia, rating medio) como benchmark.
- **Filtrado basado en contenido:** características de ítems, similaridad (coseno, distancias), perfil de usuario; limitaciones (cold discovery, dependencia de features).
- **Filtrado colaborativo**
  - *Basado en memoria:* user-based, item-based, matriz usuario–ítem, similitud (coseno, Pearson).
  - *Basado en modelos:* factorización de matrices (SVD), representaciones latentes, predicción de ratings.
- **Problemas comunes:** cold start, sparsity, escalabilidad.
- **Enfoques híbridos:** contenido + colaborativo, reglas de negocio.

#### Métricas
- **Offline:** RMSE / MAE (ratings), Precision@K, Recall@K, MAP, NDCG.
- **Limitaciones:** brecha entre métricas offline y comportamiento real del usuario.

#### Python
- Librerías: `surprise` (SVD, KNN), `sklearn.metrics.pairwise`, `pandas` (matriz usuario–ítem).
- Buenas prácticas: matrices dispersas, normalización de ratings, evaluación con partición temporal train/test.

### Laboratorio
- **NTB 1 —** Construcción de matriz usuario–ítem, baseline de popularidad, filtrado colaborativo user/item y evaluación con Precision@K y Recall@K.
- **NTB 2 —** Factorización (SVD), generación de top-N por usuario e interpretación de aciertos y errores (opcional: hincapié en contenido o reglas de negocio).

---

## Semana 12: Graph ML y geodata

### Logro de la sesión
Modelar y analizar datos estructurados como grafos y datos geoespaciales, aplicando machine learning para extraer relaciones, patrones espaciales y estructuras complejas.

#### Problemática de negocio
- Relaciones entre entidades (usuarios, productos, transacciones) y redes difíciles de tabularizar.
- Explotar información geográfica (ubicación, proximidad).
- Patrones no evidentes en datos no puramente tabulares.

#### Aplicaciones y casos típicos
- **Graph ML:** recomendación (grafo usuario–ítem), fraude (red de transacciones), redes sociales, análisis de conexiones.
- **Geodata:** segmentación geográfica, rutas y logística, hotspots, ubicación de negocios.

#### Modelado
- **Grafos:** nodos y aristas, dirigidos/no dirigidos, ponderados, matriz de adyacencia.
- **Features en grafos:** grado, centralidad básica, vecindad, conteos de conexiones.
- **ML sobre grafos (sin GNN profundo):** grafo → features tabulares; link prediction y clasificación de nodos (nivel conceptual).
- **Geodata:** coordenadas, distancia euclidiana vs Haversine, escalamiento espacial.
- **Técnicas:** clustering geoespacial (p. ej. DBSCAN con coordenadas), features de distancia a POI, densidad, heatmaps.
- **Problemas comunes:** dimensionalidad, escalabilidad, dispersión, ruido geográfico.

#### Métricas
- Graph ML (básico): accuracy / F1 en nodos, AUC en link prediction (conceptual).
- Geodata: silueta, índice de Davies–Bouldin, evaluación visual en mapas.

#### Python
- `networkx`, `geopandas`, `sklearn` (clustering), `haversine` u otras distancias.
- Buenas prácticas: conversión grafo→features, escalado antes de clustering, visualización geoespacial.

### Laboratorio
- **NTB 1 —** Graph ML: construir un grafo (p. ej. usuarios–productos), extraer features (grado, vecinos), entrenar un modelo de clasificación o scoring.
- **NTB 2 —** Geodata: dataset con coordenadas, clustering espacial (DBSCAN), zonas densas; integración opcional de features relacionales y espaciales.

---

## Semana 13: Interpretabilidad de modelos

### Logro de la sesión
Comprender, analizar e interpretar el comportamiento de modelos de machine learning, explicando predicciones a nivel global y local para generar confianza, detectar sesgos y extraer insights accionables.

#### Problemática de negocio
- Modelos “caja negra”, falta de confianza, necesidad de justificar predicciones (regulación, negocio), detección de errores y sesgos.

#### Aplicaciones y casos típicos
- Explicación en crédito y riesgo, auditoría de modelos, comunicación con stakeholders no técnicos.

#### Modelado
- **Tipos:** interpretabilidad global vs local.
- **Modelos interpretables vs opacos:** regresión, árboles frente a Random Forest o boosting.
- **Importancia de variables:** feature importance intrínseco; limitaciones (correlación, interpretaciones erróneas).
- **Técnicas:** permutation importance; SHAP (contribuciones locales/globales); LIME (aproximación local).
- **Visualización:** importancia, SHAP summary, dependence plots.
- **Problemas comunes:** correlación vs causalidad, coste computacional, exceso de confianza en explicaciones.

#### Métricas
- La interpretabilidad no tiene métricas universales; se valora estabilidad de explicaciones, consistencia entre métodos y simplicidad del modelo.

#### Python
- `sklearn.inspection` (permutation importance), `shap`, `lime`.
- Buenas prácticas: combinar métodos, interpretabilidad post-entrenamiento, validar con dominio.

### Laboratorio
- **NTB 1 —** Interpretabilidad global: entrenar un modelo supervisado (p. ej. Random Forest o boosting), feature importance, permutation importance, SHAP summary.
- **NTB 2 —** Interpretabilidad local: explicar predicciones individuales (SHAP/LIME), casos acertados vs erróneos y lectura de posibles sesgos.

---

## Semana 14: Evaluación en producción y ciclo de vida del ML

### Logro de la sesión
Comprender cómo se evalúan y gestionan los modelos de machine learning en entornos reales, mediante experimentación (A/B testing), monitoreo y mantenimiento para asegurar el desempeño en el tiempo.

#### Problemática de negocio
- Buen rendimiento offline y malo en producción; medir impacto en negocio; drift y degradación; monitoreo insuficiente.

#### Aplicaciones y casos típicos
- A/B testing en producción, recomendadores, scoring (fraude, crédito), personalización, mejora continua.

#### Modelado (conceptual)
- **Offline vs online:** métricas clásicas en dataset frente a CTR, conversión, revenue.
- **A/B testing:** diseño, grupos A/B, métricas online, tamaño de muestra, duración, significancia (conceptual).
- **Experimentación:** multivariante (mención), bandits (opcional).
- **Drift:** data drift (entrada) y concept drift (relación X–y).
- **Monitoreo:** métricas en producción, degradación, alertas.
- **Ciclo de vida:** entrenamiento, validación, despliegue, retraining, versionado, iteración.
- **Problemas comunes:** desalineación offline/online, experimentos mal diseñados, retraining ineficiente.

#### Métricas
- Online: CTR, conversión, retención, revenue (según caso).
- Experimento: diferencia A vs B, intervalos de confianza (conceptual).

#### Python (ligero, conceptual)
- Simulación con `numpy`, `pandas`; visualización con `matplotlib`, `seaborn`.
- Buenas prácticas: separar evaluación offline y online.

### Laboratorio
- **NTB 1 —** Simulación de A/B: modelo o política base, división A/B, métricas de conversión o CTR y conclusión sobre el ganador.
- **NTB 2 —** Drift y decisiones: simular cambio en los datos, medir degradación del modelo y plantear reentrenamiento o mantenimiento.

