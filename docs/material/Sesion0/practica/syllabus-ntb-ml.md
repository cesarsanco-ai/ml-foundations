# Curso de Machine Learning
## Syllabus de Notebooks por Semana

**Carlos César Sánchez Coronel**

2026

---

## Introducción

Cada semana se trabajará con dos tipos de notebooks:

- **Notebook Conceptual (NB1):** Enfocado en la comprensión profunda. Se utilizan datos *dummy* generados en el momento para operaciones matemáticas, visualización de ecuaciones, implementación de algoritmos desde cero (cuando sea pertinente) y experimentación con los parámetros de los modelos de scikit-learn. Aquí se exploran conceptos como complejidad computacional, efecto de hiperparámetros, representaciones gráficas de funciones de coste, etc.
- **Notebook de Ejercicios (NB2):** Aplicación práctica sobre conjuntos de datos reales (de scikit-learn, Kaggle, UCI, etc.). Se plantean problemas concretos de regresión, clasificación, clustering, etc., donde el estudiante debe aplicar lo aprendido, evaluar métricas, comparar modelos y justificar decisiones.

---

## Semana 1: Herramientas y Fundamentos Matemáticos

### Propósito
Familiarizar al estudiante con el entorno de trabajo (NumPy, Pandas, Matplotlib, Seaborn) y con las funciones matemáticas que aparecerán a lo largo del curso. Se busca que el alumno gane soltura en la manipulación de arrays, operaciones vectoriales y visualización, y que reconozca gráficamente las funciones clave de machine learning.

### Notebook Único (NB1) – Herramientas y Ecuaciones

- **Datos:** Datos sintéticos generados con NumPy (secuencias lineales, muestras aleatorias, etc.).
- **Actividades:**
  1. Repaso de NumPy: creación de arrays, indexado, operaciones elemento a elemento, broadcasting.
  2. Cálculo de normas (L1, L2) y distancias (euclidiana, Manhattan) entre vectores.
  3. Multiplicación de matrices y resolución de sistemas lineales sencillos.
  4. Visualización con Matplotlib: curvas de funciones (lineal, cuadrática, exponencial, logarítmica, sigmoide, ReLU, tanh, etc.).
  5. Representación de la función de pérdida MSE en 2D y 3D (variando coeficientes).
  6. Gráficas de distribuciones (histogramas, boxplots) con Seaborn sobre datos simulados.
  7. Ejemplo práctico: generar datos con ruido y ajustar una recta "a ojo" variando pendiente e intercepto.
- **Conceptos matemáticos:** Álgebra lineal básica, cálculo de gradientes (nociones), funciones convexas.
- **Herramientas:** NumPy, Matplotlib, Seaborn.

---

## Semana 2: Análisis Exploratorio y Feature Engineering

### Propósito
Aprender a explorar, limpiar y transformar datos reales. Se introduce el concepto de pipeline y transformadores de scikit-learn, sentando las bases para los modelos.

### Notebook Conceptual (NB1) – Manipulación de Datos Dummy

- **Datos:** DataFrames creados con Pandas a partir de diccionarios, incluyendo valores nulos, outliers y variables categóricas.
- **Actividades:**
  1. Generar un dataset dummy con variables numéricas, categóricas y nulos.
  2. Aplicar estadísticas descriptivas y visualizaciones (histogramas, boxplots, countplots).
  3. Detectar outliers usando el rango intercuartílico.
  4. Imputar valores nulos con SimpleImputer (media, mediana, moda).
  5. Codificar variables categóricas con OneHotEncoder y LabelEncoder.
  6. Escalar variables con StandardScaler y MinMaxScaler.
  7. Construir un pipeline con ColumnTransformer para preprocesar datos mixtos.
- **Conceptos:** Correlación de Pearson y Spearman, sesgo en variables, importancia de la escala.
- **Herramientas:** Pandas, Seaborn, scikit-learn (preprocessing, pipeline).

### Notebook de Ejercicios (NB2) – EDA sobre Dataset Real

- **Dataset:** "House Prices" (Kaggle) o "Titanic" (Kaggle).
- **Actividades:**
  1. Cargar datos y realizar un EDA completo: estadísticos, detección de nulos, análisis de correlaciones, visualizaciones multivariadas.
  2. Crear nuevas características (feature engineering) como: superficie por habitación, interacciones, variables de tipo de cambio.
  3. Aplicar transformaciones (log, Box-Cox) para normalizar distribuciones.
  4. Construir un pipeline de preprocesamiento y guardarlo.
- **Entrega:** Notebook con el análisis y las transformaciones justificadas.

---

## Semana 3: Regresión Lineal y Regularización

### Propósito
Comprender la regresión lineal desde sus fundamentos matemáticos, implementarla manualmente, y luego usar scikit-learn explorando el efecto de la regularización.

### Notebook Conceptual (NB1) – Experimentación con Datos Dummy

- **Datos:** Datos sintéticos univariados y multivariados con relación lineal más ruido gaussiano.
- **Actividades:**
  1. Implementar la solución de la ecuación normal: $\hat{\beta} = (X^TX)^{-1}X^Ty$.
  2. Calcular el error cuadrático medio (MSE) y visualizar la recta ajustada.
  3. Derivar el gradiente del MSE y dar un paso de gradiente descendente manualmente.
  4. Usar LinearRegression de scikit-learn y comparar coeficientes.
  5. Añadir regularización Ridge (L2) y Lasso (L1): variar $\alpha$ y observar cómo cambian los coeficientes (mostrar la trayectoria de regularización).
  6. Visualizar la función de coste con y sin regularización en 2D (para dos coeficientes).
  7. Medir tiempos de ejecución de la ecuación normal vs gradiente descendente para diferentes tamaños de datos.
- **Conceptos:** Normal equations, complejidad $O(n^3)$, bias-variance tradeoff, interpretación de coeficientes.
- **Herramientas:** NumPy, scikit-learn (LinearRegression, Ridge, Lasso), timeit.

### Notebook de Ejercicios (NB2) – Predicción de Precios de Viviendas

- **Dataset:** California Housing (sklearn) o Ames Housing (Kaggle).
- **Actividades:**
  1. Dividir en train/test.
  2. Entrenar regresión lineal y evaluar con RMSE, MAE, $R^2$.
  3. Probar Ridge, Lasso y ElasticNet con validación cruzada (GridSearchCV).
  4. Analizar los coeficientes resultantes y discutir la selección de características (en Lasso).
  5. Comparar el rendimiento de los modelos regularizados vs el lineal simple.

---

## Semana 4: Regresión Logística y Métricas de Clasificación

### Propósito
Entender la regresión logística como modelo de clasificación, la función sigmoide, la log-loss, y aprender a evaluar clasificadores con métricas adecuadas.

### Notebook Conceptual (NB1) – Datos Dummy y Experimentación

- **Datos:** Datos sintéticos para clasificación binaria (dos clases separables linealmente y no linealmente).
- **Actividades:**
  1. Visualizar la función sigmoide y su derivada.
  2. Implementar la log-loss y su gradiente; comprobar con un ejemplo pequeño.
  3. Entrenar LogisticRegression de scikit-learn variando C (inverso de regularización) y observar el cambio en la frontera de decisión.
  4. Visualizar la matriz de confusión y calcular precisión, recall, F1-score manualmente y con sklearn.
  5. Dibujar la curva ROC y calcular AUC para diferentes umbrales.
  6. Experimentar con clases desbalanceadas: usar class_weight y comparar con sobremuestreo (SMOTE con imbalanced-learn).
- **Conceptos:** Odds, log-odds, entropía cruzada, sensibilidad/especificidad, AUC.
- **Herramientas:** sklearn.linear_model, sklearn.metrics, imbalanced-learn.

### Notebook de Ejercicios (NB2) – Detección de Spam

- **Dataset:** SMS Spam Collection (UCI) o correos de spam.
- **Actividades:**
  1. Convertir texto a características (CountVectorizer, TF-IDF).
  2. Entrenar regresión logística y evaluar con precisión, recall, F1.
  3. Analizar la matriz de confusión y decidir qué métrica es más importante para el negocio.
  4. Probar diferentes valores de C y umbral de decisión.
  5. (Opcional) Aplicar SMOTE y ver impacto.

---

## Semana 5: KNN, Naive Bayes y SVM

### Propósito
Conocer tres algoritmos clásicos de clasificación, sus fundamentos y cuándo aplicarlos. En el notebook conceptual se profundiza en la geometría del espacio de características y el efecto de hiperparámetros.

### Notebook Conceptual (NB1) – Experimentación

- **Datos:** Datos sintéticos en 2D (lunas, círculos, linealmente separables).
- **Actividades:**
  1. **KNN:** Implementar manualmente la distancia euclidiana y el voto mayoritario. Probar distintos k y visualizar la frontera de decisión. Mostrar cómo cambia el tiempo de inferencia con el tamaño del dataset.
  2. **Naive Bayes:** Para GaussianNB, visualizar las distribuciones condicionales estimadas (campanas de Gauss) sobre los datos. Probar con datos donde la independencia condicional se viola y observar el efecto.
  3. **SVM:** Visualizar el margen máximo y los vectores soporte con kernel lineal. Probar kernel RBF y variar C y gamma, viendo cómo se modifica la frontera. Comparar con SVM lineal.
  4. Medir complejidad computacional: tiempo de entrenamiento y predicción para diferentes tamaños.
- **Conceptos:** Distancias, teorema de Bayes, margen máximo, kernel trick.
- **Herramientas:** sklearn.neighbors, sklearn.naive_bayes, sklearn.svm, matplotlib.

### Notebook de Ejercicios (NB2) – Clasificación de Dígitos y Textos

- **Datasets:** Digits (sklearn) para KNN y SVM; 20 Newsgroups para Naive Bayes.
- **Actividades:**
  1. Clasificar dígitos con KNN: encontrar el mejor k con validación cruzada.
  2. Clasificar textos con MultinomialNB y comparar accuracy.
  3. Usar SVM con kernel RBF en dígitos y optimizar C y gamma con GridSearchCV.
  4. Comparar tiempos de inferencia entre KNN y SVM.

---

## Semana 6: Árboles de Decisión y Random Forest

### Propósito
Entender cómo funcionan los árboles de decisión, su interpretabilidad y el ensamble bagging (Random Forest) para reducir varianza.

### Notebook Conceptual (NB1) – Construcción Manual y Visualización

- **Datos:** Datos sintéticos en 2D (por ejemplo, dos variables, tres clases).
- **Actividades:**
  1. Implementar un árbol de decisión recursivo (con profundidad limitada) usando el índice de Gini.
  2. Visualizar las particiones del espacio y el árbol resultante.
  3. Usar DecisionTreeClassifier de sklearn y variar max_depth, min_samples_split; observar sobreajuste.
  4. Visualizar la importancia de características (Gini importance) en un dataset dummy con variables ruidosas.
  5. Construir un Random Forest con pocos árboles y visualizar las fronteras de cada árbol y la combinada.
  6. Comparar el error OOB (Out-of-Bag) con validación cruzada.
- **Conceptos:** Entropía, ganancia de información, bagging, muestreo bootstrap.
- **Herramientas:** sklearn.tree, sklearn.ensemble, graphviz (opcional).

### Notebook de Ejercicios (NB2) – Riesgo Crediticio

- **Dataset:** German Credit o Give Me Some Credit (Kaggle).
- **Actividades:**
  1. Entrenar un árbol de decisión y visualizarlo (interpretabilidad).
  2. Entrenar Random Forest y comparar con el árbol simple en términos de AUC.
  3. Analizar la importancia de características y discutir implicaciones de negocio.
  4. Ajustar hiperparámetros (n_estimators, max_features) con GridSearchCV.

---

## Semana 7: Gradient Boosting (XGBoost, LightGBM)

### Propósito
Introducir el concepto de boosting secuencial y las implementaciones modernas eficientes, mostrando cómo ajustan errores residuales.

### Notebook Conceptual (NB1) – Boosting desde Cero (sencillo) y Experimentación

- **Datos:** Datos sintéticos de regresión (una variable) para visualizar la mejora iterativa.
- **Actividades:**
  1. Implementar un gradient boosting muy simple: en cada iteración ajustar un árbol pequeño a los residuos y actualizar la predicción.
  2. Visualizar cómo las predicciones se acercan al valor real a medida que se añaden árboles.
  3. Usar XGBoost o LightGBM con datos dummy y variar learning_rate, n_estimators, max_depth; mostrar curvas de entrenamiento y validación.
  4. Observar early stopping en acción.
  5. Comparar tiempos de entrenamiento entre XGBoost y Random Forest para un dataset mediano.
- **Conceptos:** Boosting, shrinkage, regularización en XGBoost.
- **Herramientas:** xgboost, lightgbm, sklearn.metrics.

### Notebook de Ejercicios (NB2) – Predicción de Clics o Competencia Kaggle

- **Dataset:** Avazu (submuestra) o Porto Seguro (Kaggle).
- **Actividades:**
  1. Preprocesar datos (codificación, manejo de nulos).
  2. Entrenar XGBoost y LightGBM con validación cruzada.
  3. Optimizar hiperparámetros (learning_rate, max_depth, subsample, etc.) usando RandomizedSearchCV.
  4. Evaluar con log-loss o AUC y comparar con modelos anteriores (Random Forest, etc.).

---

## Semana 8: Evaluación y Validación de Modelos

### Propósito
Consolidar las técnicas de evaluación, diagnóstico de overfitting y selección de hiperparámetros. Se trabaja con modelos ya conocidos.

### Notebook Conceptual (NB1) – Curvas de Aprendizaje y Validación Cruzada

- **Datos:** Datos sintéticos con ruido (por ejemplo, función seno con ruido).
- **Actividades:**
  1. Ajustar modelos de distinta complejidad (lineal, polinómico, árbol profundo) y dibujar curvas de aprendizaje (tamaño de entrenamiento vs error) para detectar high bias o high variance.
  2. Implementar manualmente k-fold cross-validation y comparar con cross_val_score.
  3. Visualizar el bias-variance tradeoff con modelos polinómicos de diferente grado.
  4. Usar GridSearchCV y RandomizedSearchCV en un modelo (por ejemplo, Random Forest) y comparar resultados.
  5. Analizar la importancia de la estratificación en clasificación.
- **Conceptos:** Underfitting, overfitting, curvas de aprendizaje, sesgo-varianza.
- **Herramientas:** sklearn.model_selection (learning_curve, validation_curve, GridSearchCV).

### Notebook de Ejercicios (NB2) – Comparación de Modelos en un Problema Real

- **Dataset:** Churn prediction (Telco) o cualquier dataset de clasificación.
- **Actividades:**
  1. Evaluar varios modelos (regresión logística, SVM, Random Forest, XGBoost) con validación cruzada.
  2. Dibujar curvas ROC y comparar AUC.
  3. Seleccionar los mejores hiperparámetros para cada uno.
  4. Elegir el modelo final justificando con métricas y consideraciones de negocio.

---

## Semana 9: Aprendizaje No Supervisado: Clustering y PCA

### Propósito
Introducir técnicas para datos no etiquetados: reducción de dimensionalidad (PCA) y clustering (K-Means, DBSCAN).

### Notebook Conceptual (NB1) – Visualización de Componentes y Clusters

- **Datos:** Datos sintéticos en 2D y 3D (por ejemplo, blobs, círculos concéntricos, moons).
- **Actividades:**
  1. **PCA:** Calcular autovalores y autovectores de la matriz de covarianza de los datos (centrados). Proyectar los datos y visualizar la varianza explicada.
  2. **K-Means:** Implementar el algoritmo paso a paso (inicialización aleatoria, asignación, actualización) y visualizar la evolución de centroides. Mostrar el método del codo (inercia vs k) y el silhouette score.
  3. **DBSCAN:** Visualizar el efecto de eps y minPts en la formación de clusters y detección de ruido. Comparar con K-Means en datos con formas no esféricas.
- **Conceptos:** Varianza, componentes principales, inercia, densidad.
- **Herramientas:** sklearn.decomposition, sklearn.cluster, sklearn.metrics.silhouette_score.

### Notebook de Ejercicios (NB2) – Segmentación de Clientes

- **Dataset:** Mall Customers (Kaggle) o datos de compras.
- **Actividades:**
  1. Aplicar K-Means para segmentar clientes; elegir k óptimo.
  2. Visualizar los segmentos en 2D usando PCA.
  3. Interpretar los perfiles de cada cluster (media de cada variable).
  4. Probar DBSCAN y comparar resultados (posibles outliers).

---

## Semana 10: Series de Tiempo

### Propósito
Modelar datos temporales, descomponer series y aplicar tanto modelos clásicos (ARIMA) como ML con características temporales.

### Notebook Conceptual (NB1) – Descomposición y Features Temporales

- **Datos:** Serie temporal sintética (tendencia + estacionalidad + ruido) o serie real pequeña (como pasajeros de avión).
- **Actividades:**
  1. Descomposición aditiva y multiplicativa con statsmodels.
  2. Creación de lags y ventanas móviles usando shift y rolling.
  3. Visualizar autocorrelación (ACF, PACF).
  4. Ajustar un modelo ARIMA manualmente y evaluar.
  5. Convertir la serie en problema supervisado: crear X con lags y usar regresión (por ejemplo, Random Forest) para predecir.
- **Conceptos:** Estacionariedad, diferenciación, ACF/PACF.
- **Herramientas:** pandas, statsmodels.tsa, sklearn.

### Notebook de Ejercicios (NB2) – Predicción de Ventas Diarias

- **Dataset:** Store Sales (Kaggle) o datos de demanda.
- **Actividades:**
  1. Análisis exploratorio de la serie.
  2. Feature engineering temporal (día de semana, mes, festivos, lags).
  3. Entrenar XGBoost con características temporales y evaluar con MAPE.
  4. Comparar con un modelo ARIMA/SARIMA.

---

## Semana 11: Modelos Complementarios (Regresión Robusta, SVR, Clustering Jerárquico)

### Propósito
Cubrir técnicas adicionales que pueden ser útiles en situaciones específicas, como outliers o estructuras jerárquicas.

### Notebook Conceptual (NB1) – Comparación en Datos con Outliers

- **Datos:** Datos sintéticos con outliers añadidos.
- **Actividades:**
  1. Comparar regresión lineal, HuberRegressor y RANSAC en presencia de outliers. Visualizar las rectas ajustadas.
  2. Usar SVR con kernel RBF y comparar con regresión lineal en datos no lineales.
  3. Clustering jerárquico: generar dendrograma y comparar cortes con K-Means.
- **Conceptos:** Robustez, funciones de pérdida robustas, dendrogramas.
- **Herramientas:** sklearn.linear_model.HuberRegressor, RANSACRegressor, sklearn.svm.SVR, sklearn.cluster.AgglomerativeClustering, scipy.cluster.hierarchy.

### Notebook de Ejercicios (NB2) – Aplicación en Datos Atípicos

- **Dataset:** Precios de viviendas con outliers (o dataset de calidad de vinos con datos ruidosos).
- **Actividades:**
  1. Detectar outliers y aplicar regresión robusta.
  2. Evaluar la mejora en RMSE frente a regresión lineal.
  3. (Opcional) Usar clustering jerárquico para segmentar vinos.

---

## Semana 12: Sistemas de Recomendación

### Propósito
Implementar filtrado colaborativo y factorización matricial, entender el problema de cold start y evaluar con métricas top-k.

### Notebook Conceptual (NB1) – Factorización Matricial Manual

- **Datos:** Matriz usuario-ítem pequeña (generada con valores aleatorios o ejemplo clásico de ratings).
- **Actividades:**
  1. Implementar SVD (descomposición en valores singulares) con numpy.linalg.svd y reconstruir la matriz.
  2. Calcular predicciones de ratings y comparar con original.
  3. Usar la librería `surprise` para cargar un dataset pequeño y probar SVD.
  4. Variar el número de factores latentes y observar el error de reconstrucción.
  5. Calcular precisión@k y recall@k manualmente para un usuario.
- **Conceptos:** Factorización matricial, sesgo usuario/ítem, métricas top-k.
- **Herramientas:** numpy, surprise (opcional), sklearn.metrics (para RMSE).

### Notebook de Ejercicios (NB2) – Recomendación de Películas (MovieLens)

- **Dataset:** MovieLens 100k.
- **Actividades:**
  1. Explorar los datos (distribución de ratings, usuarios, películas).
  2. Implementar filtrado colaborativo basado en ítems (usando similitud de coseno) y hacer recomendaciones para un usuario.
  3. Entrenar SVD con surprise y evaluar con RMSE en validación cruzada.
  4. Simular un escenario de cold start: recomendar a un nuevo usuario usando popularidad o metadata.

---

## Semana 13: Interpretabilidad de Modelos (SHAP, LIME, PDP)

### Propósito
Aprender a explicar modelos complejos, tanto global como localmente, cumpliendo con necesidades de negocio y regulatorias.

### Notebook Conceptual (NB1) – Explicaciones con Modelos Sencillos

- **Datos:** Dataset de Boston (o California) con Random Forest entrenado.
- **Actividades:**
  1. Calcular importancia de características (permutation importance) y comparar con la importancia del árbol.
  2. Dibujar Partial Dependence Plots (PDP) para una o dos variables.
  3. Usar SHAP: calcular valores SHAP para una predicción local y resumen global (summary plot, dependence plot).
  4. Usar LIME para explicar una instancia específica.
  5. Comparar las explicaciones de SHAP y LIME.
- **Conceptos:** Shapley values, explicaciones locales vs globales.
- **Herramientas:** sklearn.inspection, shap, lime.

### Notebook de Ejercicios (NB2) – Explicación de un Modelo de Crédito

- **Dataset:** German Credit o Lending Club.
- **Actividades:**
  1. Entrenar un modelo (XGBoost o Random Forest) para aprobación de crédito.
  2. Generar explicaciones globales (importancia SHAP) y locales para solicitudes rechazadas/aprobadas.
  3. Redactar un breve informe para un cliente simulando la explicación de por qué se le negó el crédito.

---

## Semana 14: Despliegue de Modelos (MLOps)

### Propósito
Llevar un modelo a producción: serialización, creación de API, contenerización básica y monitoreo.

### Notebook Conceptual (NB1) – De Modelo a API Local

- **Datos:** Modelo entrenado previamente (por ejemplo, regresión logística con iris).
- **Actividades:**
  1. Serializar el modelo con pickle y joblib.
  2. Crear un script simple con Flask que cargue el modelo y exponga un endpoint /predict.
  3. Probar el endpoint con requests desde el mismo notebook.
  4. Escribir un Dockerfile básico para contenerizar la API.
  5. (Opcional) Introducir MLflow para registrar experimentos.
- **Conceptos:** Serialización, API REST, contenedores, registro de modelos.
- **Herramientas:** pickle, joblib, Flask, requests, Docker (conceptos), MLflow.

### Notebook de Ejercicios (NB2) – Despliegue y Monitoreo Simulado

- **Dataset:** Modelo de recomendación o clasificación de la semana anterior.
- **Actividades:**
  1. Crear una API para el modelo y probarla localmente.
  2. Simular un drift de datos (modificar ligeramente las entradas) y usar evidently para detectar el cambio.
  3. Discutir estrategias de actualización del modelo.