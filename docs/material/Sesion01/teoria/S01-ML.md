# Semana 1: Introducción al Machine Learning

## Logro de la sesión
Al finalizar la sesión, el estudiante será capaz de:
- Comprender el flujo completo de un proyecto de machine learning.
- Identificar los tipos de aprendizaje y los modelos más representativos.
- Reconocer casos de uso en la industria y el rol de los distintos perfiles profesionales.

## Conceptos fundamentales

### Definición de Machine Learning, Deep Learning y su relación con la IA
La **Inteligencia Artificial (IA)** es el campo que busca crear sistemas capaces de realizar tareas que requieren inteligencia humana. El **Machine Learning (ML)** es una rama de la IA que permite a las máquinas aprender patrones a partir de datos sin ser explícitamente programadas. El **Deep Learning** es una subárea del ML que utiliza redes neuronales profundas para modelar representaciones complejas, especialmente útil en visión por computador, procesamiento de lenguaje natural y otras tareas con grandes volúmenes de datos.

### Problemas de negocios y Tipos de aprendizaje
Los proyectos de ML suelen originarse en necesidades de negocio. Según la naturaleza de los datos y el objetivo, se clasifican en:
- **Aprendizaje supervisado:** Se dispone de datos etiquetados. Incluye regresión (valores continuos) y clasificación (categorías discretas).
- **Aprendizaje no supervisado:** No hay etiquetas; se busca estructuras ocultas. Ejemplos: clustering, reducción de dimensionalidad, reglas de asociación.
- **Aprendizaje por refuerzo:** Un agente aprende a través de interacciones con un entorno, recibiendo recompensas o castigos.
- **Aprendizaje semi-supervisado y autosupervisado:** Combinan datos etiquetados y no etiquetados; el autosupervisado genera etiquetas a partir de los propios datos (muy usado en NLP).

### Métodos paramétricos y no paramétricos en ML
Los modelos paramétricos asumen una forma funcional fija para la relación entre entradas y salidas (por ejemplo, regresión lineal). Su complejidad no crece con el tamaño de los datos. Los modelos no paramétricos (como k-Vecinos o árboles de decisión) no presuponen una forma funcional y su complejidad aumenta con los datos, lo que les da más flexibilidad pero también riesgo de sobreajuste.


## Historia y auge de los Modelos de Machine Learning

El **Machine Learning (ML)** se desarrolla a partir de la estadística, la cibernética y la inteligencia artificial de mediados del siglo XX, buscando que las máquinas aprendan patrones de los datos sin depender únicamente de reglas explícitas.

---

### 1950s – 1960s: Los comienzos

* **Perceptrón (1958, Frank Rosenblatt)**: Primera red neuronal capaz de clasificación binaria. Limitado a problemas lineales.
* **Regresión Lineal (desconocido, usada desde principios del siglo XX)**: Predicción de variables continuas.
* **k-vecinos más cercanos – k-NN (1967, Thomas Cover & Peter Hart)**: Clasificación basada en proximidad de los datos.
* **Regresión Logística (desconocido, formalizada en ML 1960s)**: Clasificación binaria probabilística.

---

### 1970s – 1980s: Árboles y modelos probabilísticos

* **Árboles de Decisión**:

  * **ID3 (1986, J. Ross Quinlan)**
  * **C4.5 (1993, J. Ross Quinlan)**
* **Naive Bayes (desconocido; teorema de Thomas Bayes 1763; aplicado en ML formalmente 1970s–1980s)**: Clasificador probabilístico simple pero eficiente.
* **Redes Bayesianas (1980s, Judea Pearl)**: Extienden Naive Bayes para dependencias entre variables.
* **K-means (desconocido, popularizado en ML 1980s)**: Clustering particional clásico.

---

### 1990s: Modelos de margen y ensemble

* **Support Vector Machines – SVM (1992, Vladimir Vapnik & Alexey Chervonenkis)**: Clasificador de margen máximo, útil en alta dimensión.
* **Random Forest (2001, Leo Breiman)**: Ensemble de árboles de decisión, robusto y preciso.
* **Gradient Boosting (1999, Jerome Friedman)**: Base de modelos potentes de boosting.
* **DBSCAN (1996, Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu)**: Clustering basado en densidad.
* **Redes Neuronales Multicapa – MLP (1986, Rumelhart, Hinton & Williams)**: Extiende el perceptrón a múltiples capas con backpropagation.
* **Regresión Logística multiclase (desconocido, formalización en ML 1990s)**

---

### 2000s: Optimización y nuevos enfoques

* **XGBoost (2014, Tianqi Chen & Carlos Guestrin)**: Gradient boosting eficiente, ampliamente usado en datos tabulares.
* **LightGBM (2017, Microsoft, Ke et al.)**: Variante de boosting rápida y eficiente.
* **Prophet (2017, Sean J. Taylor & Benjamin Letham, Facebook)**: Predicción de series temporales basada en descomposición y regresión.
* **Ensemble stacking y bagging (desconocido, popularización 2000s)**: Combina varios modelos para mejorar desempeño.

---

### 2010s – presente: Deep Learning y transformers

* **AlexNet (2012, Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton)**: CNN que revolucionó visión por computadora.
* **ResNet (2015, Kaiming He et al.)**: Introduce conexiones residuales, facilita redes profundas.
* **LSTM – Long Short-Term Memory (1997, Sepp Hochreiter & Jürgen Schmidhuber)**: Redes recurrentes mejoradas para secuencias largas.
* **BERT (2018, Jacob Devlin et al., Google)**: Transformer para NLP, aprendizaje de contexto bidireccional.
* **GPT (2018–presente, OpenAI, Alec Radford et al.)**: Modelos de lenguaje generativo basados en transformers.



### Ciclo de vida y metodología CRISP-DM
El estándar de facto para proyectos de ML es CRISP-DM (Cross-Industry Standard Process for Data Mining), que consta de seis fases iterativas:
1. Comprensión del negocio
2. Comprensión de los datos
3. Preparación de los datos
4. Modelado
5. Evaluación
6. Despliegue

Este ciclo asegura que las soluciones de ML estén alineadas con los objetivos de negocio y sean sostenibles en el tiempo.

### Diferencias: Data Engineer, Data Scientist, ML Engineer, MLOps
- **Data Engineer:** Construye y mantiene la infraestructura de datos (pipelines, almacenes, calidad).
- **Data Scientist:** Explora datos, construye modelos y extrae insights; suele tener perfil estadístico/analítico.
- **ML Engineer:** Se enfoca en la implementación, escalado y puesta en producción de modelos; conoce de ingeniería de software y MLOps.
- **MLOps:** Prácticas que combinan ML, DevOps y automatización para gestionar el ciclo de vida completo de los modelos en producción (CI/CD, monitoreo, gobernanza).

## Aplicaciones y casos típicos
El ML está presente en numerosos sectores. Algunos ejemplos representativos:
- **Sistemas de recomendación:** Netflix, Amazon, Spotify personalizan contenidos y productos.
- **Detección de fraude:** Bancos y tarjetas de crédito identifican transacciones sospechosas en tiempo real.
- **Mantenimiento predictivo:** Maquinaria industrial sensorizada anticipa fallos y optimiza paradas.
- **Clasificación de imágenes:** Diagnóstico médico (radiografías, resonancias), vehículos autónomos (detección de objetos).
- **Predicción de series temporales:** Ventas minoristas, demanda energética, tráfico.

## Fundamentos matemáticos y computacionales

Los modelos de **Machine Learning** se fundamentan en conceptos matemáticos sólidos: funciones de coste, optimización y regularización para lograr buena **generalización**. A continuación se detallan los más importantes, con ejemplos por modelo.

---


### 1. Funciones de coste

Las funciones de coste cuantifican la discrepancia entre las predicciones $\hat{y}$ y los valores reales $y$:

* **Error cuadrático medio (MSE, para regresión lineal):**
  $$ J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
  donde $\hat{y}_i = \theta_0 + \theta_1 x_i + \dots + \theta_p x_i^{(p)}$.

* **Entropía cruzada (cross-entropy, para regresión logística y clasificación):**
  $$ J(\theta) = - \frac{1}{n} \sum_{i=1}^{n} \left[y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)\right] $$
  donde $\hat{y}_i = \sigma(\theta^T x_i)$ y $\sigma(z) = \frac{1}{1+e^{-z}}$ es la función sigmoide.

* **Función de coste para SVM (hinge loss):**
  $$ J(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (\mathbf{w}^T x_i + b)) $$

* **Clustering (k-means, suma de distancias al cuadrado):**
  $$ J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2 $$
  donde $\mu_k$ es el centroide del cluster $C_k$.

---

### 2. Optimización

Para minimizar la función de coste se usan distintos métodos:

* **Gradiente descendente (Gradient Descent):**
  $$ \theta := \theta - \alpha \nabla_\theta J(\theta) $$
  donde $\alpha$ es la tasa de aprendizaje.

* **Gradiente descendente estocástico (SGD):** Ajusta parámetros por cada muestra o minibatch, más rápido en datasets grandes.

* **Métodos cerrados (Closed-form solution):** Para **regresión lineal**, la solución óptima puede encontrarse sin iteraciones:
  $$ \hat{\theta} = (X^T X)^{-1} X^T y $$

* **Métodos de optimización convexa:** Usados en **regresión logística** y SVM, aprovechando que la función de coste es convexa.

---

### 3. Regularización y generalización

Para evitar **overfitting**, se introducen penalizaciones:

* **Ridge (L2):**
  $$ J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \theta_j^2 $$

* **Lasso (L1):**
  $$ J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\theta_j| $$

* **Elastic Net (combinación L1 + L2)** también es común.

---

### 4. Matemáticas específicas por modelo

| Modelo                      | Conceptos matemáticos clave                   | Función de coste / ecuación                                                                      |
| --------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Regresión Lineal**        | Álgebra lineal, MSE                           | $J(\theta) = \frac{1}{n} \sum (y_i - \theta^T x_i)^2$                                            |
| **Regresión Logística**     | Probabilidad, optimización convexa            | $J(\theta) = -\frac{1}{n} \sum [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$                 |
| **SVM**                     | Optimización convexa, geometría de margen     | $J(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum \max(0, 1 - y_i (\mathbf{w}^T x_i + b))$ |
| **k-NN**                    | Distancia euclidiana                          | $d(x_i, x_j) = \|x_i - x_j\|$                                                                      |
| **k-means**                 | Distancias, centroides                        | $J = \sum_{k} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$                                              |
| **DBSCAN**                  | Distancias y densidad                         | $N_\epsilon(x) = \{ y \in D \mid \|x - y\| \le \epsilon \}$                                       |
| **Random Forest / XGBoost** | Árboles, splits, reducción de entropía / Gini | Funciones de coste según impurity (Gini/Entropy) y suma de errores de predicción                 |

---





### Complejidad algorítmica de los modelos
La eficiencia computacional es clave en ML. Se analiza tanto en entrenamiento como en inferencia:
- Regresión lineal: entrenamiento $O(n^3)$ (ecuación normal) o $O(n\cdot\text{iter})$ con gradiente descendente.
- k-NN: entrenamiento $O(1)$ (solo almacena datos), inferencia $O(N\cdot n)$ sin optimizaciones.
- Árboles de decisión: entrenamiento $O(N\cdot n\cdot\log N)$, inferencia $O(\text{profundidad})$.
- Redes neuronales: depende del número de capas, neuronas y épocas; altamente paralelizable en GPUs.

### Hardware: CPU, GPU, TPU

En el contexto de **Machine Learning clásico**, la selección del hardware es fundamental para el rendimiento de los modelos, especialmente según el tamaño del dataset. La mayoría de los modelos clásicos **no requieren GPU o TPU** y funcionan eficientemente en CPU, salvo casos de datasets muy grandes o modelos específicos como XGBoost.

---

### 1. CPU (Central Processing Unit)

La CPU es un procesador de propósito general, ideal para tareas secuenciales y modelos de ML clásico. Es suficiente para la mayoría de los modelos, incluyendo **regresión lineal, regresión logística, Naive Bayes, árboles de decisión, Random Forest, k-NN, k-means, DBSCAN y Prophet**.

* Para **datasets pequeños y medianos**, la CPU permite entrenamientos rápidos y precisos.
* Para **datasets muy grandes**, algunos modelos como Random Forest pueden beneficiarse de optimizaciones adicionales, pero la CPU sigue siendo funcional.

---

### 2. GPU (Graphics Processing Unit)

Las GPU, originalmente diseñadas para gráficos, cuentan con miles de núcleos paralelos que permiten ejecutar operaciones matriciales simultáneamente. Su uso en ML clásico es más limitado y específico:

* **Aplicable principalmente a:**

  * Gradient Boosting, XGBoost y LightGBM en datasets grandes, donde acelera cálculos de histogramas y splits.
  * Random Forest con árboles muy profundos.
* **No necesario para:** Regresión lineal/logística, Naive Bayes, k-means pequeño-mediano, árboles de decisión simples.
* En datasets pequeños, la CPU suele ser más eficiente debido al overhead de transferencia de datos hacia la GPU.

---

### 3. TPU (Tensor Processing Unit)

Las TPU, desarrolladas por Google, están optimizadas para **operaciones tensoriales masivas**, típicas en deep learning. En ML clásico, su uso **no es relevante**, dado que los modelos como regresión lineal, regresión logística, Prophet, Random Forest, k-means y DBSCAN no requieren operaciones tensoriales intensivas.

* Su aplicación se limita a experimentos avanzados que combinan ML clásico con deep learning, lo cual está fuera del alcance de un curso de ML clásico.

---

### 4. Resumen por modelo y tamaño de datos

| Modelo / Técnica                       | Dataset pequeño | Dataset grande            | CPU          | GPU                            | TPU           |
| -------------------------------------- | --------------- | ------------------------- | ------------ | ------------------------------ | ------------- |
| Regresión lineal                       | ✅ ideal         | ⚪ hasta millones de filas | ✅ suficiente | ⚪ opcional                     | ⚪ innecesario |
| Regresión logística                    | ✅ ideal         | ⚪ hasta millones          | ✅ suficiente | ⚪ opcional                     | ⚪ innecesario |
| Naive Bayes                            | ✅ ideal         | ⚪ hasta millones          | ✅ suficiente | ⚪ innecesario                  | ⚪ innecesario |
| Árbol de decisión                      | ✅ ideal         | ⚪ grandes árboles         | ✅ suficiente | ⚪ acelera                      | ⚪ innecesario |
| Random Forest                          | ✅ ideal         | ✅ grandes datasets        | ✅ posible    | ✅ recomendable                 | ⚪ innecesario |
| Gradient Boosting / XGBoost / LightGBM | ✅ pequeño       | ✅ muy grande              | ✅ posible    | ✅ recomendable                 | ⚪ limitado    |
| k-NN                                   | ✅ ideal         | ⚪ hasta decenas de miles  | ✅ suficiente | ⚪ útil en datasets muy grandes | ⚪ innecesario |
| k-means                                | ✅ ideal         | ✅ millones                | ✅ suficiente | ⚪ útil                         | ⚪ innecesario |
| DBSCAN                                 | ✅ ideal         | ⚪ hasta cientos de miles  | ✅ suficiente | ⚪ útil                         | ⚪ innecesario |
| Prophet                                | ✅ ideal         | ⚪ hasta cientos de miles  | ✅ suficiente | ⚪ innecesario                  | ⚪ innecesario |

---

### Conclusión

Para **ML clásico**, la **CPU** es suficiente para casi todos los modelos y tamaños de datos. La **GPU** se recomienda únicamente para Gradient Boosting/XGBoost/LightGBM con datasets grandes o Random Forest muy profundos. Las **TPU** no se utilizan en ML clásico y su aplicación se limita al entrenamiento de modelos de deep learning.

---


## Métricas
Seleccionar la métrica adecuada es fundamental para evaluar el rendimiento del modelo y su alineación con el negocio.
- **Clasificación:** Precisión, recall, F1-score, AUC-ROC, matriz de confusión.
- **Regresión:** Error absoluto medio (MAE), error cuadrático medio (RMSE), $R^2$.
- **Clustering:** Coeficiente de silueta, índice de Davies–Bouldin, inercia.

La métrica debe reflejar el coste real de los errores en el contexto de la aplicación (por ejemplo, en detección de fraudes es más crítico minimizar falsos negativos).

