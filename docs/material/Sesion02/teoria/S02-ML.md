# Semana 2: Análisis Exploratorio y Feature Engineering

## Logro de la sesión  
Al finalizar esta semana, el estudiante será capaz de realizar un análisis exploratorio de datos (EDA) completo y aplicar técnicas de feature engineering para transformar datos crudos en conjuntos de datos de alta calidad, listos para modelos de machine learning. Se busca que el alumno desarrolle una visión crítica para identificar problemas en los datos, generar hipótesis de negocio y crear variables predictivas que maximicen el rendimiento de los modelos.

---

## Conceptos

### Importancia del EDA  
El Análisis Exploratorio de Datos (EDA) es el primer paso esencial en cualquier proyecto de ciencia de datos. Su objetivo es comprender la estructura, patrones, relaciones y peculiaridades de los datos antes de construir modelos. Un EDA riguroso permite:
- Detectar errores, inconsistencias y valores atípicos.
- Generar hipótesis iniciales sobre el comportamiento de los datos.
- Guiar la selección de técnicas de modelado y feature engineering.
- Comunicar hallazgos preliminares a las partes interesadas.

### Casos famosos de éxito y fracaso en negocios  
- **Éxito:**  
  - **Netflix:** El uso de EDA en los hábitos de visualización permitió crear el sistema de recomendaciones que hoy genera más del 80% del contenido visto.  
  - **Amazon:** El análisis de patrones de compra y navegación impulsó su motor de recomendaciones y la optimización de precios dinámicos.  
- **Fracaso:**  
  - **Google Flu Trends:** Falló al no considerar estacionalidades y cambios en el comportamiento de búsqueda, lo que llevó a predicciones erróneas.  
  - **Proyecto de predicción de reincidencia criminal (COMPAS):** Un EDA deficiente ocultó sesgos raciales en los datos, generando controversia y desconfianza.

### Pipeline de EDA  

Te propongo una **versión mejorada y más técnica**, incorporando fundamentos matemáticos, conceptos del documento y algunos criterios prácticos usados en ciencia de datos. Esto queda más **alineado con un curso avanzado de ML/EDA** como el de tu documento. 

---

# Pipeline de Análisis Exploratorio de Datos (EDA)

El **Exploratory Data Analysis (EDA)** es el proceso mediante el cual el científico de datos investiga, resume y visualiza un conjunto de datos para comprender su estructura, calidad y patrones antes del modelado. John Tukey lo describió como un enfoque para **"escuchar a los datos"** mediante estadísticas y visualizaciones. 

Un flujo típico de trabajo incluye:

1. **Comprensión del problema y de los datos**
   - Identificación de fuentes de datos.
   - Revisión del diccionario de datos.
   - Determinación del tipo de variables (numéricas, categóricas, temporales).
   - Evaluación de tamaño del dataset y granularidad.

2. **Evaluación de calidad de datos**
   - Detección de valores faltantes.
   - Identificación de duplicados.
   - Verificación de consistencia lógica.

3. **Limpieza y preprocesamiento**
   - Corrección de tipos de datos.
   - Manejo de valores faltantes.
   - Tratamiento de outliers.

4. **Análisis univariado**
   - Análisis estadístico de cada variable individualmente.
   - Estudio de distribuciones y dispersión.

5. **Análisis bivariado**
   - Estudio de relaciones entre pares de variables.
   - Identificación de correlaciones y dependencias.

6. **Análisis multivariado**
   - Identificación de patrones complejos entre múltiples variables.
   - Detección de redundancias o estructuras latentes.

7. **Documentación de hallazgos**
   - Visualizaciones.
   - Conclusiones que orienten el feature engineering o modelado.

---

# Limpieza de datos (Data Cleaning)

La limpieza de datos busca mejorar la **calidad y confiabilidad del dataset**, eliminando errores o inconsistencias.

## 1. Valores faltantes

Primero se debe identificar el **mecanismo de ausencia**:

- **MCAR (Missing Completely At Random):** La ausencia es completamente aleatoria.
- **MAR (Missing At Random):** Depende de otras variables observadas.
- **MNAR (Missing Not At Random):** Depende del valor faltante mismo.

### Estrategias de imputación

**Imputación simple**

Media o mediana:

$$ x_{imp} = \begin{cases} \bar{x} & \text{media} \\ \text{mediana}(x) & \text{mediana} \end{cases} $$

Problema: subestima la varianza.

**Imputación por regresión**

$$ x_{faltante} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots $$

**k-Nearest Neighbors**

Se calcula la media de los $k$ vecinos más cercanos en el espacio de variables.

**Imputación múltiple (MICE)**

Genera múltiples datasets imputados y combina resultados para incorporar incertidumbre.

---

## 2. Duplicados

Se detectan mediante:
- coincidencia exacta de filas
- coincidencia parcial de claves

La eliminación se justifica cuando el duplicado **no representa observaciones distintas**.

---

## 3. Errores de tipo de dato

Ejemplos comunes:

| Error                | Corrección              |
| -------------------- | ----------------------- |
| fechas como texto    | convertir a datetime    |
| categorías numéricas | convertir a categorical |
| números con símbolos | limpieza previa         |

---

## 4. Tratamiento de outliers

### Método del rango intercuartílico (IQR)

$$ IQR = Q3 - Q1 $$

Límites:

$$ \text{Inferior} = Q1 - 1.5 \cdot IQR $$

$$ \text{Superior} = Q3 + 1.5 \cdot IQR $$

Los valores fuera de estos límites se consideran **outliers potenciales**.

---

### Z-score clásico

$$ z = \frac{x - \mu}{\sigma} $$

Umbral típico:

$$ |z| > 3 $$

---

### Z-score robusto (basado en MAD)

$$ M_i = \frac{0.6745(x_i - \text{mediana}(x))}{MAD} $$

donde

$$ MAD = \text{mediana}(|x_i - \text{mediana}(x)|) $$

Es más robusto ante valores extremos. 

---

# Análisis univariado

Analiza **cada variable individualmente**.

---

## Variables numéricas

### Medidas de tendencia central

Media:

$$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $$

Mediana: valor central de la distribución.

Moda: valor más frecuente.

---

### Medidas de dispersión

Rango:

$$ R = x_{max} - x_{min} $$

Varianza:

$$ \sigma^2 = \frac{1}{n-1} \sum (x_i - \bar{x})^2 $$

Desviación estándar:

$$ \sigma = \sqrt{\sigma^2} $$

IQR:

$$ IQR = Q3 - Q1 $$

---

### Medidas de forma

**Asimetría (Skewness)**

Indica si la distribución tiene cola hacia izquierda o derecha.

**Curtosis**

Mide la concentración en las colas.

---

### Visualizaciones

- Histogramas
- Kernel Density Estimation (KDE)

Estimación de densidad:

$$ \hat f(x)=\frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right) $$

donde $K$ es el kernel (normalmente gaussiano). 

- Boxplots
- ECDF (Empirical CDF)

$$ F_n(x)=\frac{1}{n}\sum_{i=1}^{n} I(x_i \le x) $$

---

## Variables categóricas

Análisis típico:
- frecuencia absoluta
- frecuencia relativa
- moda

Herramientas:
- bar plots
- diagramas de Pareto
- tablas de contingencia

---

## Variables temporales

Se analizan mediante:
- tendencia
- estacionalidad
- ruido

Modelo aditivo:

$$ Y_t = T_t + S_t + R_t $$

Modelo multiplicativo:

$$ Y_t = T_t \cdot S_t \cdot R_t $$

---

# Análisis bivariado

Estudia **relaciones entre dos variables**.

---

## Numérica vs Numérica

### Correlación de Pearson

$$ r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}} $$

Interpretación:

| $r$     | relación |
| ----- | -------- |
| 0     | ninguna  |
| $\pm 0.3$  | débil    |
| $\pm 0.5$  | moderada |
| $\pm 0.7+$ | fuerte   |

---

### Correlación de Spearman

Basada en **rangos**, útil para relaciones monotónicas.

---

### Visualizaciones

- scatter plots
- pairplots
- heatmaps de correlación

---

## Categórica vs Categórica

Herramientas:
- tablas de contingencia
- prueba **Chi-cuadrado**

$$ \chi^2 = \sum \frac{(O - E)^2}{E} $$

---

## Numérica vs Categórica

Se analizan diferencias entre grupos.

Métodos:
- boxplots por categoría
- t-test
- ANOVA

Estadístico ANOVA:

$$ F = \frac{\text{varianza entre grupos}}{\text{varianza dentro de grupos}} $$

---

# Análisis multivariado

Busca **relaciones complejas entre múltiples variables**.

---

## Matriz de correlación

Permite detectar:
- multicolinealidad
- redundancia de variables

Regla común:

$$ |r| > 0.9 $$

→ posible eliminación de una variable.

---

## Pairplots

Exploran todas las combinaciones entre variables numéricas.

Muy útil para:
- detectar clusters
- detectar no linealidad
- identificar outliers multivariados.

---

## Detección multivariada de outliers

### Distancia de Mahalanobis

$$ D_M(x) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)} $$

Considera **correlación entre variables**. 

---

## Reducción de dimensionalidad

### PCA

Transforma variables correlacionadas en componentes ortogonales.

Primera componente:

$$ z_1 = w_1^T x $$

donde $w_1$ maximiza la varianza.

---

# Conclusión

El EDA cumple tres objetivos fundamentales:

1. **Comprender la estructura de los datos**
2. **Detectar problemas de calidad**
3. **Generar hipótesis para feature engineering**

Un EDA profundo puede determinar el **límite superior del rendimiento del modelo**, ya que los algoritmos de machine learning dependen críticamente de la calidad y representación de los datos. 

---

A continuación tienes una **versión más rigurosa y alineada con el enfoque del documento**, incorporando fundamentos matemáticos, criterios de uso y técnicas modernas de ingeniería de características. 

---

# Importancia del Feature Engineering

El **Feature Engineering** es el proceso de transformar datos crudos en representaciones (features) que permitan a los algoritmos de aprendizaje automático capturar de manera más eficiente los patrones subyacentes del problema.

Formalmente, si el conjunto de datos original es

$$ X \in \mathbb{R}^{n \times p} $$

donde $n$ es el número de observaciones y $p$ el número de variables, el feature engineering busca una transformación

$$ \phi(X) \rightarrow Z $$

donde

$$ Z \in \mathbb{R}^{n \times k} $$

y $k$ puede ser mayor o menor que $p$, pero las nuevas variables $Z$ representan mejor la estructura relevante del problema.

En muchos casos, el rendimiento de un modelo depende más de **la calidad de las características que del algoritmo utilizado**. De hecho, modelos simples como regresión logística o árboles de decisión pueden superar a modelos complejos si se utilizan **features bien diseñadas**. 

---

# Pipeline de Feature Engineering

Un pipeline típico de ingeniería de características incluye:

1. **Creación de nuevas características** a partir de las variables existentes.
2. **Transformación de variables** (log, Box-Cox, polinomios).
3. **Codificación de variables categóricas**.
4. **Escalado o normalización de variables numéricas**.
5. **Imputación avanzada de valores faltantes**.
6. **Tratamiento robusto de outliers**.
7. **Selección de características relevantes**.
8. **Automatización mediante pipelines reproducibles**.

---

# Creación de nuevas características

La creación de nuevas características consiste en generar variables derivadas que capturen relaciones más informativas que las variables originales.

## 1. Transformaciones de variables numéricas

### Transformación logarítmica

Útil para variables con **distribuciones altamente asimétricas**:

$$ x' = \log(x) $$

Reduce la asimetría y estabiliza la varianza.

---

### Transformación Box-Cox

Generalización de la transformación logarítmica:

$$ x' = \begin{cases} \frac{x^\lambda -1}{\lambda} & \lambda \neq 0 \\ \log(x) & \lambda = 0 \end{cases} $$

El parámetro $\lambda$ se estima maximizando la verosimilitud.

---

### Transformación Yeo-Johnson

Extensión de Box-Cox que permite valores negativos.

---

## 2. Interacciones y polinomios

Las interacciones capturan efectos combinados entre variables:

$$ x_{ij}^{(int)} = x_i \cdot x_j $$

Ejemplo: $\text{habitaciones} \times \text{metros}^2$

Los términos polinomiales permiten modelar relaciones no lineales:

$$ x^2, x^3, \dots $$

Ejemplo: relación entre edad y riesgo médico.

---

## 3. Binning o discretización

Consiste en dividir una variable continua en intervalos.

**Igual ancho**

$$ bin_k = [a + k\Delta, a + (k+1)\Delta] $$

**Igual frecuencia**

Cada intervalo contiene aproximadamente el mismo número de observaciones.

También se pueden usar **árboles de decisión** para encontrar puntos de corte óptimos.

---

## 4. Variables temporales

En datos de series temporales se generan nuevas características como:

### Rezagos (lags)

$$ y_{t-1}, y_{t-2}, \dots $$

Permiten utilizar valores pasados como predictores.

---

### Ventanas móviles

Media móvil:

$$ MA_t = \frac{1}{k} \sum_{i=0}^{k-1} y_{t-i} $$

Desviación móvil:

$$ SD_t = \sqrt{\frac{1}{k-1}\sum_{i=0}^{k-1}(y_{t-i}-MA_t)^2} $$

Capturan tendencias y volatilidad.

---

### Codificación cíclica

Variables como hora o día de la semana son **cíclicas**, por lo que se representan como:

$$ \sin\left(\frac{2\pi t}{T}\right), \quad \cos\left(\frac{2\pi t}{T}\right) $$

Esto evita discontinuidades artificiales (por ejemplo entre 23h y 0h). 

---

## 5. Agregaciones por grupo

En datos jerárquicos se calculan estadísticas por entidad:

Ejemplo para clientes:
- gasto promedio
- número de transacciones
- desviación de gasto
- máximo o mínimo histórico

Esto resume el comportamiento histórico de cada entidad.

---

## 6. Texto

Las características textuales pueden representarse mediante:

### Bag of Words

Representación basada en conteos de palabras.

---

### TF-IDF

$$ TFIDF(t,d) = TF(t,d) \times \log\left(\frac{N}{DF(t)}\right) $$

donde $TF$ = frecuencia de término, $DF$ = número de documentos que contienen el término.

---

### Embeddings

Representaciones densas aprendidas mediante redes neuronales.

Ejemplos: Word2Vec, BERT embeddings.

---

# Codificación de variables categóricas

Los algoritmos de machine learning requieren variables numéricas, por lo que las categorías deben transformarse.

---

## One-Hot Encoding

Se crean variables binarias para cada categoría:

$$ x_{cat} \rightarrow (x_1, x_2, ..., x_k) $$

Problema: alta dimensionalidad cuando hay muchas categorías.

---

## Label Encoding

Se asigna un número entero a cada categoría:

$$ \{A,B,C\} \rightarrow \{0,1,2\} $$

Adecuado para **variables ordinales**.

---

## Target Encoding

Cada categoría se reemplaza por la media del target:

$$ x_{cat} = \frac{1}{n_{cat}} \sum_{i \in cat} y_i $$

Debe aplicarse con **validación cruzada** para evitar *data leakage*. 

---

## Frequency Encoding

Se reemplaza la categoría por su frecuencia de aparición:

$$ x_{cat} = \frac{n_{cat}}{n} $$

---

## Binary Encoding

Las categorías se transforman en representación binaria para reducir dimensionalidad.

---

## Hash Encoding

Aplica una función hash:

$$ h(x) \rightarrow [0, k-1] $$

Útil en variables de **alta cardinalidad**.

---

# Escalado y Normalización

Algunos algoritmos dependen de la escala de las variables.

---

## Estandarización (Z-score)

$$ x' = \frac{x-\mu}{\sigma} $$

Media 0 y desviación estándar 1.

Necesaria para: regresión lineal, SVM, PCA.

---

## Normalización Min-Max

$$ x' = \frac{x-x_{min}}{x_{max}-x_{min}} $$

Escala al rango $[0,1]$.

---

## Escalado robusto

$$ x' = \frac{x - \text{mediana}(x)}{IQR} $$

Menos sensible a outliers. 

---

# Manejo de datos desbalanceados

En clasificación, cuando una clase es muy poco frecuente, los modelos tienden a ignorarla.

---

## Oversampling

Se aumentan ejemplos de la clase minoritaria.

### SMOTE

Genera ejemplos sintéticos:

$$ x_{new} = x_i + \lambda (x_{NN} - x_i) $$

donde $x_{NN}$ es un vecino cercano.

---

## Undersampling

Se reduce la clase mayoritaria.

Problema: pérdida de información.

---

## Métricas adecuadas

En datasets desbalanceados se prefieren métricas como:

- **Precision**
- **Recall**
- **F1-score**

$$ F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall} $$

- **ROC-AUC**

---

# Imputación de datos

### Imputación simple

Media, mediana o moda.

---

### KNN Imputer

Se imputan valores usando los vecinos más cercanos.

---

### Iterative Imputer (MICE)

Modela cada variable faltante como función de las demás:

$$ x_j = f(x_{-j}) $$

El proceso se repite iterativamente hasta converger.

---

# Manejo de outliers

Los outliers pueden contener **información importante** o ser errores.

Estrategias:

### Variable indicadora

$$ outlier_i = \begin{cases} 1 & \text{si } x_i \text{ es extremo} \\ 0 & \text{caso contrario} \end{cases} $$

---

### Winsorización

Limitar valores extremos:

$$ x = \begin{cases} P_{1\%} & x < P_{1\%} \\ P_{99\%} & x > P_{99\%} \end{cases} $$

---

### Transformaciones

- log
- raíz cuadrada
- Box-Cox

---

# Selección de características

La selección de variables busca **reducir dimensionalidad y mejorar generalización**.

---

## Métodos filtro

Basados en métricas estadísticas.

### Correlación

Eliminar variables altamente correlacionadas.

---

### Información mutua

$$ I(X;Y) = \sum p(x,y) \log \frac{p(x,y)}{p(x)p(y)} $$

Mide dependencia general entre variables.

---

## Métodos wrapper

Evalúan subconjuntos de variables mediante el rendimiento del modelo.

Ejemplo: **Recursive Feature Elimination (RFE)**

---

## Métodos embebidos

La selección ocurre durante el entrenamiento.

### Lasso

$$ \min_{\beta} \sum (y - X\beta)^2 + \lambda \sum |\beta_j| $$

Fuerza coeficientes irrelevantes a cero. 

---

# Variables derivadas del negocio

Uno de los aspectos más importantes del feature engineering es incorporar **conocimiento del dominio**.

Ejemplos:

### Ratios financieros
- liquidez
- endeudamiento
- margen de beneficio

---

### Variables temporales agregadas
- ventas últimos 7 días
- promedio móvil
- crecimiento mensual

---

### Variables de comportamiento
- recencia (tiempo desde última interacción)
- frecuencia de compra
- valor monetario total

Estas variables suelen ser extremadamente predictivas en problemas como: **churn**, **fraude**, **recomendación**, **scoring crediticio**.

---

## Comunicación  

Aquí tienes una versión **mejorada y más práctica**, incluyendo técnicas recomendadas y ejemplos concretos:

---

### Storytelling e insights 

La capacidad de comunicar hallazgos es **tan importante como el análisis en sí**. Los objetivos incluyen:

- **Destacar patrones relevantes:** Usar gráficas comparativas, heatmaps de correlación o scatter plots con color según categoría para resaltar relaciones importantes.
- **Explicar el impacto de variables en el negocio:** Aplicar técnicas como **feature importance** de Random Forest o **SHAP values** para cuantificar contribuciones.
- **Presentar visualizaciones claras y atractivas:** Preferir gráficos simples y legibles; combinar colores y anotaciones estratégicas.
- **Adaptar el mensaje según la audiencia:** Para técnicos: enfatizar metodología y métricas. Para no técnicos: enfocarse en impacto y decisiones accionables.

---

### Elevator pitch a equipo técnico y no técnico

El **elevator pitch** permite transmitir hallazgos en 2-3 minutos, adaptando el contenido según el interlocutor:

- **Equipo técnico:** Detallar **metodologías utilizadas**, métricas de rendimiento ($R^2$, AUC, $F1$, etc.), hipótesis validadas y retos de implementación.
- **Equipo de negocio / no técnico:** Resumir **impacto y valor del análisis**, recomendaciones claras y lenguaje sencillo.

**Tip recomendado:** Utilizar **estructura narrativa en tres pasos**: Contexto → Hallazgos → Acción. Esto asegura claridad y conexión con la audiencia.

