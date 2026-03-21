
# Semana 5: Clasificación II - Algoritmos Clásicos

## Logro de la sesión

Construir, comparar e interpretar modelos de clasificación utilizando algoritmos clásicos como **K-Nearest Neighbors (KNN)**, **Naive Bayes** y **Support Vector Machines (SVM)**, comprendiendo sus supuestos, ventajas y limitaciones en distintos contextos de negocio.

---

## Problemática de negocio

* Selección del algoritmo de clasificación más adecuado según el tipo y cantidad de datos.
* Comparación de desempeño entre distintos modelos para tomar decisiones basadas en métricas.
* Situaciones donde la **regresión logística** no es suficiente:

  * No linealidad en los datos.
  * Alta dimensionalidad.
  * Grandes volúmenes de datos con restricciones computacionales.
* Impacto del tamaño de datos y escalamiento en el rendimiento.
* Ejemplos de aplicación:

  * **KNN**: sistemas de recomendación simples, detección de similitud entre clientes o productos.
  * **Naive Bayes**: clasificación de texto, spam, análisis de sentimiento.
  * **SVM**: problemas con fronteras no lineales, visión por computadora, bioinformática.

---


## Modelado

### K-Nearest Neighbors (KNN)

* **Requisitos del modelo:** datos numéricos escalados, elección de $k$.

* **Concepto de distancia:** Euclidiana, Manhattan u otras métricas según contexto. La distancia euclidiana entre dos puntos $p$ y $q$ se define como:
  
  $$d(p,q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

* **Elección de $k$ y trade-off bias-varianza:**
  * $k$ pequeño → modelo muy sensible al ruido (alta varianza).
  * $k$ grande → modelo más estable pero menos flexible (alto sesgo).

* **Sensibilidad a la escala de variables:** necesario normalizar o estandarizar.

* **Plantilla base en Python con scikit-learn:**

```python
# Importaciones necesarias
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamiento (fundamental para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definición del modelo
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

# Entrenamiento
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)
```

---

### Naive Bayes

* **Requisitos del modelo:** variables categóricas o continuas según la variante.

* **Teorema de Bayes:**
  
  $$P(C_k|X) = \frac{P(X|C_k)P(C_k)}{P(X)}$$

* **Supuesto de independencia condicional:** cada feature contribuye de forma independiente al cálculo de la probabilidad de clase, lo que permite expresar:
  
  $$P(X|C_k) = \prod_{i=1}^{n} P(x_i|C_k)$$

* **Variantes:** Gaussian (para continuas), Multinomial (para conteos), Bernoulli (para binarias).

* **Interpretación probabilística:** salida como probabilidad de pertenencia a cada clase mediante `predict_proba()`.

* **Plantilla base en Python con scikit-learn:**

```python
# Importaciones necesarias
from sklearn.naive_bayes import GaussianNB  # Para variables continuas
# from sklearn.naive_bayes import MultinomialNB  # Para datos de conteo
# from sklearn.naive_bayes import BernoulliNB  # Para datos binarios
from sklearn.model_selection import train_test_split

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definición del modelo (elegir según tipo de datos)
model = GaussianNB()  # Para características continuas

# Entrenamiento
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Probabilidades por clase
```

---

### Support Vector Machines (SVM)

* **Requisitos del modelo:** datos escalados, posibilidad de kernel no lineal.

* **Concepto clave:** margen máximo y vectores de soporte. El hiperplano óptimo se define como:
  
  $$w \cdot x + b = 0$$
  
  con el objetivo de maximizar $\frac{2}{\|w\|}$ (el margen).

* **Kernel lineal vs no lineal:** RBF, polinomial, sigmoide, dependiendo de la complejidad de la frontera. La transformación mediante kernel $K(x_i, x_j)$ permite trabajar en espacios de mayor dimensión sin calcular explícitamente la transformación.

* **Hiperparámetros importantes:**
  * $C$ → penalización por error de clasificación (regularización).
  * $\gamma$ → alcance de influencia de cada punto en kernels RBF.

* **Manejo de no linealidad:** uso de kernels para transformar implícitamente el espacio de características.

* **Plantilla base en Python con scikit-learn:**

```python
# Importaciones necesarias
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamiento (crucial para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definición del modelo
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

# Entrenamiento
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)  # Requiere probability=True
```

---

## Comparación conceptual entre algoritmos

* **Paramétricos vs no paramétricos:**
  * **Paramétricos (Naive Bayes):** hacen supuestos sobre la distribución de los datos y tienen un número fijo de parámetros independiente del tamaño de la muestra.
  * **No paramétricos (KNN, SVM):** no asumen una forma funcional específica; su complejidad crece con los datos (KNN almacena todas las instancias; SVM el número de vectores soporte depende de los datos).

* **Interpretabilidad vs performance:**
  * **Naive Bayes:** altamente interpretable (probabilidades condicionales por feature).
  * **KNN:** interpretable (podemos examinar los vecinos más cercanos).
  * **SVM (lineal):** moderadamente interpretable (coeficientes del hiperplano).
  * **SVM (no lineal):** baja interpretabilidad (caja negra).

* **Escalabilidad computacional:**
  * **KNN:** entrenamiento O(1) (solo almacena), predicción O(n) costoso.
  * **Naive Bayes:** entrenamiento O(n·d) rápido, predicción O(d) eficiente.
  * **SVM:** entrenamiento O(n²·d) a O(n³·d) costoso, predicción O(n_sv·d) eficiente.

* **Casos de uso recomendados:**
  * **KNN:** datos de baja dimensión, relaciones locales importantes, necesidad de explicabilidad.
  * **Naive Bayes:** clasificación de texto, datos de alta dimensionalidad, tiempo real.
  * **SVM:** problemas con fronteras complejas, datasets medianos, máxima precisión requerida.



---






# Semana 5: Clasificación II - Algoritmos Clásicos

## Logro de la sesión

Construir, comparar e interpretar modelos de clasificación utilizando algoritmos clásicos como **K-Nearest Neighbors (KNN)**, **Naive Bayes** y **Support Vector Machines (SVM)**, comprendiendo sus supuestos, ventajas y limitaciones en distintos contextos de negocio.

---

## Problemática de negocio

* Selección del algoritmo de clasificación más adecuado según el tipo y cantidad de datos.
* Comparación de desempeño entre distintos modelos para tomar decisiones basadas en métricas.
* Situaciones donde la **regresión logística** no es suficiente:
  * No linealidad en los datos.
  * Alta dimensionalidad.
  * Grandes volúmenes de datos con restricciones computacionales.
* Impacto del tamaño de datos y escalamiento en el rendimiento.
* Ejemplos de aplicación:
  * **KNN**: sistemas de recomendación simples, detección de similitud entre clientes o productos.
  * **Naive Bayes**: clasificación de texto, spam, análisis de sentimiento.
  * **SVM**: problemas con fronteras no lineales, visión por computadora, bioinformática.

---

## Modelado

### K-Nearest Neighbors (KNN)

* **Requisitos del modelo:** datos numéricos escalados, elección de $k$.

* **Concepto de distancia:** Euclidiana, Manhattan u otras métricas según contexto. La distancia euclidiana entre dos puntos $p$ y $q$ se define como:
  
  $$d(p,q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

* **Elección de $k$ y trade-off bias-varianza:**
  * $k$ pequeño → modelo muy sensible al ruido (alta varianza).
  * $k$ grande → modelo más estable pero menos flexible (alto sesgo).

* **Sensibilidad a la escala de variables:** necesario normalizar o estandarizar.

* **Plantilla base en Python con scikit-learn:**

```python
# Importaciones necesarias
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamiento (fundamental para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definición del modelo
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

# Entrenamiento
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)
```

---

### Naive Bayes

* **Requisitos del modelo:** variables categóricas o continuas según la variante.

* **Teorema de Bayes:**
  
  $$P(C_k|X) = \frac{P(X|C_k)P(C_k)}{P(X)}$$

* **Supuesto de independencia condicional:** cada feature contribuye de forma independiente al cálculo de la probabilidad de clase, lo que permite expresar:
  
  $$P(X|C_k) = \prod_{i=1}^{n} P(x_i|C_k)$$

* **Variantes:** Gaussian (para continuas), Multinomial (para conteos), Bernoulli (para binarias).

* **Interpretación probabilística:** salida como probabilidad de pertenencia a cada clase mediante `predict_proba()`.

* **Plantilla base en Python con scikit-learn:**

```python
# Importaciones necesarias
from sklearn.naive_bayes import GaussianNB  # Para variables continuas
# from sklearn.naive_bayes import MultinomialNB  # Para datos de conteo
# from sklearn.naive_bayes import BernoulliNB  # Para datos binarios
from sklearn.model_selection import train_test_split

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definición del modelo (elegir según tipo de datos)
model = GaussianNB()  # Para características continuas

# Entrenamiento
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Probabilidades por clase
```

---

### Support Vector Machines (SVM)

* **Requisitos del modelo:** datos escalados, posibilidad de kernel no lineal.

* **Concepto clave:** margen máximo y vectores de soporte. El hiperplano óptimo se define como:
  
  $$w \cdot x + b = 0$$
  
  con el objetivo de maximizar $\frac{2}{\|w\|}$ (el margen).

* **Kernel lineal vs no lineal:** RBF, polinomial, sigmoide, dependiendo de la complejidad de la frontera. La transformación mediante kernel $K(x_i, x_j)$ permite trabajar en espacios de mayor dimensión sin calcular explícitamente la transformación.

* **Hiperparámetros importantes:**
  * $C$ → penalización por error de clasificación (regularización).
  * $\gamma$ → alcance de influencia de cada punto en kernels RBF.

* **Manejo de no linealidad:** uso de kernels para transformar implícitamente el espacio de características.

* **Plantilla base en Python con scikit-learn:**

```python
# Importaciones necesarias
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamiento (crucial para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definición del modelo
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

# Entrenamiento
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)  # Requiere probability=True
```

---

## Métricas de Evaluación

La evaluación de modelos de clasificación requiere un conjunto de métricas que permitan medir su desempeño desde diferentes perspectivas. La elección de la métrica adecuada depende del contexto de negocio y del costo asociado a cada tipo de error.

### 4.1 Matriz de Confusión

La matriz de confusión es la base para todas las métricas de clasificación. Para un problema binario, tiene la siguiente estructura:

| | **Predicción: Positivo** | **Predicción: Negativo** |
|---|---|---|
| **Real: Positivo** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Real: Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN) |

**Interpretación:**
- **Verdaderos Positivos (VP):** casos positivos correctamente identificados.
- **Verdaderos Negativos (VN):** casos negativos correctamente identificados.
- **Falsos Positivos (FP):** casos negativos incorrectamente clasificados como positivos (error Tipo I).
- **Falsos Negativos (FN):** casos positivos incorrectamente clasificados como negativos (error Tipo II).

**Ejemplo de negocio (detección de fraude):**
- **VP:** transacciones fraudulentas correctamente detectadas.
- **VN:** transacciones legítimas correctamente identificadas.
- **FP:** transacciones legítimas marcadas como fraude (cliente insatisfecho).
- **FN:** transacciones fraudulentas no detectadas (pérdida económica).

### 4.2 Métricas Derivadas

#### Accuracy (Exactitud)

$$Accuracy = \frac{VP + VN}{VP + VN + FP + FN}$$

**Interpretación:** proporción de predicciones correctas sobre el total.

**Cuándo usarla:** cuando las clases están balanceadas y los errores tienen costo similar.

**Limitación:** engañosa en clases desbalanceadas. Ejemplo: si 95% de transacciones son legítimas, un modelo que siempre predice "no fraude" tendría 95% de accuracy pero es inútil.

#### Precision (Precisión)

$$Precision = \frac{VP}{VP + FP}$$

**Interpretación:** de todas las predicciones positivas, ¿cuántas son realmente positivas? Mide la **exactitud de las predicciones positivas**.

**Cuándo usarla:** cuando el costo de los falsos positivos es alto.
- **Ejemplo:** diagnóstico de cáncer. Un falso positivo genera estrés innecesario y pruebas adicionales costosas.

#### Recall (Sensibilidad o TPR)

$$Recall = \frac{VP}{VP + FN}$$

**Interpretación:** de todos los casos positivos reales, ¿cuántos fueron detectados? Mide la **capacidad de encontrar todos los positivos**.

**Cuándo usarla:** cuando el costo de los falsos negativos es alto.
- **Ejemplo:** detección de fraude. Un falso negativo significa no detectar una transacción fraudulenta, generando pérdida económica directa.

#### F1-Score

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Interpretación:** media armónica entre precisión y recall. Combina ambas en una sola métrica.

**Cuándo usarla:** cuando necesitas un balance entre precisión y recall, especialmente en clases desbalanceadas.

#### Especificidad (Specificity o TNR)

$$Especificidad = \frac{VN}{VN + FP}$$

**Interpretación:** de todos los casos negativos reales, ¿cuántos fueron correctamente identificados?

### 4.3 Curva ROC y AUC

La curva ROC (Receiver Operating Characteristic) y el AUC (Area Under the Curve) evalúan la capacidad discriminativa del modelo independientemente del umbral de clasificación.

**Curva ROC:** gráfica la **tasa de verdaderos positivos (Recall)** vs la **tasa de falsos positivos (FPR)** para diferentes umbrales de clasificación.

$$FPR = \frac{FP}{FP + VN} = 1 - Especificidad$$

**AUC (Área bajo la curva):** valor entre 0.5 y 1 que resume la curva ROC.

| AUC | Interpretación |
|-----|----------------|
| 0.5 | Clasificador aleatorio (sin poder discriminativo) |
| 0.6 - 0.7 | Pobre |
| 0.7 - 0.8 | Aceptable |
| 0.8 - 0.9 | Excelente |
| 0.9 - 1.0 | Sobresaliente |

**Interpretación:** probabilidad de que el modelo asigne una probabilidad más alta a un positivo aleatorio que a un negativo aleatorio.

### 4.4 Guía para Seleccionar la Métrica Adecuada

| Escenario de Negocio | Métrica Principal | Justificación |
|---------------------|-------------------|---------------|
| **Detección de fraude** | Recall (o F1) | Es crítico detectar la mayor cantidad de fraudes posibles; los falsos negativos son muy costosos. |
| **Diagnóstico médico (enfermedad grave)** | Recall | No queremos dejar de diagnosticar a un enfermo (falso negativo es fatal). |
| **Diagnóstico médico (enfermedad benigna)** | Precision | Evitar tratamientos innecesarios por falsos positivos. |
| **Filtro de spam** | Precision o F1 | Los falsos positivos (correo importante a spam) son muy molestos. |
| **Recomendación de productos** | Precision | Mostrar recomendaciones relevantes; los falsos positivos (productos no relevantes) molestan al usuario. |
| **Clases balanceadas, errores similares** | Accuracy | Simple e interpretable cuando todas las clases importan igual. |
| **Clases desbalanceadas** | F1, AUC | Accuracy es engañosa; F1 balancea precision y recall. |
| **Comparación general de modelos** | AUC | Independiente del umbral, mide capacidad discriminativa. |

### 4.5 Ejemplo de Interpretación con Datos

**Contexto:** Banco implementa modelo para detectar transacciones fraudulentas. Dataset con 10,000 transacciones, 2% fraudulentas. Resultados en test:

**Matriz de confusión del modelo:**
| | Pred: Fraude | Pred: Normal |
|---|---|---|
| **Real: Fraude** | 180 (VP) | 20 (FN) |
| **Real: Normal** | 300 (FP) | 9,500 (VN) |

**Cálculo de métricas:**
- **Accuracy:** (180 + 9500) / 10,000 = 96.8%
- **Precision:** 180 / (180 + 300) = 37.5%
- **Recall:** 180 / (180 + 20) = 90%
- **F1-Score:** 2 × (0.375 × 0.9) / (0.375 + 0.9) = 0.53
- **Especificidad:** 9500 / (9500 + 300) = 96.9%

**Interpretación para el negocio:**
- El modelo detecta el **90% de los fraudes** (Recall alto), lo cual es bueno porque cada fraude no detectado es una pérdida.
- Sin embargo, de todas las alertas de fraude, solo el **37.5% son realmente fraudes** (Precision baja). Esto significa que por cada fraude detectado, se generan aproximadamente 1.7 falsas alarmas.
- **Impacto operativo:** el equipo de seguridad debe investigar muchas transacciones legítimas (300 falsos positivos), lo que aumenta costos operativos y puede afectar la experiencia del cliente si se bloquean transacciones legítimas.
- **Decisión de negocio:** ¿Es aceptable el costo operativo de los falsos positivos frente al beneficio de detectar el 90% de fraudes? Si el costo de investigar es bajo, el modelo es bueno. Si es alto, quizás se prefiera un modelo con mayor precisión aunque detecte menos fraudes.

### 4.6 Comparación de Modelos con Múltiples Métricas

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|--------|----------|-----------|--------|-----|-----|
| KNN (k=5) | 0.94 | 0.62 | 0.78 | 0.69 | 0.88 |
| Naive Bayes | 0.91 | 0.45 | 0.92 | 0.60 | 0.85 |
| SVM (RBF) | 0.96 | 0.71 | 0.85 | 0.77 | 0.93 |

**Análisis comparativo:**
- **SVM** tiene el mejor equilibrio: mayor F1 (0.77) y AUC (0.93). Es la mejor opción si se busca rendimiento general.
- **Naive Bayes** tiene el recall más alto (0.92), ideal si priorizamos detectar todos los positivos, aunque con muchos falsos positivos (baja precisión).
- **KNN** está en un punto intermedio, con buen balance pero inferior a SVM.

---

## Comunicación de Resultados

### 5.1 Comparación de Modelos para la Toma de Decisiones

Al comunicar resultados a stakeholders, es fundamental traducir las métricas técnicas a **impacto de negocio**.

**Ejemplo de reporte ejecutivo:**

> **Resumen Ejecutivo: Modelos de Detección de Fraude**
>
> Hemos evaluado tres modelos para detectar transacciones fraudulentas. El modelo SVM supera a los demás en capacidad discriminativa (AUC 0.93) y balance entre precisión y recall (F1 0.77).
>
> **Impacto operativo estimado:**
> - **SVM:** detectaría 850 de cada 1,000 fraudes, generando 350 falsas alarmas por cada 1,000 transacciones.
> - **Naive Bayes:** detectaría 920 fraudes (mejor), pero con 1,100 falsas alarmas (más del triple que SVM).
>
> **Recomendación:** Implementar SVM. Aunque detecta ligeramente menos fraudes que Naive Bayes, reduce significativamente las falsas alarmas, disminuyendo costos operativos y mejorando la experiencia del cliente.

### 5.2 Explicación de Trade-offs

| Trade-off | Explicación para Negocio |
|-----------|--------------------------|
| **Precision vs Recall** | "No podemos maximizar ambos. Si queremos detectar más fraudes (mayor recall), aceptamos más falsas alarmas (menor precisión), lo que aumenta el trabajo del equipo de seguridad. La decisión depende del costo de cada tipo de error." |
| **Interpretabilidad vs Precisión** | "Naive Bayes es más simple de explicar (podemos decir qué palabras indican spam), pero SVM es más preciso aunque funciona como 'caja negra'. Para cumplimiento regulatorio, a veces necesitamos el modelo interpretable." |
| **Complejidad vs Tiempo de Respuesta** | "KNN es simple pero lento en predicción; para un sistema de tiempo real que debe responder en milisegundos, SVM o Naive Bayes son mejores opciones." |

### 5.3 Justificación del Modelo Seleccionado

**Template para justificar la selección:**

> **Modelo seleccionado:** [Nombre del modelo]
>
> **Justificación basada en:**
> 1. **Requisitos de negocio:** [ej. priorizamos recall porque cada fraude no detectado cuesta $X]
> 2. **Características de los datos:** [ej. alta dimensionalidad, relaciones no lineales, etc.]
> 3. **Restricciones operativas:** [ej. tiempo de predicción < 100ms, capacidad computacional]
> 4. **Métricas comparativas:** [ej. F1-score superior en 15%, AUC 0.93 vs 0.88 del segundo mejor]
>
> **Impacto esperado:** [cuantificar en términos de negocio: fraudes detectados, ahorro estimado, clientes afectados]

**Ejemplo completo:**

> **Modelo seleccionado:** SVM con kernel RBF
>
> **Justificación:**
> - **Negocio:** Necesitamos balancear la detección de fraudes (Recall) con la experiencia del cliente (evitar falsos bloqueos). SVM ofrece el mejor F1-score (0.77), indicando el mejor balance.
> - **Datos:** Las transacciones tienen relaciones no lineales que SVM captura mejor mediante kernel RBF, mientras que Naive Bayes asume independencia (no realista aquí).
> - **Operaciones:** El sistema puede tolerar hasta 200ms por predicción; SVM cumple (80ms en pruebas de carga).
> - **Métrica:** SVM supera a KNN en AUC (0.93 vs 0.88) y a Naive Bayes en F1 (0.77 vs 0.60).
>
> **Impacto estimado:** Reducción del 35% en falsos positivos respecto al modelo anterior (Naive Bayes), manteniendo tasa de detección de fraudes >85%. Ahorro estimado: $200,000 anuales en investigación manual y 15,000 clientes no afectados por bloqueos incorrectos.

---

## Comparación Conceptual entre Algoritmos

* **Paramétricos vs no paramétricos:**
  * **Paramétricos (Naive Bayes):** hacen supuestos sobre la distribución de los datos y tienen un número fijo de parámetros independiente del tamaño de la muestra.
  * **No paramétricos (KNN, SVM):** no asumen una forma funcional específica; su complejidad crece con los datos.

* **Interpretabilidad vs performance:**
  * **Naive Bayes:** altamente interpretable (probabilidades condicionales por feature).
  * **KNN:** interpretable (podemos examinar los vecinos más cercanos).
  * **SVM (lineal):** moderadamente interpretable (coeficientes del hiperplano).
  * **SVM (no lineal):** baja interpretabilidad (caja negra).

* **Escalabilidad computacional:**
  * **KNN:** entrenamiento rápido, predicción lenta (costosa).
  * **Naive Bayes:** entrenamiento y predicción muy rápidos.
  * **SVM:** entrenamiento costoso, predicción eficiente.

* **Casos de uso recomendados:**
  * **KNN:** datos de baja dimensión, relaciones locales, necesidad de explicabilidad.
  * **Naive Bayes:** clasificación de texto, alta dimensionalidad, tiempo real.
  * **SVM:** problemas con fronteras complejas, datasets medianos, máxima precisión.



---

## Reto (1 punto)

* Comparar KNN, Naive Bayes y SVM en un mismo dataset.
* Evaluar métricas y tiempos de entrenamiento.
* Justificar cuál modelo es más adecuado según el caso de negocio.

---

## Laboratorio

* Implementación de **KNN, Naive Bayes y SVM** usando Scikit-learn.
* **Pipeline recomendado:**

  1. Datos preprocesados (EDA + feature engineering ya realizado).
  2. Escalamiento de variables (crucial para KNN y SVM).
  3. Entrenamiento de múltiples modelos.
  4. Evaluación usando métricas consistentes.
  5. Comparación de resultados y selección final del modelo.

---






## Anexo: Fundamento matemático y computacional

### 1. KNN – Distancias y espacios métricos

**Definición del espacio:**

* Sea un conjunto de puntos ({x_1, x_2, ..., x_n}) en (\mathbb{R}^d), donde cada punto (x_i = (x_{i1}, x_{i2}, ..., x_{id})) es un vector de características.
* Cada punto tiene una etiqueta de clase (y_i \in {1,2,...,K}).

**Distancias:**

1. **Euclidiana (L2)**:
   [
   d(x_i, x_j) = \sqrt{\sum_{k=1}^{d} (x_{ik} - x_{jk})^2}
   ]

2. **Manhattan (L1)**:
   [
   d(x_i, x_j) = \sum_{k=1}^{d} |x_{ik} - x_{jk}|
   ]

3. **Minkowski (Lp)**:
   [
   d(x_i, x_j) = \left( \sum_{k=1}^{d} |x_{ik} - x_{jk}|^p \right)^{1/p}, \quad p \ge 1
   ]

**Clasificación:**

* Sea (N_k(x)) el conjunto de los (k) vecinos más cercanos de un punto (x).
* El punto se clasifica asignándole la clase más frecuente entre los vecinos:
  [
  \hat{y} = \arg\max_{c \in {1,...,K}} \sum_{x_j \in N_k(x)} \mathbf{1}*{{y_j = c}}
  ]
  donde (\mathbf{1}*{{y_j = c}}) es la función indicadora.

**Complejidad computacional:**

* Entrenamiento: (O(1)) (no hay parámetros que ajustar).
* Predicción: (O(n \cdot d)) por punto (distancia a cada ejemplo).
* Optimización: estructuras como KD-trees o Ball-trees reducen la predicción a (O(\log n \cdot d)) en espacios de baja dimensión.

---

### 2. Naive Bayes – Teorema de Bayes y probabilidades condicionales

**Probabilidad condicional y Bayes:**

* Para un vector de características (X = (x_1,...,x_d)) y una clase (C_k):
  [
  P(C_k | X) = \frac{P(X | C_k) P(C_k)}{P(X)}
  ]
* (P(C_k)) es la probabilidad a priori de la clase.
* (P(X | C_k)) es la probabilidad de observar (X) dado que pertenece a la clase (C_k).

**Supuesto de independencia condicional (Naive):**
[
P(X | C_k) = \prod_{i=1}^{d} P(x_i | C_k)
]

**Decisión de clasificación (MAP):**
[
\hat{y} = \arg\max_k P(C_k) \prod_{i=1}^{d} P(x_i | C_k)
]

**Ejemplo concreto (Gaussian Naive Bayes):**

* Para variables continuas:
  [
  P(x_i | C_k) = \frac{1}{\sqrt{2 \pi \sigma_{ik}^2}} \exp \left( -\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2} \right)
  ]
* (\mu_{ik}) y (\sigma_{ik}^2) se estiman con la media y varianza de los datos de entrenamiento de clase (C_k).

**Complejidad computacional:**

* Entrenamiento: (O(n \cdot d)) (conteo de frecuencias o cálculo de media/varianza).
* Predicción: (O(d \cdot K)) por punto, con (K) clases.

---

### 3. Support Vector Machines (SVM) – Optimización de margen máximo

**Problema linealmente separable:**

* Para un dataset ({(x_i, y_i)}_{i=1}^n) con (y_i \in {-1,1}), el **margen** es la distancia entre las fronteras de separación de clases.
* El **margen máximo** se obtiene resolviendo:
  [
  \max_{\mathbf{w}, b} \frac{2}{|\mathbf{w}|}
  ]
* Equivalente a minimizar la norma del vector de pesos:
  [
  \min_{\mathbf{w},b} \frac{1}{2} |\mathbf{w}|^2 \quad
  \text{sujeto a } y_i (\mathbf{w}^T x_i + b) \ge 1, \forall i
  ]

**Lagrangiano y dual:**

* Introducimos multiplicadores (\alpha_i \ge 0):
  [
  \mathcal{L}(\mathbf{w}, b, \alpha) = \frac{1}{2} |\mathbf{w}|^2 - \sum_{i=1}^{n} \alpha_i [y_i (\mathbf{w}^T x_i + b) - 1]
  ]
* Derivadas parciales ((\nabla_\mathbf{w}), (\partial \mathcal{L}/\partial b = 0)) dan la solución:
  [
  \mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i x_i, \quad \sum_{i=1}^{n} \alpha_i y_i = 0
  ]

**Kernels para no linealidad:**

* Se define un mapeo (\phi: \mathbb{R}^d \to \mathbb{R}^p) y kernel (K(x_i, x_j) = \phi(x_i)^T \phi(x_j)).
* Predicción para un nuevo punto (x):
  [
  f(x) = \text{sign} \left( \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b \right)
  ]
  donde (SV) son los vectores de soporte ((\alpha_i > 0)).

**Regularización para margen suave (soft-margin SVM):**
[
\min_{\mathbf{w},b,\xi} \frac{1}{2}|\mathbf{w}|^2 + C \sum_{i=1}^{n} \xi_i
]
[
\text{sujeto a } y_i(\mathbf{w}^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
]

* (C) controla el trade-off entre margen amplio y errores de clasificación.

**Complejidad computacional:**

* Entrenamiento: SVM lineal (O(n \cdot d)) a (O(n^2 \cdot d)), SVM no lineal ~ (O(n^3)) para métodos clásicos.
* Predicción: (O(|SV| \cdot d)) por punto.

---
