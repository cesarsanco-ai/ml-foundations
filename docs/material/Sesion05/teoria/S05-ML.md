---
layout: default
---
# Semana 5: kNN, Naive Bayes, SVM

## Logro de la sesión

Construir, comparar e interpretar modelos de clasificación utilizando algoritmos clásicos como **K-Nearest Neighbors (KNN)**, **Naive Bayes** y **Support Vector Machines (SVM)**, comprendiendo sus supuestos, ventajas y limitaciones en distintos contextos de negocio.

---

## Problemática de negocio

En diversos contextos de negocio, la tarea de clasificación adquiere matices particulares que condicionan la elección del algoritmo adecuado. Por ejemplo, en sectores como la banca o el comercio electrónico, es común enfrentarse a datasets con cientos de variables (alta dimensionalidad), relaciones complejas entre ellas (no linealidad) o volúmenes masivos de datos que imponen restricciones computacionales. En estos escenarios, la regresión logística, si bien es un excelente punto de partida, puede quedar limitada. Por ello, se hace necesario explorar algoritmos clásicos de clasificación que se adapten a diferentes casuísticas:

- **K-Nearest Neighbors (KNN):** ideal para sistemas de recomendación simples o detección de similitud entre clientes/productos, donde la relación local entre puntos es relevante.
- **Naive Bayes:** muy utilizado en clasificación de texto (spam, análisis de sentimiento) por su eficiencia en alta dimensionalidad y su capacidad de trabajar con probabilidades.
- **Support Vector Machines (SVM):** sobresale en problemas con fronteras de decisión no lineales, aplicándose con éxito en visión por computadora y bioinformática.

La selección del modelo más adecuado requiere comparar su desempeño mediante métricas robustas, considerar el impacto del escalamiento de los datos y evaluar el trade-off entre precisión, interpretabilidad y costo computacional.

---


## Modelado

# K‑Nearest Neighbors (KNN) – Ficha técnica para modelado

## 1. Concepto fundamental
KNN es un algoritmo **no paramétrico** y **basado en instancias** (lazy learning). No aprende parámetros durante el entrenamiento; simplemente almacena todos los datos. Para hacer una predicción, calcula la distancia entre el nuevo punto y **todos** los puntos del conjunto de entrenamiento, selecciona los \(k\) más cercanos y devuelve:
- **Clasificación**: la clase mayoritaria entre esos \(k\) vecinos.
- **Regresión**: el promedio (o mediana) de los valores de esos \(k\) vecinos.

**Fórmula de distancia más común (Euclidiana):**  
\[
d(p,q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
\]

---

## 2. Problemas de negocio típicos (¿cuándo es candidato?)

KNN es especialmente adecuado cuando el problema gira en torno al concepto de **similitud** y los datos permiten definir una distancia significativa. A continuación se presentan casos de negocio donde KNN suele ser una excelente primera opción (baseline) o incluso la solución final si la dimensionalidad y el volumen de datos son moderados.

| Categoría | Ejemplo de problema de negocio | ¿Por qué KNN encaja? |
|-----------|--------------------------------|----------------------|
| **Recomendación / Personalización** | “Clientes similares a ti también compraron…” (sistemas de recomendación básicos) | La similitud entre usuarios (basada en compras, clics, valoraciones) es el núcleo del problema. KNN mide directamente esa cercanía. |
| **Análisis espacial / Geográfico** | Estimación de precios de inmuebles basada en propiedades vecinas | La distancia geográfica (coordenadas) es una métrica natural. Además, características como metros cuadrados o antigüedad se pueden combinar. |
| **Detección de anomalías** | Identificar transacciones fraudulentas (un fraude suele estar “lejos” de las transacciones normales) | KNN puede calcular la distancia media a los \(k\) vecinos más cercanos; un valor anómalo tendrá una distancia grande. |
| **Segmentación / Perfilado** | Asignar automáticamente un nuevo cliente a un perfil de riesgo (bajo, medio, alto) basado en su historial financiero | Se puede explicar fácilmente: “su perfil es idéntico al de estos 5 clientes históricos que sabemos que son de alto riesgo”. |
| **Diagnóstico / Clasificación médica** | Clasificar un tumor como benigno o maligno según parámetros de imagen | Cuando hay pocas variables (dimensiones) y la relación es no lineal, KNN puede alcanzar muy buen rendimiento con interpretabilidad total. |
| **Control de calidad / Manufactura** | Detectar productos defectuosos según medidas de sensores | Los productos similares (con medidas cercanas) suelen tener el mismo estado. KNN permite clasificar por “parecido” sin suposiciones de forma. |

**Idea clave:** Si el problema se puede plantear como *“dame los casos más parecidos a este y dime qué pasa con ellos”*, KNN es un candidato natural.

---

## 3. Requisitos y características de los datos

| Característica | Condición / Impacto en KNN |
|----------------|----------------------------|
| **Escalado de variables** | **Obligatorio**. Sin escalado, las variables con mayor rango dominan la distancia. Usar estandarización (Z‑score) o normalización (Min‑Max). |
| **Tipo de datos** | Nativamente **numéricos continuos**. Las categóricas deben transformarse (One‑Hot Encoding, codificación ordinal, o usar métricas como Hamming). |
| **Dimensionalidad** | **Sensible a la maldición de la dimensionalidad**. A medida que aumentan las variables, la distancia entre puntos tiende a igualarse. Funciona mejor con dimensionalidad baja o media (< 20). Si hay muchas, aplicar reducción (PCA, selección de características). |
| **Desbalance de clases** | **Muy sensible**. La clase mayoritaria domina los vecindarios. Requiere balanceo (SMOTE, undersampling) o usar pesos por distancia. |
| **Outliers** | **Altamente sensible**. Un outlier cercano puede distorsionar la predicción. Limpiar datos o usar distancias robustas. |
| **Volumen de datos** | Para inferencia, necesita recorrer todos los datos de entrenamiento. Conjuntos muy grandes (> 100k) pueden volverse lentos en producción. |

---

## 4. Hiperparámetros críticos y su impacto

| Hiperparámetro | Descripción | Impacto en sesgo‑varianza |
|----------------|-------------|---------------------------|
| **\(k\)** (número de vecinos) | Número de puntos vecinos que participan en la decisión. | \(k\) pequeño → alta varianza (sobreajuste). \(k\) grande → alto sesgo (subajuste). Se optimiza con validación cruzada. |
| **Métrica de distancia** | Euclidiana, Manhattan, Minkowski, coseno, etc. | Define qué se considera “cercano”. Elegir según la naturaleza de los datos (Manhattan es más robusta a outliers; coseno funciona bien en alta dimensión dispersa). |
| **Pesos (weights)** | `uniform` (todos votan igual) o `distance` (los más cercanos tienen más peso). | `distance` mitiga el efecto de vecinos lejanos y puede ayudar en datos desbalanceados. |
| **Estructura de aceleración** | KD‑Tree, Ball Tree (opcional). | Reduce la complejidad de búsqueda en baja dimensionalidad. No suele mejorar en alta dimensión. |

---

## 5. Ventajas y desventajas

| Ventajas | Desventajas |
|----------|-------------|
| Fácil de entender e implementar. | Coste de predicción alto: \(O(n \cdot d)\) por punto. |
| No asume ninguna distribución subyacente (no paramétrico). | Necesita almacenar todo el dataset en memoria. |
| Puede adaptarse a fronteras de decisión complejas. | Muy sensible a la escala, outliers y dimensionalidad alta. |
| Altamente interpretable (explicaciones basadas en vecinos). | Sufre con clases desbalanceadas sin ajustes. |
| No requiere entrenamiento (fase de entrenamiento instantánea). | En producción, cada predicción es costosa. |

---

## 6. Interpretabilidad
**Muy alta (modelo de caja blanca).**  
Cada predicción se puede justificar mostrando los \(k\) vecinos que la determinaron. Por ejemplo:  
> *“Este préstamo se clasifica como de alto riesgo porque sus características (ingreso, deuda, historial) son prácticamente idénticas a las de los 5 casos de impago que tenemos en el histórico.”*  

Esta transparencia es una ventaja competitiva en sectores regulados (finanzas, salud) donde se exige explicabilidad.

---

## 7. Escalabilidad y poder de cómputo

| Fase | Complejidad | Observaciones |
|------|-------------|---------------|
| **Entrenamiento** | \(O(1)\) (solo almacenar) | No hay cómputo de parámetros. |
| **Predicción (inferencia)** | \(O(n \cdot d)\) | Para cada nuevo punto, calcula distancias con todos los \(n\) puntos de entrenamiento y \(d\) dimensiones. Con estructuras de indexación (KD‑Tree) puede bajar a \(O(\log n \cdot d)\) en casos favorables (baja \(d\)). |
| **Memoria** | \(O(n \cdot d)\) | Necesita mantener todo el conjunto de entrenamiento en RAM. |

**Consejos de escalado:**  
- Para conjuntos grandes, considerar **búsqueda aproximada de vecinos** (ANNOY, HNSW, Faiss).  
- Reducir el dataset usando técnicas de **prototipos** (Condensed Nearest Neighbor).  
- Si la dimensionalidad es alta, aplicar **PCA** antes de KNN para ganar velocidad y mitigar la maldición.

---

## 8. Consideraciones prácticas

- **Siempre normalizar/estandarizar** antes de entrenar.
- **Optimizar \(k\) y la métrica** con validación cruzada estratificada.
- **Balancear** el dataset si las clases son desiguales.
- **Evaluar el coste de predicción** en producción si el volumen de entrenamiento es grande.
- **Aprovechar su interpretabilidad** para explicar predicciones a stakeholders.
- **Usar como baseline** antes de probar modelos más complejos (bosques, redes neuronales). Si KNN ya da un buen resultado, es una señal de que la similitud es un factor relevante.

---




**Plantilla en Python con scikit‑learn**

```python
# ==============================================
# PLANTILLA PARA KNN (K-NEAREST NEIGHBORS) - CLASIFICACIÓN
# ==============================================
# Esta plantilla está optimizada para KNN, pero mantiene una estructura
# que podrás replicar para otros modelos de ML.
# Comentarios "FIXED" = partes específicas del algoritmo KNN.
# Comentarios "VARIABLE" = partes que debes ajustar según tu problema/datos.
# ==============================================

# 1. IMPORTS NECESARIOS
# -----------------------
# FIXED: KNN requiere estos imports de scikit-learn.
# VARIABLE: Si usas otro modelo, cambiarías KNeighborsClassifier por otro.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier      # FIXED: Modelo KNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # opcional: para guardar modelo

# 2. CARGA DE DATOS
# -----------------
# VARIABLE: Ajusta la ruta y nombre de tu archivo, así como la columna objetivo.
df = pd.read_csv('tu_dataset.csv')          # <-- Cambia aquí
X = df.drop('target', axis=1)               # <-- Reemplaza 'target' por tu columna objetivo
y = df['target']                            # <-- Columna objetivo

# 3. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
# --------------------------------------
# VARIABLE: Ajusta test_size, random_state y stratify (útil para clasificación desbalanceada).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. PREPROCESAMIENTO: ESCALADO
# -------------------------------
# FIXED: KNN ES OBLIGATORIO escalar las variables porque se basa en distancias.
# VARIABLE: Elige StandardScaler (media=0, std=1) o MinMaxScaler (rango [0,1]).
# Nota: Guarda el scaler para usarlo en producción.
scaler = StandardScaler()                   # <-- O MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. DEFINICIÓN DEL MODELO BASE
# -----------------------------
# FIXED: Configura los hiperparámetros iniciales de KNN.
# VARIABLE: Puedes cambiar los valores por defecto según conocimiento previo.
model_base = KNeighborsClassifier(
    n_neighbors=5,          # <-- k (número de vecinos)
    metric='minkowski',     # <-- métrica de distancia: 'euclidean', 'manhattan', etc.
    p=2,                    # p=2 → Euclidiana; p=1 → Manhattan
    weights='uniform'       # 'uniform' (todos votan igual) o 'distance' (peso inverso)
)

# 6. (OPCIONAL) BÚSQUEDA DE HIPERPARÁMETROS CON VALIDACIÓN CRUZADA
# -----------------------------------------------------------------
# FIXED: Define el grid típico para KNN.
# VARIABLE: Ajusta los rangos según tu dataset (ej. k máximo puede ser √n).
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],          # <-- Valores de k a probar
    'weights': ['uniform', 'distance'],       # <-- Tipo de votación
    'metric': ['euclidean', 'manhattan']      # <-- Métricas de distancia
}

# FIXED: GridSearchCV con validación cruzada (5 folds) y métrica 'accuracy'.
grid_search = GridSearchCV(
    estimator=model_base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Entrenar la búsqueda (descomenta si quieres optimizar automáticamente)
# grid_search.fit(X_train_scaled, y_train)
# mejor_modelo = grid_search.best_estimator_
# print("Mejores hiperparámetros:", grid_search.best_params_)

# Si no usas búsqueda, el modelo_base es el que entrenamos.
# Aquí asumimos que usamos el modelo_base (sin optimización) para simplificar.
mejor_modelo = model_base

# 7. ENTRENAMIENTO DEL MODELO
# ----------------------------
# FIXED: KNN no entrena parámetros, solo almacena los datos.
mejor_modelo.fit(X_train_scaled, y_train)

# 8. PREDICCIONES
# ---------------
y_pred = mejor_modelo.predict(X_test_scaled)

# 9. EVALUACIÓN DEL MODELO
# ------------------------
# VARIABLE: Puedes añadir AUC, F1, etc. según el problema.
print("Accuracy en test:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. (OPCIONAL) VALIDACIÓN CRUZADA EN ENTRENAMIENTO
# --------------------------------------------------
# FIXED: cross_val_score para estimar estabilidad.
cv_scores = cross_val_score(mejor_modelo, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Validación cruzada (5 folds): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 11. (OPCIONAL) GUARDAR MODELO Y ESCALER PARA PRODUCCIÓN
# -------------------------------------------------------
# joblib.dump(mejor_modelo, 'modelo_knn.pkl')
# joblib.dump(scaler, 'scaler_knn.pkl')
# print("Modelo y escaler guardados.")

# ==============================================
# NOTAS ESPECÍFICAS PARA KNN:
# - El escalado es OBLIGATORIO. Si no escalas, las variables con mayor rango dominarán.
# - KNN sufre la "maldición de la dimensionalidad": evita usar muchas variables (>20).
# - Es sensible a outliers: limpia los datos antes si es necesario.
# - La fase de predicción es lenta (O(n*d)) porque calcula distancias con todos los puntos.
# - Para datasets grandes (>100k), considera búsqueda aproximada (ANNOY, HNSW) o reducir datos.
# ==============================================
```

---

### Naive Bayes

# Naive Bayes – Ficha técnica para modelado

## 1. Concepto fundamental

Naive Bayes es un algoritmo de clasificación **supervisado**, **probabilístico** y **paramétrico** basado en el **teorema de Bayes**. Asume que los predictores son **condicionalmente independientes** dada la clase (supuesto “naive”). Esta independencia condicional simplifica drásticamente el cálculo de la probabilidad posterior:

\[
P(C_k \mid \mathbf{x}) = \frac{P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)}{P(\mathbf{x})}
\]

donde \(P(C_k)\) es la probabilidad a priori de la clase, \(P(x_i \mid C_k)\) es la probabilidad condicional de la característica \(i\) dada la clase, y el denominador es un factor de normalización.

El modelo asume una distribución específica para las características según el tipo de datos:
- **GaussianNB**: variables continuas con distribución normal.
- **MultinomialNB**: variables de frecuencia (counts), típico en texto.
- **BernoulliNB**: variables binarias (presencia/ausencia).

---

## 2. Problemas de negocio típicos (¿cuándo es candidato?)

Naive Bayes brilla cuando se necesita un modelo rápido, con buena interpretabilidad y que funcione bien con datos de alta dimensionalidad, especialmente en clasificación de texto. Es un excelente baseline antes de probar modelos más complejos.

| Categoría | Ejemplo concreto de problema de negocio | Razón por la que este modelo encaja |
|-----------|----------------------------------------|-------------------------------------|
| **Clasificación de texto / NLP** | Clasificar emails como spam / no spam, análisis de sentimiento en reseñas de clientes, categorización de tickets de soporte | Naive Bayes Multinomial/Bernoulli son extremadamente eficaces con bolsas de palabras y alta dimensionalidad dispersa. |
| **Filtrado de contenido / moderación** | Detectar comentarios ofensivos en redes sociales, clasificar contenido adulto | Modelo rápido y con buena precisión en presencia de muchas características binarias. |
| **Diagnóstico médico / riesgo** | Clasificar pacientes en grupos de riesgo basado en síntomas o resultados de pruebas | Aunque el supuesto de independencia es fuerte, en la práctica funciona bien y ofrece una base probabilística clara. |
| **Segmentación de clientes** | Predecir abandono (churn) en telecomunicaciones usando variables demográficas y de uso | Permite incorporar fácilmente nuevas variables y ajustar probabilidades a priori. |
| **Recomendación simple / personalización** | Predecir si un usuario hará clic en un anuncio según su historial de navegación | Rápido de entrenar y actualizar en entornos con cambios frecuentes de datos. |

**Idea clave:** Naive Bayes es ideal cuando necesitas un modelo **rápido de entrenar**, **escalable a muchas variables** y con una **base probabilística interpretable**, incluso si los datos no cumplen estrictamente el supuesto de independencia.

---

## 3. Requisitos y características de los datos

| Característica | Condición / Impacto en Naive Bayes |
|----------------|-------------------------------------|
| **Escalado de variables** | **Irrelevante** para las versiones basadas en frecuencia (MultinomialNB, BernoulliNB) porque se basan en conteos o presencia. Para GaussianNB el escalado no afecta a las estimaciones de media y varianza, aunque puede ayudar en la interpretación de coeficientes. |
| **Tipo de datos** | Depende de la variante: GaussianNB requiere variables **continuas** (idealmente normales); MultinomialNB requiere **conteos no negativos** (frecuencias); BernoulliNB requiere **variables binarias** (0/1). Pueden combinarse transformando variables categóricas con One‑Hot Encoding y usando BernoulliNB. |
| **Dimensionalidad** | **Muy robusto** a la alta dimensionalidad gracias al supuesto de independencia. Es uno de los pocos modelos que funcionan bien con cientos de miles de características (ej. texto). |
| **Desbalance de clases** | **Moderadamente sensible**. El modelo utiliza las probabilidades a priori de las clases; si una clase es muy minoritaria, la predicción puede verse afectada. Se puede mitigar ajustando las probabilidades a priori o usando técnicas de balanceo. |
| **Outliers** | **Poco sensible** (especialmente en Multinomial y Bernoulli) porque se basan en distribuciones discretas. En GaussianNB, los outliers pueden afectar las estimaciones de media y varianza, pero el efecto es limitado si se usa una variante robusta (ej. complement Naive Bayes). |
| **Volumen de datos** | Entrenamiento muy rápido (una sola pasada para calcular probabilidades). Inferencia extremadamente rápida (producto de probabilidades). Escala linealmente con el número de muestras y características. |

---

## 4. Hiperparámetros críticos y su impacto

| Hiperparámetro | Descripción | Impacto en sesgo‑varianza |
|----------------|-------------|---------------------------|
| **`alpha`** (suavizado de Laplace) | Parámetro de suavizado para evitar probabilidades cero. `alpha=1` es Laplace, `alpha<1` es Lidstone. | Un `alpha` grande introduce sesgo al alejar las probabilidades de los valores observados; un `alpha` muy pequeño puede causar sobreajuste a frecuencias raras. |
| **`fit_prior`** | Indica si se deben aprender las probabilidades a priori de las clases a partir de los datos (`True`) o usar uniformes (`False`). | Si las clases están desbalanceadas, `fit_prior=True` permite capturar la desproporción. Si se fija uniforme, se evita sesgo hacia la clase mayoritaria pero puede perjudicar si las clases son muy desiguales. |
| **`var_smoothing`** (GaussianNB) | Parámetro de suavizado para la varianza, evita divisiones por cero y estabiliza las estimaciones. | Valores pequeños pueden hacer el modelo más sensible a fluctuaciones (mayor varianza); valores grandes aumentan el sesgo. |
| **`binarize`** (BernoulliNB) | Umbral para convertir variables continuas en binarias. | Define el punto de corte entre presencia/ausencia. Impacta directamente la interpretación de las características. |

**Optimización:** `alpha` y `var_smoothing` se optimizan con validación cruzada, típicamente en escala logarítmica.

---

## 5. Ventajas y desventajas

| Ventajas | Desventajas |
|----------|-------------|
| Extremadamente rápido de entrenar e inferir (adecuado para tiempo real). | Supuesto de independencia condicional muy fuerte, raramente se cumple en la realidad. |
| Escala muy bien con el número de características (alta dimensionalidad). | Puede tener un rendimiento inferior a modelos más complejos cuando hay correlaciones importantes entre variables. |
| Funciona bien incluso con conjuntos de datos pequeños. | Las estimaciones de probabilidad pueden no estar bien calibradas (sobreconfianza). |
| Muy interpretable: las probabilidades condicionales revelan la influencia de cada variable. | Sensible a la presencia de características irrelevantes (aunque menos que otros modelos). |
| Maneja naturalmente datos faltantes (se ignoran en la suma de probabilidades). | Para datos continuos, la asunción de normalidad en GaussianNB puede no ajustarse. |

---

## 6. Interpretabilidad

**Muy alta (caja blanca).** Cada predicción se basa en la probabilidad posterior calculada explícitamente. Se puede descomponer la contribución de cada variable: las probabilidades condicionales \(P(x_i \mid C_k)\) muestran cómo cada característica apoya o contradice cada clase.

**Explicación para negocio:**  
> *“El modelo predice que este email es spam porque, dado que es spam, la probabilidad de que contenga la palabra ‘ganaste’ es 0.8, mientras que para correo normal es solo 0.02; además, la probabilidad a priori de spam en nuestros datos es del 40%.”*  

Esta transparencia es clave en entornos donde se necesita justificar decisiones (cumplimiento normativo, auditoría).

---

## 7. Escalabilidad y poder de cómputo

| Fase | Complejidad | Observaciones |
|------|-------------|---------------|
| **Entrenamiento** | \(O(n \cdot d)\) | Una sola pasada para calcular frecuencias (o estimar medias/varianzas). Es uno de los algoritmos más rápidos en entrenamiento. |
| **Predicción (inferencia)** | \(O(d \cdot k)\) | Multiplicación de probabilidades condicionales para cada clase. Extremadamente rápida, incluso para millones de características. |
| **Memoria** | \(O(d \cdot k)\) | Solo almacena las probabilidades condicionales (matriz de tamaño características × clases). Muy bajo consumo. |

**Escalado:** Naive Bayes es inherentemente escalable. Puede entrenarse en streaming (aprendizaje online) actualizando las probabilidades sin necesidad de reprocesar todo el dataset.

---

## 8. Consideraciones prácticas

- **Elegir la variante correcta:**  
  - **GaussianNB** para variables continuas con distribución aproximadamente normal.  
  - **MultinomialNB** para datos de frecuencia (texto, conteos).  
  - **BernoulliNB** para datos binarios (presencia/ausencia).  
  - **ComplementNB** (variante) para clases muy desbalanceadas.

- **Preprocesamiento:**  
  - En texto, aplicar vectorización (CountVectorizer, TfidfVectorizer) y luego usar MultinomialNB.  
  - Para GaussianNB, aunque el escalado no es obligatorio, ayuda a la interpretación si se estandarizan las variables.

- **Manejo de clases desbalanceadas:**  
  - Ajustar `fit_prior=False` y usar una prior uniforme, o bien balancear el dataset antes del entrenamiento.  
  - ComplementNB está diseñado para mejorar el rendimiento en datos desbalanceados.

- **Suavizado (`alpha`):**  
  - Probar valores pequeños (0.1, 0.5, 1) con validación cruzada. Para texto, a menudo `alpha=1` es un buen punto de partida.

- **Evaluación de probabilidades:**  
  - Las probabilidades predichas suelen estar mal calibradas. Si se necesitan probabilidades bien calibradas, aplicar **calibración de probabilidades** (CalibratedClassifierCV).

- **Limitación de variables independientes:**  
  - Aunque el modelo es robusto, eliminar variables claramente redundantes (mediante selección de características) puede mejorar la interpretabilidad y reducir el ruido.

- **Uso en producción:**  
  - Modelo ideal para sistemas de baja latencia (p.ej., clasificación en tiempo real).  
  - Se puede actualizar incrementalmente añadiendo nuevas observaciones sin reentrenar desde cero.

**Plantilla en Python con scikit‑learn**

```python
# ==============================================
# PLANTILLA PARA NAIVE BAYES (GAUSSIANNB) - CLASIFICACIÓN
# ==============================================
# Esta plantilla está optimizada para GaussianNB (variables continuas).
# Para datos de texto/frecuencia usa MultinomialNB.
# Para datos binarios usa BernoulliNB.
# Comentarios "FIXED" = partes específicas del algoritmo Naive Bayes.
# Comentarios "VARIABLE" = partes que debes ajustar según tu problema/datos.
# ==============================================

# 1. IMPORTS NECESARIOS
# -----------------------
# FIXED: Imports para Naive Bayes (GaussianNB como ejemplo).
# VARIABLE: Si usas otra variante, cambia el import correspondiente.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler   # No necesario para NB, pero se deja por si acaso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# FIXED: Import del modelo elegido (GaussianNB, MultinomialNB, BernoulliNB)
from sklearn.naive_bayes import GaussianNB   # <-- Cambia a MultinomialNB o BernoulliNB según datos
import joblib

# 2. CARGA DE DATOS
# -----------------
# VARIABLE: Ajusta la ruta y nombre de tu archivo, así como la columna objetivo.
df = pd.read_csv('tu_dataset.csv')          # <-- Cambia aquí
X = df.drop('target', axis=1)               # <-- Reemplaza 'target' por tu columna objetivo
y = df['target']                            # <-- Columna objetivo

# 3. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
# --------------------------------------
# VARIABLE: Ajusta test_size, random_state y stratify (útil para clasificación desbalanceada).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. PREPROCESAMIENTO: ESCALADO (NO NECESARIO PARA NB)
# ----------------------------------------------------
# FIXED: Naive Bayes NO requiere escalado. Las distribuciones se estiman a partir de los valores originales.
# VARIABLE: Si por alguna razón necesitas escalar (ej. para interpretación), puedes activarlo,
# pero no afecta al rendimiento del modelo.
scaler = None   # <-- Mantén None para no escalar
if scaler is not None:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled, X_test_scaled = X_train, X_test

# 5. DEFINICIÓN DEL MODELO BASE
# -----------------------------
# FIXED: Configura los hiperparámetros iniciales de Naive Bayes.
# VARIABLE: Ajusta según la variante elegida.
# Para GaussianNB:
model_base = GaussianNB(
    var_smoothing=1e-9        # Parámetro de suavizado de varianza (evita divisiones por cero)
)
# Para MultinomialNB (descomentar):
# model_base = MultinomialNB(alpha=1.0, fit_prior=True)
# Para BernoulliNB (descomentar):
# model_base = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)

# 6. (OPCIONAL) BÚSQUEDA DE HIPERPARÁMETROS CON VALIDACIÓN CRUZADA
# -----------------------------------------------------------------
# FIXED: Grid típico para cada variante.
# VARIABLE: Ajusta los rangos según tu dataset.
param_grid = {
    # Para GaussianNB:
    'var_smoothing': np.logspace(-12, -3, 10)   # valores desde 1e-12 hasta 1e-3
    # Para MultinomialNB:
    # 'alpha': [0.1, 0.5, 1.0, 2.0],
    # 'fit_prior': [True, False]
    # Para BernoulliNB:
    # 'alpha': [0.1, 0.5, 1.0],
    # 'binarize': [0.0, 0.5, 1.0],
    # 'fit_prior': [True, False]
}

grid_search = GridSearchCV(
    estimator=model_base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',      # Para clasificación; para regresión usa 'neg_mean_squared_error'
    n_jobs=-1,
    verbose=1
)

# Entrenar la búsqueda (descomenta si quieres optimizar automáticamente)
# grid_search.fit(X_train_scaled, y_train)
# mejor_modelo = grid_search.best_estimator_
# print("Mejores hiperparámetros:", grid_search.best_params_)

# Si no usas búsqueda, el modelo_base es el que entrenamos.
mejor_modelo = model_base

# 7. ENTRENAMIENTO DEL MODELO
# ----------------------------
# FIXED: Naive Bayes calcula probabilidades condicionales en una sola pasada.
mejor_modelo.fit(X_train_scaled, y_train)

# 8. PREDICCIONES
# ---------------
y_pred = mejor_modelo.predict(X_test_scaled)
# Para obtener probabilidades (útil para calibrar):
y_proba = mejor_modelo.predict_proba(X_test_scaled)

# 9. EVALUACIÓN DEL MODELO
# ------------------------
print("Accuracy en test:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. (OPCIONAL) VALIDACIÓN CRUZADA EN ENTRENAMIENTO
# --------------------------------------------------
cv_scores = cross_val_score(mejor_modelo, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Validación cruzada (5 folds): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 11. GUARDAR MODELO Y PREPROCESADOR PARA PRODUCCIÓN
# --------------------------------------------------
# FIXED: Guarda el modelo y el preprocesador (si se usó). Naive Bayes no requiere escalador.
joblib.dump(mejor_modelo, 'modelo_naive_bayes.pkl')
if scaler is not None:
    joblib.dump(scaler, 'scaler_naive_bayes.pkl')
print("Modelo guardado (y escalador si aplica).")

# ==============================================
# NOTAS ESPECÍFICAS PARA NAIVE BAYES:
# - **Escalado**: NO necesario. Las estimaciones de probabilidad no dependen de la escala.
# - **Dimensionalidad**: Excelente desempeño en alta dimensión (texto, genómica).
# - **Outliers**: Poco sensible en Multinomial/Bernoulli; en GaussianNB puede afectar las medias y varianzas.
# - **Desbalance**: Se puede manejar con `fit_prior=False` o ajustando las prioridades manualmente.
# - **Calibración**: Las probabilidades predichas suelen estar mal calibradas; usar `CalibratedClassifierCV` si se necesitan probabilidades precisas.
# - **Velocidad**: Entrenamiento e inferencia extremadamente rápidos (escala lineal con n y d).
# - **Elección de variante**:
#   - `GaussianNB`: variables continuas.
#   - `MultinomialNB`: variables de frecuencia (conteos, tf-idf).
#   - `BernoulliNB`: variables binarias (presencia/ausencia).


```

---

### Support Vector Machines (SVM)


# Support Vector Machine (SVM) – Ficha técnica para clasificación

## 1. Concepto fundamental

Support Vector Machine (SVM) es un algoritmo de aprendizaje supervisado **paramétrico** que busca encontrar el **hiperplano óptimo** que separa las clases con el **margen máximo**. Para datos no linealmente separables, SVM utiliza el **kernel trick**, que mapea los datos a un espacio de mayor dimensión donde se vuelven linealmente separables.

**Fórmula básica (caso lineal separable):**

\[
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{sujeto a} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i
\]

donde \(\mathbf{w}\) es el vector normal al hiperplano, \(b\) el sesgo, e \(y_i \in \{-1, +1\}\). Los puntos con \(y_i(\mathbf{w} \cdot \mathbf{x}_i + b) = 1\) son los **vectores de soporte**.

Para datos no separables, se introducen variables de holgura \(\xi_i\) y un parámetro de regularización \(C\). Con el kernel, se reemplaza el producto punto \(\mathbf{x}_i \cdot \mathbf{x}_j\) por \(K(\mathbf{x}_i, \mathbf{x}_j)\).

---

## 2. Problemas de negocio típicos (¿cuándo es candidato?)

SVM destaca cuando se necesita un modelo con buen poder predictivo en problemas de dimensionalidad media, con fronteras de decisión complejas y donde el margen de separación es importante. Es ideal para problemas con muestras moderadas (no demasiado grandes) y donde la interpretabilidad no es la prioridad máxima.

| Categoría | Ejemplo concreto de problema de negocio | Razón por la que este modelo encaja |
|-----------|----------------------------------------|-------------------------------------|
| **Clasificación de imágenes / visión** | Detectar objetos en imágenes, clasificar radiografías, reconocimiento facial | SVM con kernel RBF puede capturar patrones complejos; funciona bien con características extraídas (HOG, SIFT) o con reducción de dimensionalidad previa. |
| **Biomedicina / genómica** | Clasificar tumores como benignos/malignos según expresión genética (muchas variables, pocas muestras) | SVM es robusto en escenarios de alta dimensionalidad con pocas muestras, gracias a la regularización y el kernel. |
| **Detección de fraudes / anomalías** | Identificar transacciones fraudulentas en tiempo real | SVM con kernel RBF puede aprender fronteras no lineales; el parámetro C controla el equilibrio entre ajuste y generalización. |
| **Marketing / segmentación** | Clasificar clientes en grupos de alto/bajo valor de vida (LTV) basado en comportamiento | SVM puede manejar datos mixtos (numéricos y categóricos transformados) y funciona bien cuando hay una separación clara de clases. |
| **Análisis de texto / sentimiento** | Clasificar reseñas en positivas/negativas (con representación vectorial como TF‑IDF) | SVM lineal (kernel lineal) es muy efectivo en texto de alta dimensionalidad, con buen rendimiento y rapidez en inferencia. |

**Idea clave:** SVM es el modelo de elección cuando se tienen **datos de dimensionalidad media (cientos a miles de variables) y muestras moderadas (miles a decenas de miles)** , especialmente si se esperan **fronteras de decisión no lineales** y se busca **maximizar el margen de separación**.

---

## 3. Requisitos y características de los datos

| Característica | Condición / Impacto en SVM |
|----------------|----------------------------|
| **Escalado de variables** | **Obligatorio e indispensable**. SVM es extremadamente sensible a la escala de las variables porque el margen se mide en el espacio de características. Sin escalado, las variables con mayor rango dominan la función de decisión. |
| **Tipo de datos** | **Numéricos**. Las categóricas deben codificarse adecuadamente (One‑Hot Encoding, etc.). El kernel trabaja sobre distancias o productos punto, por lo que los datos deben ser numéricos. |
| **Dimensionalidad** | **Robusto** gracias a la regularización y al kernel. Puede manejar alta dimensionalidad (miles de variables) incluso con pocas muestras, aunque la complejidad computacional aumenta. |
| **Desbalance de clases** | **Sensible**. El algoritmo tiende a favorecer la clase mayoritaria. Se puede mitigar usando `class_weight='balanced'` o técnicas de balanceo. |
| **Outliers** | **Moderadamente sensible**. Los outliers pueden influir en la ubicación del hiperplano, especialmente con C grande. Se recomienda limpieza previa o usar un kernel robusto (ej. RBF con parámetros adecuados). |
| **Volumen de datos** | Entrenamiento: complejidad entre \(O(n^2)\) y \(O(n^3)\) dependiendo del kernel y la implementación (SMO). No escala bien a datasets muy grandes (>100k muestras). Para grandes volúmenes, usar SVM lineal con SGDClassifier o LinearSVC. |

---

## 4. Hiperparámetros críticos y su impacto

| Hiperparámetro | Descripción | Impacto en sesgo‑varianza |
|----------------|-------------|---------------------------|
| **`C`** (regularización) | Controla el compromiso entre maximizar el margen y minimizar el error de clasificación. | `C` pequeño → margen amplio, alto sesgo (subajuste). `C` grande → margen estrecho, alta varianza (sobreajuste). |
| **`kernel`** | Tipo de función que mapea los datos a un espacio de mayor dimensión. | `linear` → modelo lineal, bajo sesgo si los datos son linealmente separables. `rbf` (gaussiano) → no lineal, puede ajustar fronteras complejas pero con más riesgo de sobreajuste. |
| **`gamma`** (kernel RBF) | Controla la influencia de un solo ejemplo de entrenamiento. | `gamma` pequeño → influencia amplia, decisión suave (alto sesgo). `gamma` grande → influencia local, ajusta ruido (alta varianza). |
| **`degree`** (kernel polinomial) | Grado del polinomio. | A mayor grado, mayor flexibilidad (mayor varianza). |
| **`coef0`** | Término independiente en kernel polinomial o sigmoide. | Ajusta la influencia de términos de alto orden. |

**Optimización:** Se realiza mediante búsqueda en rejilla (GridSearchCV) sobre \(C\) y `gamma` (escala logarítmica). Para kernel lineal, solo se optimiza \(C\).

---

## 5. Ventajas y desventajas

| Ventajas | Desventajas |
|----------|-------------|
| Efectivo en espacios de alta dimensión y con pocas muestras. | No escala bien a grandes conjuntos de datos (más de 100k muestras). |
| El kernel trick permite aprender fronteras no lineales complejas. | Sensible a la elección de hiperparámetros (C, gamma), requiere ajuste fino. |
| Robusto ante overfitting gracias a la regularización. | Las predicciones no son probabilísticas directamente (aunque se puede calibrar). |
| La solución es única (optimización convexa). | Interpretabilidad baja (especialmente con kernels no lineales). |
| Funciona bien con datos mixtos después de codificación adecuada. | Requiere escalado obligatorio de variables. |

---

## 6. Interpretabilidad

**Media / Baja** (depende del kernel).  
- **Kernel lineal**: los coeficientes \(\mathbf{w}\) indican la importancia de cada característica (similar a regresión logística). Se puede explicar qué variables contribuyen a la decisión.  
- **Kernel RBF o polinomial**: la decisión se basa en similitudes con los vectores de soporte, lo que dificulta una explicación simple.  

**Explicación para negocio (kernel lineal):**  
> *“El modelo clasifica este cliente como de alto riesgo porque las variables ingreso y deuda tienen un peso elevado y su combinación supera el umbral definido por los vectores de soporte.”*

En kernels no lineales, a menudo se recurre a técnicas de explicabilidad post-hoc (SHAP, LIME) para interpretar las predicciones.

---

## 7. Escalabilidad y poder de cómputo

| Fase | Complejidad | Observaciones |
|------|-------------|---------------|
| **Entrenamiento** | Entre \(O(n^2 \cdot d)\) y \(O(n^3 \cdot d)\) según el kernel y la implementación. El algoritmo SMO (Sequential Minimal Optimization) suele ser \(O(n^2)\) a \(O(n^3)\) en la práctica. | Para grandes conjuntos, usar LinearSVC o SGDClassifier con pérdida hinge. |
| **Predicción (inferencia)** | \(O(n_{sv} \cdot d)\) donde \(n_{sv}\) es el número de vectores de soporte. | Para kernel lineal, es \(O(d)\); para kernel RBF, requiere evaluar el kernel con todos los vectores de soporte, lo que puede ser costoso si \(n_{sv}\) es grande. |
| **Memoria** | Almacena los vectores de soporte: \(O(n_{sv} \cdot d)\). Puede ser grande si el número de vectores de soporte es alto. |

**Técnicas para escalar:**  
- Para datasets grandes: usar **LinearSVC** (kernel lineal) o **SGDClassifier** con pérdida hinge, que escalan linealmente con \(n\).  
- Para kernels no lineales, considerar **aproximaciones de kernel** (Nystroem, Random Fourier Features) para reducir la complejidad.

---

## 8. Consideraciones prácticas

- **Escalado obligatorio**: usar `StandardScaler` o `MinMaxScaler` antes de entrenar. La no estandarización puede hacer que el modelo converja mal o dé resultados subóptimos.
- **Elección del kernel**:  
  - **lineal**: cuando el número de características es muy grande (>10k) o cuando los datos son aproximadamente lineales.  
  - **RBF**: opción predeterminada para problemas no lineales; optimizar C y gamma.  
  - **polinomial**: cuando se conoce que la frontera es de un grado bajo; requiere ajuste de grado.
- **Manejo de desbalance**: usar `class_weight='balanced'` o ajustar el peso de las clases manualmente.
- **Optimización de hiperparámetros**: usar GridSearchCV con escalas logarítmicas para C y gamma (ej. \(10^{-3}\) a \(10^3\)). Para kernel lineal, solo C.
- **Interpretación**: Si se necesita explicabilidad, optar por kernel lineal o combinar SVM con herramientas como SHAP.
- **Producción**: Guardar el modelo y el scaler con `joblib`. En inferencia, evaluar la latencia: si se requiere tiempo real, considerar LinearSVC o reducir el número de vectores de soporte mediante técnicas de compresión.

---

# Plantilla en Python para SVM (clasificación)

```python
# ==============================================
# PLANTILLA PARA SVM (SUPPORT VECTOR MACHINE) - CLASIFICACIÓN
# ==============================================
# Esta plantilla está optimizada para SVC (kernel RBF y lineal).
# Comentarios "FIXED" = partes específicas del algoritmo SVM.
# Comentarios "VARIABLE" = partes que debes ajustar según tu problema/datos.
# ==============================================

# 1. IMPORTS NECESARIOS
# -----------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler   # FIXED: SVM requiere escalado obligatorio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# FIXED: Import del modelo SVC (kernel RBF por defecto)
from sklearn.svm import SVC
import joblib

# 2. CARGA DE DATOS
# -----------------
# VARIABLE: Ajusta la ruta y nombre de tu archivo, así como la columna objetivo.
df = pd.read_csv('tu_dataset.csv')          # <-- Cambia aquí
X = df.drop('target', axis=1)               # <-- Reemplaza 'target' por tu columna objetivo
y = df['target']                            # <-- Columna objetivo

# 3. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
# --------------------------------------
# VARIABLE: Ajusta test_size, random_state y stratify (útil para clasificación desbalanceada).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. PREPROCESAMIENTO: ESCALADO OBLIGATORIO
# -----------------------------------------
# FIXED: SVM necesita escalado para que todas las variables contribuyan por igual.
# VARIABLE: StandardScaler es el más común; MinMaxScaler también es válido.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. DEFINICIÓN DEL MODELO BASE
# -----------------------------
# FIXED: Configura los hiperparámetros iniciales de SVM.
# VARIABLE: Ajusta según tu conocimiento previo (kernel, C, gamma).
model_base = SVC(
    C=1.0,                      # regularización
    kernel='rbf',               # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',              # 'scale', 'auto', o un valor fijo
    probability=False,          # si True, permite predict_proba (más lento)
    class_weight=None           # 'balanced' para manejar desbalance
)

# 6. (OPCIONAL) BÚSQUEDA DE HIPERPARÁMETROS CON VALIDACIÓN CRUZADA
# -----------------------------------------------------------------
# FIXED: Grid típico para SVM con kernel RBF.
# VARIABLE: Ajusta los rangos según tu dataset y tiempo disponible.
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
    # Si usas kernel lineal, descomenta:
    # 'C': [0.1, 1, 10, 100],
    # y comenta 'gamma'
}

grid_search = GridSearchCV(
    estimator=model_base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',      # para clasificación; ajusta según métrica deseada
    n_jobs=-1,
    verbose=1
)

# Entrenar la búsqueda (descomenta si quieres optimizar automáticamente)
# grid_search.fit(X_train_scaled, y_train)
# mejor_modelo = grid_search.best_estimator_
# print("Mejores hiperparámetros:", grid_search.best_params_)

# Si no usas búsqueda, el modelo_base es el que entrenamos.
mejor_modelo = model_base

# 7. ENTRENAMIENTO DEL MODELO
# ----------------------------
mejor_modelo.fit(X_train_scaled, y_train)

# 8. PREDICCIONES
# ---------------
y_pred = mejor_modelo.predict(X_test_scaled)
# Si probability=True, también se puede obtener:
# y_proba = mejor_modelo.predict_proba(X_test_scaled)

# 9. EVALUACIÓN DEL MODELO
# ------------------------
print("Accuracy en test:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. (OPCIONAL) VALIDACIÓN CRUZADA EN ENTRENAMIENTO
# --------------------------------------------------
cv_scores = cross_val_score(mejor_modelo, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Validación cruzada (5 folds): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 11. GUARDAR MODELO Y PREPROCESADOR PARA PRODUCCIÓN
# --------------------------------------------------
# FIXED: Guarda tanto el modelo como el scaler con joblib.
joblib.dump(mejor_modelo, 'modelo_svm.pkl')
joblib.dump(scaler, 'scaler_svm.pkl')
print("Modelo y escalador guardados.")

# ==============================================
# NOTAS ESPECÍFICAS PARA SVM:
# - **Escalado**: OBLIGATORIO. Sin escalado, el modelo no converge adecuadamente.
# - **Kernel**:
#   * 'linear': para alta dimensionalidad (>10k) o datos linealmente separables.
#   * 'rbf': opción por defecto para no lineal; requiere optimizar C y gamma.
#   * 'poly': si se conoce un grado bajo; optimizar degree y coef0.
# - **Sensibilidad**: Los outliers afectan más cuando C es grande; limpiar datos antes.
# - **Desbalance**: Usar class_weight='balanced' o ajustar manualmente.
# - **Tiempo de entrenamiento**: Para datasets grandes (>50k muestras), considera LinearSVC o SGDClassifier.
# - **Probabilidades**: Activar `probability=True` si se necesitan, pero incrementa el costo.
# ==============================================
```




---

## Métricas de Evaluación

# Interpretación de métricas de clasificación (aplicable a cualquier modelo)

## 1. Marco teórico

Para evaluar un modelo de clasificación, disponemos de varias métricas que miden distintos aspectos del rendimiento. Todas parten de la **matriz de confusión** (para un problema binario):

|                 | Predicho Positivo | Predicho Negativo |
|-----------------|-------------------|-------------------|
| **Real Positivo** | VP (Verdadero Positivo) | FN (Falso Negativo) |
| **Real Negativo** | FP (Falso Positivo) | VN (Verdadero Negativo) |

### 1.1 Accuracy (Exactitud)
\[
\text{Accuracy} = \frac{VP + VN}{VP + VN + FP + FN}
\]
**Interpretación**: Proporción de aciertos sobre el total. Es una métrica global, pero puede ser engañosa si las clases están desbalanceadas.

### 1.2 Precision (Precisión)
\[
\text{Precision} = \frac{VP}{VP + FP}
\]
**Interpretación**: De todas las predicciones positivas, ¿cuántas fueron realmente positivas? Mide la fiabilidad de la clase positiva predicha. Es clave cuando los **falsos positivos** son costosos (ej. un spam que se clasifica como correo importante).

### 1.3 Recall (Sensibilidad, Tasa de Verdaderos Positivos)
\[
\text{Recall} = \frac{VP}{VP + FN}
\]
**Interpretación**: De todos los casos positivos reales, ¿cuántos fueron correctamente identificados? Es crucial cuando los **falsos negativos** son muy costosos (ej. no detectar una enfermedad).

### 1.4 F1‑Score
\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]
**Interpretación**: Media armónica entre precisión y recall. Es útil cuando se busca un balance entre ambos y se tiene una única métrica para comparar modelos.

### 1.5 ROC‑AUC (Área bajo la curva ROC)
La curva ROC representa el recall (TPR) frente a la tasa de falsos positivos (FPR) para diferentes umbrales. El AUC es el área bajo esa curva.
- **AUC = 1**: separación perfecta.
- **AUC = 0.5**: modelo aleatorio.
**Interpretación**: Probabilidad de que el modelo asigne una probabilidad más alta a un positivo aleatorio que a un negativo aleatorio. Es independiente del umbral y robusto ante desbalance.

### 1.6 Otras métricas: especificidad, sensibilidad balanceada, etc.

---

## 2. Interpretación práctica y guía de selección

| Métrica | Cuándo priorizarla | Interpretación en negocio |
|---------|--------------------|---------------------------|
| **Accuracy** | Clases balanceadas, costes simétricos. | "El modelo acierta el X% de las veces." |
| **Precision** | Los falsos positivos son costosos. | "Cuando el modelo dice que algo es positivo, acierta el X% de las veces." |
| **Recall** | Los falsos negativos son costosos. | "El modelo detecta el X% de los casos positivos reales." |
| **F1‑Score** | Se necesita un balance entre precisión y recall. | "El modelo logra un equilibrio de X entre detectar positivos y no alarmar falsamente." |
| **ROC‑AUC** | Evaluar capacidad discriminativa independiente del umbral. | "El modelo distingue entre clases en el X% de los casos." |

**Para comparar modelos** (ej. KNN vs. Random Forest):
- Si el problema es balanceado y el coste de error es simétrico, puedes usar **accuracy** como métrica principal y complementar con F1.
- Si el problema tiene desbalance o costes asimétricos, compara **AUC** y **F1** (o la métrica que mejor refleje el coste).
- También es útil mirar la matriz de confusión y las métricas por clase para identificar debilidades específicas.

---

## 3. Ejemplos de interpretación con KNN (basados en los resultados de la sección anterior)

A continuación, tres ejemplos concretos utilizando las métricas obtenidas en la evaluación del KNN con el dataset de cáncer de mama.

### Ejemplo 1: Interpretación de recall (clase maligna)
> **Contexto**: Detectar tumores malignos es prioritario porque un falso negativo podría retrasar un tratamiento.  
> **Resultado**: Recall para malignos = 0.93 (93%).  
> **Interpretación**:  
> *“De cada 100 tumores malignos reales, el modelo identifica correctamente 93. Los 7 que no detecta son falsos negativos. Si se considera que el costo de no detectar un cáncer es muy alto, deberíamos evaluar si este nivel de recall es aceptable o si necesitamos ajustar el modelo para priorizar la detección a costa de más falsos positivos.”*

### Ejemplo 2: Interpretación de precisión y F1‑score
> **Contexto**: Queremos que las predicciones positivas (benigno) sean confiables para evitar biopsias innecesarias.  
> **Resultado**: Precisión para benigno = 0.96, F1‑score global = 0.96.  
> **Interpretación**:  
> *“Cuando el modelo predice que un tumor es benigno, acierta el 96% de las veces, lo que reduce el sobretratamiento. El F1‑score de 0.96 indica un excelente equilibrio entre precisión y recall, reflejando que el modelo es confiable tanto para detectar malignos como para no alarmar falsamente sobre benignos.”*

### Ejemplo 3: Interpretación del AUC‑ROC
> **Contexto**: Se necesita evaluar la capacidad general de discriminación sin fijar un umbral.  
> **Resultado**: AUC = 0.9923.  
> **Interpretación**:  
> *“El AUC de 0.99 significa que, si se toma un par de tumores aleatorios (uno maligno y uno benigno), el modelo asignará una probabilidad más alta al maligno en el 99.2% de las veces. Esto indica una excelente capacidad para separar las clases, independientemente del umbral elegido.”*

---

## 4. Plantilla de código para métricas (solo parte de evaluación)

A continuación se muestra un bloque de código autónomo que puedes incluir en cualquier notebook después de entrenar un clasificador. Contiene los imports necesarios y una función que imprime un reporte completo. Es independiente del modelo; solo necesita que el objeto `model` tenga `predict` y `predict_proba`.

```python
# ==============================================
# EVALUACIÓN DE MÉTRICAS DE CLASIFICACIÓN
# ==============================================
# Esta plantilla calcula y muestra las métricas clave para un clasificador.
# Solo requiere los datos de prueba y el modelo ya entrenado.

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

def print_classification_metrics(model, X_test, y_test, model_name="Modelo"):
    """
    Calcula e imprime las métricas de clasificación más importantes.
    
    Parámetros:
    - model: clasificador entrenado (debe tener .predict y .predict_proba)
    - X_test: características de prueba
    - y_test: etiquetas reales
    - model_name: nombre del modelo para mostrar en el reporte
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # asume problema binario, clase positiva en columna 1
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"=== Métricas de {model_name} ===\n")
    print("Matriz de confusión:")
    print(cm)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    
    print("\nClassification Report (por clase):")
    print(classification_report(y_test, y_pred))

# EJEMPLO DE USO (después de entrenar un modelo, por ejemplo KNN)
# print_classification_metrics(knn, X_test_scaled, y_test, model_name="KNN (k=5)")
```

**Notas sobre el código**:
- Se asume problema binario. Para multiclase, habría que modificar `predict_proba` y usar métricas con `average` (macro, weighted, etc.).
- El `classification_report` ya incluye precision, recall y f1 por clase, por lo que puede ser suficiente para un resumen completo.
- La función es genérica y funciona con cualquier clasificador de scikit‑learn que implemente `predict_proba`.

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










