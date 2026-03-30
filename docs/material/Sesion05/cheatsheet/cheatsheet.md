---
layout: default
---

# Cheatsheet: kNN, Naive Bayes, SVM
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-05](../../../sesiones/sesion-05.md)

---


## 1. K‑Nearest Neighbors (KNN)

### Concepto
- **No paramétrico** y **basado en instancias** (lazy learning).  
- Almacena todos los datos de entrenamiento.  
- Para predecir, calcula distancia a todos los puntos, selecciona los \(k\) más cercanos y vota la clase mayoritaria.

**Fórmula distancia euclidiana**:  

$$ d(p,q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2} $$

### Contexto de negocio – ¿Cuándo usarlo?

| Categoría | Ejemplo | Por qué encaja |
|-----------|--------|----------------|
| **Recomendación / Personalización** | “Clientes similares a ti también compraron…” | La similitud entre usuarios es el núcleo del problema. |
| **Análisis espacial / Geográfico** | Valoración de inmuebles por propiedades vecinas | Distancia geográfica es una métrica natural. |
| **Detección de anomalías** | Transacciones fraudulentas | Un fraude suele estar “lejos” de las transacciones normales. |
| **Segmentación / Perfilado** | Asignar nuevo cliente a perfil de riesgo | Explicable: “su perfil es idéntico al de estos 5 clientes de alto riesgo”. |
| **Diagnóstico médico** | Clasificar tumor benigno/maligno con pocas variables | Relación no lineal, interpretabilidad total. |

### Requisitos y características

| Característica | Condición / Impacto |
|----------------|---------------------|
| **Escalado** | **Obligatorio**. Sin escalado, variables con mayor rango dominan la distancia. |
| **Tipo de datos** | Numéricos continuos. Categóricas → One‑Hot Encoding o métricas especiales. |
| **Dimensionalidad** | Sensible a la maldición. Funciona mejor con <20 variables. |
| **Desbalance** | Muy sensible. Requiere balanceo (SMOTE, undersampling) o pesos por distancia. |
| **Outliers** | Altamente sensible. Limpiar datos o usar distancias robustas. |
| **Volumen** | Inferencia lenta: \(O(n \cdot d)\). No apto para >100k muestras en producción. |

### Hiperparámetros críticos

| Parámetro | Valores típicos | Impacto |
|-----------|-----------------|---------|
| `n_neighbors` (k) | 3, 5, 7, 9, √n | Pequeño → alta varianza (sobreajuste). Grande → alto sesgo. |
| `weights` | 'uniform', 'distance' | 'distance' mitiga vecinos lejanos. |
| `metric` | 'euclidean', 'manhattan' | Manhattan más robusta a outliers. |

### Ventajas vs Desventajas

| ✅ Ventajas | ❌ Desventajas |
|-------------|---------------|
| Fácil de entender e implementar | Coste de predicción alto  ($O(n \cdot d)$) |
| No asume distribución subyacente | Sensible a escala, outliers y alta dimensionalidad |
| Altamente interpretable (explicación por vecinos) | Almacena todo el dataset en memoria |
| No requiere entrenamiento | Muy sensible a clases desbalanceadas |

### Código mínimo (scikit‑learn)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
```

---

## 2. Naive Bayes

### Concepto
- **Probabilístico** basado en el teorema de Bayes.  
- Supone **independencia condicional** de los predictores dada la clase (supuesto “naive”).  
- Variantes según tipo de datos: **GaussianNB** (continuas), **MultinomialNB** (conteos), **BernoulliNB** (binarias).

**Fórmula**:  

$$
P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)
$$




### Contexto de negocio – ¿Cuándo usarlo?

| Categoría | Ejemplo | Por qué encaja |
|-----------|--------|----------------|
| **Clasificación de texto / NLP** | Spam vs. no spam, análisis de sentimiento | Multinomial/Bernoulli muy eficaces con bolsas de palabras y alta dimensionalidad. |
| **Filtrado de contenido** | Moderación de comentarios ofensivos | Rápido y preciso con muchas características binarias. |
| **Diagnóstico / riesgo** | Clasificar pacientes en grupos de riesgo | Base probabilística clara; funciona bien incluso con supuesto fuerte. |
| **Marketing** | Predicción de clics en anuncios | Escala bien con datos de navegación de alta dimensión. |
| **Segmentación rápida** | Churn prediction en telecom | Incorpora fácilmente nuevas variables. |

### Requisitos y características

| Característica | Condición / Impacto |
|----------------|---------------------|
| **Escalado** | **No necesario**. Las estimaciones no dependen de la escala. |
| **Tipo de datos** | Según variante: Gaussian → continuas, Multinomial → conteos, Bernoulli → binarias. |
| **Dimensionalidad** | **Muy robusto**. Excelente con miles de características (texto). |
| **Desbalance** | Moderadamente sensible. Ajustar `fit_prior` o usar `class_weight`. |
| **Outliers** | Poco sensible (especialmente en Multinomial/Bernoulli). |
| **Volumen** | Entrenamiento e inferencia extremadamente rápidos ($(O(n \cdot d)$) y $(O(d \cdot k)$)). |

### Hiperparámetros críticos

| Parámetro | Variante | Impacto |
|-----------|----------|---------|
| `alpha` | Multinomial/Bernoulli | Suavizado Laplace (evita probabilidades cero). Alto → más sesgo. |
| `var_smoothing` | GaussianNB | Estabiliza varianzas. |
| `fit_prior` | Todas | Aprende probabilidades a priori de las clases. |

### Ventajas vs Desventajas

| ✅ Ventajas | ❌ Desventajas |
|-------------|---------------|
| Extremadamente rápido (entrenamiento e inferencia) | Supuesto de independencia muy fuerte (rara vez real) |
| Escala excelente con alta dimensionalidad | Probabilidades mal calibradas (necesitan calibración) |
| Funciona bien con datasets pequeños | Sensible a clases desbalanceadas sin ajustes |
| Muy interpretable (probabilidades condicionales) | |

### Código mínimo (scikit‑learn)

```python
from sklearn.naive_bayes import GaussianNB

# Para variables continuas
nb = GaussianNB()
nb.fit(X_train, y_train)  # No requiere escalado
y_pred = nb.predict(X_test)
```

---

## 3. Support Vector Machines (SVM)

### Concepto

- **Paramétrico** que busca el **hiperplano de margen máximo** que separa las clases.  
- Usa **kernel trick** para mapear datos a espacio de mayor dimensión y lograr separabilidad lineal.

**Fórmula básica (caso separable)**: 

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.a.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1
$$

### Contexto de negocio – ¿Cuándo usarlo?

| Categoría | Ejemplo | Por qué encaja |
|-----------|--------|----------------|
| **Clasificación de imágenes / visión** | Detección de objetos, clasificación de radiografías | Kernel RBF captura patrones complejos. |
| **Biomedicina / genómica** | Clasificar tumores según expresión genética | Robusto con muchas variables y pocas muestras. |
| **Detección de fraudes** | Identificar transacciones fraudulentas en tiempo real | Aprende fronteras no lineales; C controla equilibrio. |
| **Marketing / segmentación** | Clasificar clientes de alto/bajo valor | Maneja datos mixtos tras codificación. |
| **Análisis de texto** (kernel lineal) | Clasificación de documentos con TF‑IDF | Alta precisión y rapidez. |

### Requisitos y características

| Característica | Condición / Impacto |
|----------------|---------------------|
| **Escalado** | **Obligatorio e indispensable**. Sin escalado, las variables con mayor rango dominan. |
| **Tipo de datos** | Numéricos. Categóricas → One‑Hot Encoding. |
| **Dimensionalidad** | Robusto gracias a regularización; puede manejar miles de variables. |
| **Desbalance** | Sensible. Usar `class_weight='balanced'` o balancear datos. |
| **Outliers** | Moderadamente sensible. Limpiar datos o usar kernel robusto. |
| **Volumen** | Entrenamiento $(O(n^2)$) a $(O(n^3)$); no escala bien >50k muestras. |

### Hiperparámetros críticos

| Parámetro | Valores típicos | Impacto |
|-----------|-----------------|---------|
| `C` | 0.1, 1, 10, 100 | Pequeño → margen amplio (sesgo), grande → margen estrecho (varianza). |
| `gamma` (RBF) | 0.001, 0.01, 0.1, 1 | Pequeño → influencia amplia, grande → influencia local (riesgo sobreajuste). |
| `kernel` | 'linear', 'rbf' | Lineal para alta dimensión o linealidad; RBF para no linealidad. |

### Ventajas vs Desventajas

| ✅ Ventajas | ❌ Desventajas |
|-------------|---------------|
| Efectivo en alta dimensión con pocas muestras | No escala bien a grandes datasets (>50k muestras) |
| Kernel trick permite fronteras no lineales | Sensible a elección de hiperparámetros (C, gamma) |
| Robusto ante overfitting (regularización C) | Baja interpretabilidad (especialmente kernel RBF) |
| Solución única (optimización convexa) | |

### Código mínimo (scikit‑learn)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
```

---

## 4. Métricas de Evaluación (con enfoque de negocio)

| Métrica | Fórmula | Interpretación en negocio |
|---------|---------|---------------------------|
| **Accuracy** | (VP+VN)/Total | “El modelo acierta el X% de las veces.” Útil cuando clases balanceadas y costes simétricos. |
| **Precision** | VP/(VP+FP) | “Cuando el modelo predice positivo, acierta el X%.” Priorizar si los **falsos positivos** son costosos (ej. spam filtrado como importante). |
| **Recall** | VP/(VP+FN) | “El modelo detecta el X% de los casos positivos reales.” Priorizar si los **falsos negativos** son costosos (ej. no detectar fraude o enfermedad). |
| **F1‑Score** | 2·(P·R)/(P+R) | Balance entre precisión y recall. Útil cuando ambos importan. |
| **ROC‑AUC** | Área bajo curva ROC | “El modelo distingue entre clases en el X% de los casos.” Independiente del umbral, robusto ante desbalance. |

### Ejemplo de reporte ejecutivo
> **Resumen**: SVM supera a Naive Bayes en F1 (0.77 vs 0.60) y AUC (0.93 vs 0.88).  
> **Impacto operativo**: SVM detecta 850 de cada 1000 fraudes con 350 falsas alarmas; Naive Bayes detecta 920 fraudes pero con 1100 falsas alarmas.  
> **Recomendación**: Implementar SVM por menor costo operativo y mejor experiencia de cliente.

---

## 5. ¿Cómo seleccionar el algoritmo adecuado? (Marco de decisión)

| Factor de negocio / dato | Algoritmo recomendado | Razón |
|--------------------------|------------------------|-------|
| **Interpretabilidad prioritaria** (regulación, auditoría) | KNN o Naive Bayes | Ambos son caja blanca; KNN explica por vecinos, NB por probabilidades. |
| **Alta dimensionalidad** (miles de variables) | Naive Bayes (Multinomial/Bernoulli) o SVM lineal | Escalan linealmente; NB es más rápido. |
| **Relaciones no lineales complejas** | SVM con kernel RBF | Kernel trick captura fronteras complejas sin explotar el espacio. |
| **Dataset pequeño** (<1000 muestras) | SVM o Naive Bayes | SVM robusto; NB no requiere muchos datos. |
| **Dataset grande** (>100k filas) | Naive Bayes o SVM lineal (LinearSVC) | KNN y SVM kernel no escalan en tiempo/memoria. |
| **Tiempo real / baja latencia** | Naive Bayes o SVM lineal | Inferencia rápida; KNN es lento. |
| **Datos desbalanceados** | Ajustar pesos: SVM `class_weight`, NB `fit_prior`, KNN balancear previamente | Ninguno maneja desbalance por defecto. |
| **Costes asimétricos** (FP vs FN) | Cualquiera, evaluando con métrica adecuada | Elegir modelo que optimice la métrica que refleja el coste de negocio. |

### Flujo práctico de selección

1. **Define objetivo de negocio** → ¿prioridad: explicabilidad, velocidad, precisión? ¿coste de FP vs FN?
2. **Analiza los datos** → ¿dimensionalidad? ¿volumen? ¿linealidad?
3. **Evalúa restricciones** → ¿latencia máxima? ¿recursos computacionales?
4. **Prueba baseline** → Naive Bayes (rápido) y KNN (interpretable) como puntos de partida.
5. **Ajusta y compara** → SVM si los baselines no alcanzan rendimiento esperado y datos lo permiten.
6. **Selecciona** con base en la métrica que mejor refleje el impacto de negocio (no solo accuracy).

---

## Resumen Visual

```
               Interpretabilidad alta
               ┌─────────────────────────────┐
               │  KNN  │  Naive Bayes        │
               │ (vecinos) │ (probabilidades)│
               └─────────────────────────────┘
                      ↑           ↑
               Similitud │    Texto/alta dim
                      │           │
               ┌──────┴───────────┴──────┐
               │   SVM (kernel lineal)    │
               │   para alta dim + rápido │
               └──────────────────────────┘
                      ↑
               No lineal / muestra moderada
                      │
               ┌───────────────────────────┐
               │   SVM con kernel RBF       │
               │   (fronteras complejas)    │
               └───────────────────────────┘
```

**Pro tip:** Siempre comienza con un modelo simple (Naive Bayes o KNN) como baseline. Si supera expectativas de negocio, mantén la simplicidad. Si no, avanza a SVM u otros, documentando claramente el trade-off entre ganancia predictiva y coste de implementación/mantenimiento.






***
### Extra

### ¿Algoritmos de Clasificación adaptados a Regresión?

| Algoritmo | Nombre Técnico | Nivel de Uso | Casos de Uso (Si aplica) | Nota Técnica Clave |
| :--- | :--- | :--- | :--- | :--- |
| **KNN** | `KNeighborsRegressor` | **Bajo** | Sistemas de recomendación simples, imputación de valores faltantes (KNN Imputer). | Predice el **promedio** de los $K$ vecinos. Muy sensible a la escala y outliers. |
| **SVM** | `SVR` | **Medio** | Series de tiempo financieras, biometría, sensores con datos de alta dimensionalidad. | Busca que los puntos caigan dentro de un **"tubo" de error ($\epsilon$)**. Usa Kernels para no linealidad. |
| **Naive Bayes** | N/A | **Nulo** | No aplica para regresión estándar. | Es puramente **probabilístico para categorías** (clases discretas). |



---

### Consideraciones:

1.  **¿Por qué el nivel de uso de KNN es bajo?**
    * Porque es un "estudiante de memoria": si el valor real está fuera del rango que vio en el entrenamiento, el KNN jamás podrá predecirlo (no extrapola). Además, en banca o empresas grandes, calcular distancias contra millones de filas es demasiado lento.

2.  **¿Cuándo brilla el SVR?**
    * Cuando tienes **pocos datos pero muchas columnas** (pocas muestras, alta dimensionalidad). Es muy robusto en esos casos donde los modelos de árboles (como Random Forest) podrían sobreajustar.





