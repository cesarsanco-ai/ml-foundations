---
layout: default
---

# Cheatsheet: Clasificación II - Algoritmos Clásicos
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-05](../../../sesiones/sesion-05.md)

---

## 🚀 Algoritmos Clásicos

### 1. K-Nearest Neighbors (KNN)
* **Tipo:** No paramétrico, basado en instancias.
* **Requisitos:** Datos numéricos escalados, elegir $k$.
* **Distancia:** Euclidiana: $d(p,q)=\sqrt{\sum (p_i-q_i)^2}$.
* **Bias-Varianza:** $k$ pequeño $\to$ alta varianza; $k$ grande $\to$ alto sesgo.
* **Pros:** Simple, interpretable, no requiere fase de entrenamiento.
* **Contras:** Lento en predicción, sensible a la maldición de la dimensionalidad.

**Código base:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

---

### 2. Naive Bayes
* **Tipo:** Paramétrico, probabilístico.
* **Base:** Teorema de Bayes: $P(C_k|X) = \frac{P(X|C_k)P(C_k)}{P(X)}$.
* **Supuesto:** Independencia condicional: $P(X|C_k)=\prod P(x_i|C_k)$.
* **Variantes:** Gaussian (continuas), Multinomial (conteos), Bernoulli (binarias).
* **Pros:** Muy rápido, eficiente en alta dimensión, fácil de interpretar.
* **Contras:** El supuesto de independencia rara vez se cumple en la realidad.

**Código base:**
```python
from sklearn.naive_bayes import GaussianNB # Para variables continuas

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

---

### 3. Support Vector Machines (SVM)
* **Tipo:** No paramétrico, basado en margen máximo.
* **Objetivo:** Maximizar $\frac{2}{\|w\|}$ sujeto a $y_i(w\cdot x_i+b)\ge 1$.
* **Kernels:** Lineal, RBF (Gaussiano), polinomial, sigmoide.
* **Hiperparámetros:** $C$ (regularización), $\gamma$ (alcance del kernel RBF).
* **Pros:** Muy efectivo en alta dimensión, maneja fronteras no lineales complejas.
* **Contras:** Difícil de interpretar con kernels no lineales, costoso en memoria y tiempo para datasets grandes.

**Código base:**
```python
from sklearn.svm import SVC

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)
```

---

## 📊 Comparación Conceptual

| Característica | KNN | Naive Bayes | SVM |
| :--- | :--- | :--- | :--- |
| **Paramétrico** | No | Sí | No |
| **Interpretabilidad** | Alta | Alta | Baja (no lineal) |
| **Entrenamiento** | $O(1)$ | $O(n \cdot d)$ | $O(n^2 d)$ a $O(n^3)$ |
| **Predicción** | $O(n \cdot d)$ | $O(d \cdot K)$ | $O(\|SV\| \cdot d)$ |
| **Escalamiento** | Requerido | No | Requerido |
| **Mejor para** | Baja dimensión | Texto, alta dimensión | Fronteras complejas |

---

## 💡 Guía de Selección

| Escenario | Algoritmo Recomendado |
| :--- | :--- |
| Clasificación de texto, detección de spam | **Naive Bayes** |
| Sistemas de recomendación por similitud | **KNN** |
| Alta precisión en datos medianos | **SVM (Kernel RBF)** |
| Datos con relación lineal clara | **SVM Lineal** |
| Explicabilidad inmediata requerida | **Naive Bayes o KNN** |
| Necesidad de predicciones en tiempo real | **Naive Bayes** |

---

## 🛠️ Pipeline Recomendado

```python
# 1. División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Escalamiento (Vital para KNN y SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Modelos
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale')
}

# 4. Evaluación
for name, model in models.items():
    if name in ['KNN','SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## 📈 Métricas Clave

| Métrica | Fórmula | Cuándo usar |
| :--- | :--- | :--- |
| **Accuracy** | $(VP+VN)/total$ | Clases balanceadas. |
| **Precision** | $VP/(VP+FP)$ | Cuando el costo de un Falso Positivo es alto. |
| **Recall** | $VP/(VP+FN)$ | Cuando el costo de un Falso Negativo es alto. |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P+R}$ | Cuando buscas balance entre Precision y Recall. |
| **AUC-ROC** | Área bajo la curva | Para medir capacidad discriminativa del modelo. |

---

## ⚠️ Puntos Críticos
* **Siempre escalar** las características antes de usar KNN o SVM.
* **Validación Cruzada:** Úsala siempre para elegir el mejor valor de $k$ o los parámetros $C$ y $\gamma$.
* **Kernel:** La elección del kernel en SVM define la forma de la frontera de decisión.
* **Probabilidades:** Usa `predict_proba` para obtener la certeza del modelo (en SVM requiere `probability=True`).
* **Clases Desbalanceadas:** El Accuracy es engañoso; prioriza el F1-Score o el área bajo la curva (AUC).

---

## 💼 Contexto de Negocio

| Algoritmo | Casos de Uso Comunes |
| :--- | :--- |
| **KNN** | Motores de recomendación, segmentación de clientes. |
| **Naive Bayes** | Clasificación de correos, análisis de sentimiento. |
| **SVM** | Reconocimiento de imágenes, detección de anomalías médicas. |

> *“El mejor algoritmo depende de los datos, el contexto de negocio y las restricciones operativas.”*

