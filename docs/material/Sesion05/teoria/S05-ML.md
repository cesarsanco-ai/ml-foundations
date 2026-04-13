---
layout: default
---
# Sesión 5: kNN, Naive Bayes y SVM

### 1. Logro de la sesión

Dominar los fundamentos, **hiperparámetros** y **patrones de uso en Python** de **k vecinos más cercanos (kNN)**, **Naive Bayes** y **máquinas de vectores de soporte (SVM)** en clasificación y regresión, incluyendo el papel del **preprocesamiento**, las **métricas** adecuadas y la intuición **geométrica** de fronteras de decisión.

---

### 2. Historia y línea temporal (síntesis)

| Algoritmo | Hitos | Lectura |
|-----------|-------|---------|
| **kNN** | Fix & Hodges (1951) análisis no paramétrico; Cover & Hart (1967) formalización | “Lazy learning”: el entrenamiento almacena datos; la complejidad se traslada a la predicción. |
| **Naive Bayes** | Origen en **Teorema de Bayes** (1763); independencia condicional como aproximación práctica desde spam filtering (1990s–2000s) | Extremadamente rápido en alta dimensión dispersa (texto). |
| **SVM** | Vapnik–Chervonenkis (1960s–1990s); Cortes & Vapnik (1995) SVM no lineal con kernels | Fronteras con **margen máximo** y truco del kernel para separación en espacio de características. |

---

### 3. k-Nearest Neighbors (kNN)

#### 3.1 Idea central

No estima una función global explícita en la fase de “entrenamiento”: **memoriza** el conjunto de entrenamiento y predice por **consenso local** de vecinos.

- **Clasificación:** voto mayoritario entre los $k$ vecinos más cercanos (o ponderado por $1/\mathrm{dist}$).  
- **Regresión:** media o mediana de los $y$ de los $k$ vecinos.

#### 3.2 Distancias comunes

Para vectores $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^p$:

| Métrica | Fórmula | Comentario |
|---------|---------|------------|
| **Euclídea** ($L_2$) | $\|\mathbf{x}-\mathbf{x}'\|_2$ | Por defecto en muchos problemas tabulares escalados |
| **Manhattan** ($L_1$) | $\|\mathbf{x}-\mathbf{x}'\|_1$ | Robusta a outliers en una dimensión |
| **Minkowski** | $\|\mathbf{x}-\mathbf{x}'\|_q$ | Generaliza $L_1$ y $L_2$ |
| **Mahalanobis** (avanzado) | $(\mathbf{x}-\mathbf{x}')^\top \mathbf{S}^{-1}(\mathbf{x}-\mathbf{x}')$ | Correlaciones entre features (no siempre en kNN estándar) |

**Crítico:** sin **escalado**, variables con mayor rango numérico **dominan** la distancia.

#### 3.3 Elección de $k$

| $k$ | Comportamiento típico |
|-----|------------------------|
| **Pequeño** | Fronteras flexibles; **alta varianza**, sensible a ruido |
| **Grande** | Superficies más suaves; **mayor sesgo**, riesgo de perder detalle local |

Selección por **validación cruzada** (Sesión 8). Regla heurística: $k$ impar en binario para evitar empates en voto.

#### 3.4 Ventajas y limitaciones

**Ventajas:**

- Implementación conceptual simple; captura **no linealidad** local.  
- No asume forma paramétrica global.

**Limitaciones:**

- **Maldición de la dimensionalidad:** en $p$ alto, distancias pierden discriminación (vecinos equidistantes).  
- **Coste predictivo** $O(n)$ por consulta en forma ingenua (estructuras de ayuda: KD-ball, approximate NN).  
- Requiere **memoria** para almacenar train completo.

#### 3.5 Plantilla Python (clasificación)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",  # pondera por 1/distancia
            metric="minkowski",
            p=2,  # 2=Euclídea
            n_jobs=-1,
        ),
    ),
])
knn_pipe.fit(X_train, y_train)
y_pred = knn_pipe.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

#### 3.6 Plantilla Python (regresión)

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance")),
])
knn_reg.fit(X_train, y_train)
y_hat = knn_reg.predict(X_test)
```

---

### 4. Naive Bayes

#### 4.1 Teorema de Bayes (recordatorio)

$$ P(y=c \mid \mathbf{x}) = \frac{P(y=c)\, P(\mathbf{x} \mid y=c)}{P(\mathbf{x})} $$

El denominador es constante respecto a $c$ al comparar clases → basta maximizar el **numerador** o su log.

#### 4.2 Supuesto “ingenuo”

$$ P(\mathbf{x} \mid y=c) = \prod_{j=1}^{p} P(x_j \mid y=c) $$

**Independencia condicional** dada la clase: raramente es cierta en la vida real, pero el clasificador puede ser **muy competitivo** porque la decisión solo requiere que las **densidades relativas** estén bien ordenadas, no perfectas.

#### 4.3 Variantes

| Variante | Modelo de $P(x_j\mid c)$ | Cuándo usarla |
|----------|---------------------------|---------------|
| **GaussianNB** | Gaussian por dimensión y clase | Features continuas aproximadamente normales por clase |
| **MultinomialNB** | Conteos discretos (palabras) | Texto, conteos de eventos |
| **BernoulliNB** | Binario por dimensión | Presencia/ausencia de términos, features 0/1 |
| **ComplementNB** | Variante que usa complementos | A veces mejor con datos desbalanceados |

**Ventajas:**

- Entrenamiento **extremadamente rápido** (conteos + parámetros cerrados en muchos casos).  
- Funciona bien en **alta dimensión** si la dispersidad es manejable.

**Limitaciones:**

- Si el supuesto de independencia falla fuerte, puede haber **sesgo** sistemático.  
- **Probabilidades calibradas** pueden ser pobres → calibración posterior si se usan como scores.

#### 4.4 Plantilla Python (tabular Gaussiano)

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

gnb = GaussianNB()
gnb.fit(X_train, y_train)
p = gnb.predict_proba(X_test)[:, 1]
y_hat = gnb.predict(X_test)
print("AUC:", roc_auc_score(y_test, p))
```

#### 4.5 Texto (esquema)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
    ("nb", MultinomialNB(alpha=1.0)),  # suavizado Laplace
])
text_clf.fit(text_train, y_train)
```

`alpha` evita probabilidades cero por palabras no vistas en una clase (*smoothing*).

---

### 5. Support Vector Machines (SVM)

#### 5.1 Clasificación: margen máximo y variables duales

Para datos **linealmente separables** (idealización), la SVM busca el hiperplano que **maximiza el margen** entre clases. Para solapamiento, se introducen **variables de holgura** $\xi_i$ y penalización $C$ que controla el trade-off **margen vs violaciones** (Cortes & Vapnik, 1995).

**Kernel trick:** productos escalares $K(\mathbf{x},\mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$ permiten fronteras no lineales sin calcular $\phi$ explícitamente.

#### 5.2 Kernels habituales

| Kernel | Expresión | Intuición |
|--------|-----------|-----------|
| **Lineal** | $K(\mathbf{x},\mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$ | Baseline en datos escalados |
| **RBF (Gaussiano)** | $\exp(-\gamma \|\mathbf{x}-\mathbf{x}'\|^2)$ | Muy flexible; $\gamma$ alto → fronteras muy locales |
| **Polinomial** | $(\gamma \mathbf{x}^\top \mathbf{x}' + r)^d$ | Interacciones hasta grado $d$ |

**Hiperparámetros clave:**

- **C:** mayor C → menos tolerancia a errores de entrenamiento → puede **sobreajustar**.  
- **$\gamma$ (RBF):** mayor $\gamma$ → influencia local de cada soporte → riesgo de overfitting.

#### 5.3 Regresión: SVR

**$\epsilon$-SVR:** no penaliza errores menores que $\epsilon$ en valor absoluto (zona insensible), lo que produce soluciones **sparse** en dual (Vapnik et al., *estimation of dependences*).

#### 5.4 Ventajas y limitaciones

**Ventajas:**

- Efectivas en **medias dimensiones** y fronteras no lineales con RBF.  
- Fundamentación teórica fuerte (margen, dualidad).

**Limitaciones:**

- En **muy grandes $n$**, entrenamiento costoso (aunque existen aproximaciones).  
- **Calibración** de C y $\gamma$ es crítica; requiere búsqueda sistemática.  
- Salidas no siempre calibradas como probabilidades (`predict_proba` solo en envoltorios tipo `CalibratedClassifierCV` o `probability=True` en sklearn con coste).

#### 5.5 Plantilla Python (clasificación RBF)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    (
        "svc",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        ),
    ),
])
svm_pipe.fit(X_train, y_train)
```

#### 5.6 Plantilla Python (regresión)

```python
from sklearn.svm import SVR

svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
])
svr.fit(X_train, y_train)
y_hat = svr.predict(X_test)
```

---

### 6. Métricas (clasificación y regresión)

- **Clasificación:** matriz de confusión, precisión, recall, F1, AUC-ROC, PR-AUC si hay desbalance (Sesión 4).  
- **Regresión:** MAE, RMSE, $R^2$ (Sesión 3).

---

### 7. Laboratorio (según sílabo)

- **NTB 1 —** Clasificación con kNN, Naive Bayes y SVM (comparación y ajuste).  
- **NTB 2 —** Regresión con Naive Bayes y SVM.

---

## Referencias bibliográficas principales

1. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.  
2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.  
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.  
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.  
5. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.  
