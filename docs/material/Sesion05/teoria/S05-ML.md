---
layout: default
---
# Semana 5: kNN, Naive Bayes, SVM

## 1. K-Nearest Neighbors (KNN)

### 1.1 Fundamento Teórico

KNN es un algoritmo **no paramétrico** y **basado en instancias** (lazy learning). No construye un modelo explícito durante el entrenamiento; simplemente almacena todos los datos. La predicción se realiza en el momento de la inferencia.

**Principio fundamental:** "Dime quiénes son tus vecinos y te diré quién eres"

### 1.2 Matemáticas del Algoritmo

#### Distancia Euclidiana (la más común)
$$d(p,q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

#### Distancia Manhattan
$$d(p,q) = \sum_{i=1}^{n} |p_i - q_i|$$

#### Distancia Minkowski (generalización)
$$d(p,q) = \left(\sum_{i=1}^{n} |p_i - q_i|^p\right)^{1/p}$$

Donde:
- $p$ = parámetro de la métrica Minkowski
- Si $p=1$: Manhattan
- Si $p=2$: Euclidiana

### 1.3 Clasificación con KNN

Para un nuevo punto $x$:
1. Calcular distancias a todos los puntos de entrenamiento
2. Seleccionar los $k$ puntos más cercanos
3. Asignar la clase mayoritaria:

$$\hat{y} = \text{mode}(y_i) \quad \forall i \in \text{vecinos}(x)$$

### 1.4 Regresión con KNN

Para un nuevo punto $x$:
1. Calcular distancias a todos los puntos de entrenamiento
2. Seleccionar los $k$ puntos más cercanos
3. Promediar sus valores:

$$\hat{y} = \frac{1}{k} \sum_{i \in \text{vecinos}(x)} y_i$$

**Variante con pesos por distancia:**
$$\hat{y} = \frac{\sum_{i \in \text{vecinos}(x)} w_i \cdot y_i}{\sum w_i}$$

donde $w_i = \frac{1}{d(x, x_i)}$ (inverso de la distancia)

### 1.5 Hiperparámetros de KNN

| Hiperparámetro | Valores típicos | Efecto | Cómo optimizar |
|----------------|----------------|--------|----------------|
| **$k$** (n_neighbors) | 3, 5, 7, 9, 11, 15 | Controla la suavidad de la frontera | Validación cruzada |
| **Métrica** (metric) | 'euclidean', 'manhattan', 'minkowski' | Define qué es "cercano" | Según naturaleza datos |
| **Pesos** (weights) | 'uniform', 'distance' | Importancia relativa de vecinos | Validación cruzada |
| **$p$** (si metric='minkowski') | 1, 2, 3... | Exponente de Minkowski | Validación cruzada |

**Interpretación de $k$:**
- $k$ pequeño → modelo complejo, alta varianza (sobreajuste)
- $k$ grande → modelo simple, alto sesgo (subajuste)
- Regla empírica: $k \approx \sqrt{n}$ (raíz del número de muestras)

### 1.6 Ventajas y Limitaciones

**Ventajas:**
- No asume distribución de datos
- Fácil de implementar
- Naturalmente multiclase
- Interpretable (basado en ejemplos)

**Limitaciones:**
- Maldición de la dimensionalidad
- Sensible a escala de variables
- Costoso en predicción $O(n \cdot d)$
- Requiere normalización obligatoria

### 1.7 Plantilla Base

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Clasificación
knn_clf = KNeighborsClassifier(
    n_neighbors=5,      # k: número de vecinos
    weights='uniform',  # 'uniform' o 'distance'
    metric='minkowski', # 'euclidean', 'manhattan', 'minkowski'
    p=2                 # para minkowski: 1=Manhattan, 2=Euclidiana
)

# Regresión
knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform',
    metric='minkowski',
    p=2
)

# Escalado OBLIGATORIO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn_clf.fit(X_scaled, y)
predicciones = knn_clf.predict(X_test_scaled)
```

---

## 2. Naive Bayes

### 2.1 Fundamento Teórico

Naive Bayes es un clasificador **probabilístico** basado en el **Teorema de Bayes** con un supuesto "naive": **independencia condicional** de las características dada la clase.

### 2.2 Teorema de Bayes

$$P(C_k | \mathbf{x}) = \frac{P(C_k) \cdot P(\mathbf{x} | C_k)}{P(\mathbf{x})}$$

Donde:
- $P(C_k | \mathbf{x})$: **Probabilidad posterior** (lo que queremos)
- $P(C_k)$: **Probabilidad a priori** de la clase
- $P(\mathbf{x} | C_k)$: **Verosimilitud** (probabilidad de los datos dada la clase)
- $P(\mathbf{x})$: **Evidencia** (factor de normalización)

### 2.3 El Supuesto Naive (Independencia Condicional)

$$P(\mathbf{x} | C_k) = P(x_1, x_2, ..., x_n | C_k) = \prod_{i=1}^{n} P(x_i | C_k)$$

**Interpretación:** Dada la clase, la probabilidad de observar todas las características juntas es el producto de las probabilidades de cada característica individual.

### 2.4 Clasificación Final

$$\hat{y} = \arg\max_{C_k} P(C_k) \prod_{i=1}^{n} P(x_i | C_k)$$

El denominador $P(\mathbf{x})$ se omite porque es constante para todas las clases.

### 2.5 Variantes según distribución de datos

#### Gaussian Naive Bayes (variables continuas)
Asume distribución normal para cada característica:

$$P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)$$

Donde $\mu_{ik}$ y $\sigma_{ik}^2$ son la media y varianza de $x_i$ en la clase $C_k$.

#### Multinomial Naive Bayes (conteos/frecuencias)
Para datos que representan frecuencias (ej. conteo de palabras):

$$P(x_i | C_k) = \frac{N_{ik} + \alpha}{N_k + \alpha \cdot n}$$

Donde:
- $N_{ik}$: número de veces que la característica $i$ aparece en clase $k$
- $N_k$: total de características en clase $k$
- $\alpha$: parámetro de suavizado (Laplace)
- $n$: número de características

#### Bernoulli Naive Bayes (variables binarias)
Para características binarias (0/1):

$$P(x_i | C_k) = p_{ik}^{x_i} (1-p_{ik})^{1-x_i}$$

Donde $p_{ik}$ es la probabilidad de que $x_i=1$ en clase $C_k$.

### 2.6 Hiperparámetros de Naive Bayes

| Hiperparámetro | Variante | Valores típicos | Efecto |
|----------------|----------|----------------|--------|
| **alpha** ($\alpha$) | Multinomial, Bernoulli | 0.1, 0.5, 1.0, 2.0 | Suavizado de Laplace (evita probabilidades cero) |
| **var_smoothing** | Gaussian | 1e-12 a 1e-3 | Suavizado de varianza (estabilidad numérica) |
| **fit_prior** | Todas | True, False | Aprender probabilidades a priori vs uniformes |
| **binarize** | Bernoulli | 0.0, 0.5, 1.0 | Umbral para binarizar características |

**Interpretación de alpha ($\alpha$):**
- $\alpha = 1$: Suavizado de Laplace (estándar)
- $\alpha < 1$: Suavizado de Lidstone (menos sesgo)
- $\alpha$ grande → más suavizado, más sesgo
- $\alpha$ pequeño → menos suavizado, más varianza

### 2.7 Ventajas y Limitaciones

**Ventajas:**
- Extremadamente rápido (entrenamiento $O(n \cdot d)$)
- Funciona bien con alta dimensionalidad
- Maneja datos faltantes naturalmente
- Base probabilística interpretable

**Limitaciones:**
- Supuesto de independencia muy fuerte (raramente cierto)
- Malo cuando hay correlaciones fuertes entre variables
- Estimaciones de probabilidad mal calibradas
- No funciona para regresión (solo clasificación)

### 2.8 Plantilla Base

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Para variables continuas (distribución normal)
gnb = GaussianNB(
    var_smoothing=1e-9    # Suavizado de varianza
)

# Para conteos/frecuencias (texto, genes)
mnb = MultinomialNB(
    alpha=1.0,            # Suavizado de Laplace
    fit_prior=True        # Aprender probabilidades a priori
)

# Para variables binarias
bnb = BernoulliNB(
    alpha=1.0,            # Suavizado de Laplace
    binarize=0.0,         # Umbral para binarizar
    fit_prior=True
)

# Escalado NO necesario
gnb.fit(X_train, y_train)
predicciones = gnb.predict(X_test)
probabilidades = gnb.predict_proba(X_test)  # Obtener probabilidades
```

---

## 3. Support Vector Machines (SVM)

### 3.1 Fundamento Teórico

SVM es un algoritmo **paramétrico** que busca encontrar el **hiperplano óptimo** que maximiza el **margen** entre clases. Para datos no lineales, utiliza el **kernel trick** para mapear los datos a un espacio de mayor dimensión.

### 3.2 Caso Linealmente Separable

**Objetivo:** Encontrar el hiperplano $f(x) = \mathbf{w}^T \mathbf{x} + b = 0$ que maximiza el margen.

**Problema de optimización primal:**
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

Sujeto a:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i$$

Donde:
- $\mathbf{w}$: vector de pesos (normal al hiperplano)
- $b$: sesgo (offset)
- $y_i \in \{-1, +1\}$: etiquetas de clase
- $\|\mathbf{w}\|$: norma euclidiana de $\mathbf{w}$

**Margen:** $\frac{2}{\|\mathbf{w}\|}$

**Vectores de soporte:** Puntos donde $y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1$

### 3.3 Caso No Linealmente Separable (Soft Margin)

Introducimos **variables de holgura** $\xi_i$ para permitir errores de clasificación:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

Sujeto a:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$$
$$\xi_i \geq 0 \quad \forall i$$

Donde:
- $C$: parámetro de regularización (balance entre margen y error)
- $\xi_i$: penalización por punto mal clasificado

**Interpretación de $C$:**
- $C$ grande → margen estrecho, menos errores (alta varianza)
- $C$ pequeño → margen amplio, más errores (alto sesgo)

### 3.4 Kernel Trick (Mapeo a Espacios de Mayor Dimensión)

Para datos no lineales, mapeamos $\mathbf{x}$ a un espacio de características $\phi(\mathbf{x})$ donde sean linealmente separables.

**Función kernel:** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$

Permite calcular productos punto en el espacio transformado sin calcular explícitamente $\phi$.

#### Kernels comunes

**Kernel Lineal:**
$$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$

**Kernel Polinomial:**
$$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$

Donde:
- $d$: grado del polinomio
- $\gamma$: escala (coeficiente)
- $r$: término independiente (coef0)

**Kernel RBF (Radial Basis Function) - el más popular:**
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

Donde $\gamma$ controla la influencia de cada punto:
- $\gamma$ pequeño → influencia amplia, decisión suave (alto sesgo)
- $\gamma$ grande → influencia local, ajusta ruido (alta varianza)

**Kernel Sigmoide:**
$$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$$

### 3.5 SVM para Regresión (SVR)

En lugar de maximizar el margen, SVR intenta ajustar la mayoría de los puntos dentro de un tubo de ancho $\epsilon$:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$

Sujeto a:
$$y_i - (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \leq \epsilon + \xi_i$$
$$(\mathbf{w}^T \phi(\mathbf{x}_i) + b) - y_i \leq \epsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

Donde:
- $\epsilon$: ancho del tubo (error tolerable)
- $\xi_i, \xi_i^*$: variables de holgura para errores por encima/debajo

### 3.6 Hiperparámetros de SVM

| Hiperparámetro | Valores típicos | Efecto | Optimización |
|----------------|----------------|--------|--------------|
| **$C$** (regularización) | 0.1, 1, 10, 100 | Controla trade-off margen-error | Logarítmica |
| **kernel** | 'linear', 'rbf', 'poly', 'sigmoid' | Tipo de transformación | Según datos |
| **$\gamma$** (RBF/poly) | 0.001, 0.01, 0.1, 1 | Influencia de cada punto | Logarítmica |
| **degree** (poly) | 2, 3, 4, 5 | Grado del polinomio | Enteros pequeños |
| **coef0** (poly/sigmoid) | 0.0, 1.0, 2.0 | Término independiente | Lineal |
| **$\epsilon$** (SVR) | 0.1, 0.2, 0.5 | Ancho del tubo | Validación cruzada |

**Regla empírica para $\gamma$:**
$$\gamma = \frac{1}{n \cdot \text{var}(X)}$$

### 3.7 Ventajas y Limitaciones

**Ventajas:**
- Efectivo en alta dimensión
- Memoria eficiente (usa vectores soporte)
- Versátil (diferentes kernels)
- Robusto a overfitting (regularización)

**Limitaciones:**
- No escala bien a grandes datasets ($O(n^2)$ a $O(n^3)$)
- Sensible a escala de variables
- Interpretabilidad baja (especialmente kernels no lineales)
- Requiere ajuste fino de hiperparámetros

### 3.8 Plantilla Base

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# ========== CLASIFICACIÓN ==========
# Kernel RBF (por defecto)
svm_clf = SVC(
    C=1.0,                    # Regularización
    kernel='rbf',             # 'linear', 'rbf', 'poly', 'sigmoid'
    gamma='scale',            # 'scale', 'auto' o valor numérico
    degree=3,                 # Solo para kernel='poly'
    coef0=0.0,                # Solo para 'poly' y 'sigmoid'
    probability=False,        # Si True, permite predict_proba
    class_weight=None         # 'balanced' para clases desbalanceadas
)

# Kernel Lineal (más rápido, interpretable)
svm_linear = SVC(
    C=1.0,
    kernel='linear'
)

# ========== REGRESIÓN ==========
svr = SVR(
    C=1.0,                    # Regularización
    kernel='rbf',             # Tipo de kernel
    gamma='scale',            # Parámetro del kernel
    epsilon=0.1,              # Ancho del tubo
    degree=3,
    coef0=0.0
)

# ========== ESCALADO OBLIGATORIO ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenamiento y predicción
svm_clf.fit(X_scaled, y)
predicciones = svm_clf.predict(X_test_scaled)

# Para obtener probabilidades (si probability=True)
probabilidades = svm_clf.predict_proba(X_test_scaled)
```

---

## Resumen de Cuándo Usar Cada Modelo

| Modelo | Mejor para | Evitar cuando |
|--------|-----------|---------------|
| **KNN** | Datos de baja dimensión, similitud local, interpretabilidad | Alta dimensión, datos grandes (>100k), outliers |
| **Naive Bayes** | Texto, alta dimensionalidad, clasificación rápida | Características correlacionadas, regresión |
| **SVM** | Dimensionalidad media, fronteras complejas, precisión | Datos muy grandes (>100k), necesidad de interpretabilidad |

## Requisitos Clave por Modelo

| Requisito | KNN | Naive Bayes | SVM |
|-----------|-----|-------------|-----|
| **Escalado** | OBLIGATORIO | NO necesario | OBLIGATORIO |
| **Normalidad** | No | GaussianNB sí | No |
| **Independencia** | No | SÍ (supuesto) | No |
| **Linealidad** | No | No | Depende kernel |