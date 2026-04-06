---
layout: default
---
# Semana 11: Modelos Complementarios

En problemas reales surgen situaciones que requieren enfoques especializados: outliers severos, relaciones no lineales difíciles para árboles, clusters arbitrarios o interpretación jerárquica. Esta sesión presenta algoritmos complementarios frecuentes en entrevistas y en casos concretos.

---

## Logro de la sesión

Conocer algoritmos y técnicas complementarias, y entender cuándo aplicarlas según los datos y el problema.

---

## Regresión polinómica

### Idea

Añadir términos $x, x^2, \dots, x^d$ como features. El modelo sigue siendo **lineal en los parámetros** $\beta$:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d + \varepsilon
$$

### Riesgo de overfitting

Grados $d$ altos ajustan ruido en los extremos; validar $d$ con validación cruzada.

### Cuándo usarla

- Relación no lineal pero suave.
- Pocas variables y necesidad de interpretabilidad relativa.

---

## Regresión robusta

### Huber

Pérdida que combina cuadrática (errores pequeños) y lineal (errores grandes):

$$
L_{\delta}(r) = \begin{cases}
\frac{1}{2} r^2 & \text{si } |r| \le \delta \\
\delta(|r| - \frac{1}{2}\delta) & \text{en otro caso}
\end{cases}
$$

$r = y - \hat{y}$; $\delta$ controla la transición.

### RANSAC

1. Subconjunto aleatorio mínimo para ajustar el modelo.
2. Contar inliers bajo umbral de distancia.
3. Si hay suficientes inliers, reajustar con todos los inliers.
4. Repetir y quedarse con el mejor modelo.

**Ventaja:** muy robusto, tolera muchos outliers.  
**Desventaja:** hiperparámetros y no determinismo.

### Comparación de pérdidas

| Función | Ecuación | Sensibilidad a outliers |
| :--- | :--- | :--- |
| MSE | $r^2$ | Alta |
| MAE | $|r|$ | Moderada |
| Huber | $L_{\delta}(r)$ | Baja (con $\delta$ adecuado) |

---

## Support Vector Regression (SVR)

Extiende SVM a regresión: función $f(x)$ con margen $\varepsilon$ y máxima “planitud”.

### Idea de formulación (lineal)

Minimizar $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i(\xi_i + \xi_i^*)$ con restricciones que acotan desviaciones fuera de la banda $\varepsilon$.

- **$\varepsilon$:** tolerancia (errores pequeños no penalizan).
- **$C$:** penalización por puntos fuera de la banda.

### Kernel

Como en SVM clasificación: lineal, polinomial, RBF.

---

## DBSCAN en profundidad

### Parámetros

- **eps:** gráfico de distancia al $k$-ésimo vecino (codo); $k$ suele ser `minPts`.
- **minPts:** típicamente $\ge$ dimensión + 1; valores 3–10 habituales.

### Métricas

En alta dimensión la euclídea pierde significado; probar coseno o Manhattan.

### Variantes

- **OPTICS:** densidad variable; diagrama de alcanzabilidad.
- **HDBSCAN:** jerárquico, distintas densidades.

### vs K-Means

- Formas no esféricas, sin fijar $k$, etiqueta ruido.
- Limitación: sensibilidad a eps/minPts y densidades muy variables.

---

## Clustering jerárquico

### Tipos de enlace

- **Single:** mínima distancia entre clusters → cadenas.
- **Complete:** máxima → clusters compactos.
- **Average:** promedio de distancias.
- **Ward:** minimiza incremento de SS intra-cluster → tamaños más parejos.

### Corte del dendrograma

Saltos grandes en altura de fusión o barrido con silueta.

### Complejidad

Aglomerativo: $O(n^3)$ ingenuo o $O(n^2 \log n)$ con estructuras adecuadas; no ideal para datos enormes.

---

## Métricas externas de clustering

### Rand Index (RI)

$$
RI = \frac{a + b}{\binom{n}{2}}
$$

$a$: pares en el mismo cluster en ambas particiones; $b$: en distinto en ambas.

### Adjusted Rand Index (ARI)

Corrige por azar; puede ser negativo si peor que aleatorio.

### Mutual Information (MI) y NMI

MI mide información compartida entre particiones; NMI normaliza a $[0,1]$ aprox.

| Métrica | Rango | Corrige azar | Requiere etiquetas |
| :--- | :--- | :--- | :--- |
| Silueta | $[-1,1]$ | No | No |
| Davies-Bouldin | $[0,\infty)$ menor mejor | No | No |
| RI | $[0,1]$ | No | Sí |
| ARI | $[-1,1]$ | Sí | Sí |
| NMI | $[0,1]$ | Aprox. | Sí |

---

## Guía rápida

| Técnica | Cuándo usarla |
| :--- | :--- |
| Polinómica | No lineal suave, pocas variables |
| Huber | Outliers moderados |
| RANSAC | Muchos outliers (>50%) |
| SVR | No linealidad, control de banda $\varepsilon$ |
| DBSCAN | Formas arbitrarias, ruido, $k$ desconocido |
| Jerárquico | Jerarquía interpretable, $n$ moderado |

---

## Caso integrado: sensores industriales

RANSAC para temperatura vs presión (outliers = anomalías); DBSCAN en (T, P, vibración); regresión polinómica por cluster para límites de control; concordancia outliers con ARI entre métodos.

---

## Implementación en Python

### Polinómica

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("linear", LinearRegression()),
])
model.fit(X_train, y_train)
```

### Huber

```python
from sklearn.linear_model import HuberRegressor

huber = HuberRegressor(epsilon=1.35)
huber.fit(X_train, y_train)
```

### RANSAC

```python
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(min_samples=50, residual_threshold=5.0, max_trials=100)
ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
```

### SVR

```python
from sklearn.svm import SVR

svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)
```

### DBSCAN y gráfico k-distancia

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X)
distances, _ = neigh.kneighbors(X)
k_dist = np.sort(distances[:, -1])
plt.plot(k_dist)
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
```

### Jerárquico

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

Z = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.show()
clusters = fcluster(Z, t=3, criterion="maxclust")
```

### ARI / NMI

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)
```

---

## Resumen

- Polinómica: flexibilidad con riesgo de overfitting.
- Huber y RANSAC: robustez frente a outliers.
- SVR: regresión con margen y kernels.
- DBSCAN y variantes: densidad y ruido.
- Jerárquico: dendrograma y elección de enlace.
- ARI y NMI evalúan clustering con ground truth.
