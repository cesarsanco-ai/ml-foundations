---
layout: default
---

# Cheatsheet: Modelos Complementarios
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-11](../../../sesiones/sesion-11.md)

---

## Regresión polinómica

* Features $x, x^2, \dots, x^d$; sigue siendo lineal en $\beta$.  
* **Riesgo:** overfitting si $d$ es alto → validar grado.  

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("lin", LinearRegression()),
])
pipe.fit(X_train, y_train)
```

---

## Regresión robusta

| Método | Idea |
| :--- | :--- |
| **Huber** | Pérdida mixta cuadrática/lineal |
| **RANSAC** | Ajusta inliers; tolera muchos outliers |

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

huber = HuberRegressor(epsilon=1.35).fit(X_train, y_train)
ransac = RANSACRegressor(min_samples=50, residual_threshold=5.0).fit(X_train, y_train)
```

---

## SVR

* Margen $\varepsilon$ + penalización $C$; kernels como en SVM.  
* Escalar features.  

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1))
model.fit(X_train, y_train)
```

---

## DBSCAN (repaso avanzado)

* Elegir **eps** con gráfico k-distancia (`NearestNeighbors`).  
* **minPts** ≈ dim + 1 típicamente.  
* Variantes: **OPTICS**, **HDBSCAN** para densidad variable.  

---

## Clustering jerárquico

* Enlaces: **single** (cadenas), **complete** (compactos), **average**, **ward**.  
* Coste alto: $O(n^3)$ ingenuo → datasets grandes con cautela.  

---

## Métricas externas (con etiquetas)

* **Rand**, **ARI** (ajustado por azar), **NMI**.  

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)
```

---

## Guía rápida

| Problema | Prueba |
| :--- | :--- |
| Outliers en regresión | Huber / RANSAC |
| No lineal suave, pocas $x$ | Polinómica |
| No lineal + margen | SVR RBF |
| Clusters raros + ruido | DBSCAN |
| Jerarquía interpretable | Aglomerativo + dendrograma |

---

## Puntos críticos

* RANSAC **no determinista**; fijar semillas y revisar inliers.  
* SVR puede ser costoso en $n$ muy grande.  
* ARI/NMI requieren **etiquetas** de referencia (semi-supervisado o benchmark).  

> *“Cuando lo clásico falla, estos métodos salvan casos borde.”*
