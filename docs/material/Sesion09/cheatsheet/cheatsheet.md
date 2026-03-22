---
layout: default
---

# Cheatsheet: Aprendizaje No Supervisado (PCA y Clustering)
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-09](../../../sesiones/sesion-09.md)

---

## PCA (reducción de dimensionalidad)

* Encuentra direcciones de **máxima varianza** (componentes principales).  
* Útil para visualización, ruido, acelerar modelos y multicolinealidad.  

**Scores:** proyección $Z = X W$. Elegir $k$ con **scree plot** o varianza acumulada.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
pca = PCA(n_components=0.95)  # o entero k
Z = pca.fit_transform(X_s)
```

---

## K-Means

* Minimiza inercia intra-cluster.  
* Requiere **especificar k**; sensible a escala e inicialización.  

**Elegir k:** método del codo, **silueta**, criterio de negocio.

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, random_state=42, n_init="auto")
labels = km.fit_predict(X_scaled)
```

---

## DBSCAN

* Clusters por **densidad**; detecta **ruido** (-1).  
* Parámetros: `eps`, `min_samples`.  

**Pros:** formas arbitrarias, sin k. **Contras:** sensibles a densidad heterogénea.

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)
```

---

## Clustering jerárquico

* **Aglomerativo:** fusiona clusters; visualizar con **dendrograma**.  
* Enlaces: single, complete, average, **ward** (varianza intra).  

```python
from scipy.cluster.hierarchy import linkage, fcluster

Z = linkage(X_scaled, method="ward")
clusters = fcluster(Z, t=4, criterion="maxclust")
```

---

## Métricas (sin etiquetas)

| Métrica | Idea |
| :--- | :--- |
| **Silueta** | Cohesión vs separación; en [-1, 1] |
| **Davies-Bouldin** | Menor mejor; clusters compactos y separados |
| **Inercia** | Solo comparable con mismo k |

---

## Con etiquetas (si existen)

* **ARI**, **NMI** para comparar con ground truth (sesión 11 profundiza).  

---

## Puntos críticos

* **Estandarizar** antes de K-Means, PCA y DBSCAN con euclídea.  
* PCA asume relaciones **lineales**; no sustituye modelos no lineales para todo.  
* “Cluster” ≠ segmento de negocio sin validación cualitativa.  

> *“No supervisado descubre estructura; tú le das sentido.”*
