---
layout: default
---
# Sesión 9: Técnicas de Clustering, PCA y t-SNE

### 1. Logro de la sesión

Dominar el **aprendizaje no supervisado** desde tres frentes: **reducción de dimensionalidad lineal (PCA)**, **visualización no lineal (t-SNE)** y **clustering particional, densidad y jerárquico** (K-Means, DBSCAN, aglomerativo), incluyendo **criterios de elección de hiperparámetros**, **métricas internas sin etiquetas** y **patrones de código en Python** alineados al temario.

---

### 2. Historia y línea temporal (panorama)

| Año / periodo | Tema | Referencia representativa |
|---------------|------|-----------------------------|
| **1901** | **PCA** (Pearson) y luego **Hotelling** (1933): direcciones de varianza máxima | Álgebra lineal aplicada a datos |
| **1950s–60s** | Métodos factoriales y componentes en psicometría | Jolliffe (2002) |
| **1957** | **Lloyd** publica (1982) el algoritmo k-means | Clustering particional |
| **1996** | **DBSCAN** (Ester et al.): densidad y ruido | KDD |
| **2002** | **Spectral clustering**, reformulaciones de corte de grafos | Ng, Jordan, Weiss |
| **2008** | **t-SNE** (van der Maaten & Hinton) | Visualización no lineal |
| **Actualidad** | UMAP (2018) como alternativa a t-SNE en algunos dominios | McInnes et al. (mención) |

**Lectura:** el no supervisado **no tiene una métrica universal de “acierto”**; la validez depende de **estabilidad**, **utilidad de negocio** y, cuando sea posible, validación externa.

---

### 3. PCA: fundamento matemático detallado

#### 3.1 Objetivo

Dado $\mathbf{X} \in \mathbb{R}^{n \times p}$ **centrado** por columnas, buscar proyecciones ortogonales $\mathbf{Z} = \mathbf{X}\mathbf{V}$ tales que las columnas de $\mathbf{Z}$ tengan **varianza máxima** sucesivamente y sean **no correlacionadas**.

#### 3.2 Vía SVD (enfoque estándar)

Sea la descomposición en valores singulares:

$$ \mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top $$

Las **direcciones principales** son columnas de $\mathbf{V}$; las **scores** son $\mathbf{U}\boldsymbol{\Sigma}$. Los **autovalores** de la matriz de covarianza muestral $ \mathbf{S} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X}$ son $\lambda_j = \sigma_j^2/(n-1)$.

**Varianza explicada** por el componente $j$: $\lambda_j / \sum_k \lambda_k$.

#### 3.3 ¿Cuántos componentes retener?

- **Scree plot:** gráfico de $\lambda_j$ vs $j$; buscar “codo”.  
- **Umbral acumulado:** p.ej. 90–95 % de varianza explicada (depende del uso).  
- **Validación supervisada indirecta:** PCA como preprocesamiento y medir rendimiento downstream.

#### 3.4 Limitaciones

- **Linealidad:** solo rota/ proyecta linealmente.  
- **Escalado:** obligatorio si las unidades difieren.  
- **Interpretación:** loadings mezclan variables originales; interpretar “componentes” requiere cuidado.

#### 3.5 Plantilla Python (`PCA`)

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe_pca = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, svd_solver="full")),  # 95% varianza
])
X_proj = pipe_pca.fit_transform(X)
explained = np.cumsum(pipe_pca.named_steps["pca"].explained_variance_ratio_)
print("Componentes usados:", pipe_pca.named_steps["pca"].n_components_)
```

---

### 4. t-SNE: visualización no lineal

#### 4.1 Idea

Construir distribución de similitud en espacio alto $p_{ij}$ y otra en baja dimensión $q_{ij}$, minimizando **divergencia KL**:

$$ C = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}} $$

Con $q_{ij}$ basado en **t-Student** para evitar compresión excesiva en el centro (crowding problem).

#### 4.2 Parámetros

- **perplexity:** número efectivo de vecinos (típico 5–50); controla equilibrio local/global.  
- **learning rate**, **iteraciones**: afectan convergencia.

#### 4.3 Limitaciones (críticas)

- **No** hay transformación trivial para **nuevos puntos** (a diferencia de PCA).  
- **Estocástico:** distintas semillas → distintos layouts.  
- **No interpretar distancias globales** entre clusters lejanos como significativas.

#### 4.4 Plantilla Python

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
)
X_emb = tsne.fit_transform(X_scaled)
```

---

### 5. K-Means: algoritmo y teorema de convergencia

#### 5.1 Objetivo

Minimizar inercia intra-cluster:

$$ \sum_{k=1}^{K} \sum_{i \in C_k} \| x_i - \mu_k \|^2 $$

#### 5.2 Lloyd (k-means estándar)

1. Inicializar centroides $\mu_k$.  
2. **Asignación:** cada punto al centroide más cercano.  
3. **Actualización:** recalcular $\mu_k$ como media de puntos asignados.  
4. Repetir hasta convergencia o máximo de iteraciones.

**Converge** a mínimo local (no global). Múltiples **restarts** (`n_init` en sklearn).

#### 5.3 Elección de $K$

- **Codo:** inercia vs $K$.  
- **Silueta media** por $K$.  
- **Davies–Bouldin** (menor es mejor).

#### 5.4 Sensibilidad

- **Escala** de variables (usar `StandardScaler`).  
- **Outliers** pueden desplazar centroides.  
- Asume clusters **convexos** de forma aproximada (fallos en anillos concéntricos).

#### 5.5 Plantilla Python

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

km = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=5, n_init="auto", random_state=42)),
])
labels = km.fit_predict(X)
```

---

### 6. DBSCAN: densidad y ruido

#### 6.1 Conceptos

- **$\epsilon$-vecindad** alrededor de $x$.  
- **Punto núcleo** si tiene al menos `min_samples` puntos en su $\epsilon$-vecindad.  
- **Borde** si no es núcleo pero está en vecindad de un núcleo.  
- **Ruido** si no pertenece a ningún cluster.

**Ventajas:** no requiere $K$ **a priori**; encuentra formas arbitrarias; identifica **outliers** como ruido.

**Desventajas:** sensible a **$\epsilon$** y densidad variable; combina mal con **alta dimensión** (“maldición”).

#### 6.2 Plantilla Python

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
labels = db.fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

---

### 7. Clustering jerárquico aglomerativo

#### 7.1 Enfoque bottom-up

Comienza con cada punto como cluster y **fusiona** el par más cercano según **enlace**:

| Enlace | Distancia entre clusters $A,B$ |
|--------|----------------------------------|
| **Single** | $\min_{a\in A, b\in B} d(a,b)$ — puede encadenar (*chaining*) |
| **Complete** | $\max$ — clusters compactos |
| **Average** | media de pares |
| **Ward** | minimiza incremento de varianza intra (tendencia esférica) |

**Dendrograma:** cortar a altura $h$ para obtener $K$ clusters.

#### 7.2 Plantilla Python

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = agg.fit_predict(X_scaled)
```

---

### 8. Métricas sin *ground truth*

#### 8.1 Silueta (Rousseeuw, 1987)

Para punto $i$:

$$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$

donde $a(i)$ es distancia media intra-cluster y $b(i)$ la menor distancia media a otro cluster.

Rango $[-1,1]$; valores altos → clusters bien separados y cohesionados.

#### 8.2 Davies–Bouldin

Promedio sobre clusters de razones dispersión / separación; **menor** valor suele indicar mejor partición.

#### 8.3 Limitaciones

Todas las métricas internas tienen **supuestos** (p.ej. silueta favorece formas convexas). Validar con **negocio** y estabilidad (bootstrap).

---

### 9. Comparación de métodos y guía de uso

| Método | Fortaleza | Debilidad típica |
|--------|-----------|------------------|
| K-Means | Rápido, escalable | Formas convexas, necesita $K$ |
| DBSCAN | Formas arbitrarias, ruido | Parámetros, alta dimensión |
| Jerárquico | Dendrograma interpretable | Coste en $n$ grande |
| PCA | Preprocesamiento, denoising | Lineal |
| t-SNE | Visualización | No proyecta nuevos puntos fácilmente |

#### 9.1 *Kernel PCA* (mención avanzada)

Si la estructura es **no lineal**, PCA lineal puede fallar. **Kernel PCA** aplica el truco del kernel igual que SVM: encuentre direcciones de varianza máxima en un espacio de características inducido por $K(\mathbf{x},\mathbf{x}')$. Coste computacional mayor; útil antes de clustering en datos curvados.

#### 9.2 Clustering espectral (idea)

Construye un grafo de similitud entre puntos y usa **autovectores del Laplaciano** para embeder datos en dimensiones bajas donde **K-Means** puede separar estructuras no convexas (Ng et al., 2002). `sklearn.cluster.SpectralClustering` lo implementa.

#### 9.3 Selección automática de $K$ con silueta (ejemplo de bucle)

```python
from sklearn.metrics import silhouette_score

sil = []
K_range = range(2, 15)
for k in K_range:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    lab = km.fit_predict(X_scaled)
    sil.append(silhouette_score(X_scaled, lab))

best_k = K_range[int(np.argmax(sil))]
```

**Advertencia:** maximizar silueta **no garantiza** clusters útiles para negocio; puede favorecer particiones equilibradas artificialmente.

#### 9.4 *Clustering* y escalado: check-list

1. `StandardScaler` (o `RobustScaler` con outliers fuertes).  
2. Si usas **Gower** o distancias mixtas (no está en sklearn puro), considerar bibliotecas especializadas.  
3. Para **alta dimensión**, considerar PCA previa solo si preserva varianza relevante.

#### 9.5 Buenas prácticas con t-SNE

- Correr **varias semillas** y comparar estabilidad visual.  
- Probar **perplexidades** distintas; documentar la elegida.  
- No usar t-SNE como **única** evidencia de número de clusters: puede separar o fusionar grupos según parámetros.

---

### 10. Laboratorio (según sílabo)

- **NTB 1 —** PCA y t-SNE: reducción de dimensionalidad y visualización de estructura.  
- **NTB 2 —** Clustering: K-Means, DBSCAN y/o jerárquico con métricas de cohesión y silueta.

---

## Referencias bibliográficas principales

1. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.  
2. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.  
3. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*.  
4. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.  
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (cap. 14). Springer.  
6. Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). On spectral clustering: Analysis and an algorithm. *NeurIPS*.  
