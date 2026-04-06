---
layout: default
---
# Sesión 9: Clustering, Reducción de Dimensionalidad y Visualización

En sesiones anteriores se trabajó con datos etiquetados, donde la variable objetivo guiaba el aprendizaje. Sin embargo, en muchos escenarios prácticos no se dispone de etiquetas. El **aprendizaje no supervisado** busca descubrir estructuras ocultas en los datos: agrupaciones naturales (clustering), representaciones de menor dimensionalidad que retengan información esencial (PCA) o visualizaciones que preserven la estructura de vecindad (t-SNE).

Esta sesión aborda:
- **Reducción de dimensionalidad**: PCA (lineal) y t-SNE (no lineal para visualización).
- **Clustering**: K-Means, DBSCAN y clustering jerárquico.
- Fundamentos matemáticos, criterios de evaluación y aplicaciones reales.

## Logro de la sesión

Aplicar técnicas de reducción de dimensionalidad y clustering para explorar datos no etiquetados, interpretar sus resultados y evaluar su calidad mediante métricas apropiadas.

## Introducción: aprendizaje no supervisado

El aprendizaje no supervisado se utiliza cuando no hay etiquetas. Los objetivos principales son:
- Descubrir grupos o segmentos naturales (clustering).
- Reducir la dimensionalidad para visualización, compresión o como paso previo a otros algoritmos.
- Detectar anomalías o valores atípicos.

A diferencia del aprendizaje supervisado, no hay una métrica de rendimiento única; la evaluación depende del objetivo y del conocimiento del dominio.

---

## Reducción de dimensionalidad

### Análisis de Componentes Principales (PCA)

PCA es una técnica lineal que transforma las variables originales en un nuevo conjunto de variables no correlacionadas llamadas **componentes principales**, ordenadas por la cantidad de varianza que retienen.

#### Fundamento matemático

Dado un conjunto de datos centrados $\mathbf{X}$ de dimensiones $n \times p$:

1. **Centrado**: se resta la media de cada característica.
2. **Matriz de covarianza**:  
   $$ \mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X} $$
3. **Descomposición espectral**: autovalores $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p$ y autovectores asociados $\mathbf{v}_1, \dots, \mathbf{v}_p$.

#### Interpretación
- Los autovalores representan la varianza explicada por cada componente.
- **Scree plot**: gráfico de autovalores en orden descendente. Se busca un "codo" para decidir cuántos componentes retener.

#### Aplicaciones
- Visualización (proyectar a 2D/3D).
- Reducción de ruido.
- Preprocesamiento antes de clustering.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE es una técnica **no lineal** especializada en visualización. Preserva las distancias locales, es decir, puntos cercanos en el espacio original permanecen cercanos en la proyección, pero no preserva distancias globales.

#### Fundamento conceptual
1. Convierte distancias entre puntos en probabilidades condicionales de similitud.
2. En el espacio de baja dimensión, modela las similitudes con una distribución $t$ de Student.
3. Minimiza la divergencia de Kullback-Leibler entre las dos distribuciones.

#### Parámetros clave
- **perplexity**: balance entre atención a vecinos locales y globales (típico: 5–50).
- **n_iter**: número de iteraciones.

#### Ventajas y limitaciones
- **Ventaja**: excelente para visualizar clusters en 2D/3D.
- **Limitación**: no es determinista, no permite proyectar nuevos datos fácilmente, es computacionalmente costoso.

#### Comparación PCA vs t-SNE

| Característica | PCA | t-SNE |
|----------------|-----|-------|
| Linealidad | Lineal | No lineal |
| Preservación | Varianza global | Distancias locales |
| Determinista | Sí | No |
| Proyección de nuevos datos | Sí | No directamente |
| Uso típico | Reducción general, preprocesamiento | Visualización de clusters |

---

## Algoritmos de Clustering

### K-Means

K-Means particiona los datos en $k$ grupos, donde cada punto pertenece al cluster con el centroide más cercano.

#### Algoritmo
1. Inicializar $k$ centroides (K-Means++).
2. Asignar cada punto al centroide más cercano.
3. Recalcular centroides como la media de los puntos asignados.
4. Repetir hasta convergencia.

#### Función objetivo (inercia)
$$ J = \sum_{j=1}^k \sum_{x \in C_j} \|x - \mu_j\|^2 $$

#### Elección de $k$
- **Método del codo**: graficar inercia vs $k$, buscar el codo.
- **Coeficiente de silueta**: mide cohesión y separación.

#### Limitaciones
- Asume clusters esféricos y de tamaño similar.
- Sensible a outliers.
- Requiere especificar $k$.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN agrupa puntos basándose en densidad. No requiere número de clusters y detecta ruido.

#### Parámetros
- $\varepsilon$ (eps): radio de vecindad.
- `minPts`: mínimo de puntos para formar una región densa.

#### Clasificación de puntos
- **Núcleo**: al menos `minPts` vecinos dentro de $\varepsilon$.
- **Borde**: vecino de un núcleo pero no es núcleo.
- **Ruido**: no es núcleo ni borde.

#### Ventajas
- No requiere especificar número de clusters.
- Detecta clusters de forma arbitraria.
- Identifica outliers como ruido.

#### Desventajas
- Sensible a los parámetros $\varepsilon$ y `minPts`.
- Difícil de ajustar con densidades muy variables.

### Clustering Jerárquico (Aglomerativo)

Construye una jerarquía de clusters fusionando iterativamente los pares más cercanos.

#### Algoritmo
1. Cada punto es un cluster individual.
2. Fusionar los dos clusters más cercanos según un **enlace**.
3. Repetir hasta un solo cluster.

#### Tipos de enlace
- **Simple**: distancia mínima entre puntos.
- **Completo**: distancia máxima.
- **Promedio**: promedio de distancias entre todos los pares.
- **Ward**: minimiza incremento de varianza intra-cluster.

#### Dendrograma
Representación gráfica de la jerarquía. Cortando a cierta altura se obtienen clusters.

#### Ventajas y desventajas
- No requiere $k$ previo.
- Jerarquía interpretable.
- Costoso computacionalmente ($O(n^3)$).

### Comparación de algoritmos

| Algoritmo | Forma de clusters | Número de clusters | Ruido | Escalabilidad |
|-----------|-------------------|---------------------|-------|----------------|
| K-Means | Esféricos | Fijo $k$ | No | Alta |
| DBSCAN | Arbitraria | Automático | Sí | Media |
| Jerárquico | Cualquier | Corte en dendrograma | No | Baja |

---

## Métricas de evaluación para clustering

- **Coeficiente de silueta**
- **Índice de Davies-Bouldin**: menor valor indica mejor separación.
- **Inercia**: útil para método del codo, pero no absoluta.

---

## Caso integrado: Segmentación de clientes con PCA, t-SNE y múltiples clustering

### Problema
Una empresa de retail desea segmentar a sus clientes. Dispone de 5000 registros con:
- Monto total gastado en último año.
- Frecuencia de compra.
- Antigüedad como cliente (meses).
- Edad.

### Preprocesamiento
Estandarización (Z-score) para que todas las variables tengan igual influencia.

### Reducción y visualización
1. **PCA**: se retienen las 2 primeras componentes (explican 75% de varianza).
2. **t-SNE**: se aplica sobre los datos estandarizados (perplexity=30) para visualizar estructura local de clusters.

### Clustering comparativo

#### K-Means
- Se prueba $k=2..10$ con inercia y silueta.
- Mejor $k=4$ (silueta = 0.52).
- Clusters interpretados:
  - Cluster 0: jóvenes, gasto medio, alta frecuencia.
  - Cluster 1: mayores, alto gasto, baja frecuencia.
  - Cluster 2: edad media, bajo gasto, baja frecuencia.
  - Cluster 3: alta antigüedad, alto gasto, alta frecuencia (premium).

#### DBSCAN
- Se ajusta $\varepsilon=0.5$, `minPts=5`.
- Detecta 3 clusters más ruido (aprox. 5% de clientes atípicos).

#### Clustering Jerárquico (Ward)
- Se construye el dendrograma.
- Cortando a una altura que produce 4 clusters, se obtienen grupos consistentes con K-Means.

### Validación final
- Silueta promedio: K-Means 0.52, DBSCAN 0.48 (excluyendo ruido), Jerárquico 0.51.
- Visualización con t-SNE confirma la separación de los 4 grupos.

### Acciones de marketing por cluster
- **Cluster 0**: ofertas en moda juvenil, fidelización.
- **Cluster 1**: promociones para aumentar frecuencia.
- **Cluster 2**: campañas de reactivación.
- **Cluster 3**: trato VIP, descuentos exclusivos.

---

## Implementación en Python (plantilla base)

### PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### t-SNE

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```

### K-Means

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
# Los puntos con label -1 son ruido
```

### Clustering jerárquico y dendrograma

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.show()
```

---

## Resumen y buenas prácticas

1. **Preprocesa siempre** con escalado si usas distancias.
2. Usa **PCA** para reducción general y **t-SNE** solo para visualización (no para preprocesamiento).
3. Prueba múltiples algoritmos de clustering: K-Means (rápido, esférico), DBSCAN (ruido, formas arbitrarias), Jerárquico (jerarquías).
4. Evalúa con **silueta** y valida con **visualización** (t-SNE o PCA).
5. Interpreta los clusters con el dominio del problema.