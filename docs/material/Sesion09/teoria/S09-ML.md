
# Sesión 9: APRENDIZAJE NO SUPERVISADO
## Clustering y Análisis de Componentes Principales

**Autor:** Carlos César Sánchez Coronel  
**Fecha:** 2026

---

## Índice

- [Sesión 9: APRENDIZAJE NO SUPERVISADO](#sesión-9-aprendizaje-no-supervisado)
  - [Clustering y Análisis de Componentes Principales](#clustering-y-análisis-de-componentes-principales)
  - [Índice](#índice)
- [Aprendizaje No Supervisado: Clustering y PCA](#aprendizaje-no-supervisado-clustering-y-pca)
  - [Logro de la sesión](#logro-de-la-sesión)
  - [Introducción: aprendizaje no supervisado](#introducción-aprendizaje-no-supervisado)
  - [Análisis de Componentes Principales (PCA)](#análisis-de-componentes-principales-pca)
    - [Fundamento matemático](#fundamento-matemático)
    - [Interpretación de los componentes](#interpretación-de-los-componentes)
    - [Scree plot](#scree-plot)
    - [Aplicaciones de PCA](#aplicaciones-de-pca)
    - [Ejemplo numérico](#ejemplo-numérico)
  - [K-Means Clustering](#k-means-clustering)
    - [Algoritmo](#algoritmo)
    - [Función objetivo](#función-objetivo)
    - [Elección de (k)](#elección-de-k)
      - [Método del codo](#método-del-codo)
      - [Coeficiente de silueta](#coeficiente-de-silueta)
    - [Inicialización: K-Means++](#inicialización-k-means)
    - [Limitaciones de K-Means](#limitaciones-de-k-means)
    - [Aplicaciones](#aplicaciones)
  - [DBSCAN (Density-Based Spatial Clustering of Applications with Noise)](#dbscan-density-based-spatial-clustering-of-applications-with-noise)
    - [Parámetros](#parámetros)
    - [Clasificación de puntos](#clasificación-de-puntos)
    - [Algoritmo](#algoritmo-1)
    - [Ventajas](#ventajas)
    - [Desventajas](#desventajas)
    - [Aplicaciones](#aplicaciones-1)
  - [Clustering Jerárquico](#clustering-jerárquico)
    - [Aglomerativo](#aglomerativo)
    - [Tipos de enlace](#tipos-de-enlace)
    - [Dendrograma](#dendrograma)
    - [Ventajas y desventajas](#ventajas-y-desventajas)
  - [Métricas de evaluación para clustering](#métricas-de-evaluación-para-clustering)
    - [Coeficiente de silueta](#coeficiente-de-silueta-1)
    - [Índice de Davies-Bouldin](#índice-de-davies-bouldin)
    - [Inercia (suma de cuadrados intra-cluster)](#inercia-suma-de-cuadrados-intra-cluster)
    - [Comparación de algoritmos](#comparación-de-algoritmos)
  - [Caso integrado: Segmentación de clientes](#caso-integrado-segmentación-de-clientes)
    - [Problema](#problema)
    - [Preprocesamiento](#preprocesamiento)
    - [PCA](#pca)
    - [Selección de (k) con K-Means](#selección-de-k-con-k-means)
    - [K-Means con (k=4)](#k-means-con-k4)
    - [Validación con silueta](#validación-con-silueta)
    - [Acciones de marketing](#acciones-de-marketing)
    - [Visualización final](#visualización-final)
  - [Implementación en Python](#implementación-en-python)
    - [PCA con scikit-learn](#pca-con-scikit-learn)
    - [K-Means](#k-means)
    - [DBSCAN](#dbscan)
    - [Clustering jerárquico y dendrograma](#clustering-jerárquico-y-dendrograma)

---

# Aprendizaje No Supervisado: Clustering y PCA

En las sesiones anteriores se trabajó con datos etiquetados, donde la variable objetivo guiaba el aprendizaje. Sin embargo, en muchos escenarios prácticos no se dispone de etiquetas. El aprendizaje no supervisado busca descubrir estructuras ocultas en los datos: agrupaciones naturales (clustering) o representaciones de menor dimensionalidad que retengan la información esencial (reducción de dimensionalidad). Esta sesión aborda dos de las técnicas más importantes: Análisis de Componentes Principales (PCA) para reducción de dimensionalidad y varios algoritmos de clustering (K-Means, DBSCAN, jerárquico) para segmentación. Se incluyen fundamentos matemáticos, criterios de evaluación y aplicaciones reales.

## Logro de la sesión

Aplicar técnicas de reducción de dimensionalidad y clustering para explorar datos no etiquetados, interpretar sus resultados y evaluar su calidad mediante métricas apropiadas.

## Introducción: aprendizaje no supervisado

El aprendizaje no supervisado se utiliza cuando se tienen datos sin etiquetas. Los objetivos principales son:
- Descubrir grupos o segmentos naturales (clustering).
- Reducir la dimensionalidad para visualización, compresión o como paso previo a otros algoritmos.
- Detectar anomalías o valores atípicos.

A diferencia del aprendizaje supervisado, no hay una métrica de rendimiento única; la evaluación depende del objetivo y del conocimiento del dominio.

## Análisis de Componentes Principales (PCA)

PCA es una técnica de reducción de dimensionalidad que transforma las variables originales en un nuevo conjunto de variables no correlacionadas llamadas componentes principales, ordenadas por la cantidad de varianza de los datos que retienen.

### Fundamento matemático

Dado un conjunto de datos centrados \(\mathbf{X}\) de dimensiones \(n \times p\) (filas: observaciones, columnas: características), se busca una proyección lineal que maximice la varianza.

1. **Centrado**: se resta la media de cada característica para que \(\mathbf{X}\) tenga media cero por columnas.
2. **Matriz de covarianza**:
   \[
   \mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}
   \]
3. **Descomposición espectral**: se calculan los autovalores \(\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0\) y los autovectores asociados \(\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_p\) de \(\mathbf{\Sigma}\).
4. **Componentes principales**: el primer componente es la combinación lineal \(\mathbf{X} \mathbf{v}_1\), que tiene la máxima varianza (\(\lambda_1\)). El segundo componente es \(\mathbf{X} \mathbf{v}_2\), ortogonal al primero, con la segunda máxima varianza, y así sucesivamente.
5. **Proyección a \(k\) dimensiones**: se seleccionan los primeros \(k\) autovectores y se forma la matriz \(\mathbf{V}_k\) de dimensiones \(p \times k\). La representación reducida es:
   \[
   \mathbf{Z} = \mathbf{X} \mathbf{V}_k
   \]
   donde \(\mathbf{Z}\) es \(n \times k\).

### Interpretación de los componentes

- Los autovalores \(\lambda_j\) representan la varianza explicada por cada componente.
- La proporción de varianza explicada por el componente \(j\) es \(\lambda_j / \sum_{i=1}^p \lambda_i\).
- Los autovectores (loadings) indican la contribución de cada variable original al componente. Valores altos en magnitud indican variables importantes para ese componente.

### Scree plot

Un scree plot representa los autovalores en orden descendente. Se utiliza para decidir cuántos componentes retener: se busca un "codo" donde la varianza explicada adicional sea pequeña.

*[En el documento original se incluye una gráfica de scree plot con autovalores: 2.5, 1.2, 0.6, 0.3, 0.2 para 5 componentes]*

**Figura 1:** Ejemplo de scree plot. El codo se observa entre el segundo y tercer componente.

### Aplicaciones de PCA

- **Visualización**: proyectar datos a 2D o 3D para inspección visual.
- **Reducción de ruido**: descartar componentes de baja varianza que suelen corresponder a ruido.
- **Preprocesamiento**: reducir dimensionalidad antes de aplicar modelos que sufren la maldición de la dimensionalidad (ej. KNN, clustering).
- **Compresión**: almacenar solo las primeras componentes.

### Ejemplo numérico

Supongamos dos variables \(x_1\) y \(x_2\) con datos centrados:
\[
\mathbf{X} = \begin{pmatrix}
1 & 2 \\
2 & 1 \\
3 & 4 \\
4 & 3
\end{pmatrix}
\]

Matriz de covarianza:
\[
\mathbf{\Sigma} = \begin{pmatrix}
1.67 & 1.00 \\
1.00 & 1.67
\end{pmatrix}
\]

Autovalores: \(\lambda_1 = 2.67\), \(\lambda_2 = 0.67\). Autovectores: \(\mathbf{v}_1 = (0.707, 0.707)\), \(\mathbf{v}_2 = (-0.707, 0.707)\). La primera componente explica el 80% de la varianza. Proyectando, se obtienen las nuevas coordenadas.

## K-Means Clustering

K-Means es el algoritmo de clustering más popular. Particiona los datos en \(k\) grupos, donde cada punto pertenece al cluster con el centroide más cercano.

### Algoritmo

1. **Inicialización**: seleccionar \(k\) centroides iniciales (aleatorios o con K-Means++).
2. **Asignación**: asignar cada punto al centroide más cercano (generalmente distancia euclidiana).
3. **Actualización**: recalcular cada centroide como la media de los puntos asignados.
4. Repetir 2 y 3 hasta convergencia (los centroides no cambian o se alcanza un número máximo de iteraciones).

### Función objetivo

K-Means minimiza la suma de distancias cuadráticas intra-cluster (inercia):
\[
J = \sum_{j=1}^k \sum_{x \in C_j} \|x - \mu_j\|^2
\]
donde \(\mu_j\) es el centroide del cluster \(C_j\).

### Elección de \(k\)

#### Método del codo

Se ejecuta K-Means para distintos valores de \(k\) y se grafica la inercia. Se busca el punto donde la reducción de inercia se vuelve marginal (codo).

*[En el documento original se incluye una gráfica de inercia para k=1..10 con valores: 90, 60, 40, 30, 25, 22, 20, 19, 18, 17]*

**Figura 2:** Método del codo. El codo se observa en \(k=3\) o \(k=4\).

#### Coeficiente de silueta

Para cada punto, se define:
- \(a(i)\): distancia media a los puntos de su mismo cluster (cohesión).
- \(b(i)\): distancia media al cluster vecino más cercano (separación).

La silueta de un punto es:
\[
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
\]

Varía entre -1 y 1. Valores cercanos a 1 indican clusters bien separados y compactos. El coeficiente de silueta promedio para todo el dataset se usa para elegir \(k\).

### Inicialización: K-Means++

Para evitar mínimos locales pobres, K-Means++ elige centroides iniciales de manera que estén dispersos:
1. Elegir el primer centroide aleatoriamente.
2. Para cada punto, calcular la distancia al centroide más cercano ya elegido.
3. Elegir el siguiente centroide con probabilidad proporcional al cuadrado de esa distancia.
4. Repetir hasta tener \(k\) centroides.

Es la implementación por defecto en scikit-learn.

### Limitaciones de K-Means

- Asume clusters esféricos y de tamaño similar.
- Sensible a outliers.
- Requiere especificar \(k\).
- Puede converger a óptimos locales.

### Aplicaciones

- Segmentación de clientes en marketing.
- Compresión de imágenes (reducir colores a \(k\)).
- Agrupamiento de documentos.

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN agrupa puntos basándose en la densidad: las regiones densas se expanden y los puntos aislados se consideran ruido.

### Parámetros

- \(\varepsilon\) (eps): radio de vecindad.
- `minPts`: número mínimo de puntos requeridos para formar una región densa.

### Clasificación de puntos

- **Punto núcleo**: tiene al menos `minPts` vecinos dentro de \(\varepsilon\) (incluyéndose a sí mismo).
- **Punto borde**: no es núcleo pero es vecino de un núcleo.
- **Ruido**: no es núcleo ni borde.

### Algoritmo

1. Para cada punto no visitado, encontrar sus vecinos en radio \(\varepsilon\).
2. Si es punto núcleo, formar un nuevo cluster y expandirlo recursivamente incluyendo todos los puntos densidad-alcanzables.
3. Si es borde, asignarlo al cluster correspondiente.
4. Si es ruido, marcarlo como tal.

### Ventajas

- No requiere especificar número de clusters.
- Detecta clusters de forma arbitraria.
- Identifica outliers como ruido.

### Desventajas

- Sensible a los parámetros \(\varepsilon\) y `minPts`.
- Difícil de ajustar cuando las densidades varían.

### Aplicaciones

- Detección de anomalías (fraudes, fallos).
- Datos geoespaciales (clusters de ubicaciones).
- Datos con formas no convexas.

## Clustering Jerárquico

El clustering jerárquico construye una jerarquía de clusters mediante enfoques aglomerativos (de abajo arriba) o divisivos (de arriba abajo).

### Aglomerativo

1. Comenzar con cada punto como un cluster individual.
2. En cada paso, fusionar los dos clusters más cercanos según una medida de distancia entre clusters (enlace).
3. Repetir hasta que todos los puntos estén en un único cluster.

### Tipos de enlace

- **Enlace simple**: distancia mínima entre puntos de los dos clusters. Tiende a formar cadenas.
- **Enlace completo**: distancia máxima. Tiende a formar clusters compactos.
- **Enlace promedio**: promedio de distancias entre todos los pares.
- **Enlace de Ward**: minimiza el incremento de varianza intra-cluster.

### Dendrograma

Representación gráfica de la jerarquía. Cortando el dendrograma a cierta altura se obtienen clusters.

*[En el documento original se incluye un diagrama de dendrograma con 4 puntos A, B, C, D]*

**Figura 3:** Ejemplo de dendrograma con 4 puntos.

### Ventajas y desventajas

- No requiere \(k\) previo.
- Proporciona jerarquías interpretables.
- Costoso computacionalmente (\(O(n^3)\) o \(O(n^2)\)).

## Métricas de evaluación para clustering

Dado que no hay etiquetas, la evaluación es interna (basada en los datos) o externa (si se dispone de ground truth).

### Coeficiente de silueta

Ya descrito. Promedio sobre todos los puntos. Bueno para comparar diferentes \(k\) o algoritmos.

### Índice de Davies-Bouldin

Mide la similitud promedio entre cada cluster y su más similar. Para un cluster \(i\), se define \(R_i = \max_{j \neq i} (s_i + s_j) / d_{ij}\), donde \(s_i\) es la dispersión intra-cluster (distancia media al centroide) y \(d_{ij}\) es la distancia entre centroides. El índice final es el promedio de \(R_i\) sobre todos los clusters. Valores más bajos indican mejor separación.

### Inercia (suma de cuadrados intra-cluster)

Utilizada en K-Means. Útil para el método del codo, pero no es una métrica absoluta (depende de la escala).

### Comparación de algoritmos

| **Algoritmo** | **Forma clusters** | **Número de clusters** | **Ruido** | **Escalabilidad** |
|---------------|-------------------|----------------------|-----------|------------------|
| K-Means | Esféricos | Fijo \(k\) | No | Alta |
| DBSCAN | Arbitraria | Automático | Sí | Media |
| Jerárquico | Cualquier | Corte en dendrograma | No | Baja |

**Tabla 1:** Comparación de algoritmos de clustering.

## Caso integrado: Segmentación de clientes

### Problema

Una empresa de retail desea segmentar a sus clientes para personalizar campañas de marketing. Dispone de datos transaccionales con variables como:
- Monto total gastado en el último año.
- Frecuencia de compra (número de compras).
- Antigüedad como cliente (meses).
- Edad del cliente.

Se tienen 5000 clientes.

### Preprocesamiento

- Estandarizar las variables (escalado Z-score) para que todas tengan la misma influencia.
- Aplicar PCA para visualizar y reducir dimensionalidad.

### PCA

Se calculan las componentes principales. Las dos primeras componentes explican el 75% de la varianza. Se proyectan los datos a 2D para visualización.

### Selección de \(k\) con K-Means

Se prueba K-Means con \(k\) de 2 a 10, calculando inercia y silueta. El método del codo sugiere \(k=4\) y la silueta máxima también en \(k=4\).

### K-Means con \(k=4\)

Se ejecuta K-Means++ con 4 clusters. Se obtienen los centroides en el espacio original (interpretando medias por variable). Se caracterizan los clusters:
- **Cluster 0**: clientes jóvenes, gasto medio, alta frecuencia (jóvenes leales).
- **Cluster 1**: clientes mayores, alto gasto, baja frecuencia (compradores esporádicos de alto valor).
- **Cluster 2**: clientes de edad media, gasto bajo, frecuencia baja (clientes ocasionales).
- **Cluster 3**: clientes con alta antigüedad, gasto alto, frecuencia alta (clientes premium).

### Validación con silueta

El coeficiente de silueta promedio es 0.52, indicando una estructura razonable.

### Acciones de marketing

- Para el cluster 0: ofertas en productos de moda, programas de fidelización.
- Para el cluster 1: promociones especiales para incentivar la frecuencia.
- Para el cluster 2: campañas de reactivación.
- Para el cluster 3: trato VIP, descuentos exclusivos.

### Visualización final

Se grafican los clientes en el espacio de las dos primeras componentes, coloreados por cluster, confirmando la separación visual.

## Implementación en Python

### PCA con scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Varianza explicada: {pca.explained_variance_ratio_}")
```

### K-Means

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Elegir k
best_k = np.argmax(silhouettes) + 2
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
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
