---
layout: default
---
# Sesión 12: Graph ML y Geodata

### 1. Logro de la sesión

Representar problemas como **grafos**, extraer **features tabulares de nodos y enlaces** (sin necesidad de GNN profundo en el núcleo del temario), aplicar **link prediction** como clasificación sobre pares, y combinar **datos geoespaciales** con modelos clásicos de ML entendiendo **autocorrelación espacial**, **métricas de distancia** y **validación espacial**. Reconocer **límites de escala** y cuándo la literatura apunta a **embeddings** (*DeepWalk*, *Node2vec*) o **GNN**.

---

### 2. Historia y contexto

| Periodo | Hito |
|---------|------|
| **1736** | Euler y los siete puentes de Königsberg: nacimiento de la teoría de grafos |
| **1990s** | Web como grafo; algoritmos de centralidad en sociología y WWW |
| **1998** | **PageRank**: centralidad por recurrencia estocástica |
| **2000s** | *Link prediction* formalizado en redes sociales; **preferential attachment** |
| **2010s** | *DeepWalk*, *node2vec*: caminatas aleatorias + skip-gram |
| **2017+** | **GCN**, **GraphSAGE**: aprendizaje de representaciones con agregación de vecinos |
| **Paralelo** | **SIG** y datos raster/vectorial; **Moran’s I**, *hotspots* con **DBSCAN** |

**Lectura:** muchos problemas “no son grafos hasta que se elige la arista”: la definición de $E$ **determina** la señal disponible.

---

### 3. Grafos: definiciones esenciales

Un **grafo** $G=(V,E)$ consta de nodos $V$ y aristas $E \subseteq V \times V$.

- **No dirigido:** $(u,v) \equiv (v,u)$. **Dirigido:** el orden importa (seguimiento, citas).
- **Ponderado:** peso $w_{uv}$ (fuerza, distancia, número de interacciones).
- **Multigrafo:** múltiples aristas entre el mismo par (llamadas repetidas).
- **Bipartito:** $V = U \cup I$, aristas solo entre $U$ e $I$ (usuarios–ítems).

**Matriz de adyacencia** $\mathbf{A}$: $A_{uv} = w_{uv}$ si hay arista, 0 si no. En no ponderado, $A_{uv} \in \{0,1\}$.

**Lista de adyacencia:** representación dispersa eficiente cuando el grafo es ralo.

---

### 4. Features de nodos (enfoque tabular clásico)

Sin entrenar una GNN, es común calcular **descriptores** por nodo y alimentar Random Forest, gradient boosting o regresión logística.

| Familia | Ejemplos |
|---------|----------|
| **Grado** | $k_u = \sum_v A_{uv}$ (entrante/saliente si es dirigido) |
| **Centralidad de grado** | normalización por $|V|-1$ |
| **Betweenness** (concepto) | fracción de caminos mínimos que pasan por $u$ — costosa en grafos grandes; existen aproximaciones |
| **Eigenvector / PageRank** | importancia recursiva “quién se conecta con importantes” |
| **Estructura local** | *clustering coefficient* (transitividad), conteo de triángulos |
| **Vecindad a 2 saltos** | número de nodos alcanzables, suma de grados de vecinos |
| **Atributos de vecinos** | media/máx de una variable en $\mathcal{N}(u)$ |

Estas features capturan **posición estructural** y **entorno local** transferibles a tabular.

---

### 5. Link prediction

#### 5.1 Formulación

Se observa $G_{\mathrm{obs}}$ y se quiere estimar la probabilidad de aristas **no observadas** o futuras. Típicamente se genera un conjunto de **pares negativos** (no arista) y se entrena un clasificador con features del par $(u,v)$.

#### 5.2 Features clásicas de pares

| Feature | Idea |
|---------|------|
| **Common neighbors** | $|\mathcal{N}(u) \cap \mathcal{N}(v)|$ |
| **Jaccard** | $\frac{|\mathcal{N}(u) \cap \mathcal{N}(v)|}{|\mathcal{N}(u) \cup \mathcal{N}(v)|}$ |
| **Preferential attachment** | $k_u \cdot k_v$ |
| **Adamic/Adar** | $\sum_{w \in \mathcal{N}(u)\cap\mathcal{N}(v)} \frac{1}{\log k_w}$ |
| **Katz** (parámetro $\beta$) | suma de caminos de longitud variable con decaimiento |
| **Shortest path length** | si es finito en $G_{\mathrm{obs}}$ |

En grafos dirigidos, distinguir **predecesores** y **sucesores**.

---

### 6. Embeddings de grafos (referencia para lectura)

**DeepWalk:** caminatas aleatorias desde cada nodo → secuencias → **Word2Vec** (Skip-gram).  
**Node2vec:** sesga la caminata con parámetros $p,q$ para explorar **BFS/DFS-like** y capturar homofilia o equivalencias estructurales.

**GNN (visión de alto nivel):** en cada capa, el embedding del nodo se actualiza agregando transformaciones de embeddings de vecinos (**mensaje + agregación + no-linealidad**). Requiere frameworks especializados (*PyTorch Geometric*, *DGL*) y hardware adecuado para grafos grandes.

---

### 7. Geodata: coordenadas, distancias y proyecciones

#### 7.1 Latitud y longitud

Son coordenadas **esféricas**. La distancia geodésica corta sobre la esfera se aproxima con **Haversine**:

$$ d = 2r \arcsin\left(\sqrt{\sin^2\frac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\frac{\Delta\lambda}{2}}\right) $$

con $r$ radio terrestre, $\phi$ latitud, $\lambda$ longitud.

#### 7.2 Proyecciones métricas

Para regiones **locales**, proyectar a un CRS métrico (p. ej. UTM) permite usar **Euclides** en metros con error controlado. Para **distancias intercontinentales**, Haversine u otras fórmulas esféricas son más coherentes.

#### 7.3 Autocorrelación espacial (*Tobler’s first law*)

“Todo está relacionado con todo lo demás, pero las cosas cercanas más que las lejanas.” Los puntos cercanos suelen tener **valores de target correlacionados** → violación de independencia si se trata el problema como i.i.d.

**Implicación ML:** métricas de validación pueden ser **optimistas** si el train y test están geográficamente mezclados pero en despliegue se generaliza a **nuevas regiones**.

#### 7.4 Moran’s I (idea)

Estadístico global de autocorrelación espacial de una variable $z$:

$$I = \frac{n}{\sum_{i,j} w_{ij}} \frac{\sum_{i,j} w_{ij}(z_i-\bar{z})(z_j-\bar{z})}{\sum_i (z_i-\bar{z})^2}$$

con pesos $w_{ij}$ (contigüidad, inverso de distancia). Valores altos sugieren **agrupación** de valores similares.

---

### 8. Clustering espacial y DBSCAN

**DBSCAN** agrupa puntos densamente conectados y marca **ruido**. Aplicado a $(x,y)$ proyectados o a coordenadas con métrica adecuada, detecta **hotspots**.

**Cuidados:**

- Lat/lon crudos en latitudes medias: un grado de longitud **no** equivale a la misma distancia en metros que un grado de latitud.
- Elegir $\varepsilon$ en **metros** tras proyección o con distancia Haversine si la librería lo soporta.

---

### 9. Validación espacial (bloques)

- **K-fold por bloques geográficos:** polígonos o teselas; todo lo que cae en una tesela va al mismo fold.
- **Holdout por región:** entrenar en ciudades A,B; test en ciudad C.

Esto aproxima el riesgo de **generalización geográfica** mejor que un split aleatorio por filas.

---

### 10. Python: ecosistema

| Librería | Uso |
|----------|-----|
| `networkx` | Construcción, métricas, algoritmos clásicos |
| `igraph` / `graph-tool` | Rendimiento en grafos grandes (opcional) |
| `geopandas` | Geometrías, CRS, operaciones espaciales |
| `scikit-learn` | Modelos sobre tablas de features derivadas |

---

### 11. Laboratorio (según sílabo)

- **NTB 1 —** Grafo usuarios–ítems → features → modelo supervisado.  
- **NTB 2 —** Coordenadas, DBSCAN espacial, zonas densas.

---

### 12. Plantilla `networkx` + tabla de features

```python
import networkx as nx
import pandas as pd

G = nx.Graph()
# G.add_weighted_edges_from([(u, v, w), ...])

rows = []
for n in G.nodes():
    rows.append({
        "node": n,
        "degree": G.degree[n],
        "clustering": nx.clustering(G, n),
    })
feat = pd.DataFrame(rows)
```

Para **PageRank** (dirigido o no según definición):

```python
pr = nx.pagerank(G, alpha=0.85)
feat["pagerank"] = feat["node"].map(pr)
```

---

### 13. Grafo usuarios–ítem como bipartito

Las aristas conectan $u \in U$ con $i \in I$. Features útiles:

- Grado de usuario = número de ítems consumidos.
- Grado de ítem = popularidad.
- Proyecciones: vecinos de usuario son ítems; “2 saltos” conectan usuarios que comparten ítems (útil para link prediction entre usuarios o entre ítems por **co-compra**).

---

### 14. Profundización: complejidad y muestreo

Calcular betweenness exacto es $O(|V||E|)$ en algoritmo de Brandes — prohibitivo en millones de nodos. En la práctica industrial se usan **muestreos**, **aproximaciones**, o se evita esa feature en favor de **PageRank aproximado** o **embeddings**.

---

### 15. Profundización: features de caminos sin almacenar todo

Para pares candidatos en link prediction, no hace falta el diámetro global: a menudo basta **common neighbors** y **Adamic/Adar**, computables desde listas de adyacencia.

---

### 16. DBSCAN en lat/lon: patrón práctico

1. Convertir a GeoDataFrame con CRS WGS84.  
2. Reproyectar a CRS métrico local (metros).  
3. Ejecutar `DBSCAN(eps=500, min_samples=10)` si `eps` son 500 metros (ejemplo ilustrativo).

```python
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN

gdf = gdf.to_crs(epsg=XXXX)  # CRS métrico apropiado
coords = np.stack([gdf.geometry.x, gdf.geometry.y], axis=1)
labels = DBSCAN(eps=500, min_samples=10).fit_predict(coords)
gdf["cluster"] = labels
```

---

### 17. Errores frecuentes

| Error | Consecuencia |
|-------|--------------|
| Usar Euclides en lat/lon globales | Distorsión de clusters y vecinos |
| Ignorar autocorrelación en validación | Test leakage espacial implícito |
| Confundir grafo dirigido y no dirigido | Features de grado y vecindad incorrectas |
| Link prediction sin negativos representativos | Clasificador trivial que aprende “popularidad” |

---

### 18. Casos de uso

| Dominio | Grafo / espacio | Features típicas |
|---------|-----------------|------------------|
| Fraude | transacciones entre cuentas | grado, PageRank, vecindad |
| Retail | co-compra | common neighbors entre ítems |
| Movilidad | trayectos entre celdas | densidad DBSCAN, flujo entre zonas |
| Epidemiología | contactos | k-core, centralidad (con cautela ética) |

---

### 19. Ética y privacidad (nota breve)

Los grafos sociales pueden **re-identificar** individuos; las ubicaciones precisas son datos sensibles. Anonimización y agregación son obligatorias antes de compartir datos en ejercicios realistas.

---

### 20. Checklist de proyecto grafo → tabla

1. ¿Definición explícita de nodos y aristas?  
2. ¿Grafo dirigido/ponderado/multipartito?  
3. ¿Features escalables o necesidad de muestreo?  
4. ¿Validación estándar o **espacial/temporal**?  
5. ¿Baseline simple (grado, popularidad) antes del modelo complejo?  

---

### 21. GNN: paso de mensajes (intuición sin implementación)

En una capa de GNN, el embedding del nodo $u$ se actualiza **agregando mensajes** de vecinos $v \in \mathcal{N}(u)$:

$$\mathbf{h}_u^{(l+1)} = \sigma\left( \mathbf{W}^{(l)} \mathbf{h}_u^{(l)} + \sum_{v \in \mathcal{N}(u)} \mathbf{M}^{(l)} \mathbf{h}_v^{(l)} \right)$$

con $\sigma$ no lineal, matrices aprendibles y variantes (GCN normaliza por grado, GraphSAGE muestrea vecinos). Tras $L$ capas, cada nodo incorpora información de vecindad a distancia $L$ (*receptive field*). El entrenamiento suele ser **semi-supervisado** (etiquetas en un subconjunto de nodos) o **supervisado** sobre grafos completos.

**Cuándo saltar a GNN:** cuando las features manuales dejan de capturar la señal y hay **GPU**, **framework** especializado y datos limpios de topología.

---

### 22. Validación espacial: *blocking* por teselas

Un patrón reproducible:

1. Particionar el espacio en **teselas** (cuadrícula, hexágonos H3, comunas administrativas).  
2. Asignar cada tesela a un fold de forma que **toda** una tesela quede solo en train o solo en test.  
3. Entrenar modelos y reportar error **por fold** y **agregado**.

Si el error en teselas “lejanas” al train es mucho mayor, el modelo **no generaliza geográficamente** — información crítica para despliegue regional.

---

### 23. Datos puntuales vs poligonales

- **Puntos** (POI, eventos): features de distancia a centroides, densidad local, cluster ID.  
- **Polígonos** (barrios, parcelas): agregaciones de población, uso de suelo, contigüidad como grafo de vecinos espaciales.

`geopandas` permite *spatial join* punto-en-polígono para enriquecer la tabla antes del ML clásico.

---

### 24. Kriging y modelos geoestadísticos (límite del temario)

**Kriging** predice en ubicaciones no muestreadas asumiendo **estacionariedad espacial** de la covarianza. Es potente pero exige modelado explícito y supuestos. En este curso se menciona como **contraste**: cuando la interpolación espacial es el objetivo, el ML tabular “punto a punto” puede ser subóptimo frente a geoestadística clásica.

---

### 25. Construcción de aristas desde datos tabulares

A veces no hay grafo explícito: se puede definir arista entre dos entidades si:

- comparten **≥ k** eventos en ventana temporal,  
- tienen similitud coseno **> umbral** en vectores de co-ocurrencia,  
- están a **distancia < ε** en espacio geográfico.

La elección del umbral altera **densidad** del grafo y por tanto todas las métricas derivadas — documentar siempre la regla.

---

### 26. Matriz de pesos espaciales $\mathbf{W}$

En econometría espacial, $\mathbf{W}$ codifica vecindad: $w_{ij}=1$ si $i$ y $j$ son contiguos, o $w_{ij} \propto 1/d_{ij}$. Modelos como **SAR** o **SEM** extienden regresión con términos de rezago espacial — útiles cuando el ML tabular ignora estructura y el error muestra patrón geográfico. Para el curso, basta saber que **incorporar explícitamente** la vecindad puede ser alternativa a solo añadir features de distancia.

---

### 27. LISA y hotspots locales

**LISA** (*Local Indicators of Spatial Association*, Anselin) descompone Moran en contribuciones por región, clasificando cuadrantes tipo **HH** (alto rodeado de alto), **LL**, **HL**, **LH**. Es complementario a DBSCAN: LISA resalta **autocorrelación estadística** local; DBSCAN resalta **densidad** geométrica de puntos.

---

### 28. Comunidades (*Louvain* / modularidad — referencia)

Para grafos grandes, detectar comunidades optimizando **modularidad** produce clusters útiles como **features categóricas** (“comunidad 3”) en tabular. `networkx` expone algoritmos de detección de comunidades en versiones recientes; la literatura clásica incluye **Louvain** y **Leiden**.

---

## Referencias bibliográficas principales

1. Easley, D., & Kleinberg, J. (2010). *Networks, Crowds, and Markets*. Cambridge University Press.  
2. Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). *Mining of Massive Datasets* (3rd ed.). Cambridge University Press.  
3. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online learning of social representations. *KDD*.  
4. Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. *KDD*.  
5. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.  
6. Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis*, 27(2), 93–115.  
