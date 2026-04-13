---
layout: default
---
# Sesión 12: Graph ML y Geodata


### 1. Logro de la sesión

Extraer **features tabulares desde grafos** y desde **datos geoespaciales**, entrenar modelos supervisados o clustering, y entender **limitaciones** de escala y ruido geográfico.

---

### 2. Grafos: definiciones

**Grafo** $G=(V,E)$: nodos $V$, aristas $E$ (dirigidas o no, ponderadas). **Matriz de adyacencia** $\mathbf{A}$ con $A_{ij}$ peso o presencia.

---

### 3. Features de nodos (sin GNN profundo)

- **Grado** entrante/saliente.  
- **Centralidad** (grado, betweenness aproximada, eigenvector — concepto).  
- **Conteos** de vecinos a 2 saltos, agregados de atributos vecinos.

Estas features alimentan **Random Forest**, **logística**, etc.

---

### 4. Link prediction (idea)

Predecir aristas faltantes como problema binario sobre pares $(u,v)$ con features de caminos cortos, preferential attachment, Adamic/Adar, etc.

---

### 5. Geodata

**Distancia Haversine** entre lat/lon en esfera:

$$ d = 2r \arcsin\left(\sqrt{\sin^2\frac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\frac{\Delta\lambda}{2}}\right) $$

**DBSCAN** en $(x,y)$ proyectado localmente para *hotspots*.

**Problemas:** autocorrelación espacial, unidades, elección de proyección para distancias largas.

---

### 6. Python

- `networkx` para grafos y métricas.  
- `geopandas` si hay geometrías; `sklearn` para modelos tabulares resultantes.

---

### 7. Laboratorio (según sílabo)

- **NTB 1 —** Grafo usuarios–ítems → features → modelo supervisado.  
- **NTB 2 —** Coordenadas, DBSCAN espacial, zonas densas.



### 8. Features de grafo avanzadas (nivel introductorio)

- **Clustering coefficient**, **PageRank** aproximado (centralidad de importancia).  
- **Embeddings** tipo DeepWalk/Node2vec: random walks → Word2Vec — útiles cuando hay millones de nodos (fuera del alcance detallado de esta sesión, pero referencia para lectura).

### 9. Autocorrelación espacial (idea)

Datos geográficos suelen violar independencia: puntos cercanos se parecen (**Tobler**). Modelos espaciales completos (kriging) están fuera del temario; en ML tabular basta reconocer el sesgo y usar validación espacial (bloques por región).

### 10. Plantilla `networkx` + tabla de features

```python
import networkx as nx
import pandas as pd

G = nx.Graph()
# G.add_edges_from([...])
rows = []
for n in G.nodes():
    rows.append({"node": n, "degree": G.degree[n]})
feat = pd.DataFrame(rows)
```

### 11. DBSCAN en lat/lon

Escalar lat/lon con cuidado: en latitudes medias, longitud se “comprime”. Usar proyección métrica local o Haversine como métrica custom si la librería lo permite.


---

## Referencias bibliográficas principales

1. Easley, D., & Kleinberg, J. (2010). *Networks, Crowds, and Markets*. Cambridge University Press.  
2. Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). *Mining of Massive Datasets* (3rd ed.). Cambridge UP.  
