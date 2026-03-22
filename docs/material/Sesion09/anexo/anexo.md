
## Anexo
# Fundamento Matemático y Computacional del Aprendizaje No Supervisado (PCA y Clustering)
#### Autor: Carlos César Sánchez Coronel

*(Alineado con la Sesión 9: PCA, k-means, DBSCAN, clustering jerárquico y métricas.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

**Sin etiquetas:** datos $\{\mathbf{x}_i\}_{i=1}^n \subset \mathbb{R}^p$.

- **PCA:** encontrar proyección lineal $\mathbf{V}_k \in \mathbb{R}^{p \times k}$ que maximice varianza retenida.
- **k-means:** particionar en $K$ grupos minimizando inercia intra-cluster.
- **DBSCAN:** agrupar por densidad con parámetros $\varepsilon$, `minPts`.

### 1.2 Notación

- $\tilde{\mathbf{X}}$: matriz centrada $n \times p$.
- $\boldsymbol{\mu}_k$: centroide del cluster $k$.
- $C_k$: conjunto de índices asignados a $k$.

### 1.3 Supuestos

- PCA: linealidad; k-means: clusters esféricos de tamaño similar (idealizado); DBSCAN: densidad homogénea por cluster.

---

## 2. Fundamento Matemático

### 2.1 PCA como maximización de varianza

Centrar columnas. Buscar $\mathbf{v}_1$ con $\|\mathbf{v}_1\|=1$:

$$
\boxed{\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^\top \mathbf{S} \mathbf{v}}
$$

$\mathbf{S} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}$: covarianza muestral.

**Solución:** $\mathbf{v}_1$ es autovector de $\mathbf{S}$ asociado al mayor autovalor $\lambda_1$.

**Proyección:** $\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{V}_k$, varianza total explicada $\sum_{j=1}^k \lambda_j / \sum_{\ell=1}^p \lambda_\ell$.

### 2.2 k-means — función objetivo

$$
\boxed{J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2, \quad \boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i}
$$

**Algoritmo Lloyd:** alternar asignación (mínima distancia a centroides) y actualización de $\boldsymbol{\mu}_k$. Descenso en $J$ monótono; mínimo local.

### 2.3 DBSCAN

- **Vecindad:** $N_\varepsilon(\mathbf{x}) = \{\mathbf{x}' : d(\mathbf{x},\mathbf{x}') \le \varepsilon\}$.
- **Punto núcleo:** $|N_\varepsilon(\mathbf{x})| \ge \text{minPts}$.
- Expansión de clusters desde núcleos; puntos no alcanzables → ruido.

### 2.4 Clustering jerárquico aglomerativo

Matriz de enlaces: **single** (mín inter-cluster), **complete** (máx), **average**, **Ward** (minimizar incremento de suma de cuadrados intra-cluster).

### 2.5 Optimización

- k-means: no convexo global; múltiples reinicios (K-means++ mejora inicialización).
- PCA: problema de autovalores, convexo en $\mathbf{v}$ unitario (esfera).

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo k-means

```
Inicializar centroides μ_1…μ_K
Repetir hasta convergencia:
  Asignar cada x_i al μ_k más cercano
  Recalcular μ_k como media de su cluster
```

- **Tiempo típico:** $O(n \cdot K \cdot p \cdot \text{iter})$.
- **Espacio:** $O(n p + K p)$.

### 3.2 PCA vía SVD

```python
import numpy as np

def pca_fit_transform(X, k):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :k] * s[:k]
    components = Vt[:k].T
    return Z, components, (s[:k] ** 2) / (X.shape[0] - 1)
```

### 3.3 Escalamiento

- **Grandes $n,p$:** randomized SVD; mini-batch k-means.
- **Alta dimensión:** PCA previo o métricas alternativas en DBSCAN.

---

## 4. Métricas de Evaluación Específicas

**Silueta** para punto $i$:

$$
\boxed{s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}}
$$

$a(i)$: distancia media intra-cluster; $b(i)$: distancia media al cluster vecino más cercano.

**Davies–Bouldin:** menor es mejor (compactez/separación).

**Inercia:** valor de $J$ en k-means (solo comparativo con mismo $K$ y escala).

---

## 5. Descomposición Teórica

PCA minimiza error de reconstrucción en norma Frobenius entre $\tilde{\mathbf{X}}$ y proyección de rango $k$ (teorema de Eckart–Young para SVD truncada).

---

## 6. Selección de Hiperparámetros

- $K$ en k-means: codo, silueta, conocimiento de dominio.
- DBSCAN: gráfico k-distancia para $\varepsilon$; `minPts` ≈ dimensión+1 en la práctica.

---

## 7. Ecuaciones Clave (resumen)

| Método | Ecuación |
|--------|----------|
| PCA | $\mathbf{S}\mathbf{v} = \lambda\mathbf{v}$, $\mathbf{Z}=\tilde{\mathbf{X}}\mathbf{V}_k$ |
| k-means | $J = \sum_k \sum_{i\in C_k}\|\mathbf{x}_i-\boldsymbol{\mu}_k\|^2$ |
| Silueta | $s(i) = (b-a)/\max(a,b)$ |

---

## 8. Referencias y Lecturas Complementarias

- MacQueen — k-means (1967).
- Ester et al. — DBSCAN (1996).
- Jolliffe — *Principal Component Analysis*.
