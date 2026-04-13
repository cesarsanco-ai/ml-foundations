---
layout: default
---
# Sesión 11: Sistemas de recomendación


### 1. Logro de la sesión

Diseñar **sistemas de recomendación** con **baselines**, **filtrado colaborativo** (memoria y modelos) y **contenido**, midiendo **Precision@K**, **Recall@K** y métricas de ranking (MAP, NDCG) con buenas prácticas de evaluación.

---

### 2. Historia breve

| Año | Hito |
|-----|------|
| **1990s** | Amazon, Netflix popularizan personalización |
| **2006** | Netflix Prize impulsa factorización matricial |
| **2010s** | Deep learning + embeddings (word2vec, two-tower) en producción |

---

### 3. Baselines y contenido

**Popularidad (top-N):** ítems más vendidos/vistos — benchmark obligatorio.

**Basado en contenido:** perfil de usuario = agregación de vectores de ítems (TF-IDF, atributos). Recomendar ítems similares por **coseno**.

**Límites:** *cold start* de ítems nuevos con metadatos ricos; puede faltar serendipidad.

---

### 4. Filtrado colaborativo basado en memoria

Matriz usuario–ítem $R$ sparse. Predicción para usuario $u$, ítem $i$:

$$ \hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} s(u,v)\,(r_{vi}-\bar{r}_v)}{\sum_{v} |s(u,v)|} $$

con similitud $s$ coseno o Pearson.

**Item-based** a menudo más estable que user-based en producción (menos drift de usuarios).

---

### 5. Factorización de matrices / SVD truncado

Minimizar:

$$ \sum_{(u,i)\in \Omega} (r_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2 + \lambda(\|\mathbf{p}\|^2 + \|\mathbf{q}\|^2) $$

Latent factors $\mathbf{p}_u, \mathbf{q}_i \in \mathbb{R}^K$. Generaliza a usuarios/ítems sin co-ocurrencia directa.

**Librería:** `surprise` con `SVD`.

---

### 6. Problemas: cold start, sparsity, escala

| Problema | Mitigación |
|----------|------------|
| Usuario nuevo | contenido, reglas, popularidad |
| Matriz muy vacía | regularización, factores latentes, implicit feedback |
| Escala | sampling negativo, approximate nearest neighbors |

---

### 7. Métricas offline

- **RMSE/MAE** si hay ratings explícitos.  
- **Precision@K / Recall@K** en top-K.  
- **NDCG** si el orden importa y hay grados de relevancia.

**Limitación:** gap con CTR/conversión online — siempre validar A/B cuando sea posible.

---

### 8. Plantilla `surprise` (esquema)

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
svd = SVD(n_factors=50, reg_all=0.02, lr_all=0.005, n_epochs=20)
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
```

---

### 9. Laboratorio (según sílabo)

- **NTB 1 —** Matriz usuario–ítem, popularidad, KNN user/item, Precision@K y Recall@K.  
- **NTB 2 —** SVD / factorización, top-N por usuario, interpretación de errores.



### 10. Híbridos y reglas de negocio

Combinar score colaborativo con **filtros** (no recomendar productos ilegales en cierta región), **diversificación** (no solo ítems casi idénticos) y **exploración** (bandits) es práctica industrial estándar.

### 11. Evaluación con partición temporal

Si el comportamiento de usuarios **evoluciona**, el split aleatorio infla métricas. Usar **último mes** como test o **leave-last-out** por usuario.

### 12. Implicit feedback (mención)

Cuando solo hay **clics/compras** sin rating, se modelan como observaciones binarias o con confianza (Hu et al., *implicit feedback*). `surprise` puede adaptarse con datos binarios y métricas de ranking.

### 13. Código: Precision@K / Recall@K manual (idea)

```python
import numpy as np

def precision_at_k(reco, relevant, k):
    topk = set(reco[:k])
    return len(topk & set(relevant)) / k

def recall_at_k(reco, relevant, k):
    topk = set(reco[:k])
    return len(topk & set(relevant)) / max(1, len(set(relevant)))
```


---

## Referencias bibliográficas principales

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8), 30–37.  
2. Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2015). *Recommender Systems Handbook*. Springer.  
