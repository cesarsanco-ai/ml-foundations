

# Cheatsheet: Sistemas de Recomendación
**Autor:** Carlos César Sánchez Coronel  

---

## Paradigmas

| Tipo | Idea | Cold start |
| :--- | :--- | :--- |
| **Popularidad** | Top global | Usuario nuevo OK; aburrido |
| **Contenido** | Similar a lo que ya gustó | Ítem nuevo OK si hay features |
| **Colaborativo** | Usuarios/ítems similares | Usuario/ítem nuevos difícil |
| **Híbrido** | Combina lo anterior | Más robusto |

---

## Colaborativo

* **User-based:** vecinos por correlación/Pearson o coseno.  
* **Item-based:** similitud entre ítems; más estable, precomputable.  

---

## Factorización (Funk SVD)

* $\hat{r}_{u,i} = \mathbf{p}_u \cdot \mathbf{q}_i$ aprendido por SGD solo en ratings observados.  
* Con sesgos: $\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i$.  

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)
algo = SVD(n_factors=20, reg_all=0.1, n_epochs=20)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

---

## Métricas

| Objetivo | Métricas |
| :--- | :--- |
| Predecir rating | **RMSE**, **MAE** |
| Ranking | **Precision@k**, **Recall@k**, **MAP**, **NDCG@k** |
| Catálogo | **Coverage** (diversidad) |

```python
def precision_at_k(rec, rel, k):
    rec = rec[:k]
    return len(set(rec) & set(rel)) / k
```

---

## Cold start

* **Usuario nuevo:** onboarding, popular, demografía.  
* **Ítem nuevo:** contenido, metadatos.  
* **Sistema nuevo:** reglas + transfer learning / datos externos.  

---

## Escalabilidad

* Item-based + precomputo; factorización con SGD; hashing LSH para vecinos.  

---

## Puntos críticos

* Matriz **muy sparse** → regularización y más datos implícitos.  
* **Popularity bias:** métricas pueden favorecer lo mainstream; vigilar diversidad.  
* Separar train/test **por usuario o tiempo** según escenario real.  

> *“Netflix no solo predice estrellas; ordena lo que verás después.”*
