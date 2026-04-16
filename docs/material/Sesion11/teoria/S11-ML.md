---
layout: default
---
# Sesión 11: Sistemas de recomendación

### 1. Logro de la sesión

Diseñar **sistemas de recomendación** con **baselines sólidos**, **filtrado colaborativo** basado en memoria (usuario–usuario, ítem–ítem) y **modelos de factorización**, combinar con señales de **contenido** cuando proceda, y evaluar con **Precision@K**, **Recall@K**, **MAP**, **NDCG** y **RMSE/MAE** donde haya ratings explícitos. Comprender **cold start**, **sesgo de popularidad**, **feedback implícito** y la brecha entre **métricas offline** y **resultados online** (CTR, conversión).

---

### 2. Historia y línea temporal

| Periodo | Hito |
|---------|------|
| **1990s** | Comercio electrónico (Amazon) y personalización masiva |
| **1994–2000** | GroupLens, filtrado colaborativo con ratings explícitos |
| **2006** | **Netflix Prize**: factorización matricial y regularización a escala |
| **2010s** | **ALS** en feedback implícito; factorización con sesgos; escalado distribuido |
| **2015+** | **Deep learning**: embeddings, *two-tower*, secuencias (RNN/Transformers) en producción |
| **Actualidad** | Pipeline de **retrieval + ranking**; exploración con **bandits**; gobernanza y fairness |

**Lectura:** un sistema real casi nunca es “solo un algoritmo”: incorpora **reglas de negocio**, **diversidad**, **freshness** y **experimentación A/B**.

---

### 3. Formalización: matriz usuario–ítem

Sea un conjunto de usuarios $\mathcal{U}$ e ítems $\mathcal{I}$. La matriz de interacciones $\mathbf{R} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ es en general **dispersa**:

- **Feedback explícito:** ratings $r_{ui} \in \{1,\ldots,5\}$ o estrellas.
- **Feedback implícito:** clics, tiempo de visualización, compras (binarios o conteos con **confianza**).

El objetivo puede ser **predecir** $r_{ui}$ faltantes (completar la matriz) o **rankear** un subconjunto candidato $\mathcal{C}_u$ para el usuario $u$ (top-$K$).

---

### 4. Baselines y métodos basados en contenido

#### 4.1 Popularidad (top-N)

Recomendar los ítems **más consumidos** en la ventana reciente. Es un **benchmark obligatorio**: cualquier modelo sofisticado debe superarlo en métricas de ranking relevantes para el negocio.

#### 4.2 Perfil de usuario por contenido

Cada ítem $i$ tiene vector de atributos $\mathbf{z}_i$ (TF-IDF de texto, categoría one-hot, embeddings de imagen). El perfil del usuario es agregación (media ponderada) de $\mathbf{z}_i$ sobre ítems consumidos. Se recomiendan ítems con **mayor similitud coseno** al perfil.

**Ventajas:** mitiga **cold start de ítems** si hay metadatos ricos.  
**Límites:** puede faltar **serendipidad**; sesgo hacia ítems similares a la burbuja pasada.

---

### 5. Filtrado colaborativo basado en memoria

#### 5.1 Similitud entre usuarios

Métricas habituales sobre vectores de ratings **co-completados** (solo ítems calificados por ambos):

- **Coseno:** $\cos(\mathbf{r}_u, \mathbf{r}_v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\|\|\mathbf{r}_v\|}$.
- **Pearson:** correlación centrada; reduce el efecto de usuarios optimistas vs pesimistas.

#### 5.2 Predicción user-based (k-NN)

Para predecir $r_{ui}$ con vecindario $N_k(u)$ de usuarios similares:

$$ \hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} s(u,v)\,(r_{vi}-\bar{r}_v)}{\sum_{v \in N_k(u)} |s(u,v)|} $$

donde $s(u,v)$ es la similitud y $\bar{r}_u$ la media de ratings del usuario $u$.

#### 5.3 Item-based

Se invierte el rol: se buscan ítems $j$ similares a $i$ según patrones de co-rating por usuarios. En muchos sistemas a escala, **item–item** es más **estable** (los ítems cambian menos que las preferencias volátiles de usuarios) y escala mejor en ciertas arquitecturas de caché.

---

### 6. Factorización de matrices / SVD regularizado

#### 6.1 Idea central

Se buscan embeddings **$\mathbf{p}_u, \mathbf{q}_i \in \mathbb{R}^K$** tales que:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

donde $\mu$ es sesgo global, $b_u$ y $b_i$ sesgos de usuario e ítem. Los términos $\mathbf{p}_u^\top \mathbf{q}_i$ capturan **factores latentes** (género, tono, precio implícito, etc.) no observados directamente.

#### 6.2 Función objetivo (mínimos cuadrados con observaciones en $\Omega$)

$$ \min_{\{\mathbf{p},\mathbf{q},b\}} \sum_{(u,i)\in \Omega} (r_{ui} - \hat{r}_{ui})^2 + \lambda\left(\sum_u \|\mathbf{p}_u\|^2 + \sum_i \|\mathbf{q}_i\|^2 + \cdots\right) $$

Optimización: **SGD**, **ALS** (alternating least squares) según formulación.

#### 6.3 Generalización

La factorización **interpola** a pares $(u,i)$ sin co-ocurrencia directa en $\Omega$ a través de la estructura de bajo rango en factores latentes — siempre que el patrón sea aprendible y la regularización adecuada.

**Librería didáctica:** `surprise` con `SVD`, `SVD++` (incorpora ítems implícitos del historial).

---

### 7. Feedback implícito (mención operativa)

Cuando solo hay **clics/compras**, se define una matriz de preferencias $p_{ui}$ y niveles de **confianza** $c_{ui}$ (Hu et al.): interacciones observadas reciben mayor peso. Los objetivos dejan de ser solo RMSE sobre ratings y pasan a **ranking** (BPR, WARP) o reconstrucción ponderada.

En práctica curricular, basta reconocer que **muestreo negativo** (pares no observados) y **métricas de ranking** son el estándar.

---

### 8. Problemas: cold start, sparsity, escala

| Problema | Descripción | Mitigación |
|----------|-------------|------------|
| **Usuario nuevo** | Sin historial | contenido, onboarding, popularidad, reglas |
| **Ítem nuevo** | Sin interacciones | metadatos, *content-based*, exploración |
| **Matriz muy vacía** | Pocos ratings por usuario | regularización fuerte, factores latentes, implicit models |
| **Escala** | Millones de ítems | ANN sobre embeddings, dos etapas retrieval+ranking |
| **Sesgo de popularidad** | Modelo solo recomienda *hits* | re-ranking con diversidad, métricas beyond-accuracy |

---

### 9. Métricas offline

#### 9.1 RMSE y MAE

Apropiadas con **ratings explícitos** y objetivo de predicción puntual. No capturan bien la calidad del **top-K** si el negocio es ranking.

#### 9.2 Precision@K y Recall@K

Sea $\mathrm{Rel}_u$ el conjunto de ítems relevantes para $u$ (p. ej. consumidos en test) y $\mathrm{Reco}_u(K)$ la lista recomendada de tamaño $K$.

$$\mathrm{Precision@K} = \frac{|\mathrm{Reco}_u(K) \cap \mathrm{Rel}_u|}{K}, \qquad
\mathrm{Recall@K} = \frac{|\mathrm{Reco}_u(K) \cap \mathrm{Rel}_u|}{|\mathrm{Rel}_u|}$$

#### 9.3 Mean Average Precision (MAP)

Promedia **AP** (*average precision*) sobre usuarios. Para un ranking, AP pondera posiciones: aciertos arriba valen más.

#### 9.4 NDCG

Cuando hay **grados de relevancia** (no solo binario), **DCG** acumula ganancia con descuento por posición; **NDCG** normaliza respecto al orden ideal.

**Limitación:** las métricas offline **no** miden efectos de **posición en UI**, **fatiga** o **sesgo de exposición** — por eso el **A/B test** sigue siendo referencia.

---

### 10. Partición de datos y *leakage*

- **Split aleatorio** de entradas $(u,i)$ puede **inflar** métricas si del mismo usuario quedan interacciones futuras en train y “similares” en test sin orden temporal.
- **Leave-last-out** por usuario: para cada $u$, el último ítem consumido va a test; entrenar sin él.
- **Partición temporal** cuando el dominio evoluciona (noticias, moda).

---

### 11. Plantilla `surprise` (esquema)

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
svd = SVD(n_factors=50, reg_all=0.02, lr_all=0.005, n_epochs=20, random_state=42)
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
```

---

### 12. Laboratorio (según sílabo)

- **NTB 1 —** Matriz usuario–ítem, popularidad, KNN user/item, Precision@K y Recall@K.  
- **NTB 2 —** SVD / factorización, top-N por usuario, interpretación de errores.

---

### 13. Híbridos y reglas de negocio en producción

En sistemas reales se combina un **score** colaborativo o de ML con:

- **Filtros duros:** edad, jurisdicción, stock, compatibilidad.
- **Diversificación:** MMR (*Maximal Marginal Relevance*) para no mostrar diez variantes casi idénticas.
- **Exploración:** $\varepsilon$-greedy o **contextual bandits** para ítems nuevos (*cold start* activo).
- **Re-ranking** por objetivos múltiples (CTR, margen, frescura).

---

### 14. Precision@K / Recall@K manual (referencia)

```python
def precision_at_k(reco, relevant, k):
    topk = set(reco[:k])
    rel = set(relevant)
    return len(topk & rel) / k if k else 0.0

def recall_at_k(reco, relevant, k):
    topk = set(reco[:k])
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(topk & rel) / len(rel)
```

---

### 15. Profundización: por qué la factorización es bajo rango

Si existen $K$ factores latentes que gobiernan preferencias, la matriz completa (sin ruido) tendría rango $\le K$. En la vida real, $K$ es un **hiperparámetro**: demasiado bajo → **subajuste**; demasiado alto → **sobreajuste** especialmente con datos sparse. Use validación cruzada o holdout por usuario.

---

### 16. Profundización: sesgo de popularidad y fairness

Los modelos colaborativos tienden a **reforzar** ítems ya populares porque tienen más señal. Esto puede perjudicar a **creadores** o **ítems de nicho**. Métricas complementarias: **coverage** del catálogo, **diversidad**, **exposición** por grupo — temas activos en investigación y regulación.

---

### 17. Dos etapas: *retrieval* y *ranking* (visión arquitectónica)

1. **Retrieval:** recuperar cientos de candidatos baratos (similitud coseno en embeddings, ANN, inverted indexes por categoría).  
2. **Ranking:** modelo más pesado (red neuronal, GBDT sobre features cruzados usuario–ítem) reordena la lista.

Esta separación es la norma en **YouTube**, **Pinterest**, **recomendación retail** a escala.

---

### 18. Evaluación con muestreo negativo (idea)

Para entrenar modelos de clasificación *click vs no-click*, se muestrean ítems no vistos como negativos. El muestreo debe ser **representativo** (popularity-biased vs uniform) o el modelo aprende artefactos. En laboratorio, documentar la política de muestreo.

---

### 19. Errores frecuentes

| Error | Consecuencia |
|-------|--------------|
| No incluir baseline de popularidad | “Mejora” ilusoria |
| Split que filtra usuarios distintos en train/test sin cuidado | Data leakage o test trivial |
| Optimizar solo RMSE cuando el producto es top-K | Desalineación con KPI |
| Ignorar que “no visto” ≠ “no relevante” | Negativos ruidosos en implícito |

---

### 20. Checklist de diseño de un experimento offline

1. ¿Definición clara de **relevancia** (click, compra, rating ≥ 4)?  
2. ¿Protocolo de split **por usuario** o **temporal**?  
3. ¿K alineado con el tamaño de la interfaz (5, 10, 50)?  
4. ¿Baselines: popularidad, item-item, SVD?  
5. ¿Métricas complementarias: coverage, diversidad?  

---

### 21. NDCG: construcción paso a paso

Para un usuario, sea $\mathrm{rel}(i)$ la relevancia del ítem en posición $i$ del ranking (p. ej. 0/1 o escala 0–4). El **DCG** (*Discounted Cumulative Gain*) acumula relevancia con un descuento logarítmico en la posición:

$$\mathrm{DCG@K} = \sum_{i=1}^{K} \frac{2^{\mathrm{rel}(i)} - 1}{\log_2(i+1)}$$

(Variante común; existen definiciones equivalentes con $\mathrm{rel}(i)$ sin exponencial.) El **IDCG@K** es el DCG del **orden ideal** (relevancias ordenadas de mayor a menor). Entonces:

$$\mathrm{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}$$

Valores en $[0,1]$ facilitan comparar usuarios con distinto número de ítems relevantes. **NDCG** es sensible al **orden** dentro del top-K, a diferencia de Precision@K puramente conjuntista.

---

### 22. Learning to rank (visión de conjunto)

Más allá de factorizar $\mathbf{R}$, los sistemas grandes entrenan modelos que **ordenan** listas candidatas con funciones como:

- **Pointwise:** predecir $r_{ui}$ y ordenar por score (equivalente a regresión/clasificación).  
- **Pairwise:** comparar pares $(i,j)$ para usuario $u$ (RankSVM, BPR).  
- **Listwise:** optimizar directamente una métrica suavizada de ranking (LambdaMART, etc.).

En cursos introductorios basta reconocer que la **función de pérdida** debe alinearse con el **uso del ranking** en producto, no solo con RMSE.

---

### 23. *Two-tower* y recuperación aproximada

Un esquema industrial típico aprende embeddings $\mathbf{u}_u$ y $\mathbf{v}_i$ con redes separadas (una torre por usuario con su historial agregado, otra por ítem con metadatos). El score es $\mathbf{u}_u^\top \mathbf{v}_i$ o distancia coseno. Así, la fase de **retrieval** reduce a **búsqueda de vecinos más cercanos** (ANN) en espacio latente — escalable a catálogos enormes.

---

### 24. Fairness en recomendación (introducción)

Más allá de la precisión media, puede medirse:

- **Paridad de exposición** entre grupos de ítems (p. ej. proveedores, géneros).  
- **Equidad de oportunidad** en clicks condicionados a relevancia verdadera (cuando existe etiqueta).  

Las definiciones matemáticas chocan a menudo entre sí (**imposibilidades** tipo Kleinberg et al. en ciertos marcos): por eso el producto debe **priorizar** qué noción de equidad es aceptable legal y culturalmente.

---

### 25. Depuración cuando las métricas no mejoran

| Síntoma | Hipótesis a investigar |
|---------|-------------------------|
| RMSE baja pero NDCG plano | El modelo ajusta magnitudes pero no orden relativo |
| Train excelente, test pobre | Filtrado de usuarios/ítems poco frecuentes; sobreajuste |
| Offline sube, A/B plano | Sesgo de exposición; métrica offline no proxy de negocio |

---

### 26. ALS para factorización (idea algebraica)

En mínimos cuadrados alternados, se fijan temporalmente todos los $\mathbf{q}_i$ y se resuelve un problema **cuadrático convexo** para los $\mathbf{p}_u$, luego se fijan los $\mathbf{p}_u$ y se actualizan los $\mathbf{q}_i$. Iterar hasta convergencia. Con regularización L2, cada subproblema tiene solución cerrada tipo **ridge regression**. Esta vista conecta la factorización con **regresión** clásica y facilita la extensión a **implicit ALS** con pesos de confianza.

---

### 27. *Closed-loop* en logs de recomendación

Si solo se registra feedback sobre ítems **mostrados**, el modelo nunca observa utilidad de alternativas no expuestas. Técnicas como **IPS** (*inverse propensity scoring*) ponderan ejemplos observados por la inversa de la probabilidad de exposición — frágil si las propensidades están mal estimadas. Por eso la **exploración** y los experimentos aleatorizados son parte del diseño del sistema, no un “extra”.

---

## Referencias bibliográficas principales

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8), 30–37.  
2. Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2015). *Recommender Systems Handbook*. Springer.  
3. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*.  
4. Rendle, S., et al. (2009). BPR: Bayesian personalized ranking from implicit feedback. *UAI*.  
5. Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for YouTube recommendations. *RecSys*.  
