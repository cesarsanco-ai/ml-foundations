
## Anexo
# Fundamento Matemático y Computacional de la Evaluación y Validación de Modelos
#### Autor: Carlos César Sánchez Coronel

*(Alineado con la Semana 8: sesgo-varianza, splits, validación cruzada, tuning y leakage.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Sea $\mathcal{A}$ un algoritmo de aprendizaje que, dado un conjunto de entrenamiento $\mathcal{D}_{\text{train}}$, produce un modelo $\hat{f}$. El **error de generalización** es:

$$
R(\hat{f}) = \mathbb{E}_{(\mathbf{x},y)\sim P}\big[ \ell(\hat{f}(\mathbf{x}), y) \big]
$$

Solo podemos **estimar** $R$ con datos no usados en el ajuste indebido, o con CV bien anidada.

### 1.2 Notación

- $\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{test}}$ (disjuntos).
- $k$-fold: particiones $\mathcal{D}_1,\ldots,\mathcal{D}_k$.

### 1.3 Supuestos

- Muestras i.i.d. para CV estándar (violación en series temporales → CV temporal).

---

## 2. Fundamento Matemático

### 2.1 Descomposición sesgo–varianza (regresión cuadrática)

Para verdadero $f^*(x) = \mathbb{E}[y\mid x]$:

$$
\boxed{\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f^*(x)\big)^2}_{\text{sesgo}^2} + \underbrace{\mathbb{V}(\hat{f}(x))}_{\text{varianza}} + \sigma^2(x)}
$$

### 2.2 Estimador hold-out

Con test de tamaño $n_{\text{test}}$:

$$
\hat{R} = \frac{1}{n_{\text{test}}} \sum_{(\mathbf{x}_i,y_i) \in \mathcal{D}_{\text{test}}} \ell(\hat{f}(\mathbf{x}_i), y_i)
$$

Es insesgado de $R$ **si** $\hat{f}$ no depende de test (condicionalmente a train).

### 2.3 $k$-fold CV

$$
\boxed{\text{CV}_k = \frac{1}{k}\sum_{j=1}^k \frac{1}{|\mathcal{D}_j|} \sum_{i \in \mathcal{D}_j} \ell(\hat{f}^{(-j)}(\mathbf{x}_i), y_i)}
$$

donde $\hat{f}^{(-j)}$ se entrena sin fold $j$.

**Varianza:** típicamente menor que un solo split para $n$ moderado; correlación entre folds aumenta varianza del estimador CV.

### 2.4 Bootstrap .632 (idea)

Combinar error en-bag y out-of-bag para corregir sesgo optimista/pesimista; útil en bosques.

### 2.5 Optimización de hiperparámetros

Sea $\lambda \in \Lambda$ hiperparámetros. **Selección:**

$$
\hat{\lambda} = \arg\min_{\lambda \in \Lambda} \widehat{\text{CV}}(\lambda)
$$

**Riesgo:** múltiples comparaciones → optimismo; conjunto de test final **una sola vez**.

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo k-fold estratificado (clasificación)

```
Partir D_train en k estratos por clase
Para j = 1 … k:
  Entrenar en D_train \ D_j
  Evaluar en D_j
Promediar métricas
```

- **Coste:** $k$ veces el entrenamiento.

### 3.2 Nested CV (sesgo reducido en estimación de performance)

```
Para cada fold externo:
  Para cada λ en grid (en train externo):
    CV interna en train externo
  Elegir λ*
  Entrenar con λ* en train externo completo
  Evaluar en test externo
```

### 3.3 Numpy: índices k-fold

```python
import numpy as np

def kfold_indices(n, k, rng=None):
    rng = np.random.default_rng(rng)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for j in range(k):
        val_idx = folds[j]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != j])
        yield train_idx, val_idx
```

---

## 4. Métricas de Evaluación Específicas

- **Clasificación:** según coste de FP/FN; PR-AUC en desbalance.
- **Regresión:** RMSE, MAE.
- **Estabilidad:** desviación de métrica entre folds.

---

## 5. Descomposición Teórica

Complejidad del modelo ↑ → sesgo ↓, varianza ↑. Validación estima el punto operativo en datos nuevos.

---

## 6. Selección de Hiperparámetros

- Grid search, random search, Bayesian optimization.
- **Curvas de aprendizaje:** error train vs tamaño de muestra para diagnosticar sesgo/varianza.

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Fórmula |
|----------|---------|
| Sesgo–varianza | $\mathbb{E}[(y-\hat{f})^2] = \text{bias}^2 + \text{var} + \sigma^2$ |
| CV-$k$ | $\frac{1}{k}\sum_j L_j$ |
| Test hold-out | $\frac{1}{n_{\text{test}}}\sum_{i \in \text{test}} \ell_i$ |

---

## 8. Referencias y Lecturas Complementarias

- Kohavi — *A Study of Cross-Validation and Bootstrap* (1995).
- Varma & Simon — bias in CV error estimation with feature selection.
- Bergstra & Bengio — Random Search for Hyper-Parameter Optimization.
