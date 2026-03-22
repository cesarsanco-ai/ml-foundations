---
layout: default
---

# Fundamento Matemático y Computacional de k-NN, Naive Bayes y SVM
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-05](../../../sesiones/sesion-05.md)

*(Alineado con la Semana 5: clasificación con algoritmos clásicos.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Dado $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ con $y_i \in \{1,\ldots,K\}$, se busca un clasificador $\hat{y} = h(\mathbf{x})$.

- **k-NN:** $h(\mathbf{x}) = \text{voto mayoritario entre los } k \text{ vecinos más cercanos}$.
- **Naive Bayes:** $h(\mathbf{x}) = \arg\max_k P(C_k \mid \mathbf{x})$ bajo supuesto de independencia condicional.
- **SVM:** $h(\mathbf{x}) = \text{sign}(\mathbf{w}^\top\phi(\mathbf{x}) + b)$ con $\phi$ posible kernel implícito.

### 1.2 Notación

- $d(\mathbf{x}, \mathbf{x}')$: distancia (euclídea, Manhattan, Minkowski).
- **SVM:** $\mathbf{w} \in \mathbb{R}^d$, $b \in \mathbb{R}$; en dual, multiplicadores $\alpha_i$.

### 1.3 Supuestos

- **k-NN:** espacio métrico; escalado de features crítico.
- **Naive Bayes:** $P(\mathbf{x}\mid C_k) = \prod_j P(x_j \mid C_k)$.
- **SVM:** clases separables o casi separables con margen suave.

---

## 2. Fundamento Matemático

### 2.1 k-NN — distancias

**Minkowski ($L_p$):**

$$
\boxed{d_p(\mathbf{x},\mathbf{x}') = \left(\sum_{j=1}^p |x_j - x'_j|^p\right)^{1/p}}
$$

$p=2$: euclídea; $p=1$: Manhattan.

**Decisión:** $\hat{y}(\mathbf{x}) = \arg\max_c \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}[y_i = c]$.

### 2.2 Naive Bayes

**Teorema de Bayes:**

$$
P(C_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C_k) P(C_k)}{P(\mathbf{x})}
$$

**Naive factorization:**

$$
\boxed{P(\mathbf{x} \mid C_k) = \prod_{j=1}^p P(x_j \mid C_k)}
$$

**Gaussian NB** (continuas):

$$
P(x_j \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{jk}^2}} \exp\left(-\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}\right)
$$

**Decisión MAP (log para estabilidad):**

$$
\hat{y} = \arg\max_k \left[ \log P(C_k) + \sum_{j=1}^p \log P(x_j \mid C_k) \right]
$$

### 2.3 SVM — margen máximo (lineal, separable)

Primal:

$$
\boxed{\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.a. } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1}
$$

**Dual:**

$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j
$$

s.a. $\sum_i \alpha_i y_i = 0$, $\alpha_i \ge 0$.

**Solución:** $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ (solo $\alpha_i>0$ son vectores de soporte).

### 2.4 Soft-margin + hinge

$$
\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i
$$

s.a. $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i$, $\xi_i \ge 0$.

Equivale a penalizar $\sum_i \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$.

### 2.5 Kernel trick

$$
K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top \phi(\mathbf{x}')
$$

**RBF:** $K(\mathbf{x},\mathbf{x}') = \exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$.

Predicción: $f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$.

### 2.6 Optimización (SVM)

Problema cuadrático convexo; algoritmos SMO, LIBSVM, etc.

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo k-NN (predicción)

```
Para cada punto de test x:
  Calcular d(x, x_i) para todo i en train
  Ordenar y tomar k índices menores
  y_hat ← clase mayoritaria entre esos k
```

- **Predicción naive:** $O(n_{\text{train}} \cdot p)$ por punto.
- **KD-tree** (baja dimensión): ~$O(\log n)$ por consulta amortizado.

### 3.2 Naive Bayes entrenamiento

```
Para cada clase k, cada feature j:
  Estimar P(C_k) por frecuencias
  Estimar P(x_j|C_k) (Gaussiana: media/varianza; multinomial: conteos+suavizado)
```

- **Tiempo:** $O(n p)$ para Gaussian; predicción $O(pK)$.

### 3.3 Numpy: k-NN básico

```python
import numpy as np

def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        d = np.sqrt(((X_train - x) ** 2).sum(axis=1))
        idx = np.argpartition(d, k)[:k]
        vals, cnts = np.unique(y_train[idx], return_counts=True)
        preds.append(vals[cnts.argmax()])
    return np.array(preds)
```

### 3.4 Escalamiento

- **Grandes $n$:** approximate nearest neighbors (FAISS, Annoy); muestreo de prototipos.
- **Grandes $p$:** NB a veces viable; SVM kernel costoso en entrenamiento $O(n^2)$–$O(n^3)$.

---

## 4. Métricas de Evaluación Específicas

Mismas que clasificación general: accuracy, precision, recall, F1, AUC (si `predict_proba`: NB nativo; SVM con Platt/calibración; k-NN por proporción de votos).

---

## 5. Descomposición Teórica

- **k-NN:** sesgo bajo con $k$ pequeño, varianza alta; $k$ grande suaviza (más sesgo).
- **NB:** sesgo por supuesto de independencia; varianza baja con pocos parámetros.
- **SVM:** maximizar margen acota capacidad (VC) bajo ciertos supuestos.

---

## 6. Selección de Hiperparámetros

- **k:** validación cruzada; impar para binaria evita empates.
- **SVM:** grid en $C$, $\gamma$ (RBF); escalado obligatorio.
- **NB:** suavizado Laplace en conteos.

---

## 7. Ecuaciones Clave (resumen)

| Método | Ecuación central |
|--------|------------------|
| k-NN | $d_2(\mathbf{x},\mathbf{x}') = \|\mathbf{x}-\mathbf{x}'\|_2$ |
| NB | $\hat{y} = \arg\max_k \log P(C_k) + \sum_j \log P(x_j\mid C_k)$ |
| SVM dual | $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ |
| Kernel | $f(\mathbf{x}) = \sum_i \alpha_i y_i K(\mathbf{x}_i,\mathbf{x}) + b$ |

---

## 8. Referencias y Lecturas Complementarias

- Cover & Hart — k-NN (paper clásico).
- Vapnik — *The Nature of Statistical Learning Theory*.
- Murphy — *PML* (cap. SVM y generativos).
