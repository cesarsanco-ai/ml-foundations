---
layout: default
---

# Fundamento Matemático y Computacional de la Interpretabilidad de Modelos (SHAP, LIME, PDP)
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-13](../../../sesiones/sesion-13.md)

*(Alineado con la Semana 13: importancia, valores de Shapley, explicaciones locales.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Modelo black-box $f: \mathcal{X} \to \mathbb{R}$ (regresión o score de clase). Se busca **atribución** de la contribución de cada feature $j$ a la predicción $f(\mathbf{x})$ respecto a una **línea base** $f_{\text{base}}$ (ej. $\mathbb{E}[f(\mathbf{X})]$).

### 1.2 Notación

- $\mathbf{x} = (x_1,\ldots,x_p)$ instancia a explicar.
- $S \subseteq \{1,\ldots,p\}$ subconjunto de features activas; coalición en teoría de juegos.

### 1.3 Supuestos

- **SHAP:** axiomas de Shapley (eficiencia, simetría, dummy, linealidad).
- **LIME:** modelo local lineal aproxima $f$ en vecindad ponderada de $\mathbf{x}$.

---

## 2. Fundamento Matemático

### 2.1 Valores de Shapley (juego cooperativo)

Valor marginal del jugador $j$:

$$
\Delta_j(S) = v(S \cup \{j\}) - v(S)
$$

**Valor de Shapley:**

$$
\boxed{\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(p-|S|-1)!}{p!} \Delta_j(S)}
$$

### 2.2 SHAP para ML

Función característica típica (modelo condicional o marginal sobre background):

$$
v_f(S) = \mathbb{E}[f(\mathbf{X}) \mid \mathbf{X}_S = \mathbf{x}_S]
$$

(o integración sobre $\mathbf{X}_{\bar{S}}$ según definición escogida).

**Propiedad de eficiencia:**

$$
\sum_{j=1}^p \phi_j(\mathbf{x}) = f(\mathbf{x}) - \mathbb{E}[f(\mathbf{X})]
$$

### 2.3 Permutation importance

Para feature $j$, error con datos permutados:

$$
\text{Imp}(j) = \mathbb{E}\big[L(f(\mathbf{X}^{(j)}), Y)\big] - \mathbb{E}\big[L(f(\mathbf{X}), Y)\big]
$$

$\mathbf{X}^{(j)}$: columna $j$ permutada.

### 2.4 Partial Dependence

$$
\boxed{\text{PD}_j(x_j) = \mathbb{E}_{\mathbf{X}_{\setminus j}}\big[f(x_j, \mathbf{X}_{\setminus j})\big] \approx \frac{1}{n}\sum_{i=1}^n f(x_j, \mathbf{x}^{(i)}_{\setminus j})}
$$

### 2.5 LIME — regresión local ponderada

Minimizar:

$$
\boxed{\arg\min_{\mathbf{w}} \sum_{i=1}^n \pi_{\mathbf{x}}(\mathbf{z}_i)\big(f(\mathbf{z}_i) - \mathbf{w}^\top \tilde{\mathbf{z}}_i\big)^2 + \lambda\|\mathbf{w}\|_1}
$$

$\pi_{\mathbf{x}}(\mathbf{z}) = \exp(-d(\mathbf{x},\mathbf{z})^2/\sigma^2)$; $\tilde{\mathbf{z}}$ versión interpretable.

### 2.6 Optimización

- Kernel SHAP: muestreo de coaliciones; problema de regresión ponderada.
- Tree SHAP: recurrencia exacta en árboles (polinomial en hojas por camino).

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo permutation importance

```
error_base ← L(y, f(X))
Para j = 1 … p:
  X' ← copia de X; permutar columna j
  error_j ← L(y, f(X'))
  Imp_j ← error_j - error_base
```

- **Coste:** $O(p \cdot n \cdot C_f)$ con $C_f$ coste de inferencia.

### 3.2 Numpy: permutación de una columna

```python
import numpy as np

def permute_column(X, j, rng=None):
    rng = np.random.default_rng(rng)
    Xp = np.array(X, copy=True)
    Xp[:, j] = rng.permutation(Xp[:, j])
    return Xp
```

### 3.3 Escalamiento

- SHAP exacto en árboles: lineal en nodos por instancia en implementaciones eficientes.
- Kernel SHAP: muestreo; coste alto en $p$ grande.

---

## 4. Métricas de Evaluación Específicas

- **Fidelidad local:** $R^2$ del modelo surrogate LIME en vecindad.
- **Consistencia:** comparar explicaciones cuando el modelo cambia monótonamente.
- **Human evaluation** en aplicaciones.

---

## 5. Descomposición Teórica

Shapley es la **única** asignación que satisface los axiomas citados bajo la definición de $v$ elegida; interpretación causal **no** está garantizada si features correlacionadas.

---

## 6. Selección de Hiperparámetros

- LIME: ancho de kernel $\sigma$, tamaño de vecindad, complejidad (Lasso).
- SHAP: tamaño y representatividad del conjunto de fondo.

---

## 7. Ecuaciones Clave (resumen)

| Método | Fórmula |
|--------|---------|
| Shapley | $\phi_j = \sum_S w_S (v(S\cup j)-v(S))$ |
| Eficiencia SHAP | $\sum_j \phi_j = f(\mathbf{x}) - f_{\text{base}}$ |
| PDP | $\mathbb{E}[f(x_j, \mathbf{X}_{\setminus j})]$ |
| LIME | $\min_\mathbf{w} \sum_i \pi_{\mathbf{x}}(\mathbf{z}_i)(f(\mathbf{z}_i)-\mathbf{w}^\top\tilde{\mathbf{z}}_i)^2$ |

---

## 8. Referencias y Lecturas Complementarias

- Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions* (NeurIPS 2017).
- Ribeiro, Singh, Guestrin — *"Why Should I Trust You?"* (LIME, KDD 2016).
- Molnar — *Interpretable Machine Learning* (libro online).
