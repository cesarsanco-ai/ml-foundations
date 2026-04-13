---
layout: default
---
# Sesión 6: Árboles de Decisión y Random Forest

### 1. Logro de la sesión

Construir e interpretar **árboles de decisión** (CART) para clasificación y regresión, y **ensambles tipo Random Forest** mediante **bagging**, comprendiendo **criterios de impureza**, **control de complejidad**, **error OOB** e **importancia de variables** con visión crítica de sus sesgos.

---

### 2. Historia y línea temporal

| Periodo | Hito |
|---------|------|
| **1960s** | **Morgan & Sonquist (1963)** y sistemas AID: primeros árboles automáticos en ciencias sociales |
| **1980s** | **CART** (Breiman et al., 1984): binario, poda por coste-complejidad |
| **1986** | **ID3 / C4.5** (Quinlan): ganancia de información, dominio simbólico |
| **1990s–2000s** | Árboles como base de **boosting** y **bagging** |
| **2001** | **Random Forest** (Breiman): bagging + aleatorización de features |
| **Actualidad** | RF y boosting (XGBoost, etc.) dominan tabular; interpretación con SHAP |

**Lectura:** el árbol es el modelo **más interpretable** localmente (reglas), pero un árbol profundo puede ser tan opaco como cualquier otro modelo complejo.

---

### 3. Árboles CART: estructura y algoritmo

#### 3.1 Componentes

- **Nodo raíz:** toda la muestra.  
- **Nodos internos:** pregunta sobre una variable $x_j$ y un umbral $t$ (“¿$x_j \le t$?”).  
- **Hojas:** predicción = clase mayoritaria (clasificación) o media (regresión).

#### 3.2 Criterios de división

**Clasificación** (nodos $m$ con proporciones $\hat{p}_{mk}$ de clase $k$):

- **Gini:** $\sum_k \hat{p}_{mk}(1-\hat{p}_{mk})$ — mide impureza.  
- **Entropía:** $-\sum_k \hat{p}_{mk}\log \hat{p}_{mk}$ — de la teoría de información (Quinlan).

**Regresión:** minimizar **SSE** intra-nodo, equivalente a reducir **MSE** ponderado.

En cada split se evalúan muchos $(j,t)$; se elige el que **más reduce** la impurez o el error.

#### 3.3 Sobreajuste y control

Los árboles **profundos** encajan ruido → **alta varianza**. Parámetros clave en `sklearn`:

| Parámetro | Efecto al **subir** el valor |
|-----------|------------------------------|
| `max_depth` | Más profundidad → más flexibilidad (riesgo overfitting) |
| `min_samples_leaf` | Hojas más pobladas → modelo más regularizado |
| `min_samples_split` | Exige más datos para dividir un nodo |
| `max_features` | En RandomForest, no en árbol simple |

**Poda:** en CART clásico, poda por coste-complejidad; en sklearn se controla principalmente por **pre**-poda vía `max_depth` y afines.

#### 3.4 Ventajas y limitaciones del árbol solo

**Ventajas:**

- Captura **no linealidades** e **interacciones** sin especificarlas.  
- Manejo natural de **variables mixtas** (según implementación).  
- **Interpretación** vía reglas si el árbol no es enorme.

**Limitaciones:**

- **Inestabilidad:** pequeños cambios en datos pueden cambiar el árbol por completo.  
- Sesgo hacia variables con muchos niveles de corte posibles si no se controla.

#### 3.5 Plantilla Python (árbol de clasificación)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Visualización (opcional): `plot_tree` o exportar a Graphviz.

---

### 4. Random Forest: bagging + aleatorización

#### 4.1 Bagging (Bootstrap AGGregatING)

1. Generar $B$ muestras bootstrap de tamaño $n$ con reemplazo.  
2. Entrenar un árbol profundo en cada muestra.  
3. **Clasificación:** voto mayoritario. **Regresión:** media de predicciones.

**Efecto:** los árboles tienen **alta varianza** pero están **poco correlacionados** si el modelo base es inestable → el promedio **reduce varianza** sin aumentar tanto el sesgo (Breiman, 1996).

#### 4.2 Aleatorización de variables (`max_features`)

En cada split, solo se considera un **subconjunto aleatorio** de predictores → los árboles se **decorrelacionan** más, mejorando la reducción de varianza respecto a bagging puro.

#### 4.3 Out-of-Bag (OOB) error

Para cada árbol $b$, aproximadamente **37 %** de las observaciones queda fuera del bootstrap (“out-of-bag”). Se puede estimar error sin conjunto de validación explícito:

$$ \mathrm{OOB} \approx \frac{1}{n}\sum_{i=1}^{n} L\bigl(y_i, \hat{f}_{\mathrm{OOB}}(x_i)\bigr) $$

**Ventaja:** estimación interna “gratis”. **Cuidado:** no sustituye siempre un **test holdout** riguroso en producción.

#### 4.4 Hiperparámetros importantes

- `n_estimators` ($B$): más árboles → estimación más estable (rendimientos decrecientes).  
- `max_depth` / `min_samples_leaf`: controlan profundidad de cada árbol base.  
- `max_features`: típicamente `sqrt(p)` (clasificación) o `p/3` (regresión) como heurística inicial.

#### 4.5 Plantilla Python

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    oob_score=True,
)
rf.fit(X_train, y_train)
print("OOB score (accuracy):", rf.oob_score_)
```

Para regresión: `RandomForestRegressor`, criterio `squared_error`.

---

### 5. Comparación árbol vs bosque

| Criterio | Árbol único | Random Forest |
|----------|-------------|----------------|
| Interpretabilidad | Alta si poco profundo | Baja (muchas reglas) |
| Varianza | Alta | Mucho menor |
| Tiempo inferencia | Muy rápido | Más lento (promedio de $B$ árboles) |
| Necesidad de escalado | Menor que kNN/SVM | Menor (árboles por cortes ordinales) |

---

### 6. Importancia de variables

#### 6.1 Mean Decrease in Impurity (MDI) — “importancia Gini”

Suma de reducciones de impureza aportadas por $j$ en todos los splits donde participa.

**Sesgos conocidos:**

- Sesgo hacia variables **numéricas con muchos valores únicos** o alta cardinalidad.  
- **Correlación:** puede repartir importancia entre features redundantes de forma opaca.

#### 6.2 Permutation importance

Tras el modelo entrenado, se **perm**uta columna $j$ en validación y se mide caída de rendimiento. Más costoso pero a menudo más **alineado** con contribución predictiva real (Breiman, 2001; Fisher et al., 2019 en sklearn).

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
```

---

### 7. Métricas

- **Clasificación:** precisión, recall, F1, AUC (Sesión 4).  
- **Regresión:** MSE, RMSE, MAE, $R^2$ (Sesión 3).

---

### 7.1 Profundización: algoritmo CART (esquema)

**Entrenamiento (simplificado):**

1. Si el nodo cumple criterio de parada (`min_samples_leaf`, `max_depth`, impureza nula), declarar hoja.  
2. Para cada variable $j$ y cada punto de corte viable $t$, calcular la **reducción de impureza** $\Delta$ al dividir en $\{x_j \le t\}$ y $\{x_j > t\}$.  
3. Elegir $(j^\star, t^\star)$ que maximiza $\Delta$.  
4. Repetir recursivamente en hijos.

**Complejidad:** depende del número de puntos de corte evaluados; en variables continuas se suelen ordenar valores y solo considerar **cortes entre valores distintos** adyacentes.

**Poda (CART clásico):** se introduce penalización por número de hojas $|T|$:

$$ R_\alpha(T) = R(T) + \alpha |T| $$

donde $R(T)$ es el error de predicción en muestra (impureza o MSE). $\alpha$ controla trade-off ajuste–complejidad. En `sklearn` la poda post-hoc completa no está expuesta igual; se emula con `max_depth`, `ccp_alpha` (*cost complexity pruning*).

---

### 7.2 Gini vs entropía (¿cuál usar?)

Ambas son **cóncavas** y favorecen nodos puros. Diferencias prácticas:

- **Gini** suele ser más rápida de evaluar (sin logaritmos).  
- **Entropía** puede producir splits ligeramente distintos; en la práctica los resultados suelen ser **muy similares** (Breiman et al., 1984).

`sklearn` por defecto usa Gini en `DecisionTreeClassifier`.

---

### 7.3 Profundización: por qué el bosque reduce varianza

Sea $\hat{f}_1,\ldots,\hat{f}_B$ árboles entrenados con muestras bootstrap **correlacionadas** con correlación $\rho$ entre pares. La varianza del promedio $\bar{f}$ puede aproximarse (intuición de Breiman) como proporcional a:

$$ \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2 $$

- Aumentar $B$ reduce el segundo término.  
- **Reducir $\rho$** (con `max_features` < $p$) es clave: si todos los árboles son idénticos, no hay ganancia respecto a uno solo.

---

### 7.4 `max_features` en RandomForest: guía práctica

| Valor | Comportamiento típico |
|-------|------------------------|
| `"sqrt"` ($\sqrt{p}$) | Heurística clásica en clasificación |
| `"log2"` | Menos aleatorización por split |
| `0.3`–`0.7` de $p$ | Explorar en validación si $p$ es grande |

Valores **muy bajos** aumentan sesgo (cada split ve pocas variables); valores **altos** acercan los árboles y sube correlación.

---

### 8. Laboratorio (según sílabo)

- **NTB 1 —** Clasificación con árboles de decisión y Random Forest.  
- **NTB 2 —** Regresión con árboles de decisión y Random Forest (importancia de variables).

---

## Referencias bibliográficas principales

1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. Wadsworth.  
2. Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.  
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.  
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.  
5. Louppe, G. (2014). Understanding random forests: from theory to practice. *PhD thesis*, University of Liège.  
