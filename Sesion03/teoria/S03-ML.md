Aquí tienes una versión **mejorada y ampliada en Markdown** para Semana 3, con definiciones más completas, conexiones a ML clásico, regularización y fundamentos matemáticos, lista para adjuntarse a un curso:

---

# Semana 3: Regresión Lineal y Regularización

## Logro de la sesión

Construir, entrenar e interpretar modelos de **regresión lineal**, aplicando técnicas de **regularización** para mejorar la generalización y seleccionar las métricas adecuadas para evaluar su desempeño.

---

## Problemática de negocio

Los problemas de regresión buscan **predecir una variable continua** a partir de una o varias variables predictoras.

* **Tipos de problemas:**

  * Predicción de ventas mensuales según variables de marketing.
  * Estimación de precios de viviendas según características físicas y ubicación.
  * Pronóstico de demanda energética según clima y horario.
* **Solución propuesta:**

  * **Regresión simple:** una variable predictora → una variable objetivo.
  * **Regresión múltiple:** múltiples variables predictoras para capturar relaciones más complejas.

---

## Modelado

### Requisitos del modelo

* Variables numéricas y/o codificadas adecuadamente.
* Datos limpios y preprocesados (outliers tratados, valores faltantes imputados, variables escaladas si es necesario).
* Cumplimiento aproximado de supuestos estadísticos para interpretar resultados correctamente.

### Regresión lineal simple y múltiple

* **Formulación matemática:**
  [
  \hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
  ]
  donde (\beta_0) es el intercepto y (\beta_j) los coeficientes de cada variable.
* **Interpretación de coeficientes:** cada (\beta_j) representa el cambio esperado en (y) por un cambio unitario en (x_j), manteniendo constantes las demás variables.

### Supuestos del modelo

1. **Linealidad:** relación lineal entre variables predictoras y objetivo.
2. **Independencia de errores:** residuos no correlacionados.
3. **Homoscedasticidad:** varianza constante de los errores.
4. **Normalidad de residuos:** para inferencia y predicción de intervalos de confianza.

### Regularización

Evita **overfitting** y mejora generalización al penalizar magnitudes de coeficientes:

| Método          | Penalización                      | Fórmula                                                            |         |                               |
| --------------- | --------------------------------- | ------------------------------------------------------------------ | ------- | ----------------------------- |
| **Ridge (L2)**  | Suma de cuadrados de coeficientes | (\min_\beta \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j \beta_j^2) |         |                               |
| **Lasso (L1)**  | Suma de valores absolutos         | (\min_\beta \sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j            | \beta_j | )                             |
| **Elastic Net** | Combinación L1 y L2               | (\min_\beta \sum_i (y_i - \hat{y}_i)^2 + \lambda_1 \sum_j          | \beta_j | + \lambda_2 \sum_j \beta_j^2) |

* **Interpretación:**

  * Ridge reduce coeficientes grandes pero no los anula.
  * Lasso puede eliminar variables irrelevantes.
  * Elastic Net combina beneficios de ambos.
* **Selección de hiperparámetro ((\lambda))**: usualmente mediante **validación cruzada** para balancear bias y varianza.

### Plantilla base en Python

```python
# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Split de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo y entrenamiento
model = LinearRegression()  # o Ridge(alpha=0.1), Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## Métricas de evaluación

* **Error cuadrático medio (MSE):**
  [
  MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
  ]
* **Raíz del error cuadrático medio (RMSE):**
  (\text{RMSE} = \sqrt{MSE})
* **Error absoluto medio (MAE):**
  (\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|)
* **Coeficiente de determinación ((R^2)):**
  [
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  ]
* **Interpretación:** elegir métricas según sensibilidad a outliers y necesidad de interpretación en la unidad original de la variable.

---
## Comunicación: Storytelling e Insights con Data Dummy

### Ejemplo de Data Dummy

Supongamos un dataset ficticio sobre predicción de precio de casas (`Price`) en función de metros cuadrados (`Area`) y número de habitaciones (`Rooms`):

| ID | Area (m²) | Rooms | Price ($) |
| -- | --------- | ----- | --------- |
| 1  | 50        | 2     | 120,000   |
| 2  | 80        | 3     | 200,000   |
| 3  | 100       | 4     | 250,000   |
| 4  | 60        | 2     | 140,000   |
| 5  | 90        | 3     | 210,000   |

Se ajusta una **regresión lineal múltiple**:

[
\hat{y} = \beta_0 + \beta_1 \cdot Area + \beta_2 \cdot Rooms
]

Supongamos que obtenemos los siguientes coeficientes con regularización Ridge:

[
\hat{\beta}_0 = 10,000, \quad \hat{\beta}_1 = 2,000, \quad \hat{\beta}_2 = 15,000
]

Y las métricas de desempeño del modelo son:

* **MSE:** 5,000,000
* **RMSE:** 2,236
* **MAE:** 1,900
* **R²:** 0.92

---

### Storytelling con Data Dummy

**Objetivo:** Traducir métricas en insights claros.

1. **Identificar patrones:**

   * Cada metro cuadrado adicional incrementa el precio promedio en $2,000.
   * Cada habitación adicional agrega $15,000.

2. **Impacto en el negocio:**

   * Propietarios pueden usar este modelo para establecer precios de venta competitivos.
   * Agentes inmobiliarios pueden priorizar propiedades con mayor retorno por m².

3. **Visualizaciones sugeridas:**

   * Scatter plot de `Price` vs `Area`, con línea de regresión ajustada.
   * Heatmap de correlación entre `Price`, `Area` y `Rooms`.
   * Histogramas de residuales para evaluar ajuste.

---

### Elevator Pitch: Equipo Técnico

> “Se ajustó un modelo de regresión lineal múltiple para predecir el precio de casas usando `Area` y `Rooms`.
> El modelo tiene un **R² de 0.92**, RMSE de 2,236 USD y MAE de 1,900 USD, indicando excelente ajuste para este dataset.
> Se aplicó **regularización Ridge** para controlar multicolinealidad entre `Area` y `Rooms`.
> Los coeficientes indican un incremento promedio de $2,000 por m² y $15,000 por habitación.
> Próximos pasos: validar el modelo con cross-validation y probar Elastic Net para evaluar sparsity y correlaciones fuertes.”

---

### Elevator Pitch: Equipo No Técnico / Negocio

> “Este modelo nos ayuda a estimar el precio de venta de las casas de manera precisa.
> Por ejemplo, aumentar 1 metro cuadrado en el área promedio incrementa el valor de la propiedad en $2,000, y añadir una habitación suma $15,000.
> Con este modelo, podemos fijar precios competitivos, entender qué características afectan más el precio y priorizar propiedades con mayor retorno de inversión.
> Además, el modelo es confiable para nuestra cartera, con un ajuste del 92% de los precios reales.”

---

## Reto: 1 punto

* Investigar **requerimientos computacionales** para entrenar un modelo de regresión lineal con un dataset de **10 millones de filas y 1000 columnas** en Databricks, considerando optimización y paralelización.

---

## Laboratorio: Ver Colab

* Implementación de **scikit-learn:** regresión lineal, Ridge, Lasso.
* Pipeline completo:

  1. EDA + feature engineering
  2. Regresión múltiple
  3. Regularización
  4. Métricas de evaluación y visualización de resultados.

---

Perfecto. Vamos a desarrollar un **fundamento matemático completo y detallado** para regresión lineal y regularización, paso a paso, incluyendo **función de pérdida, optimización, solución cerrada, gradiente, y regularización Ridge/Lasso/Elastic Net**. Te lo pongo en un formato que puedes adjuntar tal cual en Markdown o LaTeX.

---



## Anexo: Fundamento Matemático y Computacional de la Regresión Lineal

### 1. Planteamiento de la regresión lineal

Dado un conjunto de datos ((X, y)) con (n) observaciones y (p) características, la **regresión lineal múltiple** modela la relación entre (X) y (y) como:

[
y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} + \varepsilon_i \quad i=1,\dots,n
]

o en forma vectorial:

[
\mathbf{y} = X \boldsymbol{\beta} + \boldsymbol{\varepsilon}
]

donde:

* (X \in \mathbb{R}^{n \times (p+1)}) es la matriz de diseño (incluye columna de 1s para el intercepto),
* (\boldsymbol{\beta} \in \mathbb{R}^{p+1}) son los coeficientes a estimar,
* (\boldsymbol{\varepsilon}) son los errores aleatorios.

---

### 2. Función de pérdida

Se define la **función de pérdida** como el **Error Cuadrático Medio (MSE)**:

[
J(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}*i)^2 = \frac{1}{n} \sum*{i=1}^n (y_i - X_i \boldsymbol{\beta})^2
]

En forma matricial:

[
J(\boldsymbol{\beta}) = \frac{1}{n} (\mathbf{y} - X\boldsymbol{\beta})^\top (\mathbf{y} - X\boldsymbol{\beta})
]

---

### 3. Optimización: derivación de la solución

Para encontrar los coeficientes óptimos, se minimiza (J(\boldsymbol{\beta})) tomando el gradiente respecto a (\boldsymbol{\beta}):

[
\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = -\frac{2}{n} X^\top (\mathbf{y} - X\boldsymbol{\beta})
]

Igualando a cero para minimizar:

[
X^\top (\mathbf{y} - X\boldsymbol{\beta}) = 0
]

[
X^\top \mathbf{y} = X^\top X \boldsymbol{\beta}
]

[
\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y} \quad \text{(Solución OLS cerrada)}
]

> **Nota:** La inversa ( (X^\top X)^{-1} ) existe si las columnas de (X) son linealmente independientes.

---

### 4. Regularización

Cuando (X^\top X) es singular o hay riesgo de **overfitting**, se aplican técnicas de regularización:

#### 4.1 Ridge (L2)

Agrega penalización sobre la norma cuadrada de los coeficientes:

[
J_{ridge}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - X_i \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^{p} \beta_j^2
]

Gradiente:

[
\nabla_{\boldsymbol{\beta}} J_{ridge} = -\frac{2}{n} X^\top (\mathbf{y} - X \boldsymbol{\beta}) + 2\lambda \boldsymbol{\beta}
]

Solución cerrada:

[
\hat{\boldsymbol{\beta}}_{ridge} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}
]

* (\lambda > 0) controla la fuerza de penalización.
* Ridge **no fuerza coeficientes a cero**, solo reduce su magnitud.

#### 4.2 Lasso (L1)

Agrega penalización sobre la suma absoluta de los coeficientes:

[
J_{lasso}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - X_i \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^{p} |\beta_j|
]

* No tiene solución cerrada debido a la no diferenciabilidad en (\beta_j = 0).
* Se optimiza mediante **coordinate descent** o **gradiente subdiferencial**:

[
\beta_j \leftarrow S\left(\frac{1}{n} \sum_{i=1}^n x_{ij} (y_i - \hat{y}_{i,-j}), \frac{\lambda}{2}\right)
]

donde (S(\cdot, \cdot)) es la función de **soft-thresholding**.

* Lasso **puede forzar coeficientes exactamente a cero**, permitiendo selección de variables.

#### 4.3 Elastic Net

Combina L1 y L2:

[
J_{EN}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n (y_i - X_i \boldsymbol{\beta})^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2
]

* (\lambda_1) controla sparsity (L1), (\lambda_2) controla shrinkage (L2).
* Se resuelve mediante **gradiente descendente** o **coordinate descent**.

---

### 5. Interpretación de la regularización

| Tipo        | Efecto                                            | Uso recomendado                                   |
| ----------- | ------------------------------------------------- | ------------------------------------------------- |
| Ridge       | Reduce magnitud de coeficientes, no los hace cero | Datos multicolineales, many features              |
| Lasso       | Reduce y puede eliminar coeficientes              | Selección automática de features                  |
| Elastic Net | Balance entre Ridge y Lasso                       | Datasets con correlaciones y alta dimensionalidad |

---

### 6. Optimización computacional

* **Gradiente descendente:** iterativo, útil para datasets grandes.
  [
  \boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla J(\boldsymbol{\beta}^{(t)})
  ]

* **Vectorización:** operaciones sobre matrices completas en lugar de loops para eficiencia.

* **Complejidad algorítmica:**

  * OLS cerrada: (O(p^2 n + p^3))
  * Gradiente descendente: (O(k \cdot n \cdot p)) para k iteraciones

---
