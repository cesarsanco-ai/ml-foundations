---
layout: default
---
# Semana 3: Regresión Lineal y Regularización

## Logro de la sesión

Construir, entrenar e interpretar modelos de **regresión lineal**, aplicando técnicas de **regularización** para mejorar la generalización y seleccionar las métricas adecuadas para evaluar su desempeño.

---

## Problemática de negocio

Los problemas de regresión buscan **predecir una variable continua** a partir de una o varias variables predictoras.

- **Tipos de problemas:**
  - Predicción de ventas mensuales según variables de marketing.
  - Estimación de precios de viviendas según características físicas y ubicación.
  - Pronóstico de demanda energética según clima y horario.
- **Solución propuesta:**
  - **Regresión simple:** una variable predictora → una variable objetivo.
  - **Regresión múltiple:** múltiples variables predictoras para capturar relaciones más complejas.

---

## Modelado

### Requisitos del modelo

- Variables numéricas y/o codificadas adecuadamente.
- Datos limpios y preprocesados (outliers tratados, valores faltantes imputados, variables escaladas si es necesario).
- Cumplimiento aproximado de supuestos estadísticos para interpretar resultados correctamente.

### Regresión lineal simple y múltiple

- **Formulación matemática:**

$$ \hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p $$

donde $\beta_0$ es el intercepto y $\beta_j$ los coeficientes de cada variable.

- **Interpretación de coeficientes:** cada $\beta_j$ representa el cambio esperado en $y$ por un cambio unitario en $x_j$, manteniendo constantes las demás variables.

### Supuestos del modelo

1. **Linealidad:** relación lineal entre variables predictoras y objetivo.
2. **Independencia de errores:** residuos no correlacionados.
3. **Homoscedasticidad:** varianza constante de los errores.
4. **Normalidad de residuos:** para inferencia y predicción de intervalos de confianza.

### Regularización

Evita **overfitting** y mejora generalización al penalizar magnitudes de coeficientes:

| Método | Penalización | Fórmula |
|--------|--------------|---------|
| **Ridge (L2)** | Suma de cuadrados de coeficientes | $$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$ |
| **Lasso (L1)** | Suma de valores absolutos | $$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$ |
| **Elastic Net** | Combinación L1 y L2 | $$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$$ |

- **Interpretación:**
  - Ridge reduce coeficientes grandes pero no los anula.
  - Lasso puede eliminar variables irrelevantes.
  - Elastic Net combina beneficios de ambos.
- **Selección de hiperparámetro ($\lambda$)**: usualmente mediante **validación cruzada** para balancear bias y varianza.

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

- **Error cuadrático medio (MSE):**

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

- **Raíz del error cuadrático medio (RMSE):**

$$ RMSE = \sqrt{MSE} $$

- **Error absoluto medio (MAE):**

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

- **Coeficiente de determinación ($R^2$):**

$$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$

- **Interpretación:** elegir métricas según sensibilidad a outliers y necesidad de interpretación en la unidad original de la variable.

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

$$ \hat{y} = \beta_0 + \beta_1 \cdot Area + \beta_2 \cdot Rooms $$

Supongamos que obtenemos los siguientes coeficientes con regularización Ridge:

$$ \hat{\beta}_0 = 10,000, \quad \hat{\beta}_1 = 2,000, \quad \hat{\beta}_2 = 15,000 $$

Y las métricas de desempeño del modelo son:

- **MSE:** 5,000,000
- **RMSE:** 2,236
- **MAE:** 1,900
- **$R^2$:** 0.92

---

### Storytelling con Data Dummy

**Objetivo:** Traducir métricas en insights claros.

1. **Identificar patrones:**
   - Cada metro cuadrado adicional incrementa el precio promedio en $2,000.
   - Cada habitación adicional agrega $15,000.

2. **Impacto en el negocio:**
   - Propietarios pueden usar este modelo para establecer precios de venta competitivos.
   - Agentes inmobiliarios pueden priorizar propiedades con mayor retorno por m².

3. **Visualizaciones sugeridas:**
   - Scatter plot de `Price` vs `Area`, con línea de regresión ajustada.
   - Heatmap de correlación entre `Price`, `Area` y `Rooms`.
   - Histogramas de residuales para evaluar ajuste.

---

### Elevator Pitch: Equipo Técnico

> "Se ajustó un modelo de regresión lineal múltiple para predecir el precio de casas usando `Area` y `Rooms`.
> El modelo tiene un **$R^2$ de 0.92**, RMSE de 2,236 USD y MAE de 1,900 USD, indicando excelente ajuste para este dataset.
> Se aplicó **regularización Ridge** para controlar multicolinealidad entre `Area` y `Rooms`.
> Los coeficientes indican un incremento promedio de $2,000 por m² y $15,000 por habitación.
> Próximos pasos: validar el modelo con cross-validation y probar Elastic Net para evaluar sparsity y correlaciones fuertes."

---

### Elevator Pitch: Equipo No Técnico / Negocio

> "Este modelo nos ayuda a estimar el precio de venta de las casas de manera precisa.
> Por ejemplo, aumentar 1 metro cuadrado en el área promedio incrementa el valor de la propiedad en $2,000, y añadir una habitación suma $15,000.
> Con este modelo, podemos fijar precios competitivos, entender qué características afectan más el precio y priorizar propiedades con mayor retorno de inversión.
> Además, el modelo es confiable para nuestra cartera, con un ajuste del 92% de los precios reales."

---
