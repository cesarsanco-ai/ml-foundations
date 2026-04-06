---
layout: default
---
# Semana 10: Series de Tiempo

Los datos temporales aparecen en innumerables contextos: ventas diarias, tráfico web, indicadores económicos, sensores IoT, etc. Modelar correctamente la dependencia temporal es esencial para pronósticos precisos. Esta sesión aborda modelos estadísticos clásicos (ARIMA, SARIMA) y enfoques modernos (XGBoost, Prophet), con feature engineering temporal, métricas y ejemplos prácticos.

---

## Logro de la sesión

Modelar datos temporales con técnicas estadísticas y de machine learning, comprendiendo particularidades de los datos temporales y métricas adecuadas para evaluar pronósticos.

---

## Componentes de una serie temporal

Una serie temporal $y_t$ (con $t=1,\dots,T$) puede descomponerse en:

- **Tendencia (Trend):** movimiento a largo plazo (creciente, decreciente o estable).
- **Estacionalidad (Seasonal):** patrones que se repiten en períodos fijos (diario, semanal, anual).
- **Ciclo (Cycle):** fluctuaciones sin período fijo (p. ej. ciclos económicos); a diferencia de la estacionalidad, la duración no es predecible.
- **Ruido (Noise):** variaciones aleatorias no explicadas.

### Descomposición clásica

- **Aditiva:** $y_t = T_t + S_t + C_t + R_t$
- **Multiplicativa:** $y_t = T_t \times S_t \times C_t \times R_t$

La multiplicativa conviene cuando la magnitud de fluctuaciones crece con el nivel de la serie.

### Descomposición en la práctica

Métodos como **STL** (Seasonal and Trend decomposition using Loess) estiman componentes de forma robusta. Sirve para entender la serie, ajuste estacional y crear features para ML.

---

## Feature engineering para series temporales

Para usar regresión, árboles o XGBoost hay que transformar el problema en supervisado.

### Rezagos (lags)

$y_{t-1}, y_{t-2}, \dots, y_{t-L}$ con $L$ el máximo rezago.

### Ventanas móviles

- Media móvil: $\frac{1}{w}\sum_{i=t-w}^{t-1} y_i$
- Desviación estándar, mínimo, máximo, percentiles

### Diferencias

$\Delta y_t = y_t - y_{t-1}$ (útil para estacionariedad y como features).

### Variables de calendario

- Hora, día de la semana, mes, año; festivos, fines de semana.
- Codificación cíclica: $\sin(2\pi \cdot \text{hora}/24)$, $\cos(2\pi \cdot \text{hora}/24)$.

### Agregaciones multiescala

- Media últimos 7 días; mismo día de la semana en últimas 4 semanas; mismo día del año anterior.

---

## Modelos clásicos: ARIMA y SARIMA

### ARMA $(p,q)$

$$
y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}
$$

$\varepsilon_t$: ruido blanco.

### ARIMA $(p,d,q)$

$d$ diferencias para lograr estacionariedad; equivale a ARMA sobre la serie diferenciada.

### SARIMA $(p,d,q)(P,D,Q)_s$

Estacionalidad con período $s$; operador de rezago $B$ con componentes regulares y estacionales.

### Selección de órdenes

- **ACF / PACF** para orientar $p$ y $q$.
- **AIC / BIC** para comparar modelos.
- Validación en ventana de prueba (últimos períodos).

### Ejemplo clásico

Serie de pasajeros aéreos: a menudo log + diferencias y SARIMA$(0,1,1)(0,1,1)_{12}$.

---

## Machine learning para series

### Regresión con features temporales

Dataset con lags, ventanas y exógenas; regresión lineal, árboles, etc.

### XGBoost

- Lags y ventanas cuidadosos; dummies de estacionalidad.
- **Validación cruzada temporal** (no aleatoria).

**Ventajas:** no linealidad, muchas features, robustez a outliers.  
**Desventajas:** extrapolación de tendencia limitada fuera del rango de entrenamiento.

### Prophet

Modelo aditivo:

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t
$$

- $g(t)$: tendencia (lineal por tramos o logística).
- $s(t)$: estacionalidad (Fourier).
- $h(t)$: festivos.

**Ventajas:** datos faltantes, cambios de tendencia, uso sencillo.  
**Desventajas:** menos flexible que XGBoost con muchos predictores externos.

### Comparación rápida

| Característica | ARIMA/SARIMA | XGBoost | Prophet |
| :--- | :--- | :--- | :--- |
| Fundamento | Estadístico | Boosting | Aditivo |
| Estacionalidad | Explícita (SARIMA) | Features / dummies | Fourier |
| Variables exógenas | Limitado (ARIMAX) | Sí | Limitado (festivos / regresores) |
| Extrapolación tendencia | Sí | Limitada | Sí |
| Facilidad | Media | Alta | Muy alta |

---

## Métricas de evaluación

### MAE

$$
\text{MAE} = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|
$$

### RMSE

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}
$$

### MAPE

$$
\text{MAPE} = \frac{100}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right|
$$

Problemas si $y_t \approx 0$.

### MASE

$$
\text{MASE} = \frac{\frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|}{\frac{1}{n-1} \sum_{t=2}^n |y_t - y_{t-1}|}
$$

Denominador = error del pronóstico naive $\hat{y}_t = y_{t-1}$. MASE $< 1$ mejora al naive.

---

## Caso integrado: ventas diarias retail

**Datos:** ventas 3 años, festivos/promociones, clima, opcional economía local.

**EDA:** tendencia, estacionalidad semanal, picos en festividades, efecto promociones.

**Features:** calendario, festivos, clima con posibles lags, lags de ventas (7, 14, 28 días), medias móviles.

**Modelos:** SARIMA con $s=7$; XGBoost con time series split y Optuna; Prophet con weekly/yearly y festivos.

**Evaluación:** MAE, RMSE, MAPE, MASE en últimos 28 días (ejemplo ilustrativo en la teoría original con tabla comparativa Naive / SARIMA / XGBoost / Prophet).

---

## Implementación en Python

### ARIMA (statsmodels)

```python
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
fitted = model.fit()
forecast = fitted.forecast(steps=28)
```

### XGBoost con validación temporal

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, val_index in tscv.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
```

### Prophet

```python
from prophet import Prophet
import pandas as pd

df = pd.DataFrame({"ds": fechas, "y": ventas})
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_country_holidays(country_name="PE")
model.fit(df)
future = model.make_future_dataframe(periods=28)
forecast = model.predict(future)
```

---

## Conexión con el resto del curso

- Feature engineering conecta con transformaciones y agregaciones previas.
- XGBoost se relaciona con la sesión de Gradient Boosting.
- Validación temporal complementa métodos de la sesión de validación.
- MAE y RMSE son métricas de regresión ya vistas.

---

## Resumen

- Las series se descomponen en tendencia, estacionalidad, ciclo y ruido.
- Lags, ventanas y calendario son clave para ML en series.
- ARIMA/SARIMA modelan autocorrelación y estacionalidad de forma paramétrica.
- XGBoost y Prophet son alternativas flexibles con predictores externos o uso rápido.
- Evaluar con MAE, RMSE, MAPE, MASE y validación temporal.
- La elección del modelo depende de la serie, datos externos e interpretabilidad.
