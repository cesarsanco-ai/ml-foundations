---
layout: default
---
# Sesión 10: Series de Tiempo


### 1. Logro de la sesión

Modelar series temporales integrando **descomposición**, **ARIMA/SARIMA**, **Prophet** y **modelos tabulares con features de tiempo** (lags, ventanas), con **métricas** adecuadas (MAE, RMSE, MAPE, MASE) y **validación temporal** sin *leakage*.

---

### 2. Historia y contexto

| Periodo | Hito |
|---------|------|
| **1920s–30s** | Yule, Walker: autoregresión |
| **1970** | Box & Jenkins popularizan **ARIMA** como marco integrado |
| **1990s** | Modelos de espacio de estados, SARIMA estacional |
| **2017** | **Prophet** (Taylor & Letham): tendencias y estacionalidades flexibles en industria |
| **2010s+** | ML con lags (XGBoost) compite/ complementa modelos clásicos en tabular temporal |

---

### 3. Componentes de una serie

**Tendencia** $T_t$: cambio lento del nivel medio.  
**Estacionalidad** $S_t$: patrones que se repiten con periodo $s$ (día, semana, año).  
**Ciclos**: oscilaciones sin periodo fijo (económico).  
**Ruido** $\varepsilon_t$: irregular.

Descomposición **aditiva** $Y_t = T_t + S_t + R_t$ vs **multiplicativa** $Y_t = T_t \cdot S_t \cdot R_t$ (cuando la amplitud estacional crece con el nivel).

---

### 4. Estacionariedad

Una serie es **débilmente estacionaria** si media y covarianza no dependen del tiempo (condiciones simplificadas). **Pruebas** (ADF, KPSS) orientan necesidad de **diferenciación** $d$ en ARIMA.

---

### 5. ARIMA y SARIMA

**ARIMA$(p,d,q)$:** parte autorregresiva (AR), integrada (I = diferencias), media móvil (MA).

**SARIMA** añade términos estacionales $(P,D,Q)_s$ para patrones que se repiten cada $s$ pasos.

**Diagnóstico:** residuos deben parecer **ruido blanco** (ACF de residuos sin estructura). Si queda estructura → modelo incompleto.

**Python:** `statsmodels.tsa.arima.model.ARIMA`, `SARIMAX` con exógenas.

---

### 6. Prophet (visión práctica)

Descomposición tipo:

$$ y(t) = g(t) + s(t) + h(t) + \varepsilon_t $$

donde $g$ es tendencia (piecewise), $s$ estacionalidades (Fourier), $h$ festividades.

**Ventajas:** manejo de **datos faltantes**, **cambios de tendencia**, estacionalidades múltiples.  
**Límites:** no captura bien dinámicas muy no lineales sin features adicionales; validar frente a baselines.

---

### 7. Machine learning con features temporales

Construir tabla supervisada $(X_t, y_t)$ con:

- **Lags:** $y_{t-1},\ldots,y_{t-L}$  
- **Medias móviles**, desviaciones, máximos en ventana  
- **Calendario:** día de la semana, mes (codificado)  
- **Diferencias** $\Delta y_t$

Modelar con **LightGBM/XGBoost** o regresión lineal regularizada.

**Crítico:** partición **temporal** train/test (últimos meses test), nunca mezclar aleatoriamente filas temporales.

---

### 8. Métricas

| Métrica | Fórmula / idea | Uso |
|---------|----------------|-----|
| MAE | $\frac{1}{H}\sum |e_t|$ | Robusta, misma unidad |
| RMSE | $\sqrt{\mathrm{MSE}}$ | Penaliza grandes errores |
| MAPE | media $\|e_t/y_t\|$ | Relativo; problemas si $y_t\approx 0$ |
| MASE | error vs error de modelo naive estacional | Escalable entre series |

---

### 9. Plantillas Python (esquema)

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# y: Series con índice DatetimeIndex frecuente
model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = model.fit(disp=False)
fc = res.get_forecast(steps=24)
```

```python
from prophet import Prophet

df = pd.DataFrame({"ds": fechas, "y": valores})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(periods=30, freq="D")
forecast = m.predict(future)
```

---

### 10. Laboratorio (según sílabo)

- **NTB 1 —** Series temporales con Prophet y features + modelos ML (XGBoost/LightGBM).  
- **NTB 2 —** Series temporales con ARIMA/SARIMA y diagnóstico de residuos.


---

### 11. Profundización ARIMA/SARIMA (notación y diagnóstico)

Un proceso **ARMA$(p,q)$** estacionario satisface:

$$ \phi(B) X_t = \theta(B) \varepsilon_t $$

donde $B$ es el operador retardo, $\phi$ y $\theta$ polinomios de orden $p$ y $q$. **ARIMA$(p,d,q)$** aplica $(1-B)^d$ para lograr estacionariedad en media.

**SARIMA** añade parte estacional $\Phi(B^s)$, $(1-B^s)^D$, $\Theta(B^s)$.

**ACF/PACF** sirven como guía exploratoria para órdenes (no sustituyen validación out-of-sample).

**Ljung-Box** sobre residuos: detectar autocorrelación residual.

---

### 12. Validación temporal (*time series cross-validation*)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    ...
```

Nunca usar KFold aleatorio en series ordenadas: mezcla información futura en el train.

---

### 13. Feature matrix para ML tabular (ejemplo)

```python
import pandas as pd

df = df.sort_index()
df["lag1"] = df["y"].shift(1)
df["roll7"] = df["y"].rolling(7).mean()
df["dow"] = df.index.dayofweek
df = df.dropna()
X = df[["lag1", "roll7", "dow"]]
y = df["y"]
```

---

### 14. Errores frecuentes

| Error | Consecuencia |
|-------|--------------|
| Mezclar orden temporal en split | Métricas irreales |
| Filtrar outliers sin entender estacionalidad | Pérdida de señal |
| Ignorar festivos en retail | Residuos estructurados |




### 15. Casos de uso del temario (detalle)

| Dominio | Patrón temporal | Modelo típico |
|---------|-----------------|---------------|
| Retail | ventas diarias/semanales con estacionalidad fuerte | SARIMA, Prophet, ML+lags |
| Energía | demanda con ciclo día/noche | SARIMA con $s=48$ (media hora) o ML con calendario |
| Web | tráfico con tendencia + weekly | Prophet o ML con Fourier + lags |

**Comparación estadístico vs ML:** los modelos estadísticos clásicos suelen ser más **parsimoniosos** y con intervalos de predicción mejor estudiados; el ML brilla cuando hay **covariables exógenas** ricas (promociones, clima, precios competencia) que entran como columnas.

### 16. Plantilla ML completa con `TimeSeriesSplit` + LightGBM

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for tr, va in tscv.split(X):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[va])
    scores.append(mean_absolute_error(y.iloc[va], pred))
print("MAE medio:", float(np.mean(scores)))
```

*(Requiere `import numpy as np`.)*

### 17. Métricas relativas y MASE (detalle)

**MASE** compara el MAE del modelo con el MAE de un modelo **naive** estacional (p.ej. mismo día año anterior o drift). Valores $<1$ indican mejora respecto al baseline ingenuo.

**sMAPE** (simétrica) mitiga asimetrías de MAPE pero tiene otros sesgos — Hyndman discute elección en su libro abierto.


---

## Referencias bibliográficas principales

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.  
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.  
3. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.  
