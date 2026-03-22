---
layout: default
---

# Cheatsheet: Series de Tiempo
**Autor:** Carlos César Sánchez Coronel  

---

## Componentes

* **Tendencia** + **estacionalidad** + **ciclo** + **ruido**  
* **Aditiva** $y = T+S+R$ vs **multiplicativa** $y = T \times S \times R$  

---

## Features para ML tabular

* **Lags:** $y_{t-1},\dots,y_{t-L}$  
* **Ventanas:** media móvil, std, min/max  
* **Calendario:** día semana, mes; $\sin/\cos$ para ciclos  
* **Diferencias:** $\Delta y_t = y_t - y_{t-1}$  

---

## Modelos clásicos

| Modelo | Idea |
| :--- | :--- |
| **ARIMA(p,d,q)** | AR + I (diferencias) + MA |
| **SARIMA** | + estacionalidad con período $s$ |

Selección: ACF/PACF, AIC/BIC, validación en última ventana.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
fit = model.fit()
fc = fit.forecast(steps=28)
```

---

## ML / Prophet

* **XGBoost / RF:** features de lags + **TimeSeriesSplit** (no shuffle).  
* **Prophet:** tendencia + estacionalidad + festivos; API sencilla.  

```python
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

tscv = TimeSeriesSplit(n_splits=5)
for tr, va in tscv.split(X):
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05)
    model.fit(X[tr], y[tr])
```

---

## Métricas de pronóstico

| Métrica | Uso |
| :--- | :--- |
| **MAE** | Error en unidades de $y$ |
| **RMSE** | Penaliza errores grandes |
| **MAPE** | Error relativo; cuidado con $y\approx 0$ |
| **MASE** | vs modelo naive; $<1$ mejora a “persistencia” |

---

## Puntos críticos

* **No usar split aleatorio** en el tiempo si hay autocorrelación.  
* Exógenas (promos, clima) suelen favorecer **gradient boosting** frente a ARIMA puro.  
* Reentrenar con **ventana rodante** en producción.  

> *“El tiempo impone orden: respétalo en train, validación y despliegue.”*
