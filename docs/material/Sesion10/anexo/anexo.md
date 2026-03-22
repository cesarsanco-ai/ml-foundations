---
layout: default
---

# Fundamento Matemático y Computacional de Series Temporales para ML
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-10](../../../sesiones/sesion-10.md)

*(Alineado con la Semana 10: descomposición, lags, ARIMA/SARIMA, validación temporal, Prophet.)*

---

## 1. Planteamiento General del Problema

### 1.1 Definición formal

Proceso $\{y_t\}_{t=1}^T$ indexado por tiempo. Objetivo: predecir $y_{t+h}$ dado $\{y_s, s \le t\}$ (y covariables $\mathbf{x}_t$ si hay).

**Dependencia:** viola i.i.d.; el orden temporal es esencial.

### 1.2 Notación

- Operador rezago $B$: $B y_t = y_{t-1}$.
- Ruido blanco $\{\varepsilon_t\}$, $\mathbb{E}[\varepsilon_t]=0$, $\mathbb{V}(\varepsilon_t)=\sigma^2$, incorrelados.

### 1.3 Supuestos

- **Estacionariedad** (débil): media constante, autocovarianza solo depende del lag (tras diferenciación en ARIMA).
- **ML tabular con lags:** supuesto de que el vector de features captura suficiente historia.

---

## 2. Fundamento Matemático

### 2.1 Descomposición

$$
\boxed{y_t = T_t + S_t + R_t \quad \text{(aditiva)}}, \quad
y_t = T_t \cdot S_t \cdot R_t \quad \text{(multiplicativa)}
$$

### 2.2 ARMA$(p,q)$

$$
\boxed{y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t + \sum_{j=1}^q \theta_j \varepsilon_{t-j}}
$$

### 2.3 ARIMA$(p,d,q)$

$\Delta^d y_t$ sigue un ARMA$(p,q)$, con $\Delta y_t = y_t - y_{t-1}$.

### 2.4 SARIMA — componente estacional

Operadores polinomiales en $B$ y $B^s$ ($s$: período estacional).

### 2.5 Reformulación supervisada para ML

Construir tabla:

$$
\mathbf{z}_t = \big( y_{t-1}, y_{t-2}, \ldots, y_{t-L},\, \text{medias móviles},\, \text{calendario},\, \mathbf{x}_t \big)
$$

Objetivo $y_t$; entonces regresión/clasificación estándar **pero** la partición train/test debe respetar el tiempo.

### 2.6 Prophet (esquema aditivo)

$$
\boxed{y(t) = g(t) + s(t) + h(t) + \varepsilon_t}
$$

$g$: tendencia (lineal a tramos / logística); $s$: Fourier para estacionalidad; $h$: festivos.

### 2.7 Optimización

- ARIMA: máxima verosimilitud / mínimos cuadrados condicionales; criterios AIC/BIC.
- Prophet: ajuste penalizado (L2) sobre parámetros de tendencia y estacionalidad.

---

## 3. Algoritmos Computacionales

### 3.1 Pseudocódigo: matriz de lags (univariado)

```
Para t = L+1 … T:
  fila_t ← (y_{t-1}, …, y_{t-L})
  objetivo_t ← y_t
Entrenar modelo en filas (train) predecir últimos pasos (test)
```

### 3.2 Validación temporal (walk-forward)

```
Para cada origen de tiempo t0:
  Train: datos ≤ t0
  Val: ventana (t0, t0+h]
  Registrar error
```

### 3.3 Numpy: features de rezagos

```python
import numpy as np

def lag_matrix(y, L):
    """y: (T,), devuelve X (T-L, L), target y[L:]"""
    T = len(y)
    X = np.column_stack([y[L - 1 - k : T - 1 - k] for k in range(L)])
    target = y[L:]
    return X, target
```

---

## 4. Métricas de Evaluación Específicas

$$
\text{MAE} = \frac{1}{H}\sum_{h=1}^H |y_{t+h} - \hat{y}_{t+h}|, \quad
\text{RMSE} = \sqrt{\frac{1}{H}\sum_{h=1}^H (y_{t+h} - \hat{y}_{t+h})^2}
$$

**sMAPE**, **MAPE** (cuidado con $y_t \approx 0$). **Quantile loss** para intervalos.

---

## 5. Descomposición Teórica

Descomposición bias–varianza sigue aplicando al modelo supervisado sobre $\mathbf{z}_t$, pero **distribución cambiante** (concept drift) rompe el supuesto estacionario.

---

## 6. Selección de Hiperparámetros

- Órdenes ARIMA: AIC/BIC + diagnóstico de residuos.
- $L$ (lags), ventanas: validación temporal.
- Prophet: número de términos de Fourier, changepoints.

---

## 7. Ecuaciones Clave (resumen)

| Concepto | Fórmula |
|----------|---------|
| Diferencia | $\Delta y_t = y_t - y_{t-1}$ |
| AR(1) | $y_t = \phi y_{t-1} + \varepsilon_t$ |
| MA(1) | $y_t = \varepsilon_t + \theta \varepsilon_{t-1}$ |
| Prophet | $y = g + s + h + \varepsilon$ |

---

## 8. Referencias y Lecturas Complementarias

- Box, Jenkins, Reinsel — *Time Series Analysis: Forecasting and Control*.
- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (libro online).
- Taylor & Letham — Prophet (2017).
