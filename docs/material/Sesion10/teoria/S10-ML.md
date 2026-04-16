---
layout: default
---
# Sesión 10: Series de Tiempo

### 1. Logro de la sesión

Modelar series temporales integrando **descomposición**, **estacionariedad y pruebas**, **ARIMA/SARIMA**, **Prophet** y **modelos tabulares con features de tiempo** (lags, ventanas, calendario), con **métricas** adecuadas (MAE, RMSE, MAPE, MASE, sMAPE) y **validación temporal** sin *data leakage*. El alumno debe distinguir cuándo un enfoque **estadístico clásico** aporta parsimonia e intervalos de predicción bien estudiados frente a cuándo el **ML tabular con covariables** exógenas es más competitivo.

---

### 2. Historia y contexto

| Periodo | Hito |
|---------|------|
| **1920s–30s** | Yule, Walker: autoregresión como herramienta para ciclos económicos |
| **1938** | Wold descompone series en componentes deterministas y estocásticos |
| **1970** | Box & Jenkins sistematizan **ARIMA**: identificación, estimación, verificación |
| **1980s–90s** | Modelos de **espacio de estados**, Kalman; **SARIMA** para estacionalidad fija |
| **2000s** | Software estadístico masivo (`statsmodels`, R) democratiza diagnóstico ACF/PACF |
| **2017** | **Prophet** (Taylor & Letham): API simple, festividades y *changepoints* en industria |
| **2010s+** | **ML con lags** (boosting, redes) compite o complementa ARIMA cuando hay muchas covariables |

**Lectura transversal:** en series, el **orden temporal** no es un detalle de implementación: define qué información es legítima en entrenamiento y qué constituye *leakage* si se mezcla con el futuro.

---

### 3. Componentes de una serie y descomposición

Una serie $\{Y_t\}$ suele descomponerse en piezas interpretables:

- **Tendencia** $T_t$: movimiento lento del nivel medio (crecimiento, saturación, declive).
- **Estacionalidad** $S_t$: patrón que se repite con periodo fijo $s$ (día, semana, año).
- **Ciclos**: oscilaciones sin periodo fijo estricto (ciclos económicos); a veces se absorben en tendencia flexible o en modelos con muchos grados de libertad.
- **Ruido** $R_t$ o $\varepsilon_t$: componente irregular, idealmente no predecible con la información disponible.

**Descomposición aditiva:**

$$Y_t = T_t + S_t + R_t$$

**Descomposición multiplicativa:**

$$Y_t = T_t \cdot S_t \cdot R_t$$

La forma multiplicativa es razonable cuando la **amplitud estacional crece con el nivel** (ventas que escalan). En la práctica se suele trabajar en **logaritmos** para volver a un esquema aditivo sobre $\log Y_t$, siempre que $Y_t > 0$.

**Métodos clásicos de estimación de $T_t$ y $S_t$:** medias móviles centradas, STL (*Seasonal-Trend decomposition using Loess*), o componentes paramétricas dentro de Prophet. La descomposición no sustituye un modelo predictivo completo, pero **orienta** sobre necesidad de diferenciación, periodo $s$ y presencia de outliers estructurales.

---

### 4. Estacionariedad

#### 4.1 Intuición

Una serie es **débilmente estacionaria** (simplificando) si:

1. $\mathbb{E}[Y_t]$ no depende de $t$ (media constante).
2. $\mathrm{Cov}(Y_t, Y_{t-h})$ solo depende del **rezago** $h$, no de $t$.

Muchos modelos lineales clásicos (ARMA) asumen estacionariedad **después de transformaciones** (diferencias, log).

#### 4.2 Diferenciación

El operador **retraso** $B$ cumple $B Y_t = Y_{t-1}$. La diferencia regular es:

$$\Delta Y_t = (1-B)Y_t = Y_t - Y_{t-1}$$

Aplicar $d$ veces conduce a $(1-B)^d Y_t$. En presencia de estacionalidad con periodo $s$, se usa también $(1-B^s)^D$ en SARIMA.

#### 4.3 Pruebas habituales

| Prueba | Hipótesis nula (lectura práctica) | Uso |
|--------|-----------------------------------|-----|
| **ADF** (Dickey–Fuller aumentada) | Raíz unitaria / no estacionariedad | Si se rechaza $H_0$, evidencia a favor de estacionariedad |
| **KPSS** | Estacionariedad alrededor de tendencia o nivel | A veces se contrasta con ADF: divergencias indican serie difícil |

**Cuidado:** las pruebas son sensibles a **tamaño de muestra**, **estructura estacional** y **cambios de régimen**. No sustituyen el **diagnóstico de residuos** ni la validación *out-of-sample*.

---

### 5. ARIMA y SARIMA

#### 5.1 ARMA en notación de operadores

Un proceso **ARMA$(p,q)$** (estacionario) satisface:

$$\phi(B) X_t = \theta(B) \varepsilon_t$$

donde $\varepsilon_t$ es ruido blanco, y:

$$\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p, \qquad
\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$$

**ARIMA$(p,d,q)$:** si $Y_t$ no es estacionaria en media, se modela $\phi(B)(1-B)^d Y_t = \theta(B)\varepsilon_t$ (con la notación estándar que incorpora $(1-B)^d$ en el lado izquierdo según la parametrización del software).

#### 5.2 SARIMA

**SARIMA** extiende ARIMA con polinomios en $B^s$ para capturar estacionalidad con periodo $s$:

$$\Phi(B^s)(1-B^s)^D \phi(B)(1-B)^d Y_t = \Theta(B^s)\theta(B)\varepsilon_t$$

Los órdenes se denotan **SARIMA$(p,d,q)(P,D,Q)_s$**. Ejemplo: datos mensuales con año fuerte → $s=12$.

#### 5.3 ACF y PACF como guía exploratoria

- **ACF** (*autocorrelation function*): correlación entre $Y_t$ y $Y_{t-h}$.
- **PACF** (*partial* ACF): correlación entre $Y_t$ y $Y_{t-h}$ **eliminando** la explicación lineal intermedia $Y_{t-1},\ldots,Y_{t-h+1}$.

Estas funciones **sugieren** órdenes $p,q,P,Q$ en procesos teóricos simples; en datos reales ruidosos, la selección final debe apoyarse en **AIC/BIC** y en **rendimiento predictivo** en horizonte.

#### 5.4 Diagnóstico de residuos

Tras ajustar el modelo, los **residuos** $\hat{\varepsilon}_t$ deberían comportarse como **ruido blanco**:

- ACF de residuos sin picos significativos.
- **Ljung–Box** (o Box–Pierce): contraste de autocorrelación conjunta en varios rezagos.

Si persiste estructura, el modelo está **incompleto** (falta estacionalidad, variable exógena, o cambio de régimen).

#### 5.5 Selección de órdenes y validación

En `statsmodels`, `ARIMA` y `SARIMAX` permiten buscar rejillas pequeñas de $(p,d,q)$ y $(P,D,Q)$ minimizando **AIC** o **BIC**. **BIC** tiende a penalizar más la complejidad. Para el rendimiento final, compare siempre con **error en test temporal** (no solo criterios dentro de muestra).

**Python (referencia):** `statsmodels.tsa.statespace.sarimax.SARIMAX`, o `statsmodels.tsa.arima.model.ARIMA` según versión.

---

### 6. Prophet (visión práctica y límites)

Prophet descompone:

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

- $g(t)$: tendencia **piecewise** con *changepoints* posibles.
- $s(t)$: estacionalidades **suaves** vía bases de Fourier (anual, semanal, etc.).
- $h(t)$: efectos de **festividades** y eventos puntuales.

**Ventajas en producción:** tolerancia razonable a **datos faltantes**, estacionalidades **múltiples**, incorporación de **vacaciones** por calendario.

**Límites:** dinámicas fuertemente **no lineales** o dependientes de **covariables externas** ricas pueden requerir **regresores adicionales** en Prophet o pasar a **ML tabular**. Los intervalos de predicción dependen de supuestos del modelo; conviene **calibrar** con validación temporal.

---

### 7. Machine learning con features temporales

#### 7.1 Tabla supervisada desde la serie

Se construye un par $(X_t, y_t)$ donde $y_t$ es el valor a predecir (p. ej. demanda del día $t$) y $X_t$ incluye:

- **Lags:** $y_{t-1},\ldots,y_{t-L}$.
- **Estadísticos de ventana:** medias móviles, desviaciones, máximos/mínimos en $W$ pasos.
- **Calendario:** día de la semana, mes, indicadores de festivo.
- **Diferencias:** $\Delta y_t$, aceleraciones.
- **Covariables exógenas:** precio, promoción, clima, tráfico web.

Modelos habituales: **LightGBM/XGBoost**, **Random Forest**, regresión **ridge/lasso** si la dimensionalidad es moderada y se desea interpretabilidad.

#### 7.2 Por qué el split temporal es obligatorio

Si se mezclan filas al azar, el modelo puede **“ver el futuro”** a través de lags correlacionados con el target en test, inflando métricas. La partición correcta es del tipo: entrenar hasta $T_0$, validar en $(T_0, T_1]$, test en $(T_1, T_2]$.

#### 7.3 Esquemas de validación

- **TimeSeriesSplit** (expanding window): en cada fold, el train crece y el test es un bloque posterior.
- **Rolling-origin** (*rolling forecast origin*): análogo en horizontes de pronóstico múltiples.

---

### 8. Métricas de error de pronóstico

| Métrica | Definición (horizonte $H$) | Comentario |
|---------|----------------------------|------------|
| **MAE** | $\frac{1}{H}\sum_{t=1}^H \|e_t\|$ | Misma escala que $y$; robusta a outliers moderados |
| **RMSE** | $\sqrt{\frac{1}{H}\sum e_t^2}$ | Penaliza errores grandes |
| **MAPE** | $\frac{1}{H}\sum \|e_t / y_t\|$ | Relativa; inestable si $y_t \approx 0$ |
| **sMAPE** | versión simétrica de error porcentual | Mitiga asimetrías pero tiene otros sesgos |
| **MASE** | MAE del modelo / MAE de baseline naive | Comparación **entre series** con escalas distintas |

**MASE** suele definirse respecto a un **naive** (p. ej. $\hat{y}_t = y_{t-1}$) o **naive estacional** ($\hat{y}_t = y_{t-s}$). Valores menores que 1 indican que el modelo mejora el baseline.

---

### 9. Baselines que siempre deben evaluarse

Antes de modelos complejos, incluya:

1. **Naive:** último valor observado.
2. **Naive estacional:** valor de hace $s$ periodos.
3. **Media móvil** de corto plazo.

Si ARIMA o XGBoost no superan claramente estos baselines en test temporal, el problema puede ser **poca señal**, **horizonte mal elegido** o **data leakage** en algún feature.

---

### 10. Plantillas Python (esquema)

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# y: Series con índice DatetimeIndex y frecuencia inferida o explícita
model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = model.fit(disp=False)
fc = res.get_forecast(steps=24)
mean_fc = fc.predicted_mean
ci = fc.conf_int()
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

### 11. Laboratorio (según sílabo)

- **NTB 1 —** Series temporales con Prophet y features + modelos ML (XGBoost/LightGBM).  
- **NTB 2 —** Series temporales con ARIMA/SARIMA y diagnóstico de residuos.

---

### 12. Profundización: notación ARIMA/SARIMA y Ljung–Box

Un proceso **ARMA** estacionario puede escribirse como en la sección 5.1. **ARIMA** incorpora integración $(1-B)^d$. **SARIMA** añade la parte estacional con retardo $B^s$.

**Ljung–Box** evalúa si las autocorrelaciones muestrales de los residuos hasta cierto máximo rezago son conjuntamente compatibles con ruido blanco. Rechazar $H_0$ sugiere **dependencia residual** → revisar órdenes, estacionalidad o necesidad de variables exógenas (`SARIMAX` con columnas exógenas).

---

### 13. Validación temporal con `TimeSeriesSplit`

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    # entrenar y evaluar
```

Nunca use `KFold` aleatorio sobre filas indexadas por tiempo si los features incluyen **lags** o **estadísticos de ventana** que mezclan información temporal.

---

### 14. Feature matrix para ML tabular (ejemplo mínimo)

```python
import pandas as pd

df = df.sort_index()
df["lag1"] = df["y"].shift(1)
df["lag7"] = df["y"].shift(7)
df["roll7"] = df["y"].rolling(7).mean()
df["dow"] = df.index.dayofweek
df = df.dropna()
X = df[["lag1", "lag7", "roll7", "dow"]]
y = df["y"]
```

---

### 15. Errores frecuentes

| Error | Consecuencia |
|-------|--------------|
| Mezclar orden temporal en split | Métricas irreales; modelo no desplegable |
| Filtrar outliers sin entender estacionalidad | Eliminar picos legítimos (Black Friday, calor) |
| Ignorar festivos en retail / transporte | Residuos estructurados en fechas clave |
| Usar MAPE con ceros o valores cercanos a cero | Métricas explosivas o engañosas |
| Ajustar SARIMA sin fijar o inferir bien $s$ | Estacionalidad mal especificada |

---

### 16. Casos de uso del temario (detalle)

| Dominio | Patrón temporal | Modelo típico |
|---------|-----------------|---------------|
| Retail | ventas diarias/semanales | SARIMA, Prophet, ML + lags y promociones |
| Energía | demanda con ciclo día/noche | SARIMA con $s$ adecuado (p. ej. 48 en media hora) o ML con calendario |
| Web / SaaS | tráfico con tendencia + weekly | Prophet; ML con Fourier + lags |
| Finanzas (intro) | volatilidad agrupada | Modelos GARCH fuera de alcance; ARIMA en retornos con cautela |

**Estadístico vs ML:** los modelos clásicos suelen ser **parsimoniosos** y facilitan intervalos basados en supuestos del modelo. El ML destaca con **muchas covariables** y no linealidades, a costa de más trabajo de **ingeniería de features** y **validación temporal** rigurosa.

---

### 17. Plantilla ML con `TimeSeriesSplit` + LightGBM

```python
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for tr, va in tscv.split(X):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[va])
    scores.append(mean_absolute_error(y.iloc[va], pred))
print("MAE medio (CV temporal):", float(np.mean(scores)))
```

---

### 18. Horizonte de pronóstico y recursión

En predicción **multi-paso**, puede usarse estrategia **directa** (un modelo por horizonte) o **recursiva** (alimentar predicciones como lags futuros). La recursiva acumula **error**; la directa exige más datos y modelos. Esta distinción afecta a la comparación justa entre ARIMA (típicamente recursiva en implementaciones estándar) y ML (a menudo directa por horizonte).

---

### 19. Intervalos de predicción vs incertidumbre del modelo

- **Intervalos paramétricos** (SARIMAX): asumen forma del error y correcta especificación.
- **Conformal prediction** y **bootstrap** de residuos: alternativas en la práctica cuando el modelo es aproximado.

En negocio, comunicar **banda** de error es tan importante como el punto de pronóstico.

---

### 20. Checklist antes de entregar un pronóstico

1. ¿Índice temporal **ordenado** y con **frecuencia** explícita?  
2. ¿Train/val/test **temporales**?  
3. ¿Baselines naive comparados?  
4. ¿Residuos sin estructura (si hay modelo clásico)?  
5. ¿Métrica alineada con el coste del error (sobre- vs sub-estimar)?  

---

### 21. Suavizado exponencial (puente hacia ETS y ARIMA)

El **suavizado exponencial simple** actualiza el nivel $\ell_t$ con:

$$\ell_t = \alpha y_t + (1-\alpha)\ell_{t-1}, \qquad 0 < \alpha \le 1$$

Valores altos de $\alpha$ reaccionan rápido a cambios (más varianza en el pronóstico); valores bajos producen series suavizadas. Las extensiones **Holt** (tendencia) y **Holt–Winters** (estacionalidad) son familias **ETS** (*Error, Trend, Seasonality*) que se mapean a ciertos ARIMA equivalentes en casos lineales. Conocer ETS ayuda a entender **por qué** ARIMA y métodos de suavizado a menudo compiten en benchmarks de series univariadas.

---

### 22. Variables exógenas (`SARIMAX`)

Cuando hay regresores $x_{t}$ observables (promociones, clima), **SARIMAX** extiende SARIMA con términos exógenos en la ecuación de media. La validación temporal sigue siendo obligatoria y los regresores futuros deben estar **disponibles** en el horizonte de pronóstico (o modelarse a su vez).

---

### 23. Series múltiples (*hierarchical forecasting* — mención)

En retail con miles de SKU, los pronósticos deben ser **coherentes** (la suma de hijos coincide con el padre). Existen métodos de **reconciliación** (*bottom-up*, *top-down*, *MinT*) que están fuera del detalle de esta sesión pero explican por qué el “mejor modelo por serie” no basta a escala operativa.

---

### 24. Trabajo con frecuencias irregulares

Si los timestamps no son equiespaciados, muchas rutinas clásicas fallan o requieren **re-muestreo** (p. ej. a diario con agregación). La re-muestreo introduce **agujeros** y **alias** temporal: documentar la regla (suma vs media, *forward-fill* prohibido para targets, etc.).

---

## Referencias bibliográficas principales

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.  
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.  
3. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.  
4. Cleveland, R. B., et al. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3–73.  
