---
layout: default
---
# Sesión 14: Evaluación en producción y ciclo de vida del ML

### 1. Logro de la sesión

Diseñar la **evaluación** de sistemas de ML en tres planos: **offline** (holdout, validación cruzada cuando aplique), **online** (A/B, *quasi-experimentos*), y **monitoreo continuo** (*drift*, calidad de datos, SLOs). Comprender **drift de covariables** y **drift de concepto**, el **ciclo de vida** (entrenamiento → despliegue → observación → *retraining*), y riesgos como **feedback loops** y **sesgo de selección**. Relacionar métricas de laboratorio con **KPIs de negocio** sin confundir correlación temporal con impacto causal del modelo.

---

### 2. Historia y contexto

| Periodo | Hito |
|---------|------|
| **1990s–2000s** | Experimentación web aleatorizada; métricas de conversión |
| **2009** | Kohavi et al.: marco de experimentación controlada a escala |
| **2010s** | MLOps: CI/CD para modelos; versionado de datos (DVC) y artefactos (MLflow) |
| **2014+** | Encuestas y algoritmos de **concept drift** en streams |
| **2020s** | *Model cards*, *datasheets*, gobernanza; coste de inferencia y sostenibilidad |

**Lectura:** un modelo excelente en offline puede **fracasar online** por cambio de población, UX, o porque la métrica offline no era un **proxy** válido del objetivo de negocio.

---

### 3. Brecha offline / online

| Plano | Qué se mide | Fortalezas | Debilidades |
|-------|-------------|------------|-------------|
| **Offline** | AUC, log-loss, RMSE en histórico | Barato, reproducible | No captura efectos de UI, sesgo de exposición, competencia con reglas heurísticas |
| **Shadow** | Mismo tráfico, modelo no actúa | Sin riesgo para usuario | No mide cambio de comportamiento por la recomendación |
| **A/B** | CTR, retención, revenue | Oro estándar para impacto | Coste de muestra, ética, interferencias |

**Sesgo de selección:** solo observamos etiquetas para items **mostrados** (publicidad, recomendación). El modelo entrenado con logs puede perpetuar **exclusión** de ítems nunca mostrados (*inverse propensity scoring* y diseño experimental son remedios parciales).

---

### 4. Diseño de experimentos A/B (marco)

#### 4.1 Pasos esenciales

1. **Hipótesis** y **métrica primaria** acordada con negocio (una sola primaria evita *p-hacking* informal).  
2. **Asignación aleatoria** de unidades (usuarios, sesiones, mercados) a control $A$ y tratamiento $B$ con probabilidades fijas.  
3. **Tamaño muestral** y **poder** (*power*): detectar un **MDE** (*minimum detectable effect*) con alta probabilidad.  
4. **Análisis** al cierre del experimento o con reglas secuenciales válidas (evitar *peeking* sin corrección).  
5. **Segmentación post-hoc** con cautela (múltiples tests inflan falsos positivos).

#### 4.2 Unidades y *network effects*

Si el tratamiento en un usuario afecta a otros (red social, mercados bipartitos), la unidad independiente puede violarse → **experimentos cluster** o diseños específicos.

#### 4.3 Métricas ratio y varianza

Para tasas de conversión, la varianza de $\hat{p}_B - \hat{p}_A$ escala aproximadamente como $\sqrt{p(1-p)/n}$ por brazo (bajo i.i.d. ideal). Por eso mejoras de 0,1 pp pueden requerir **millones** de eventos.

---

### 5. Drift: tipos e intuición

#### 5.1 Covariate drift (*prior shift*)

Cambia la distribución de entradas: $P(X)$ evoluciona, pero $P(Y|X)$ se mantiene (supuesto fuerte en la práctica rara vez verificable al 100 %).

**Ejemplos:** nueva cámara en inspección visual; formulario con campo opcional que deja de cumplimentarse; cambio demográfico de usuarios.

#### 5.2 Concept drift

Cambia la relación etiqueta–features: $P(Y|X)$ cambia aunque $P(X)$ sea estable.

**Ejemplos:** nueva regulación; competidor altera precios; estacionalidad no modelada; fraude adaptativo.

#### 5.3 Label drift

Cambia $P(Y)$; puede coexistir con otros tipos. Importa para **calibración** de probabilidades y umbrales.

---

### 6. Detección de drift (operativa)

| Técnica | Qué compara | Comentario |
|---------|-------------|------------|
| **KS** (Kolmogorov–Smirnov) | Distribución de una feature entre ventana referencia y reciente | Univariado; múltiples tests requieren corrección |
| **PSI** (*Population Stability Index*) | Bines de un score o feature entre esperado y actual | Heurístico; umbrales 0,1–0,25 son folklore industrial |
| **Chi-cuadrado** | Variables categóricas | Similar uso a PSI con conteos |
| **Monitoreo de rendimiento** | Caída de AUC/precision en ventana reciente si hay etiquetas tardías | Oro cuando es posible |

**PSI** típico:

$$\mathrm{PSI} = \sum_i (A_i - E_i)\ln\frac{A_i}{E_i}$$

donde $E_i$ y $A_i$ son proporciones esperadas y actuales en el bin $i$. Evitar divisiones por cero con suavizado.

---

### 7. Monitoreo y ciclo de vida (MLOps)

#### 7.1 Pipeline conceptual

**Datos** → validación de esquema → **entrenamiento** → evaluación offline → **registro** de modelo y datos → **despliegue** (canary, blue/green) → **monitoreo** (datos, predicciones, negocio) → **alertas** → **retraining** o rollback.

#### 7.2 Versionado

- **Código** (Git).  
- **Datos** (hash, DVC).  
- **Hiperparámetros y métricas** (MLflow, Weights & Biases).  
- **Artefactos** (modelo serializado, vocabularios).

#### 7.3 *Feedback loop*

Si el modelo influye en qué datos futuros se recolectan (recomendación, precios dinámicos), el siguiente entrenamiento puede **reforzar** políticas pasadas. Mitigación: **exploración** controlada, logs de **propensión**, re-etiquetado activo.

---

### 8. Estrategias de despliegue

| Estrategia | Descripción | Riesgo residual |
|------------|-------------|-----------------|
| **Big bang** | sustitución total | Alto si falla |
| **Canary** | pequeño % de tráfico al nuevo modelo | Requiere métricas en tiempo casi real |
| **Shadow** | nuevo modelo en paralelo sin efecto | No mide reacción del usuario |
| **Blue/Green** | conmutación rápida entre versiones | Infraestructura duplicada |

---

### 9. Python: simulación simple de A/B

```python
import numpy as np

rng = np.random.default_rng(0)
n = 10_000
assign = rng.binomial(1, 0.5, size=n)  # 0=A, 1=B
# Tasas verdaderas ligeramente distintas (ilustrativo)
conv_a = rng.binomial(1, 0.080, size=n)
conv_b = rng.binomial(1, 0.085, size=n)
rate_a = conv_a[assign == 0].mean()
rate_b = conv_b[assign == 1].mean()
print("Tasa A:", rate_a, "Tasa B:", rate_b)
```

Para inferencia formal: test de proporciones, *bootstrap*, o modelos bayesianos.

---

### 10. Laboratorio (según sílabo)

- **NTB 1 —** Simulación A/B y conclusión.  
- **NTB 2 —** Drift simulado y plan de retraining.

---

### 11. Tamaño de muestra e intuición estadística

La varianza de la diferencia de medias o proporciones decrece con $\sqrt{n}$. Por ello:

- Experimentos cortos con poco tráfico **no** detectan mejoras pequeñas pero económicamente relevantes.  
- **MDE** debe fijarse en unidades de negocio (“+0,5 pp de conversión”), no solo “significancia”.

---

### 12. *Peeking* y análisis interino

Mirar resultados cada día y parar cuando $p < 0{,}05$ **infla** la tasa de error tipo I. Soluciones: **pruebas secuenciales** (SPRT, *always valid inference*), o reglas pre-registradas de parada.

---

### 13. Simulación de drift en features (esquema)

```python
from sklearn.metrics import roc_auc_score

# Ilustración: desplazar una feature en "test bajo drift"
X_drift = X_test.copy()
X_drift.iloc[:, 0] = X_drift.iloc[:, 0] + 2.0
proba_orig = model.predict_proba(X_test)[:, 1]
proba_drift = model.predict_proba(X_drift)[:, 1]
print("AUC original:", roc_auc_score(y_test, proba_orig))
print("AUC bajo drift sintético:", roc_auc_score(y_test, proba_drift))
```

*(Asume clasificación binaria y `predict_proba` disponible.)*

---

### 14. Ciclo de retraining: checklist

1. ¿Empeoró una **métrica de negocio** o solo offline?  
2. ¿El drift es en **inputs**, en **etiquetas**, o en **ambos**?  
3. ¿Hay **suficientes etiquetas nuevas** o hay que activar etiquetado?  
4. ¿El rollback está **automatizado** y probado?  
5. ¿Se documentó **por qué** se reentrena (ticket, versión de datos)?  

---

### 15. SLOs y alertas para ML

**SLO** (*Service Level Objective*): p. ej. latencia p99 de inferencia < 120 ms; fracción de respuestas nulas < 0,1 %.

**Alertas de datos:** esquema, rango de valores, tasa de nulos, PSI de features clave.

**Alertas de modelo:** caída de calibración, desviación de score distribution, aumento de outliers en *embedding* space (si aplica).

---

### 16. Model cards y documentación

Un **model card** (Mitchell et al.) resume: propósito, datos de entrenamiento, métricas por subgrupo, limitaciones conocidas, consideraciones éticas. No es burocracia: reduce **deuda técnica** cuando el equipo cambia.

---

### 17. Coste total de propiedad (TCO)

Más allá del AUC: coste de **inferencia** (GPU/CPU), almacenamiento de features, pipelines de datos, y horas humanas de mantenimiento. Un modelo 1 % mejor pero 10× más caro puede no pasar el filtro de negocio.

---

### 18. Errores frecuentes

| Error | Consecuencia |
|-------|--------------|
| Optimizar métrica offline desalineada con negocio | “Éxito” técnico sin impacto |
| Retrain automático sin gates de calidad | Modelo peor en producción |
| Ignorar drift de etiquetas tardías | Señales falsas de mejora/empeora |
| Experimentos sin poder estadístico | Decisiones ruidosas |

---

### 19. Relación con gobernanza y cumplimiento

Según sector (salud, crédito), pueden aplicarse requisitos de **trazabilidad**, **explicabilidad documentada** y **pruebas de no discriminación**. El ciclo de vida debe integrar revisión legal/compliance **antes** del despliegue masivo.

---

### 20. Checklist de salida a producción

1. ¿Tests de código + tests de datos (*great expectations*, etc.)?  
2. ¿Contrato de entrada del modelo versionado?  
3. ¿Plan de **rollback** y *feature flags*?  
4. ¿Dashboards de monitoreo con owners?  
5. ¿Plantilla de **post-mortem** si hay incidente?  

---

### 21. Tests estadísticos clásicos en A/B (tasas)

Para proporciones binarias por brazo, el **test z de dos proporciones** (con o sin continuidad) es el estándar didáctico. Bajo hipótesis nula de igualdad de tasas:

$$z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A}+\frac{1}{n_B}\right)}}$$

donde $\hat{p}$ es la proporción combinada. En la práctica moderna también se usan **bootstrap** y métodos bayesianos para comunicar **intervalos creíbles** de la mejora — más intuitivos para stakeholders que el solo $p$-valor.

---

### 22. *CUPED* y reducción de varianza

**CUPED** (*Controlled-experiment Using Pre-Experiment Data*) ajusta la métrica del experimento usando covariables pre-experimento correlacionadas con el outcome, reduciendo varianza y el tamaño muestral necesario. Es estándar en grandes plataformas web; requiere datos históricos limpios y cuidado con fugas temporales.

---

### 23. Validación de esquema y calidad de datos en producción

Antes de inferencia batch o online:

- **Esquema:** tipos, rangos, columnas obligatorias.  
- **Drift de nulos:** pico de missing puede romper *imputación* entrenada.  
- **Desfase temporal:** features calculados con timestamps incorrectos generan *leakage* o pérdida de señal.

Herramientas tipo **Great Expectations**, **Evidently**, o tests custom en el pipeline CI reducen incidentes “silenciosos”.

---

### 24. Incidentes: respuesta operativa

1. **Detectar** (alerta de SLO o caída de métrica).  
2. **Mitigar** (rollback, tráfico al modelo anterior, desactivar feature dañina).  
3. **Diagnosticar** (logs de versión de datos/modelo, cambios de upstream).  
4. **Corregir** y **documentar** en post-mortem sin culpar personas individuales — enfocarse en sistemas.

---

### 25. Sombras y canarios: métricas a vigilar

En **canary**, además de la métrica de negocio primaria:

- latencia p50/p99,  
- tasa de errores HTTP,  
- distribución de scores (KS vs baseline),  
- tasa de *fallback* cuando el modelo devuelve nulo.

Un canary “exitoso” en conversión pero con latencia inaceptable **no** debe promoverse.

---

### 26. Coste de oportunidad del experimento

Cada usuario en el brazo control podría haber recibido una política mejor si el tratamiento resulta ganador — y viceversa. Los comités de experimentación ponderan **duración** vs **riesgo** y a veces emplean **asignación no uniforme** (p. ej. 90/10) en fases exploratorias.

---

### 27. Continuidad entre entrenamiento y servicio (*training-serving skew*)

Discrepancias entre código de **feature engineering** en batch de train y en **servicio online** (distinta librería, orden de operaciones, redondeo) degradan rendimiento sin drift real en $P(X)$. Mitigación: **feature store** compartido, contenedores idénticos, pruebas de consistencia end-to-end.

---

### 28. SLI, SLO y presupuesto de error

Un **SLI** (*Service Level Indicator*) es la medida bruta (p. ej. fracción de requests con latencia < 100 ms). Un **SLO** fija el objetivo (≥ 99,9 % en 30 días). El **presupuesto de error** es $1 - \text{SLO}$: cuánto “fallo” acumulado se tolera antes de congelar releases y priorizar estabilidad. Adaptar esta mentalidad a pipelines de ML evita despliegues continuos que rompen inferencia silenciosamente.

---

### 29. Observabilidad: logs estructurados de inferencia

Registrar por request (con políticas de privacidad): versión del modelo, hash del conjunto de features, latencia por etapa (preproceso → modelo → postproceso), y flags de *fallback*. Cuando un cliente reporta un error, estos campos permiten **reproducir** la traza sin adivinar.

---

### 30. Ética del experimento: consentimiento y daño

Los A/B tests en producto deben considerar **usuarios vulnerables**, **efectos irreversibles** (precios personalizados, salud) y **transparencia** cuando la ley lo exija. Un marco de revisión ética interna (aunque sea ligero) reduce riesgo reputacional y legal — complementario al poder estadístico.

---

## Referencias bibliográficas principales

1. Kohavi, R., Longbotham, R., Sommerfield, D., & Henne, R. M. (2009). Controlled experiments on the web: survey and practical guide. *Data Mining and Knowledge Discovery*, 18(1), 140–181.  
2. Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.  
3. Huyen, C. (2022). *Designing Machine Learning Systems*. O’Reilly.  
4. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37.  
5. Mitchell, M., et al. (2019). Model cards for model reporting. *FAccT*.  
6. Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS*.  
