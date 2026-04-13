---
layout: default
---
# Sesión 14: Evaluación en producción y ciclo de vida del ML


### 1. Logro de la sesión

Diseñar evaluación **offline vs online**, **experimentos A/B**, detectar **drift** de datos y de concepto, y situar el **ciclo de vida** (entrenamiento, despliegue, monitoreo, retraining) sin confundir métricas de laboratorio con impacto de negocio.

---

### 2. Brecha offline / online

**Offline:** AUC, RMSE en histórico.  
**Online:** CTR, conversión, revenue — afectados por UX, sesgo de selección, exploración.

---

### 3. A/B testing (marco)

1. Definir hipótesis y métrica primaria.  
2. Asignación aleatoria usuarios a A/B.  
3. Tamaño muestral para poder estadístico (conceptual).  
4. Evitar *peeking* continuo sin corrección.

**Referencias:** Kohavi et al. experimentación controlada en web.

---

### 4. Drift

- **Covariate drift:** $P(X)$ cambia.  
- **Concept drift:** $P(Y|X)$ cambia.

**Detección:** comparar distribuciones de features y calidad de predicción en ventanas temporales; tests KS/PSI (conceptual).

---

### 5. Monitoreo y ciclo de vida

Pipeline: datos → entrenamiento → validación → despliegue → **monitoreo** → alertas → **retraining** versionado.

**Riesgos:** entrenar sobre datos sesgados por política del modelo actual (*feedback loop*).

---

### 6. Python (simulación)

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
assign = rng.binomial(1, 0.5, size=10_000)  # 0=A, 1=B
conv_a = rng.binomial(1, 0.08, size=10_000)
conv_b = rng.binomial(1, 0.085, size=10_000)
rate_a = conv_a[assign==0].mean()
rate_b = conv_b[assign==1].mean()
print(rate_a, rate_b)
```

---

### 7. Laboratorio (según sílabo)

- **NTB 1 —** Simulación A/B y conclusión.  
- **NTB 2 —** Drift simulado y plan de retraining.



### 8. Diseño experimental: tamaño de muestra (intuición)

La varianza de la diferencia de tasas $\hat{p}_B - \hat{p}_A$ escala aproximadamente como $\sqrt{\frac{p(1-p)}{n}}$ — por eso experimentos con pocos usuarios no detectan mejoras pequeñas pero importantes.

### 9. PSI (*Population Stability Index*) — idea

Compara distribución de un score o feature entre ventana de entrenamiento y ventana reciente:

$$ \mathrm{PSI} = \sum_i (A_i - E_i)\ln\frac{A_i}{E_i} $$

Valores altos sugieren **drift** fuerte; umbrales dependen de la política interna (típicamente 0.1–0.25 como reglas heurísticas industriales).

### 10. Ciclo de retraining: checklist

1. ¿Empeoró métrica de negocio o solo offline?  
2. ¿El drift es en **inputs** o en la relación etiqueta–features?  
3. ¿Hay datos nuevos etiquetados suficientes?  
4. Versionar **datos** y **modelo** (MLflow, DVC, etc.).

### 11. Simulación de drift en features

```python
# Desplazar media de una feature en test sintético
X_drift = X_test.copy()
X_drift["f1"] = X_drift["f1"] + 2.0
proba_orig = model.predict_proba(X_test)[:, 1]
proba_drift = model.predict_proba(X_drift)[:, 1]
print("Δ AUC aprox:", roc_auc_score(y_test, proba_drift))
```


---

## Referencias bibliográficas principales

1. Kohavi, R., et al. (2009). Controlled experiments on the web. *DMKD*.  
2. Huyen, C. (2022). *Designing Machine Learning Systems*. O’Reilly.  
3. Gama, J., et al. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*.  
