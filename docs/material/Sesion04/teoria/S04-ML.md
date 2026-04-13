---
layout: default
---
# Sesión 4: Regresión Logística y Balance de Datos

### 1. Logro de la sesión

Construir, interpretar y evaluar modelos de **clasificación binaria y multiclase** mediante **regresión logística**, integrando **regularización**, **técnicas de balanceo de clases** y **métricas** (incluidas ROC–PR) alineadas al **coste de error** del dominio aplicado.

---

### 2. Historia y línea temporal

| Periodo | Hito | Notas |
|---------|------|-------|
| **Finales s. XIX – 1920s** | Modelos de **dosis-respuesta** en toxicología/biología (Pearson, **Bliss**, Berkson) | Origen de transformaciones “logit” para probabilidades acotadas a $(0,1)$. |
| **1944–1955** | Formalización estadística de modelos lineales generalizados (camino hacia GLM) | La logística como **GLM binomial** con enlace logit es estándar en estadística. |
| **1970s en adelante** | Uso masivo en medicina, econometría discreta, biometría | Interpretación en **odds ratios** muy valorada en epidemiología. |
| **1990s–2000s** | **Machine learning**: fronteras lineales en espacio de características, kernelización (SVM), vs redes | La logística sigue siendo **baseline** fuerte y calibrable. |
| **Actualidad** | Regularización L1/L2 en alta dimensión; pipelines con **imbalanced-learn**; explicabilidad (odds, SHAP) | Base para *credit scoring*, fraude, marketing response models. |

**Referencia histórica:** Cox (1958) sobre modelos lineales en probabilidades; Hosmer–Lemeshow para regresión logística aplicada.

---

### 3. Marco teórico: regresión logística binaria

#### 3.1 Motivación: modelar una probabilidad

En clasificación binaria, $y \in \{0,1\}$. Una regresión lineal directa sobre $\hat{p}$ no restringe $[0,1]$. Se introduce la **función logística** (sigmoide):

$$ \sigma(z) = \frac{1}{1+e^{-z}} $$

El modelo **logístico** postula:

$$ P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^\top \boldsymbol{\beta}}} $$

#### 3.2 Logit y odds

Definiendo $\mathrm{odds} = \frac{p}{1-p}$:

$$ \log \frac{P(y=1 \mid \mathbf{x})}{1 - P(y=1 \mid \mathbf{x})} = \mathbf{x}^\top \boldsymbol{\beta} $$

Así, los coeficientes $\beta_j$ actúan de forma **aditiva en la escala del logit**; al exponenciar, cada unidad de $x_j$ **multiplica los odds** por $e^{\beta_j}$ (factor multiplicativo), *ceteris paribus*.

#### 3.3 Estimación: máxima verosimilitud

Bajo muestreo Bernoulli independiente, la log-verosimilitud es cóncava en $\boldsymbol{\beta}$; se maximiza con **IRLS** (mínimos cuadrados reponderados iterativos) o métodos cuasi-Newton. Por eso sklearn usa solvers como `lbfgs`, `liblinear`, `saga` (según penalización y tamaño).

#### 3.4 Regularización (Ridge / Lasso / Elastic Net sobre pesos)

**Marco:** igual que en regresión lineal penalizada, se añade término $\lambda \|\boldsymbol{\beta}\|_2^2$ (Ridge), $\lambda \|\boldsymbol{\beta}\|_1$ (Lasso) o mezcla, típicamente **sin** penalizar el intercepto si se usa `fit_intercept=True`.

**Ventajas de Ridge en logística:**

- Estabiliza coeficientes con **multicolinealidad** o muchas variables dummy correlacionadas.  
- Reduce **sobreajuste** en alta dimensión relativa a muestra.

**Ventajas de Lasso:**

- Promueve **sparse** coeficientes → selección de variables interpretables.

**Elastic Net:** compromiso cuando hay **grupos correlacionados** de predictores (Zou & Hastie, 2005).

En `sklearn`, `LogisticRegression` usa parámetro `C` como **inverso** de la fuerza de regularización: **C pequeño → más regularización** (convención opuesta a $\lambda$ en muchos textos de estadística).

---

### 4. Multiclase: One-vs-Rest vs Softmax

| Enfoque | Mecanismo | Ventajas | Limitaciones |
|---------|-----------|----------|--------------|
| **One-vs-Rest (OvR)** | Un clasificador binario por clase vs el resto | Simple, paralelizable | Fronteras inconsistentes si no se calibra |
| **Multinomial (softmax)** | Un solo modelo con función softmax sobre $K$ clases | Coherencia probabilística conjunta | Más costoso en $K$ grande |

`sklearn`: `multi_class='ovr'` vs `'multinomial'` (según solver).

#### 4.1 Detalle softmax (multinomial)

Para $K$ clases, se definen scores lineales $z_k(\mathbf{x}) = \mathbf{x}^\top \boldsymbol{\beta}_k$ y probabilidades:

$$ P(y=k \mid \mathbf{x}) = \frac{e^{z_k(\mathbf{x})}}{\sum_{j=1}^{K} e^{z_j(\mathbf{x})}} $$

La función es **invariante** a sumar la misma constante a todos los $z_k$; por eso se fija una referencia (p.ej. $\boldsymbol{\beta}_K=\mathbf{0}$) en algunas parametrizaciones. La pérdida típica es la **entropía cruzada multinomial**, convex en $\boldsymbol{\beta}$ en formulación estándar.

**Ventaja conceptual:** estima **jointly** todas las fronteras; **OvR** en cambio entrena $K$ problemas binarios independientes, lo que puede producir regiones de decisión **incoherentes** si no se recalibra (en la práctica a menudo funciona bien como aproximación rápida).

---

### 5. Umbral de decisión y costes asimétricos

La regla $\hat{y}=1$ si $\hat{p} \geq \tau$ con $\tau=0.5$ solo es óptima si **FP y FN** tienen el mismo coste y no hay desbalance extremo bajo criterios bayesianos simples. En la práctica:

- **Fraude / salud:** a menudo se baja $\tau$ para subir **recall** (aceptar más FP).  
- **Marketing** (spam): a veces se sube $\tau$ para proteger inbox (menos FP).

**Curvas ROC y PR** permiten elegir $\tau$ **post hoc** según el trade-off deseado (Fawcett, 2006; Davis & Goadrich, 2006 para PR en datos desbalanceados).

---

### 6. Balanceo de datos (temario ampliado)

#### 6.1 Por qué aparece el problema

Con **clase minoritaria muy pequeña**, minimizar pérdida global o accuracy puede llevar al clasificador trivial “siempre mayoritaria”. Los modelos lineales y muchos otros **calibran mal** la probabilidad de la clase rara sin tratamiento.

#### 6.2 Oversampling: SMOTE

**SMOTE** (Chawla et al., 2002) sintetiza ejemplos minoritarios interpolando entre vecinos en el espacio de características:

$$ \mathbf{x}_{\mathrm{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{\mathrm{nn}} - \mathbf{x}_i), \quad \lambda \in (0,1) $$

**Ventajas:** aumenta densidad local de la minoritaria sin duplicar exactamente.  
**Riesgos:** puede crear ejemplos en regiones donde la clase no debería estar (ruido); sensible a outliers; **debe aplicarse solo sobre train** (evitar leakage).

#### 6.3 Undersampling

Elimina observaciones de la **mayoritaria** para equilibrar proporciones.

**Ventajas:** rápido, reduce tiempo de entrenamiento.  
**Desventajas:** **pérdida de información** de la mayoritaria; puede empeorar varianza si se descarta señal útil.

#### 6.4 Pesos de clase (`class_weight`)

Asigna mayor coste a errores en clase minoritaria sin inventar filas:

```python
LogisticRegression(class_weight="balanced")
```

**Ventajas:** no altera dimensionalidad del dataset; implementación simple.  
**Limitaciones:** no aumenta datos reales; puede no bastar si la minoritaria es casi nula.

#### 6.5 ¿Cuándo usar qué? (guía)

| Escenario | Primera línea típica | Comentario |
|-----------|----------------------|------------|
| Minoritaria pequeña pero con señal | `class_weight` + métricas adecuadas | Menos invasivo |
| Minoritaria muy escasa | SMOTE (train) + validación cuidadosa | Vigilar sobreajuste sintético |
| Dataset enorme mayoritario | Undersampling aleatorio o **edited** methods | Vigilar pérdida de información |
| Necesidad de calibración de probabilidades | Calibración (`CalibratedClassifierCV`) tras el modelo | Sesión 8/13 |

---

### 7. Métricas: matriz de confusión y derivadas

#### 7.1 Definiciones

|  | Predicho 1 | Predicho 0 |
|--|------------|------------|
| **Real 1** | TP | FN |
| **Real 0** | FP | TN |

- **Exactitud:** $(TP+TN)/(TP+TN+FP+FN)$ — engañosa con desbalance.  
- **Precisión:** $TP/(TP+FP)$ — “de los que predije positivo, cuántos lo eran”.  
- **Recall (sensibilidad):** $TP/(TP+FN)$ — “de los positivos reales, cuántos detecté”.  
- **Especificidad:** $TN/(TN+FP)$.

#### 7.2 F1 y media armónica

$$ F_1 = \frac{2 \cdot \mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}} $$

Penaliza desequilibrios entre precisión y recall (a diferencia de la media aritmética).

#### 7.3 ROC–AUC y PR–AUC

- **AUC–ROC:** probabilidad de que un par (positivo, negativo) esté ordenado correctamente por score; **invariante** a umbrales pero puede ser **optimista** con fuerte desbalance.  
- **PR–AUC:** foco en la clase positiva; a menudo más informativa cuando la positiva es rara.

---

### 8. Plantilla base en Python (`scikit-learn`)

#### 8.1 Imports

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    RocCurveDisplay,
)
```

#### 8.2 Clasificador logístico con escalado

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

log_reg = Pipeline([
    ("scaler", StandardScaler()),
    (
        "clf",
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])
log_reg.fit(X_train, y_train)
y_score = log_reg.predict_proba(X_test)[:, 1]
y_pred = log_reg.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_score))
print("PR-AUC:", average_precision_score(y_test, y_score))
```

**Objetos útiles:**

- `predict_proba`: probabilidades estimadas por clase.  
- `decision_function`: logit lineal antes de sigmoide (clase binaria).  
- `coef_`, `intercept_` en el paso `clf`.

#### 8.3 Cambiar umbral manualmente

```python
from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(y_test, y_score)
# elegir thr según objetivo de negocio, p.ej. recall mínimo
```

```python
tau = 0.35
y_pred_tau = (y_score >= tau).astype(int)
```

#### 8.4 SMOTE dentro del pipeline (concepto)

La librería **imbalanced-learn** permite `imblearn.pipeline.Pipeline` para que el resampling ocurra **solo en cada fold de CV** (Sesión 8). Esquema típico:

```python
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline
# pipe = ImbPipeline([("smote", SMOTE()), ("scaler", StandardScaler()), ("clf", LogisticRegression(...))])
```

*(Instalación: `imbalanced-learn`; usar siempre con validación anidada para evitar optimismo.)*

#### 8.5 Calibración de probabilidades (puente conceptual)

Las probabilidades $\hat{p}$ de la logística pueden estar **mal calibradas** (p.ej. predice 0.7 pero solo el 40 % de esos casos es positivo). Métodos como **Platt scaling** o **isotonic regression** sobre un conjunto de **calibración** ajustan la salida (`CalibratedClassifierCV` en sklearn). Importa en **decisiones económicas** donde el umbral se elige por coste esperado bayesiano.

---

### 9. Selección de métricas según contexto (temario)

| Contexto | Énfasis típico |
|----------|----------------|
| **Detección de amenaza / fraude** | Recall alto (aceptar revisar FP) |
| **Filtrado de spam** | Precisión alta (no bloquear correos legítimos) |
| **Modelos con scores para ranking** | AUC, PR en validación temporal si aplica |

---

### 10. Laboratorio (según sílabo)

- **NTB 1 —** Balanceo de clases (oversampling, undersampling, pesos) y efecto en métricas.  
- **NTB 2 —** Regresión logística: entrenamiento, umbral, regularización y evaluación.

---

## Referencias bibliográficas principales

1. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.  
2. McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.  
3. Chawla, N. V., et al. (2002). SMOTE: synthetic minority over-sampling technique. *JAIR*, 16, 321–357.  
4. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.  
5. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*.  
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.  
