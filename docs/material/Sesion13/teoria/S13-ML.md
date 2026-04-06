---
layout: default
---
# Semana 13: Interpretabilidad de Modelos


Modelos complejos (Random Forest, XGBoost, redes) logran alta precisión pero son “cajas negras”. En salud, finanzas o cumplimiento normativo (p. ej. GDPR) se necesitan explicaciones. Esta sesión cubre interpretabilidad global y local, intrínseca y post-hoc.

---

## Logro de la sesión

Explicar predicciones con técnicas globales y locales, entendiendo fundamentos, usos en industria y limitaciones.

---

## Enfoques

- **Global vs local:** comportamiento promedio del modelo vs una predicción concreta.
- **Intrínseco vs post-hoc:** modelo interpretable por diseño vs métodos aplicados después (SHAP, LIME).

---

## Importancia de características (árboles y más)

### Impureza (MDI)

Suma de reducciones de impureza ponderadas por muestras en nodos donde se usa la variable.

**Limitaciones:** sesgo hacia variables con muchas categorías; no indica dirección; correlaciones pueden distorsionar.

### Permutación

Importancia como aumento medio del error al permutar la variable:

$$
\text{Importancia}(j) = \frac{1}{K} \sum_{k=1}^K (\text{error}_{\text{perm}} - \text{error}_{\text{original}})
$$

Recomendado en scikit-learn: `permutation_importance`.

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(X_train, y_train)
result = permutation_importance(model, X_test, y_test, n_repeats=10)
sorted_idx = result.importances_mean.argsort()
```

---

## PDP e ICE

### Partial Dependence Plot (PDP)

$$
\text{PDP}(x_s) = \mathbb{E}_{x_c}[f(x_s, x_c)] \approx \frac{1}{n} \sum_{i=1}^n f(x_s, x_c^{(i)})
$$

### ICE

Una curva por instancia al variar $x_s$; curvas cruzadas sugieren interacción.

### Limitaciones

Correlación entre $x_s$ y $x_c$ puede producir regiones poco realistas al marginalizar.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_train, ["duration", "amount"], kind="average", grid_resolution=20
)
# kind="individual" para ICE
```

---

## SHAP

Basado en **valores de Shapley** (teoría de juegos): reparto equitativo de la contribución de cada feature a la predicción vs valor base.

Propiedades: eficiencia (suman al valor predicho), simetría, linealidad, dummy.

### Aproximaciones

- **Kernel SHAP:** agnóstico, más costoso.
- **Tree SHAP:** rápido en árboles.
- **Deep SHAP:** redes.

### Visualizaciones

Summary plot, dependence plot, force plot, waterfall.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

## LIME

Explicación local: perturbar alrededor de $x$, ponderar por proximidad y ajustar modelo interpretable (p. ej. Lasso).

Kernel de proximidad típico:

$$
\pi_x(z) = \exp\left(-\frac{d(x,z)^2}{\sigma^2}\right)
$$

**Pros:** agnóstico, rápido. **Contras:** inestabilidad, sensibilidad al tamaño de vecindad.

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=["bueno", "malo"],
    mode="classification",
)
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
```

---

## Comparación de métodos

| Método | Ámbito | Modelo-agnóstico | Velocidad | Interpretación |
| :--- | :--- | :--- | :--- | :--- |
| Permutación | Global | Sí | Rápida | Ranking variables |
| PDP | Global | Sí | Media | Relación parcial |
| ICE | Local/Global | Sí | Media | Heterogeneidad |
| SHAP | Local/Global | Depende | Rápida (Tree) | Contribuciones aditivas |
| LIME | Local | Sí | Media | Modelo local |

**Guía rápida:** permutación para ranking; PDP/ICE para forma funcional; SHAP (TreeSHAP) como estándar; LIME como alternativa local rápida.

---

## Aplicaciones

- **Finanzas:** scoring y explicaciones a cliente (force plot).
- **Salud:** PDP de factores de riesgo para el médico.
- **RRHH:** importancia y equidad entre grupos.

---

## Caso German Credit (esquema)

XGBoost; importancia por permutación; PDP en variables clave; SHAP global y local para un rechazo; comunicación en lenguaje natural; revisión de sesgos.

---

## Limitaciones y buenas prácticas

- SHAP asume aditividad; interacciones se reparten.
- LIME: repetir para comprobar estabilidad.
- PDP con correlación fuerte: considerar ALE o PDP condicionales.
- Explicación $\neq$ causalidad.
- Validar con expertos de dominio.

---

## Resumen

- Interpretabilidad global/local e intrínseca/post-hoc.
- Permutación, PDP/ICE, SHAP y LIME cubren la mayoría de necesidades.
- TreeSHAP es referencia en modelos de árboles.
- Cumplimiento y confianza del negocio requieren explicaciones auditables.
