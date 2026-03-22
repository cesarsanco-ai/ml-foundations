---
layout: default
---

# Cheatsheet: Interpretabilidad de Modelos
**Autor:** Carlos César Sánchez Coronel  

---

## Tipos de explicación

| | Global | Local |
| :--- | :--- | :--- |
| **Pregunta** | ¿Qué importa en general? | ¿Por qué esta predicción? |
| **Ejemplos** | PDP, permutación, SHAP summary | LIME, SHAP force/waterfall |

---

## Importancia por permutación

* Rompe una feature y mide subida del error → ranking **fiable** y **model-agnostic**.  

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importances = r.importances_mean
```

---

## PDP e ICE

* **PDP:** efecto medio de $x_j$ marginalizando el resto.  
* **ICE:** curvas por instancia; cruces → interacción.  
* Cuidado si hay **correlación fuerte** entre features (regiones irreales).  

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(model, X_train, features=[0, 1], kind="average")
```

---

## SHAP

* Valores **Shapley**: reparto aditivo de la contribución vs valor base.  
* **TreeSHAP:** rápido en árboles/XGBoost/LightGBM.  

```python
import shap

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test)
shap.summary_plot(sv, X_test)
```

---

## LIME

* Aproxima el modelo con uno simple **en vecindad** de $x$.  
* Rápido y agnóstico; puede ser **inestable** (repetir muestreo).  

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train.values, feature_names=cols, mode="classification")
exp = explainer.explain_instance(x0, model.predict_proba, num_features=8)
```

---

## Comparación rápida

| Método | Scope | Nota |
| :--- | :--- | :--- |
| Permutación | Global | Baseline sólido |
| PDP/ICE | Global/local | Forma funcional |
| SHAP | Ambos | Estándar en tabular con árboles |
| LIME | Local | Explicación rápida |

---

## Puntos críticos

* Explicación $\neq$ **causalidad**.  
* **GDPR / auditoría:** documentar método y limitaciones.  
* Validar narrativa con **expertos de dominio**.  

> *“Si no puedes explicarlo, quizá no debería decidir solo en riesgo alto.”*
