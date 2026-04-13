---
layout: default
---
# Sesión 13: Interpretabilidad de modelos


### 1. Logro de la sesión

Aplicar **interpretabilidad global y local** (importancia, permutación, **SHAP**, **LIME**) con conciencia de **limitaciones**, **estabilidad** y riesgo de **confundir correlación con causalidad**.

---

### 2. Taxonomía

| Tipo | Pregunta |
|------|----------|
| Global | ¿Qué variables importan en general? |
| Local | ¿Por qué esta instancia? |

**Modelos intrínsecamente interpretables:** regresión corta, árbol poco profundo.  
**Post-hoc:** aplican a cajas negras.

---

### 3. Importancia por permutación

Permutar $X_j$ en validación y medir caída de métrica. **Ventaja:** modelo-agnóstica. **Coste:** reentrenamiento no necesario pero múltiples evaluaciones.

---

### 4. SHAP (Shapley)

Valores de Shapley asignan a cada feature una contribución aditiva local que cumple axiomas de eficiencia, simetría, *dummy* (Lundberg & Lee, 2017).

**TreeSHAP:** eficiente para árboles. **KernelSHAP:** aproximación modelo-agnóstica.

---

### 5. LIME

Perturba entradas alrededor de $\mathbf{x}$ y ajusta un modelo lineal local ponderado.

**Riesgo:** inestabilidad si vecindario mal elegido.

---

### 6. Visualización

- **Summary plot** SHAP (beeswarm).  
- **Dependence plots** para interacciones.

---

### 7. Plantillas Python

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
```

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
```

---

### 8. Laboratorio (según sílabo)

- **NTB 1 —** Importancia global + SHAP summary.  
- **NTB 2 —** Explicaciones locales SHAP/LIME y sesgos.



### 9. Sesgos en explicaciones

- **Correlación fuerte:** SHAP puede repartir importancia arbitrariamente entre features colineales.  
- **Datos de entrenamiento sesgados:** explicaciones “correctas” para un modelo injusto perpetúan la injusticia.

### 10. LIME: pseudocódigo

1. Generar perturbaciones $\mathbf{x}'$ alrededor de $\mathbf{x}$.  
2. Ponderar por proximidad a $\mathbf{x}$.  
3. Ajustar modelo lineal ponderado que aproxime $f(\mathbf{x}')$ localmente.

### 11. SHAP interacción

`shap.TreeExplainer` puede calcular valores de interacción para pares de features — útil cuando el efecto de $x_j$ depende de $x_k$.

### 12. Plantilla LIME para tabular

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=list(X_train.columns),
    class_names=["neg", "pos"],
    mode="classification",
)
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
```


---

## Referencias bibliográficas principales

1. Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.).  
2. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.  
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?”. *KDD*.  
