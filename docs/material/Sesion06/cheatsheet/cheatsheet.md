---
layout: default
---

# Cheatsheet: Árboles de Decisión y Random Forest
**Autor:** Carlos César Sánchez Coronel  

---

## Árbol de decisión

* Particiones recursivas del espacio de features.  
* **Clasificación:** Gini, entropía, information gain.  
* **Regresión:** reducción de MSE en nodos.  

$$
G = 1 - \sum_i p_i^2 \quad \text{(Gini)}
$$

---

## Hiperparámetros clave

| Parámetro | Efecto |
| :--- | :--- |
| `max_depth` | Limita complejidad; menos overfitting |
| `min_samples_leaf` | Hojas más grandes → más regularización |
| `max_features` | Aleatoriedad y diversidad (en RF) |

---

## Random Forest (bagging de árboles)

* Entrena muchos árboles con **bootstrap** y submuestreo de features.  
* **Predicción:** voto mayoritario (clasificación) o media (regresión).  
* Reduce **varianza** respecto a un solo árbol.  

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
```

---

## Importancia de variables

* **`feature_importances_`:** basada en impureza (sesgos posibles).  
* Complementar con **`permutation_importance`** en test.  

---

## Comparación rápida

| Aspecto | Árbol solo | Random Forest |
| :--- | :--- | :--- |
| Varianza | Alta | Baja |
| Sesgo | Bajo–medio | Medio |
| Interpretabilidad | Alta (árbol pequeño) | Media (ensamble) |
| Velocidad predicción | Muy rápida | Rápida |

---

## Puntos críticos

* Pueden **sobreajustar** si el árbol es muy profundo sin restricciones.  
* Capturan **interacciones** sin especificarlas manualmente.  
* Menos sensible al escalado que KNN/SVM, pero orden de categorías arbitrarias puede importar si se codifica mal.  

> *“RF es el caballo de batalla tabular antes de pasar a boosting.”*
