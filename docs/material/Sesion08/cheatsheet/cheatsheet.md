---
layout: default
---

# Cheatsheet: Evaluación y Validación de Modelos
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-08](../../../sesiones/sesion-08.md)

---

## Bias vs varianza

| Situación | Train | Test | Diagnóstico |
| :--- | :--- | :--- | :--- |
| **Underfitting** | Alto error | Alto error | Modelo muy simple |
| **Overfitting** | Muy bajo | Alto | Demasiada complejidad o leakage |

---

## Validación cruzada

* **k-Fold:** estima performance con varianza controlada.  
* **StratifiedKFold:** proporciones de clase estables en cada fold.  
* **TimeSeriesSplit:** sin mezclar futuro en train.  
* **GroupKFold:** mismo grupo (usuario, paciente) no cruza folds.  

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(scores.mean(), "+/-", scores.std())
```

---

## Búsqueda de hiperparámetros

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator, param_distributions, n_iter=30, cv=5, scoring="f1", n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
best = search.best_estimator_
```

* **GridSearch:** exhaustivo, costoso.  
* **Random / Bayesian (Optuna):** mejor en muchas dimensiones.  

---

## Data leakage (fugas)

* Escalar / imputar **dentro de pipeline** con CV.  
* No incluir información del test en features (p. ej. estadísticas globales sin separar).  
* Targets derivados del futuro en series temporales.  

---

## Curvas de aprendizaje

* Si train y val convergen **altos** → más datos o más complejidad.  
* Si brecha grande → regularizar o más datos.  

---

## Selección de modelo

| Criterio | Herramienta |
| :--- | :--- |
| Comparar modelos | CV con misma partición |
| Significancia (opcional) | Test estadístico paired (con cautela) |
| Negocio | Costo asimétrico → métrica custom |

---

## Puntos críticos

* El **test** solo al final o una vez consciente de múltiples comparaciones.  
* **Reproducibilidad:** `random_state`, versiones fijadas.  
* Métrica en CV alineada con producción (p. ej. AUC-PR en desbalance extremo).  

> *“Un modelo excelente en notebook con leakage es un fracaso en producción.”*
