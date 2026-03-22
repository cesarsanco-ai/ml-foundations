---
layout: default
---

# Cheatsheet: Clasificación I — Regresión Logística y Desbalance
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-04](../../../sesiones/sesion-04.md)

---

## Regresión logística

* **Probabilidad:** $\hat{p} = \sigma(z) = \frac{1}{1+e^{-z}}$, $z = \beta_0 + \sum \beta_j x_j$  
* **Log-odds:** $\text{logit}(\hat{p}) = \log\frac{\hat{p}}{1-\hat{p}}$ (lineal en $x$)  
* **Multiclase:** OvR o **softmax**  

---

## Pérdida

**Log-loss (cross-entropy binaria):**

$$
L = -\frac{1}{N}\sum_i \big[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\big]
$$

Regularización L1/L2/Elastic como en regresión.

---

## Métricas (clases no balanceadas)

| Métrica | Fórmula corta | Cuándo |
| :--- | :--- | :--- |
| **Precision** | VP/(VP+FP) | Costo alto de FP |
| **Recall** | VP/(VP+FN) | Costo alto de FN |
| **F1** | $2PR/(P+R)$ | Balance P y R |
| **AUC-ROC** | Capacidad de ranking | Comparar modelos |

---

## Umbral (threshold)

* Por defecto 0.5 **no** es óptimo si el costo de FP/FN es asimétrico.  
* Ajustar maximizando F-beta o utilidad de negocio en validación.  

```python
from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(y_val, y_proba)
# elegir thr según objetivo
```

---

## Balanceo de clases

| Técnica | Idea |
| :--- | :--- |
| **class_weight='balanced'** | Pesa clases en el entrenamiento |
| **Undersampling** mayoritaria | Rápido; pierde datos |
| **Oversampling / SMOTE** minoritaria | Más muestras sintéticas; validar sin leakage |
| **Métricas adecuadas** | F1, AUC-PR a veces mejor que AUC-ROC |

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
```

---

## Pipeline recomendado

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(X_train, y_train)
```

---

## Puntos críticos

* **Data leakage:** no balancear mezclando train y test; SMOTE solo en train.  
* **Accuracy** puede ser alta prediciendo siempre la mayoritaria.  
* Coeficientes logísticos = cambio en log-odds **asumiendo modelo bien especificado**.  

> *“La clase minoritaria suele ser la que importa en fraude, churn y salud.”*
