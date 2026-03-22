---
layout: default
---

# Cheatsheet: Introducción al Machine Learning
**Autor:** Carlos César Sánchez Coronel  

---

## Tipos de aprendizaje

| Tipo | Datos | Objetivo típico |
| :--- | :--- | :--- |
| **Supervisado** | Etiquetas | Regresión ($y$ continua), clasificación ($y$ discreta) |
| **No supervisado** | Sin etiquetas | Clustering, reducción de dimensión, reglas de asociación |
| **Por refuerzo** | Recompensas del entorno | Políticas de decisión secuencial |
| **Semi / autosupervisado** | Mixto o pseudo-etiquetas | NLP, visión con pretext tasks |

---

## Paramétrico vs no paramétrico

* **Paramétrico:** forma funcional fija (p. ej. regresión lineal); complejidad no crece con $n$.
* **No paramétrico:** más flexible (k-NN, árboles); complejidad puede crecer con datos → riesgo de overfitting.

---

## Flujo de un proyecto ML (macro)

1. Definición del problema y métrica de negocio  
2. Datos: recolección, EDA, calidad  
3. Feature engineering y split train/val/test  
4. Modelo baseline + modelos candidatos  
5. Validación y ajuste de hiperparámetros  
6. Interpretación (si aplica) y despliegue (MLOps)  

---

## Roles frecuentes

| Rol | Enfoque |
| :--- | :--- |
| **Data Scientist** | Modelado, experimentación, comunicación |
| **ML Engineer** | Pipelines, despliegue, escalado |
| **Data Engineer** | Ingesta, almacenes, calidad de datos |

---

## Código mínimo: primer modelo (sklearn)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
```

---

## Puntos críticos

* Alinear la **métrica técnica** con el **objetivo de negocio** (no usar solo accuracy si hay desbalance).
* **Generalización** importa más que el score en entrenamiento.
* Documentar datos, versiones y decisiones (trazabilidad).

> *“El ML aprende de datos; si los datos están mal o el problema está mal planteado, el modelo no lo arregla.”*
