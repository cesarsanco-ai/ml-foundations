---
layout: default
---

# Cheatsheet: EDA y Feature Engineering
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-02](../../../sesiones/sesion-02.md)

---

## Pipeline de EDA (resumen)

1. **Contexto:** diccionario de datos, tipos de variables, granularidad.  
2. **Calidad:** faltantes, duplicados, inconsistencias.  
3. **Limpieza:** tipos, imputación, outliers.  
4. **Univariado:** distribuciones, sesgos, escalas.  
5. **Bivariado / multivariado:** correlaciones, redundancia, interacciones.  
6. **Documentar** hallazgos y decisiones para el modelado.  

---

## Valores faltantes

| Estrategia | Cuándo |
| :--- | :--- |
| Eliminar fila/columna | Pocos faltantes o columna irrelevante |
| Imputación media/mediana | Numéricas, MCAR aproximado |
| Moda / categoría “Unknown” | Categóricas |
| Modelo de imputación (KNN, Iterative) | Patrones informativos en faltantes |

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median")
X_train_imp = imp.fit_transform(X_train_num)
X_test_imp = imp.transform(X_test_num)
```

---

## Codificación categórica

* **One-hot:** pocas categorías, sin orden natural.  
* **Ordinal / target encoding:** muchas categorías (con cuidado de leakage).  
* **Frequency encoding:** alta cardinalidad.  

```python
import pandas as pd
X = pd.get_dummies(X, columns=["city"], drop_first=True)
```

---

## Escalado

| Método | Uso típico |
| :--- | :--- |
| **StandardScaler** | KNN, SVM, redes, PCA |
| **MinMaxScaler** | Rangos acotados, algunas redes |
| **RobustScaler** | Muchos outliers |

**Regla:** `fit` solo en train; `transform` en test.

---

## Outliers

* Detectar: IQR, z-score robusto, aislamiento visual.  
* Tratar: cap (winsorize), transformación log, modelo robusto, o eliminar si error de medición.  

---

## Feature engineering rápido

* Ratios, agregaciones temporales, interacciones explícitas ($x_1 \cdot x_2$).  
* Binning cuando la relación es no lineal por tramos (con validación).  

---

## Split y leakage

* **Random split** si i.i.d.  
* **Estratificado** en clasificación desbalanceada.  
* **Temporal / grupal** si hay dependencia (tiempo, usuario, tienda).  

---

## Puntos críticos

* Un EDA deficiente → sesgos ocultos (p. ej. COMPAS) o modelos frágiles.  
* No usar estadísticas del **test** para decidir imputación o escalado sobre train.  
* Correlación alta entre features → multicolinealidad en modelos lineales.  

> *“Escuchar los datos antes de forzar el algoritmo.”*
