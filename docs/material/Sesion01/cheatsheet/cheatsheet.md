---
layout: default
---

# Cheatsheet: Introducción al Machine Learning
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-01](../../../sesiones/sesion-01.md)

---

## 1. Definiciones clave

| Concepto | Definición |
|----------|------------|
| **IA** | Sistemas que realizan tareas que requieren inteligencia humana |
| **Machine Learning (ML)** | Máquinas aprenden patrones a partir de datos sin programación explícita |
| **Deep Learning** | Subárea del ML que usa redes neuronales profundas |

---

## 2. Tipos de aprendizaje

| Tipo | Características | Ejemplos |
|------|----------------|----------|
| **Supervisado** | Datos etiquetados | Regresión, Clasificación |
| **No supervisado** | Sin etiquetas | Clustering, PCA, t-SNE |
| **Por refuerzo** | Agente + entorno + recompensas | Juegos, robótica |
| **Semi-supervisado** | Mezcla etiquetados + no etiquetados | NLP, clasificación de imágenes |

---

## 3. Modelos paramétricos vs no paramétricos

| Paramétricos | No paramétricos |
|--------------|-----------------|
| Forma fija | Forma flexible |
| Complejidad no crece con datos | Complejidad crece con datos |
| Ej: Regresión lineal, Regresión logística | Ej: k-NN, Árboles de decisión, SVM |
| Menos sobreajuste | Más riesgo de sobreajuste |

---

## 4. Metodología CRISP-DM (6 fases iterativas)

```
1. Comprensión del negocio
2. Comprensión de los datos
3. Preparación de los datos
4. Modelado
5. Evaluación
6. Despliegue
```

---

## 5. Roles profesionales

| Rol | Función principal |
|-----|-------------------|
| **Data Engineer** | Infraestructura y pipelines de datos |
| **Data Scientist** | Exploración, modelos, insights |
| **ML Engineer** | Implementación, escalado, producción |
| **MLOps** | CI/CD, monitoreo, gobernanza de modelos |

---

## 6. Funciones de coste (las más importantes)

| Modelo | Función de coste |
|--------|------------------|
| Regresión lineal (MSE) | $$J(\theta) = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$ |
| Regresión logística | $$J(\theta) = -\frac{1}{n} \sum [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$ |
| SVM (Hinge loss) | $$J(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum \max(0, 1 - y_i (\mathbf{w}^T x_i + b))$$ |
| K-Means | $$J = \sum_{k} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$ |

---

## 7. Optimización

| Método | Fórmula | Uso |
|--------|---------|-----|
| **Gradiente descendente** | $$\theta := \theta - \alpha \nabla_\theta J(\theta)$$ | General |
| **SGD** | Mismo principio, por batches | Datasets grandes |
| **Solución cerrada** | $$\hat{\theta} = (X^T X)^{-1} X^T y$$ | Solo regresión lineal |

---

## 8. Regularización (evitar overfitting)

| Técnica | Penalización | Fórmula |
|---------|--------------|---------|
| **Ridge (L2)** | Cuadrática | $$J(\theta) = MSE + \lambda \sum \theta_j^2$$ |
| **Lasso (L1)** | Absoluta | $$J(\theta) = MSE + \lambda \sum |\theta_j|$$ |
| **Elastic Net** | L1 + L2 | Combinación de ambas |

---

## 9. Complejidad algorítmica

| Modelo | Entrenamiento | Inferencia |
|--------|---------------|------------|
| Regresión lineal (cerrada) | $O(n^3)$ | $O(p)$ |
| Regresión lineal (GD) | $O(n \cdot iter)$ | $O(p)$ |
| k-NN | $O(1)$ | $O(N \cdot n)$ |
| Árboles de decisión | $O(N \cdot n \cdot \log N)$ | $O(profundidad)$ |
| Random Forest | $O(árboles \times N \cdot n \cdot \log N)$ | $O(árboles \times profundidad)$ |

> $n$ = número de características, $N$ = número de muestras, $p$ = parámetros

---

## 10. Métricas por tipo de problema

### Clasificación
| Métrica | Fórmula | Cuándo usarla |
|---------|---------|----------------|
| Precisión | $$\frac{VP}{VP+FP}$$ | Minimizar falsos positivos |
| Recall | $$\frac{VP}{VP+FN}$$ | Minimizar falsos negativos |
| F1-score | $$2 \cdot \frac{precisión \cdot recall}{precisión + recall}$$ | Balance entre ambas |
| AUC-ROC | Área bajo la curva | Clasificación binaria |

### Regresión
| Métrica | Fórmula |
|---------|---------|
| MAE | $$\frac{1}{n} \sum |y_i - \hat{y}_i|$$ |
| RMSE | $$\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$$ |
| $R^2$ | $$1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$ |

### Clustering
| Métrica | Rango | Interpretación |
|---------|-------|----------------|
| Coeficiente de silueta | $[-1, 1]$ | Cercano a 1 = bueno |
| Índice Davies-Bouldin | $[0, \infty)$ | Menor = mejor |

---

## 11. Hardware para ML clásico

| Hardware | Cuándo usar | Modelos compatibles |
|----------|-------------|---------------------|
| **CPU** | Suficiente para casi todo | Regresión, árboles, k-NN, k-means, DBSCAN, Prophet |
| **GPU** | Solo datasets grandes | XGBoost, LightGBM, Random Forest profundo |
| **TPU** | No necesario en ML clásico | Solo Deep Learning |

> **Regla de oro:** Para ML clásico, empieza con CPU. Solo considera GPU si tienes millones de filas y usas Gradient Boosting.

---

## 12. Línea de tiempo de modelos clave

| Década | Modelos |
|--------|---------|
| **1950-60** | Perceptrón, k-NN, Regresión lineal/logística |
| **1970-80** | Árboles (ID3, C4.5), Naive Bayes, k-means |
| **1990** | SVM, Random Forest, DBSCAN, Gradient Boosting |
| **2000** | XGBoost, LightGBM, Prophet |
| **2010+** | Deep Learning (AlexNet, ResNet, LSTM, BERT, GPT) |

---

## 13. Aplicaciones típicas por industria

| Industria | Aplicación |
|-----------|-------------|
| Retail | Sistemas de recomendación (Netflix, Amazon, Spotify) |
| Finanzas | Detección de fraude |
| Industria | Mantenimiento predictivo |
| Salud | Diagnóstico por imágenes |
| Transporte | Predicción de tráfico, demanda |

