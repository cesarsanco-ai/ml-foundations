## 📚 PLAN DE DATASETS POR SEMANA

| Semana | Tema | Dataset | Tipo de Dato | Problema ML | Shape | ¿Qué se aprende? | Complejidad |
|:------:|------|---------|--------------|-------------|-------|------------------|:-----------:|
| **01** | Introducción al ML | **Iris** | 📊 Estructurado | Clasificación Multiclase (3 clases) | 150 × 4 | Conceptos básicos: features, target, train/test split, accuracy | ⭐ 1 |
| **02** | EDA y Feature Engineering | **Adult Census** | 📊 Estructurado | Clasificación Binaria (ingreso >50K) | 48,842 × 14 | Análisis de missing values, encoding de categóricas, correlaciones, detección de outliers | ⭐⭐ 2 |
| **03** | Regresión Lineal y Regularización | **California Housing** | 📊 Estructurado | Regresión (precios vivienda) | 20,640 × 8 | Regresión lineal, Ridge, Lasso, ElasticNet, R², RMSE | ⭐⭐ 2 |
| **04** | Regresión Logística y Balanceo | **Credit Scoring** | 📊 Estructurado | Clasificación Binaria (default) | 16,714 × 10 | Regresión logística, AUC-ROC, matrices de confusión, SMOTE, class_weight | ⭐⭐ 2 |
| **05** | kNN, Naive Bayes, SVM | **Digits** (imagen como tabla) | 📊 Estructurado (píxeles) | Clasificación Multiclase (10 dígitos) | 1,797 × 64 | kNN (distancia euclidiana), Naive Bayes (independencia), SVM (kernel lineal/rbf) | ⭐⭐⭐ 3 |
| **06** | Árboles, Random Forest, Ensambles | **Bank Churn** | 📊 Estructurado | Clasificación Binaria (abandono) | 5,000 × 19 | Árbol de decisión (Gini/entropía), Random Forest (bagging), feature importance | ⭐⭐ 2 |
| **07** | Gradient Boosting | **Covertype** | 📊 Estructurado | Clasificación Multiclase (7 clases) | 581,012 × 54 | XGBoost, LightGBM, CatBoost, early stopping, learning_rate, n_estimators | ⭐⭐⭐⭐ 4 |
| **08** | Evaluación y Validación | **Bank Marketing** | 📊 Estructurado | Clasificación Binaria (depósito) | 45,211 × 16 | Cross-validation (k-fold), curvas ROC/PR, overfitting/underfitting, métricas múltiples | ⭐⭐ 2 |
| **09** | Clustering y PCA | **Fashion MNIST** | 🖼️ No Estructurado (imagen) | No supervisado (10 clusters reales) | 70,000 × 28×28 | K-Means, DBSCAN, PCA (reducción dimensional), silhouette score, inercia | ⭐⭐⭐ 3 |
| **10** | Series de Tiempo | **Wine Quality** (con índice temporal simulado) | 📊 Estructurado | Regresión (calidad) | 6,497 × 11 | ARIMA, Prophet, tendencias, estacionalidad, rolling windows | ⭐⭐⭐ 3 |
| **11** | Optimización con C++ | **MNIST** (formato numérico) | 📊 Estructurado (píxeles) | Clasificación Multiclase (10 dígitos) | 70,000 × 784 | Implementación eficiente de distancias (kNN) o matrices (PCA) en C++ | ⭐⭐⭐⭐ 4 |
| **12** | ML Cloud y Edge | **SUPERB (KS)** | 🎵 No Estructurado (audio) | Clasificación Binaria (12 comandos) | Audio (~1-2 segundos) | Despliegue en Cloud (AWS/GCP), inferencia en Edge (Raspberry Pi), formatos ONNX | ⭐⭐⭐⭐ 4 |
| **13** | Interpretabilidad de Modelos | **Credit Scoring** (mismo de semana 4) | 📊 Estructurado | Clasificación Binaria (default) | 16,714 × 10 | SHAP, LIME, Feature importance, Partial Dependence Plots (PDP), reglas extraíbles | ⭐⭐⭐ 3 |
| **14** | Despliegue y Observabilidad | **Food101** (versión pequeña) | 🖼️ No Estructurado (imagen) | Clasificación Multiclase (101 clases) | 101,000 imágenes | API REST (FastAPI/Flask), Docker, monitoreo (drift, latency, logs) | ⭐⭐⭐⭐ 4 |

---

## 📋 Resumen Rápido por Semana

| Semana | Dataset | ¿Qué problema resuelve? |
|--------|---------|-------------------------|
| 01 | Iris | "Hola mundo" del ML - clasificación de flores |
| 02 | Adult Census | ¿Quién gana más de 50K al año? |
| 03 | California Housing | ¿Cuánto cuesta una casa en California? |
| 04 | Credit Scoring | ¿Un cliente pagará su préstamo? |
| 05 | Digits | ¿Qué dígito escribió el usuario? |
| 06 | Bank Churn | ¿Un cliente abandonará el banco? |
| 07 | Covertype | ¿Qué tipo de árbol crece aquí? (Big Data) |
| 08 | Bank Marketing | ¿Contratará un depósito a plazo? |
| 09 | Fashion MNIST | ¿Qué prenda de ropa es? (sin etiquetas) |
| 10 | Wine Quality | ¿Qué calidad tendrá este vino? (temporal) |
| 11 | MNIST | Optimización de algoritmos clásicos |
| 12 | SUPERB (KS) | ¿Qué comando de voz dijo? (Edge) |
| 13 | Credit Scoring | ¿Por qué el modelo rechazó al cliente? |
| 14 | Food101 | ¿Qué comida es esta? (API desplegada) |

---

## 🎯 Portafolio Final del Alumno (6 proyectos estrella)

| Proyecto | Semanas | Dataset | Entregable |
|----------|---------|---------|------------|
| **1. Clasificación básica** | 1, 4-6 | Bank Churn | Notebook con EDA + Random Forest + métricas |
| **2. Regresión** | 2-3 | California Housing | Notebook con feature engineering + Ridge/Lasso |
| **3. Big Data + Boosting** | 7-8 | Covertype | Comparativa XGBoost vs LightGBM vs CatBoost |
| **4. No supervisado** | 9 | Fashion MNIST | Clustering con K-Means + visualización PCA |
| **5. Edge + Audio** | 12 | SUPERB (KS) | Modelo desplegado en API + inferencia |
| **6. Interpretabilidad** | 13 | Credit Scoring | Dashboard con SHAP + LIME + reglas |

---

## 💡 Notas importantes

1. **Semana 5 (Digits)**: Aunque son imágenes, se usan como tabla de 64 píxeles → perfecto para ML clásico.
2. **Semana 10 (Series de Tiempo)**: Wine Quality no tiene fecha real; se puede simular un índice temporal o usar `fetch_openml(data_id=42712)` para dataset con fecha.
3. **Semana 11 (C++)**: MNIST en formato numérico (784 columnas) es ideal para implementar kNN o PCA desde cero en C++.
4. **Semana 12 (Edge)**: SUPERB (KS) es el único dataset de audio funcional actualmente sin scripts deprecados.

