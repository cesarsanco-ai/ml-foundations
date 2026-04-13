---
layout: default
---
# Sesión 1: Introducción al Machine Learning

### 1. Logro de la sesión

Comprender un panorama inicial del flujo de datos a nivel de ingenieria y ciencia de datos, con los modelos principales del Machine Learning.

---

### 2. Fundamentos conceptuales

#### Definición IA / ML / DL

**Marco teórico:** Basado en la **jerarquía de Russell y Norvig** (AI: A Modern Approach). La IA se divide en cuatro enfoques: sistemas que piensan como humanos, actúan como humanos, piensan racionalmente o actúan racionalmente. ML es un subconjunto del enfoque racional (aprendizaje a partir de datos). DL es un subconjunto de ML basado en representaciones jerárquicas.

**Referentes:** 
- Arthur Samuel (1959): "Campo de estudio que da a las computadoras la capacidad de aprender sin ser explícitamente programadas"
- Tom Mitchell (1997): "Un programa aprende si su rendimiento en tareas T mejora con la experiencia E"

#### Tipos de aprendizaje

**Marco teórico:** Basado en la **clasificación de Murphy (Machine Learning: A Probabilistic Perspective)**:

| Tipo | Fundamento | Autores clave |
|------|-----------|---------------|
| Supervisado | Teoría de la decisión estadística | Vapnik (Statistical Learning Theory) |
| No supervisado | Teoría de la información / clustering | Hartigan (1975) |
| Refuerzo | Procesos de decisión de Markov (MDP) | Sutton & Barto (1998) |
| Semisupervisado | Teoría del aprendizaje semi-supervisado | Chapelle, Scholkopf, Zien (2006) |

#### Métodos paramétricos vs no paramétricos

**Marco teórico:** Basado en la **teoría de la complejidad estadística** (Vapnik-Chervonenkis theory):

- **Paramétricos:** Asumen una distribución fija (ej: Gaussiana). La complejidad del modelo no crece con n. Sesgo alto, varianza baja.
- **No paramétricos:** No asumen forma funcional fija. La complejidad crece con n. Sesgo bajo, varianza alta.

**Referente:** Wasserman (All of Statistics, 2004)

---

### 3. Fundamentos matemáticos

#### Funciones de coste

**Marco teórico:** Basado en la **teoría de la estimación** (Casella & Berger, Statistical Inference). La función de coste cuantifica la discrepancia entre predicción $\hat{y}$ y valor real $y$.

| Función | Problema | Fundamento |
|---------|----------|------------|
| MSE | Regresión | Riesgo cuadrático, óptimo para errores Gaussianos |
| Entropía cruzada | Clasificación | Máxima verosimilitud para distribución Bernoulli |
| Hinge loss | SVM | Aproximación convexa del error de clasificación 0-1 |

**Referente:** Bishop (Pattern Recognition and Machine Learning, 2006)

#### Optimización

**Marco teórico:** Basado en la **optimización convexa** (Boyd & Vandenberghe):

- **Gradiente descendente:** Método iterativo de primer orden. Converge a óptimo global si la función es convexa.
- **SGD:** Robbins-Monro (1951). Permite procesar datasets que no caben en memoria.
- **Solución cerrada (Normal Equation):** Aplica solo a regresión lineal. Usa pseudoinversa de Moore-Penrose.

#### Regularización

**Marco teórico:** Basado en la **teoría de la complejidad estadística** y el **principio de la navaja de Occam**:

| Técnica | Fundamento matemático | Efecto |
|---------|----------------------|--------|
| Ridge (L2) | Norma euclidiana | Contrae coeficientes, no los lleva a cero |
| Lasso (L1) | Norma L1 | Selección de características (escasa) |
| Elastic Net | Combinación L1 + L2 | Maneja correlaciones altas |

**Referente:** Hastie, Tibshirani, Friedman (The Elements of Statistical Learning)

#### Overfitting y generalización

**Marco teórico:** Basado en la **teoría de Vapnik-Chervonenkis (VC theory)**. El error de generalización se descompone como:

$$Error_{generalización} = Error_{entrenamiento} + \epsilon_{complejidad}$$

El **sesgo** mide la capacidad del modelo, la **varianza** mide la sensibilidad a los datos.

---

### 4. Historia y evolución

**Marco teórico:** Basado en la **historiografía de la IA** (Nilsson, The Quest for Artificial Intelligence). Se organiza en "olas" o "estaciones" del ML:

| Década | Paradigma dominante | Hitos clave |
|--------|---------------------|-------------|
| 1950s-60s | Conexionismo temprano | Perceptrón (Rosenblatt) |
| 1970s-80s | Simbolismo / árboles | ID3 (Quinlan), backpropagation (Rumelhart) |
| 1990s | Teoría del aprendizaje estadístico | SVM (Vapnik), Random Forest (Breiman) |
| 2000s | Ensembles y boosting | XGBoost (Chen), LightGBM (Microsoft) |
| 2010s-presente | Deep Learning y transformers | AlexNet, ResNet, BERT, GPT |

---

### 5. Metodología y ciclo de vida

#### CRISP-DM

**Marco teórico:** Estándar de facto desde 1999, desarrollado por SPSS, NCR, DaimlerChrysler. Se basa en la **metodología de proyectos en cascada adaptada con iteraciones**.

| Fase | Propósito | Entregable típico |
|------|-----------|-------------------|
| Comprensión del negocio | Definir objetivos desde el negocio | Plan de proyecto |
| Comprensión de los datos | Recolectar y explorar datos | Reporte de calidad de datos |
| Preparación de datos | Limpiar y transformar | Dataset final |
| Modelado | Seleccionar y entrenar modelos | Modelo candidato |
| Evaluación | Validar contra objetivos | Decisión de deploy |
| Despliegue | Poner en producción | Sistema operativo |

#### Roles profesionales

**Marco teórico:** Basado en el **modelo de madurez de datos de Gartner** y los frameworks de competencias de la industria:

| Rol | Dominio primario | Responsabilidad clave |
|-----|-----------------|----------------------|
| Data Engineer | Ingeniería de datos | Pipelines ETL/ELT, Data Lakes/Warehouses |
| Data Scientist | Estadística/ML | Exploración, modelado, insights |
| ML Engineer | Ingeniería software + ML | API, escalado, versionado de modelos |
| MLOps Engineer | DevOps + ML | CI/CD, monitoreo, gobernanza |

---

### 6. Aspectos computacionales

#### Complejidad algorítmica

**Marco teórico:** Basado en la **teoría de la complejidad computacional (notación Big O)**. Distingue entre:

- **Complejidad de entrenamiento:** Lo que importa para desarrollo
- **Complejidad de inferencia:** Lo que importa para producción

| Modelo | Entrenamiento | Inferencia | Referente |
|--------|---------------|------------|-----------|
| Regresión lineal (normal) | O(n³) | O(p) | Strassen (1969) |
| k-NN | O(1) | O(n·p) | Cover & Hart (1967) |
| Árboles | O(n·p·log n) | O(profundidad) | Quinlan (1986) |
| Random Forest | O(k·n·p·log n) | O(k·profundidad) | Breiman (2001) |

#### Hardware: CPU / GPU / TPU

**Marco teórico:** Basado en la **arquitectura de computadoras (Flynn's taxonomy)** y la **ley de Moore** con evolución hacia aceleradores:

| Hardware | Arquitectura | Ideal para | Limitación |
|----------|--------------|------------|------------|
| CPU | Control + ALU, pocos núcleos potentes | Tareas secuenciales, ML clásico | Paralelismo limitado |
| GPU | Miles de núcleos simples | Operaciones matriciales (DL, XGBoost) | Overhead de transferencia |
| TPU | Matriz de multiplicación optimizada | Tensor operations (DL exclusivo) | No útil para ML clásico |

**Referente:** Hennessy & Patterson (Computer Architecture: A Quantitative Approach)

---

### 7. Evaluación (Métricas)

**Marco teórico:** Basado en la **teoría de la decisión estadística** y el **análisis de costes**. La métrica debe elegirse en función del problema de negocio, no del modelo.

| Tipo de problema | Métricas clave | Fundamento |
|-----------------|---------------|------------|
| Clasificación | Precisión, Recall, F1, AUC-ROC | Análisis de la matriz de confusión (Provost & Fawcett) |
| Regresión | MAE, RMSE, R² | Teoría del error (Hastie et al.) |
| Clustering | Silhouette, Davies-Bouldin, Inercia | Validación interna de clusters (Rousseeuw, 1987) |

**Principio fundamental:** "Lo que no se mide, no se mejora" — la métrica debe reflejar el coste real de los errores en el contexto de la aplicación.

---

### 8. Aplicaciones reales

**Marco teórico:** Basado en el **aprendizaje basado en casos (Case-Based Learning)** y la **transferencia de conocimiento**. Los casos deben:

1. Ser **auténticos** (problemas reales de la industria)
2. Cubrir **múltiples sectores**
3. Conectar cada caso con **tipos de aprendizaje y métricas**

| Industria | Caso de uso | Tipo de aprendizaje | Métrica crítica |
|-----------|-------------|---------------------|-----------------|
| Retail/E-commerce | Recomendación | Supervisado / Filtrado colaborativo | Precision@K, Recall@K |
| Finanzas | Detección de fraude | Supervisado (clasificación desbalanceada) | Recall (sobre F1) |
| Manufactura | Mantenimiento predictivo | Supervisado (series temporales) | RMSE, Precisión |
| Salud | Diagnóstico médico | Supervisado (clasificación de imágenes) | Sensibilidad/Especificidad |
| Energía | Predicción de demanda | Supervisado (series temporales) | MAE, MAPE |

---

### 9. Apéndice: línea temporal del ML en la industria (visión panorámica)

| Periodo | Tendencia | Implicación práctica |
|---------|-----------|----------------------|
| **2010–2015** | *Big data* + Hadoop/Spark; **deep learning** revoluciona visión | Pipelines distribuidos; GPUs accesibles |
| **2015–2020** | **TensorFlow/PyTorch**; transfer learning | Equipos mixtos investigación–ingeniería |
| **2020–2026** | **MLOps**, feature stores, gobernanza, LLMs | Menos “solo notebook”, más productos reproducibles |

#### 9.1 Plantilla mínima de exploración en Python (sesiones posteriores la amplían)

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
print(df.describe())
```

Este patrón (`import` → objeto `DataFrame`/`ndarray` → métodos `describe`, `plot`) es la base sobre la que se montan los laboratorios NTB.

#### 9.2 Cómo usar este documento para estudiar

1. Leer cada **Marco teórico** intentando reformular con tus palabras.  
2. Para cada tabla de métricas, escribir **un ejemplo de negocio** donde esa métrica sea la adecuada.  
3. Antes de cada laboratorio, trazar el **flujo de datos** desde archivo bruto hasta métrica final.

---

## Referencias bibliográficas principales

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
5. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Pearson.
6. Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.
7. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.