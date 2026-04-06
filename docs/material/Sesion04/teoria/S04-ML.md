---
layout: default
---
# Semana 4: Regresión Logística y Balanceo de Datos

## Logro de la sesión

El estudiante será capaz de **construir, interpretar y evaluar modelos de clasificación binaria y multiclase** mediante regresión logística, incorporando técnicas de balanceo de datos, selección de métricas adecuadas y ajuste de thresholds según objetivos de negocio.

---

## Problemática de negocio

- **Tipos de problemas de clasificación:**
  - **Binaria:** Dos clases (ej. fraude/no fraude, churn/no churn, aprobado/rechazado).
  - **Multiclase:** Más de dos clases (ej. clasificación de productos, tipos de clientes, diagnóstico médico múltiple).

- **Impacto del desbalance de clases:**
  Cuando la proporción de clases es muy desigual, modelos simples tienden a predecir siempre la clase mayoritaria, generando métricas engañosas (alta exactitud pero baja sensibilidad para la clase minoritaria).

- **Ejemplos clásicos:**
  - Detección de fraude financiero
  - Churn de clientes
  - Diagnóstico de enfermedades
  - Score crediticio
  - Clasificación de productos o comportamiento de clientes

- **Diferencias clave con regresión:**
  - **Regresión:** Output continuo $y \in \mathbb{R}$.
  - **Clasificación:** Output categórico $y \in \{0,1\}$ o $y \in \{1,...,K\}$.

---

## Modelado de clasificación

### Regresión logística binaria

Función de predicción:

$$ \hat{p} = P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \sum_{i=1}^{n} \beta_i x_i $$

- **Interpretación:** $\hat{p}$ es la probabilidad de pertenecer a la clase positiva.
- **Log-odds:**
  $$ \text{logit}(\hat{p}) = \log\frac{\hat{p}}{1-\hat{p}} = \beta_0 + \sum \beta_i x_i $$
  Cada $\beta_i$ representa el cambio en log-odds por unidad de incremento en $x_i$.

### Regresión logística multiclase

- **One-vs-Rest (OvR):** Se entrena un modelo binario por cada clase.
- **Softmax:** Probabilidad de clase $k$:
  $$ P(y=k|X) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad z_k = \beta_{0k} + \sum_i \beta_{ik} x_i $$

### Función de pérdida y optimización

- **Función de pérdida log-loss (cross-entropy):**
  $$ L(\beta) = - \frac{1}{N} \sum_{i=1}^N \big[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \big] $$

- **Optimización:** Se minimiza $L(\beta)$ usando **Gradiente Descendente** o variantes (SGD, Adam):
  $$ \beta \leftarrow \beta - \eta \nabla_\beta L(\beta) $$

- **Regularización:**

  La regularización controla overfitting y selecciona variables relevantes.

---

### Desbalanceo de datos

El desbalanceo de clases ocurre cuando una clase (mayoritaria) supera significativamente a la otra (minoritaria). En regresión logística, esto es particularmente crítico porque el algoritmo optimiza la verosimilitud global, dando igual peso a todas las observaciones.

**Contextos comunes de desbalanceo:**
- **Detección de fraude financiero:** 1-2% de transacciones fraudulentas.
- **Diagnóstico médico:** enfermedades raras con incidencia <1%.
- **Churn de clientes:** 5-10% de clientes que abandonan.
- **Campañas de marketing:** 1-5% de conversión.
- **Detección de spam:** 10-20% de correos no deseados.
- **Riesgo de crédito:** 3-5% de impagos.

**Consecuencias del desbalanceo en regresión logística:**

| Consecuencia | Explicación | Impacto en negocio |
|--------------|-------------|-------------------|
| **Sesgo hacia clase mayoritaria** | El modelo aprende que predecir "mayoría" minimiza error global | Subestima eventos raros (ej. fraudes no detectados) |
| **Probabilidades sesgadas** | Las probabilidades predichas son sistemáticamente bajas para la clase minoritaria | Umbrales estándar (0.5) no funcionan |
| **Métrica engañosa** | Accuracy alto pero modelo inútil | Falsa sensación de seguridad |
| **Convergencia lenta** | El gradiente es dominado por la clase mayoritaria | Entrenamiento ineficiente |

**Ejemplo numérico del problema:**

Supongamos 10,000 transacciones con 2% de fraude (200 fraudes, 9,800 normales). Un modelo que siempre predice "normal" tiene:
- Accuracy = 9,800/10,000 = 98%
- Pero detecta 0 fraudes → modelo completamente inútil para el negocio.

### Técnicas de Balanceo para Regresión Logística

#### Oversampling (Sobremuestreo)

**Descripción:** Aumenta la clase minoritaria replicando o generando muestras sintéticas.

**Técnicas principales:**

| Técnica | Funcionamiento | Ventajas | Desventajas |
|---------|---------------|----------|-------------|
| **Random Oversampling** | Duplica aleatoriamente instancias de la clase minoritaria | Simple, rápido | Sobreajuste por replicación exacta |
| **SMOTE** | Genera muestras sintéticas interpolando entre instancias existentes | Más variedad, reduce sobreajuste | Puede generar ruido en fronteras |
| **ADASYN** | Similar a SMOTE pero enfocado en regiones difíciles | Mejora en fronteras complejas | Más sensible a outliers |
| **Borderline-SMOTE** | SMOTE enfocado en instancias cercanas a la frontera | Mejora la frontera de decisión | Complejidad adicional |

**Cuándo usar oversampling:**
- Dataset pequeño a mediano (<10,000 muestras)
- Clase minoritaria muy pequeña (<100 muestras)
- Cuando es crítico maximizar recall
- Preferir SMOTE sobre random oversampling para evitar sobreajuste

#### Undersampling (Submuestreo)

**Descripción:** Reduce la clase mayoritaria eliminando ejemplos para equilibrar las clases.

**Técnicas principales:**

| Técnica | Funcionamiento | Ventajas | Desventajas |
|---------|---------------|----------|-------------|
| **Random Undersampling** | Elimina aleatoriamente instancias de la clase mayoritaria | Simple, reduce dataset | Puede perder información valiosa |
| **Tomek Links** | Elimina pares de instancias de diferentes clases mutuamente más cercanas | Limpia fronteras, reduce ruido | No balancea completamente por sí solo |
| **NearMiss** | Selecciona instancias mayoritarias cercanas a la minoría | Mantiene información relevante | Computacionalmente costoso |
| **Cluster Centroids** | Reemplaza clusters mayoritarios con sus centroides | Preserva estructura | Pérdida de variabilidad |

**Cuándo usar undersampling:**
- Dataset grande (>50,000 muestras) donde el costo computacional importa
- La clase mayoritaria tiene mucha redundancia o ruido
- Como complemento a oversampling (ej. SMOTE + Tomek)
- Cuando la clase mayoritaria es extremadamente grande

#### Ajuste de Pesos (Class Weights)

**Descripción:** Asigna mayor penalización a errores en la clase minoritaria durante el entrenamiento, sin modificar los datos.

**Fundamento matemático:**
La función de costo de regresión logística con pesos es:

$$ J(\theta) = -\frac{1}{n} \left[ \sum_{i=1}^{n} w_{y_i} \left( y_i \log(h_\theta(x_i)) + (1-y_i) \log(1-h_\theta(x_i)) \right) \right] $$

donde $w_{y_i}$ es el peso asignado a la clase de la observación $i$.

**Cálculo automático de 'balanced':**

$$ w_j = \frac{n}{k \cdot n_j} $$

donde:
- $n$: total de muestras
- $k$: número de clases
- $n_j$: muestras en clase $j$

**Ejemplo numérico:**
Para 10,000 muestras con 200 fraudes (clase 1) y 9,800 normales (clase 0):

$$ w_0 = \frac{10000}{2 \cdot 9800} = 0.51 $$
$$ w_1 = \frac{10000}{2 \cdot 200} = 25 $$

Los errores en fraudes se penalizan 49 veces más que errores en normales ($25/0.51 \approx 49$).

**Ventajas del ajuste de pesos:**
- No modifica los datos originales
- No aumenta el tamaño del dataset
- Preserva toda la información
- Matemáticamente elegante (modifica la función de costo)
- Generalmente la primera opción a probar

**Desventajas:**
- Menos efectivo en desbalanceos extremos (>100:1)
- Puede hacer el modelo inestable con pesos muy altos
- Requiere que el algoritmo soporte pesos (sí en regresión logística)

### Comparación de Técnicas de Balanceo

| Aspecto | Oversampling | Undersampling | Class Weights |
|---------|--------------|---------------|---------------|
| **Tamaño del dataset** | Aumenta | Disminuye | Sin cambios |
| **Información** | Genera sintética | Descarta real | Preserva toda |
| **Tiempo entrenamiento** | Aumenta | Disminuye | Sin cambios |
| **Riesgo de overfitting** | Alto | Bajo | Medio |
| **Riesgo de underfitting** | Bajo | Alto | Medio |
| **Implementación** | Requiere imbalanced-learn | Requiere imbalanced-learn | Nativa en sklearn |
| **Desbalanceo extremo** | Efectivo | Ineficiente | Limitado |

---

## Métricas de Evaluación para Regresión Logística

### Matriz de Confusión

La matriz de confusión es la base fundamental para evaluar clasificadores binarios:

| | **Predicción: Positivo** | **Predicción: Negativo** |
|---|---|---|
| **Real: Positivo** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Real: Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN) |

### Métricas Derivadas

#### Exactitud (Accuracy)
$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$

**Interpretación:** Proporción de aciertos sobre el total.

**Limitación en regresión logística:** Engañosa en desbalanceo. Un modelo que siempre predice la clase mayoritaria puede tener alta accuracy pero ser inútil.

#### Precisión (Precision)
$$ Precision = \frac{TP}{TP + FP} $$

**Interpretación:** De todas las predicciones positivas, ¿cuántas son correctas? Mide la confiabilidad de las alertas.

**Rango:** 0 a 1 (mayor es mejor)

#### Recall (Sensibilidad, TPR)
$$ Recall = \frac{TP}{TP + FN} $$

**Interpretación:** De todos los positivos reales, ¿cuántos detectamos? Mide la capacidad de encontrar la clase de interés.

**Rango:** 0 a 1 (mayor es mejor)

#### Especificidad (Specificity, TNR)
$$ Especificidad = \frac{TN}{TN + FP} $$

**Interpretación:** De todos los negativos reales, ¿cuántos identificamos correctamente?

#### F1-Score
$$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

**Interpretación:** Media armónica entre precisión y recall. Penaliza valores extremos.

**Rango:** 0 a 1 (mayor es mejor)

### Curvas de Evaluación

#### Curva ROC (Receiver Operating Characteristic)

**Construcción:** Gráfica TPR (Recall) vs FPR ($1 - \text{Especificidad}$) para todos los umbrales posibles (0 a 1).

**FPR (Tasa de Falsos Positivos):**
$$ FPR = \frac{FP}{FP + TN} = 1 - Especificidad $$

**AUC-ROC (Área bajo la curva ROC):**

| AUC | Interpretación |
|-----|----------------|
| 0.5 | Clasificador aleatorio |
| 0.6 - 0.7 | Pobre |
| 0.7 - 0.8 | Aceptable |
| 0.8 - 0.9 | Excelente |
| 0.9 - 1.0 | Sobresaliente |

#### Curva Precisión-Recall (PR Curve)

**Construcción:** Gráfica Precisión vs Recall para todos los umbrales.

**Ventaja sobre ROC:** Más informativa en datasets desbalanceados porque se enfoca en la clase minoritaria (positiva).

### Selección de Métricas según Contexto de Negocio

| Contexto | Métrica Principal | Justificación |
|----------|------------------|---------------|
| **Detección de fraude** | Recall o PR-AUC | Priorizar detección de fraudes |
| **Filtro de spam** | Precisión | Evitar falsos positivos molestos |
| **Diagnóstico COVID** | Recall (Sensibilidad) | No querer falsos negativos |
| **Churn de clientes** | F1 o Recall | Balancear retención vs recursos |
| **Recomendación** | Precisión | Mostrar solo recomendaciones relevantes |
| **Riesgo de crédito** | F1 o PR-AUC | Balancear riesgo vs clientes rechazados |

---

## Comunicación de Resultados en Regresión Logística

### Storytelling con Data Dummy

**Dataset simulado de churn de clientes:**

| Cliente | Edad | Antigüedad (años) | Consumo | Churn (0=no,1=si) |
| ------- | ---- | ----------------- | ------- | ----------------- |
| 1       | 25   | 1                 | 120     | 0                 |
| 2       | 40   | 5                 | 350     | 1                 |
| 3       | 30   | 2                 | 200     | 0                 |
| 4       | 50   | 10                | 500     | 1                 |
| 5       | 28   | 1                 | 150     | 0                 |

Modelo de regresión logística con coeficientes:

$$ \hat{p} = \frac{1}{1+e^{-(-3 + 0.02 \cdot Edad + 0.005 \cdot Consumo + 0.1 \cdot Antigüedad)}} $$

**Interpretación:**
- Cada año adicional de antigüedad aumenta log-odds de churn en 0.1.
- Cada unidad de consumo aumenta log-odds en 0.005.

**Métricas dummy:**
- Exactitud: 0.88
- Recall clase positiva: 0.92
- Precisión clase positiva: 0.85
- F1-score: 0.88
- AUC: 0.95

---

### Elevator pitch

**Equipo técnico:**

> "Se entrenó un modelo de regresión logística binaria para predecir churn de clientes. El modelo incluye features `Edad`, `Consumo` y `Antigüedad`. Se aplicó **regularización L2** y ajuste de **class weights** para balancear la clase minoritaria. Métricas obtenidas: AUC 0.95, recall 0.92, F1-score 0.88."

**Equipo no técnico / negocio:**

> "El modelo identifica clientes con alta probabilidad de abandonar el servicio. Clientes con mayor antigüedad o consumo elevado tienen más riesgo de churn. Con esta información, podemos diseñar campañas de retención dirigidas a los clientes críticos."

