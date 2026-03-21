

# Semana 4: Clasificación I – Regresión Logística y Balance de Datos

## Logro de la sesión

El estudiante será capaz de **construir, interpretar y evaluar modelos de clasificación binaria y multiclase** mediante regresión logística, incorporando técnicas de balanceo de datos, selección de métricas adecuadas y ajuste de thresholds según objetivos de negocio.

---

## Problemática de negocio

* **Tipos de problemas de clasificación:**

  * **Binaria:** Dos clases (ej. fraude/no fraude, churn/no churn, aprobado/rechazado).
  * **Multiclase:** Más de dos clases (ej. clasificación de productos, tipos de clientes, diagnóstico médico múltiple).

* **Impacto del desbalance de clases:**
  Cuando la proporción de clases es muy desigual, modelos simples tienden a predecir siempre la clase mayoritaria, generando métricas engañosas (alta exactitud pero baja sensibilidad para la clase minoritaria).

* **Ejemplos clásicos:**

  * Detección de fraude financiero
  * Churn de clientes
  * Diagnóstico de enfermedades
  * Score crediticio
  * Clasificación de productos o comportamiento de clientes

* **Diferencias clave con regresión:**

  * **Regresión:** Output continuo (y \in \mathbb{R}).
  * **Clasificación:** Output categórico (y \in {0,1}) o (y \in {1,...,K}).

---




## Modelado de clasificación

### Regresión logística binaria

Función de predicción:

[
\hat{p} = P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \sum_{i=1}^{n} \beta_i x_i
]

* **Interpretación:** (\hat{p}) es la probabilidad de pertenecer a la clase positiva.
* **Log-odds:**
  [
  \text{logit}(\hat{p}) = \log\frac{\hat{p}}{1-\hat{p}} = \beta_0 + \sum \beta_i x_i
  ]
  Cada (\beta_i) representa el cambio en log-odds por unidad de incremento en (x_i).

### Regresión logística multiclase

* **One-vs-Rest (OvR):** Se entrena un modelo binario por cada clase.
* **Softmax:** Probabilidad de clase (k):
  [
  P(y=k|X) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad z_k = \beta_{0k} + \sum_i \beta_{ik} x_i
  ]

### Función de pérdida y optimización

* **Función de pérdida log-loss (cross-entropy):**
  [
  L(\beta) = - \frac{1}{N} \sum_{i=1}^N \big[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \big]
  ]

* **Optimización:** Se minimiza (L(\beta)) usando **Gradiente Descendente** o variantes (SGD, Adam):
  [
  \beta \leftarrow \beta - \eta \nabla_\beta L(\beta)
  ]

* **Regularización:**

  * **L2 (Ridge):** (L_{reg} = L(\beta) + \lambda \sum \beta_i^2)
  * **L1 (Lasso):** (L_{reg} = L(\beta) + \lambda \sum |\beta_i|)
  * **Elastic Net:** (L_{reg} = L(\beta) + \lambda_1 \sum |\beta_i| + \lambda_2 \sum \beta_i^2)

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

### 7.2 Técnicas de Balanceo para Regresión Logística

#### 7.2.1 Oversampling (Sobremuestreo)

**Descripción:** Aumenta la clase minoritaria replicando o generando muestras sintéticas.

**Técnicas principales:**

| Técnica | Funcionamiento | Ventajas | Desventajas |
|---------|---------------|----------|-------------|
| **Random Oversampling** | Duplica aleatoriamente instancias de la clase minoritaria | Simple, rápido | Sobreajuste por replicación exacta |
| **SMOTE** (Synthetic Minority Oversampling Technique) | Genera muestras sintéticas interpolando entre instancias existentes y sus vecinos | Más variedad, reduce sobreajuste | Puede generar ruido en fronteras |
| **ADASYN** | Similar a SMOTE pero enfocado en regiones difíciles | Mejora en fronteras complejas | Más sensible a outliers |
| **Borderline-SMOTE** | SMOTE enfocado en instancias cercanas a la frontera | Mejora la frontera de decisión | Complejidad adicional |

**Implementación en regresión logística:**

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.linear_model import LogisticRegression

# Aplicar SMOTE (solo en entrenamiento)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Entrenar con datos balanceados
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)
```

**Cuándo usar oversampling:**
- Dataset pequeño a mediano (<10,000 muestras)
- Clase minoritaria muy pequeña (<100 muestras)
- Cuando es crítico maximizar recall
- Preferir SMOTE sobre random oversampling para evitar sobreajuste

#### 7.2.2 Undersampling (Submuestreo)

**Descripción:** Reduce la clase mayoritaria eliminando ejemplos para equilibrar las clases.

**Técnicas principales:**

| Técnica | Funcionamiento | Ventajas | Desventajas |
|---------|---------------|----------|-------------|
| **Random Undersampling** | Elimina aleatoriamente instancias de la clase mayoritaria | Simple, reduce dataset | Puede perder información valiosa |
| **Tomek Links** | Elimina pares de instancias de diferentes clases mutuamente más cercanas | Limpia fronteras, reduce ruido | No balancea completamente por sí solo |
| **NearMiss** | Selecciona instancias mayoritarias cercanas a la minoría | Mantiene información relevante | Computacionalmente costoso |
| **Cluster Centroids** | Reemplaza clusters mayoritarios con sus centroides | Preserva estructura | Pérdida de variabilidad |

**Implementación:**

```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Tomek Links (para limpiar frontera)
tomek = TomekLinks()
X_train_clean, y_train_clean = tomek.fit_resample(X_train, y_train)
```

**Cuándo usar undersampling:**
- Dataset grande (>50,000 muestras) donde el costo computacional importa
- La clase mayoritaria tiene mucha redundancia o ruido
- Como complemento a oversampling (ej. SMOTE + Tomek)
- Cuando la clase mayoritaria es extremadamente grande

#### 7.2.3 Ajuste de Pesos (Class Weights)

**Descripción:** Asigna mayor penalización a errores en la clase minoritaria durante el entrenamiento, sin modificar los datos.

**Fundamento matemático:**
La función de costo de regresión logística con pesos es:

$$J(\theta) = -\frac{1}{n} \left[ \sum_{i=1}^{n} w_{y_i} \left( y_i \log(h_\theta(x_i)) + (1-y_i) \log(1-h_\theta(x_i)) \right) \right]$$

donde $w_{y_i}$ es el peso asignado a la clase de la observación $i$.

**Implementación en scikit-learn:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# Opción 1: balanced automático
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Opción 2: pesos personalizados
weights = {0: 1.0, 1: 5.0}  # clase 1 (minoritaria) tiene peso 5
model = LogisticRegression(class_weight=weights)

# Opción 3: calcular pesos óptimos
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
```

**Cálculo automático de 'balanced':**
$$w_j = \frac{n}{k \cdot n_j}$$

donde:
- $n$: total de muestras
- $k$: número de clases
- $n_j$: muestras en clase $j$

**Ejemplo numérico:**
Para 10,000 muestras con 200 fraudes (clase 1) y 9,800 normales (clase 0):

$$w_0 = \frac{10000}{2 \cdot 9800} = 0.51$$
$$w_1 = \frac{10000}{2 \cdot 200} = 25$$

Los errores en fraudes se penalizan 49 veces más que errores en normales (25/0.51 ≈ 49).

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

### 7.3 Comparación de Técnicas de Balanceo

| Aspecto | Oversampling | Undersampling | Class Weights |
|---------|--------------|---------------|---------------|
| **Tamaño del dataset** | Aumenta | Disminuye | Sin cambios |
| **Información** | Genera sintética | Descarta real | Preserva toda |
| **Tiempo entrenamiento** | Aumenta | Disminuye | Sin cambios |
| **Riesgo de overfitting** | Alto (especialmente random) | Bajo | Medio |
| **Riesgo de underfitting** | Bajo | Alto (pérdida info) | Medio |
| **Implementación** | Requiere imbalanced-learn | Requiere imbalanced-learn | Nativa en sklearn |
| **Desbalanceo extremo** | Efectivo | Ineficiente | Limitado |

### 7.4 Árbol de Decisión para Selección de Técnica

```
¿Tamaño del dataset?
│
├── Pequeño (<5,000 muestras)
│   │
│   ├── ¿Clase minoritaria < 100 muestras?
│   │   ├── Sí → SMOTE + Validación cuidadosa
│   │   └── No → Class Weights
│   │
│   └── ¿Overfitting evidente?
│       ├── Sí → Class Weights
│       └── No → Probar SMOTE
│
├── Mediano (5,000 - 50,000 muestras)
│   │
│   ├── ¿Desbalanceo moderado (<10:1)?
│   │   ├── Sí → Class Weights (suficiente)
│   │   └── No → SMOTE o Combinación
│   │
│   └── ¿Costo computacional preocupación?
│       ├── Sí → Class Weights
│       └── No → SMOTE + Validación
│
└── Grande (>50,000 muestras)
    │
    ├── ¿Clase mayoritaria con ruido?
    │   ├── Sí → Undersampling inteligente (Tomek, NearMiss)
    │   └── No → Class Weights
    │
    └── ¿Desbalanceo extremo (>100:1)?
        ├── Sí → Combinación: Undersampling + SMOTE
        └── No → Class Weights o Random Undersampling
```

### 7.5 Relación entre Técnica de Balanceo y Métrica Objetivo

La elección de la técnica debe alinearse con la métrica que prioriza el negocio:

| Si priorizas... | Técnica recomendada | Explicación |
|-----------------|--------------------|-------------|
| **Recall** (detectar todos los positivos) | SMOTE o Class Weights agresivos | Ambas técnicas fortalecen la clase minoritaria, aumentando sensibilidad |
| **Precision** (minimizar falsos positivos) | Undersampling o Class Weights moderados | Evita generar instancias sintéticas que puedan ser ruido |
| **F1-Score** (balance) | Class Weights con validación | Permite ajustar finamente el trade-off |
| **AUC-ROC** | Class Weights | Menos sensible al umbral; pesos suelen funcionar bien |
| **PR-AUC** | SMOTE + Validación | PR-AUC es sensible a detección de minoría; oversampling ayuda |
| **Probabilidades calibradas** | Class Weights | No distorsiona la distribución como oversampling |

### 7.6 Buenas Prácticas en Balanceo para Regresión Logística

1. **Nunca balancear antes del split train-test:**
   - Aplicar técnicas de balanceo **después** de separar entrenamiento y prueba
   - Balancear solo el conjunto de entrenamiento
   - Mantener el test set con distribución real para evaluación honesta

2. **Validación cruzada estratificada:**
   ```python
   from sklearn.model_selection import StratifiedKFold, cross_val_score
   
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   # Asegura misma proporción de clases en cada fold
   ```

3. **Probar múltiples técnicas:**
   ```python
   techniques = {
       'none': LogisticRegression(class_weight=None),
       'balanced': LogisticRegression(class_weight='balanced'),
       'custom': LogisticRegression(class_weight={0:1, 1:10}),
       'smote': Pipeline([('smote', SMOTE()), ('lr', LogisticRegression())])
   }
   ```

4. **Ajuste de umbral posterior:**
   - El umbral por defecto (0.5) raramente es óptimo con datos balanceados
   - Optimizar umbral en validación:
   ```python
   from sklearn.metrics import f1_score
   
   thresholds = np.arange(0.1, 0.9, 0.05)
   scores = []
   for threshold in thresholds:
       y_pred_val = (model.predict_proba(X_val)[:, 1] >= threshold).astype(int)
       scores.append(f1_score(y_val, y_pred_val))
   
   best_threshold = thresholds[np.argmax(scores)]
   ```

5. **Evaluar con métricas apropiadas:**
   - Evitar accuracy como métrica principal
   - Usar F1, PR-AUC, Recall, Precision según contexto
   - Comparar curvas PR completas

6. **Considerar el costo real de negocio:**
   - Asignar costos a FP y FN según impacto económico
   - Optimizar directamente el costo esperado

### 7.7 Ejemplo Completo con Interpretación

**Contexto:** Banco con 50,000 transacciones mensuales, 2% fraudulentas. Se prueba regresión logística con diferentes técnicas.

**Resultados comparativos:**

| Técnica | Precision | Recall | F1 | AUC-ROC | PR-AUC | VP (sobre 1,000 fraudes) | FP |
|---------|-----------|--------|-----|---------|--------|--------------------------|-----|
| Sin balanceo | 0.82 | 0.35 | 0.49 | 0.88 | 0.28 | 350 | 77 |
| Class Weight (balanced) | 0.58 | 0.78 | 0.67 | 0.92 | 0.45 | 780 | 565 |
| SMOTE | 0.52 | 0.85 | 0.65 | 0.91 | 0.48 | 850 | 785 |
| Undersampling | 0.48 | 0.72 | 0.58 | 0.87 | 0.38 | 720 | 780 |
| SMOTE + Tomek | 0.55 | 0.82 | 0.66 | 0.92 | 0.47 | 820 | 670 |

**Interpretación para negocio:**

> **Análisis comparativo:**
> - **Sin balanceo:** Alta precisión (82%) pero detecta solo 35% de fraudes. Pierde 650 fraudes mensuales → pérdida económica significativa.
> - **Class Weights:** Mejor F1 (0.67). Detecta 780 fraudes (78%) con 565 falsos positivos. Balance óptimo costo-beneficio.
> - **SMOTE:** Máximo recall (85%) pero 785 falsos positivos. Ideal si cada fraude no detectado cuesta mucho.
> - **Undersampling:** Peor rendimiento; pierde información valiosa.
> - **SMOTE+Tomek:** Similar a class weights pero más complejo.
>
> **Recomendación:** Implementar regresión logística con class weights 'balanced' y umbral optimizado en 0.35. Esto maximiza F1 y proporciona el mejor balance entre detección de fraudes y carga operativa.

---

## Métricas de Evaluación para Regresión Logística

### 8.1 Matriz de Confusión

La matriz de confusión es la base fundamental para evaluar clasificadores binarios:

| | **Predicción: Positivo** | **Predicción: Negativo** |
|---|---|---|
| **Real: Positivo** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Real: Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN) |

**Interpretación en contextos de negocio:**

| Contexto | VP | VN | FP | FN |
|----------|----|----|----|-----|
| **Fraude** | Fraudes detectados | Transacciones normales | Falsas alarmas | Fraudes no detectados |
| **Marketing** | Clientes que convierten | No conversores | Oportunidades perdidas | Conversiones no captadas |
| **Crédito** | Impagos correctos | Pagadores correctos | Clientes rechazados injustamente | Impagos no detectados |
| **Diagnóstico** | Enfermos detectados | Sanos correctos | Sanos con tratamiento | Enfermos no diagnosticados |

### 8.2 Métricas Derivadas

#### Exactitud (Accuracy)
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Interpretación:** Proporción de aciertos sobre el total.

**Limitación en regresión logística:** Engañosa en desbalanceo. Un modelo que siempre predice la clase mayoritaria puede tener alta accuracy pero ser inútil.

**Ejemplo:** En fraude (2% positivos), accuracy de 98% puede significar 0 fraudes detectados.

#### Precisión (Precision)
$$Precision = \frac{TP}{TP + FP}$$

**Interpretación:** De todas las predicciones positivas, ¿cuántas son correctas? Mide la confiabilidad de las alertas.

**Rango:** 0 a 1 (mayor es mejor)

**Ejemplo negocio:** Precisión = 0.80 significa que de cada 100 alertas de fraude, 80 son reales y 20 son falsas alarmas.

**Útil cuando:** El costo de falsos positivos es alto (ej. bloquear clientes legítimos, tratamientos innecesarios).

#### Recall (Sensibilidad, TPR)
$$Recall = \frac{TP}{TP + FN}$$

**Interpretación:** De todos los positivos reales, ¿cuántos detectamos? Mide la capacidad de encontrar la clase de interés.

**Rango:** 0 a 1 (mayor es mejor)

**Ejemplo negocio:** Recall = 0.75 significa que detectamos 75 de cada 100 fraudes reales.

**Útil cuando:** El costo de falsos negativos es alto (ej. fraudes no detectados, enfermedades no diagnosticadas).

#### Especificidad (Specificity, TNR)
$$Especificidad = \frac{TN}{TN + FP}$$

**Interpretación:** De todos los negativos reales, ¿cuántos identificamos correctamente?

**Rango:** 0 a 1 (mayor es mejor)

**Relación con Recall:** Mientras recall se enfoca en positivos, especificidad se enfoca en negativos.

#### F1-Score
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Interpretación:** Media armónica entre precisión y recall. Penaliza valores extremos.

**Rango:** 0 a 1 (mayor es mejor)

**Ventaja:** Única métrica que combina ambas perspectivas. Ideal para comparación general.

**Cuándo usarla:**
- Clases desbalanceadas
- Cuando necesitas un balance entre precisión y recall
- Para comparar modelos de forma integral

### 8.3 Curvas de Evaluación

#### Curva ROC (Receiver Operating Characteristic)

**Construcción:** Gráfica TPR (Recall) vs FPR (1 - Especificidad) para todos los umbrales posibles (0 a 1).

**FPR (Tasa de Falsos Positivos):**
$$FPR = \frac{FP}{FP + TN} = 1 - Especificidad$$

**Interpretación:** Muestra el trade-off entre detectar positivos y evitar falsos positivos.

**AUC-ROC (Área bajo la curva ROC):**

| AUC | Interpretación |
|-----|----------------|
| 0.5 | Clasificador aleatorio (sin poder discriminativo) |
| 0.6 - 0.7 | Pobre |
| 0.7 - 0.8 | Aceptable |
| 0.8 - 0.9 | Excelente |
| 0.9 - 1.0 | Sobresaliente |

**Interpretación probabilística:** AUC es la probabilidad de que el modelo asigne mayor probabilidad a un positivo aleatorio que a un negativo aleatorio.

**Ejemplo:** AUC = 0.85 significa que el 85% de las veces, un positivo aleatorio tendrá mayor score que un negativo aleatorio.

#### Curva Precisión-Recall (PR Curve)

**Construcción:** Gráfica Precisión vs Recall para todos los umbrales.

**PR-AUC (Área bajo la curva PR):**

**Ventaja sobre ROC:** Más informativa en datasets desbalanceados porque se enfoca en la clase minoritaria (positiva).

**Interpretación:**
- Clasificador perfecto: PR-AUC = 1.0
- Clasificador aleatorio: PR-AUC = proporción de positivos (ej. 0.02 si hay 2% positivos)
- Útil para comparar modelos cuando la clase positiva es rara

**Ejemplo comparativo ROC vs PR:**
- En fraude (2% positivos), ROC puede verse optimista (ej. 0.95) aunque el modelo sea mediocre
- PR-AUC refleja mejor la realidad (ej. 0.45) porque se enfoca en la clase minoritaria

### 8.4 Comparación de Métricas

| Métrica | Ventajas | Desventajas | Mejor uso |
|---------|----------|-------------|-----------|
| **Accuracy** | Simple, intuitiva | Engañosa en desbalanceo | Clases balanceadas |
| **Precision** | Relevante para FP | Ignora FN | Cuando FP es costoso |
| **Recall** | Relevante para FN | Ignora FP | Cuando FN es costoso |
| **F1-Score** | Balancea ambas | No considera TN | Comparación general |
| **AUC-ROC** | Independiente umbral | Optimista en desbalanceo | Capacidad discriminativa |
| **PR-AUC** | Realista en desbalanceo | Menos conocida | Clases desbalanceadas |

### 8.5 Selección de Métricas según Contexto de Negocio

| Contexto | Métrica Principal | Justificación |
|----------|------------------|---------------|
| **Detección de fraude** | Recall o PR-AUC | Priorizar detección de fraudes; cada FN es pérdida económica |
| **Filtro de spam** | Precisión | Evitar que correos importantes vayan a spam (FP molesto) |
| **Diagnóstico COVID** | Recall (Sensibilidad) | No queremos falsos negativos (personas enfermas no aisladas) |
| **Diagnóstico enfermedad benigna** | Precisión | Evitar tratamientos innecesarios por falsos positivos |
| **Churn de clientes** | F1 o Recall | Balancear retención vs recursos invertidos |
| **Recomendación de productos** | Precisión | Mostrar solo recomendaciones relevantes |
| **Riesgo de crédito** | F1 o PR-AUC | Balancear riesgo de impago vs clientes rechazados |
| **Clases balanceadas** | Accuracy | Simple e interpretable |
| **Comparación general** | F1 + AUC-ROC | Visión completa |
| **Desbalanceo extremo** | PR-AUC | Más sensible a la clase minoritaria |

### 8.6 Interpretación de Métricas con Ejemplos de Negocio

#### Ejemplo 1: Detección de Fraude

**Resultados sobre 10,000 transacciones (200 fraudes reales):**

```
VP = 180, FN = 20, FP = 300, VN = 9,500

Accuracy = (180 + 9500)/10000 = 96.8%
Precision = 180/(180+300) = 37.5%
Recall = 180/(180+20) = 90%
F1 = 2*(0.375*0.9)/(0.375+0.9) = 0.53
```

**Interpretación para negocio:**

> "El modelo detecta el **90% de los fraudes** (Recall alto), excelente para minimizar pérdidas. Sin embargo, de cada 100 alertas de fraude, solo **38 son realmente fraudulentas** (Precisión 37.5%). Esto genera 300 falsas alarmas que el equipo debe investigar, con un costo operativo de X horas hombre. La decisión es si preferimos este nivel de detección con el costo asociado, o si ajustamos el modelo para reducir falsos positivos aunque detectemos menos fraudes."

#### Ejemplo 2: Campaña de Marketing

**Resultados sobre 50,000 clientes (2,500 conversiones):**

```
VP = 1,800, FN = 700, FP = 4,200, VN = 43,300

Precision = 1800/(1800+4200) = 30%
Recall = 1800/(1800+700) = 72%
```

**Interpretación para negocio:**

> "El modelo identifica al **72% de los clientes que convertirán** (Recall), permitiendo enfocar esfuerzos en 6,000 clientes (1,800+4,200). De estos, el **30% realmente convertirá** (Precision), superando ampliamente la tasa base del 5%. El costo por conversión se reduce significativamente al dirigirnos solo al 12% de la base de clientes."

#### Ejemplo 3: Diagnóstico Médico

**Resultados sobre 1,000 pacientes (10 con enfermedad):**

```
VP = 9, FN = 1, FP = 50, VN = 940

Recall = 9/10 = 90%
Precision = 9/(9+50) = 15.3%
```

**Interpretación para negocio:**

> "El modelo detecta el **90% de los casos positivos** (Recall), crítico para una enfermedad grave. Sin embargo, de cada 100 pacientes que alerta, solo **15 están realmente enfermos** (Precision), generando 50 falsos positivos que requieren pruebas adicionales. En este contexto, el alto recall justifica los falsos positivos dado el costo de no diagnosticar la enfermedad."

### 8.7 Comparación de Modelos con Múltiples Métricas

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC | PR-AUC |
|--------|----------|-----------|--------|-----|---------|--------|
| Regresión Logística (base) | 0.94 | 0.62 | 0.78 | 0.69 | 0.88 | 0.45 |
| RL + Class Weights | 0.91 | 0.55 | 0.88 | 0.68 | 0.91 | 0.52 |
| RL + SMOTE | 0.89 | 0.48 | 0.92 | 0.63 | 0.90 | 0.54 |
| RL con interacciones | 0.95 | 0.68 | 0.82 | 0.74 | 0.93 | 0.58 |

**Análisis comparativo:**

- **Modelo base:** Buen balance inicial, F1=0.69, AUC-ROC=0.88
- **Class Weights:** Mejora recall (0.78→0.88) y AUC-ROC (0.88→0.91), pero reduce precisión. Ideal si priorizamos detección.
- **SMOTE:** Máximo recall (0.92) pero baja precisión (0.48). Útil solo si cada FN es muy costoso.
- **Con interacciones:** Mejor F1 (0.74) y AUC-ROC (0.93), indicando que capturar relaciones no lineales mejora el modelo.

**Recomendación:** El modelo con interacciones ofrece el mejor balance (F1 más alto) y mejor capacidad discriminativa (AUC-ROC). Es la opción recomendada para implementación.

---

## Comunicación de Resultados en Regresión Logística

### 9.1 Comparación de Modelos para Toma de Decisiones

**Ejemplo de reporte ejecutivo:**

> **Resumen Ejecutivo: Modelo de Detección de Fraude**
>
> Hemos evaluado cuatro variantes de regresión logística para detectar transacciones fraudulentas. El modelo con interacciones (RL + términos polinómicos) supera a los demás en capacidad discriminativa (AUC-ROC 0.93) y balance entre precisión y recall (F1 0.74).
>
> **Impacto operativo estimado (sobre 100,000 transacciones mensuales, 2,000 fraudes):**
> - **RL base:** detectaría 1,560 fraudes (78% recall), con 950 falsas alarmas.
> - **RL + Class Weights:** detectaría 1,760 fraudes (88% recall), con 1,440 falsas alarmas.
> - **RL + SMOTE:** detectaría 1,840 fraudes (92% recall), con 2,000 falsas alarmas.
> - **RL con interacciones:** detectaría 1,640 fraudes (82% recall), con solo 770 falsas alarmas.
>
> **Recomendación:** Implementar regresión logística con interacciones. Ofrece el mejor balance: alta detección (82%) con mínimas falsas alarmas (770 vs 950-2,000 de alternativas). Reduce costos operativos en 30% respecto al modelo actual.

### 9.2 Explicación de Trade-offs en Regresión Logística

| Trade-off | Explicación para Negocio |
|-----------|--------------------------|
| **Precisión vs Recall** | "En regresión logística, podemos desplazar el umbral de decisión. Un umbral bajo detecta más fraudes (alto recall) pero genera más falsas alarmas (baja precisión). Un umbral alto hace lo contrario. La decisión depende del costo relativo de cada error." |
| **Sesgo vs Varianza** | "Un modelo simple (pocas variables) puede no capturar patrones complejos (alto sesgo). Uno muy complejo (muchas interacciones) puede ajustarse demasiado a los datos de entrenamiento y fallar en producción (alta varianza). Buscamos el equilibrio." |
| **Interpretabilidad vs Performance** | "La regresión logística simple es muy interpretable: podemos decir exactamente cómo afecta cada variable. Añadir interacciones y términos polinómicos mejora la precisión pero dificulta la explicación. Para cumplimiento regulatorio, a veces necesitamos el modelo interpretable." |
| **Costo computacional vs Precisión** | "Modelos más complejos requieren más tiempo de entrenamiento y recursos. Para predicciones en tiempo real, debemos asegurar que el modelo responda en menos de 100ms." |

### 9.3 Justificación del Modelo Seleccionado

**Template para justificar la selección:**

> **Modelo seleccionado:** [Descripción del modelo]
>
> **Justificación basada en:**
> 1. **Requisitos de negocio:** [ej. priorizamos recall porque cada fraude no detectado cuesta $X]
> 2. **Características de los datos:** [ej. relaciones no lineales, desbalanceo 98-2]
> 3. **Técnica de balanceo aplicada:** [ej. class weights por mejor F1]
> 4. **Restricciones operativas:** [ej. tiempo de predicción < 100ms, capacidad computacional]
> 5. **Métricas comparativas:** [ej. F1-score superior en 15%, AUC-ROC 0.93 vs 0.88 del segundo mejor]
> 6. **Umbral optimizado:** [ej. umbral 0.35 que maximiza F1 en validación]
>
> **Impacto esperado:** [cuantificar en términos de negocio: fraudes detectados, ahorro estimado, clientes afectados]

**Ejemplo completo:**

> **Modelo seleccionado:** Regresión Logística con interacciones de segundo orden + class weights 'balanced' + umbral 0.35
>
> **Justificación:**
> - **Negocio:** Necesitamos balancear la detección de fraudes (Recall) con la experiencia del cliente (evitar falsos bloqueos). El modelo ofrece el mejor F1 (0.74), indicando el balance óptimo.
> - **Datos:** Las transacciones presentan interacciones entre variables (ej. monto × hora del día) que el modelo captura mediante términos polinómicos, mejorando el ajuste.
> - **Balanceo:** Class weights 'balanced' superó a SMOTE y undersampling en validación cruzada, maximizando F1 sin generar ruido sintético.
> - **Operaciones:** Tiempo de predicción de 15ms por transacción, cumpliendo el requisito de tiempo real (<100ms).
> - **Métrica:** F1 de 0.74 supera al modelo base (0.69) y a alternativas (0.68, 0.63). AUC-ROC de 0.93 indica excelente capacidad discriminativa.
> - **Umbral:** Optimizado en 0.35 (vs 0.5 por defecto) para maximizar F1 en datos de validación.
>
> **Impacto estimado:** 
> - Detección: 1,640 fraudes mensuales (82% del total)
> - Falsas alarmas: 770 transacciones a investigar
> - Ahorro: $150,000 anuales en pérdidas evitadas
> - Eficiencia operativa: 30% menos falsas alarmas que modelo actual
> - Clientes afectados: 0.08% de transacciones legítimas bloqueadas (mínimo impacto)




---

## Comunicación

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

[
\hat{p} = \frac{1}{1+e^{-(-3 + 0.02 \cdot Edad + 0.005 \cdot Consumo + 0.1 \cdot Antigüedad)}}
]

**Interpretación:**

* Cada año adicional de antigüedad aumenta log-odds de churn en 0.1.
* Cada unidad de consumo aumenta log-odds en 0.005.

**Métricas dummy:**

* Exactitud: 0.88
* Recall clase positiva: 0.92
* Precisión clase positiva: 0.85
* F1-score: 0.88
* AUC: 0.95

---

### Elevator pitch

**Equipo técnico:**

> “Se entrenó un modelo de regresión logística binaria para predecir churn de clientes.
> El modelo incluye features `Edad`, `Consumo` y `Antigüedad`.
> Se aplicó **regularización L2** y ajuste de **class weights** para balancear la clase minoritaria.
> Métricas obtenidas: AUC 0.95, recall 0.92, F1-score 0.88.
> Se recomienda optimizar threshold según tolerancia al riesgo de falsos negativos.”

**Equipo no técnico / negocio:**

> “El modelo identifica clientes con alta probabilidad de abandonar el servicio.
> Clientes con mayor antigüedad o consumo elevado tienen más riesgo de churn.
> Con esta información, podemos diseñar campañas de retención dirigidas a los clientes críticos, reduciendo pérdidas y mejorando la fidelidad.”

---

## Reto: 1 punto

* Analizar un dataset desbalanceado, aplicar **oversampling, undersampling y ajuste de pesos**, comparar métricas antes y después y justificar la estrategia óptima según contexto de negocio.

---

## Laboratorio: Ver Colab

* Scikit-learn: regresión logística (binaria y multiclase)
* Pipeline completo:

  1. EDA avanzado
  2. Feature engineering
  3. Balanceo de datos
  4. Entrenamiento de modelo
  5. Evaluación con métricas
  6. Ajuste de threshold

---



## Anexo: Fundamento matemático y computacional – Regresión Logística (Paso a Paso)

### 1. Modelo básico

La regresión logística modela la probabilidad de que la variable dependiente (y_i) tome valor 1:

[
\hat{p}*i = P(y_i=1 \mid X_i) = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}, \quad z_i = \beta_0 + \sum*{j=1}^{n} \beta_j x_{ij}
]

* (\sigma(z)) = función sigmoide.
* (x_{ij}) = valor de la variable (j) para la observación (i).
* (\beta_j) = coeficiente a estimar.

**Interpretación:**
[
\text{logit}(\hat{p}_i) = \log\frac{\hat{p}_i}{1-\hat{p}*i} = z_i = \beta_0 + \sum_j \beta_j x*{ij}
]
El coeficiente (\beta_j) representa el cambio en log-odds de (y=1) por unidad de (x_j).

---

### 2. Función de verosimilitud

La verosimilitud para todo el dataset ((i=1..N)):

[
L(\beta) = \prod_{i=1}^N \hat{p}_i^{y_i} (1-\hat{p}_i)^{1-y_i}
]

* Si (y_i=1), el término relevante es (\hat{p}_i).
* Si (y_i=0), el término relevante es (1-\hat{p}_i).

**Log-verosimilitud (más conveniente para derivar):**

[
\ell(\beta) = \log L(\beta) = \sum_{i=1}^N \Big[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \Big]
]

---

### 3. Función de pérdida (log-loss / cross-entropy)

Para minimizar, definimos la **función de costo**:

[
J(\beta) = - \frac{1}{N} \sum_{i=1}^N \Big[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \Big]
]

* Minimizar (J(\beta)) ≡ maximizar la log-verosimilitud.
* Penaliza fuertemente predicciones equivocadas, especialmente cuando la probabilidad asignada es cercana a 0 o 1 y el valor real es opuesto.

---

### 4. Derivación del gradiente paso a paso

#### Paso 1: Recordar (\hat{p}_i = \sigma(z_i))

[
\hat{p}*i = \frac{1}{1 + e^{-z_i}}, \quad z_i = \beta_0 + \sum_j \beta_j x*{ij}
]

#### Paso 2: Derivar (\hat{p}_i) respecto a (\beta_j)

[
\frac{\partial \hat{p}_i}{\partial \beta_j} = \hat{p}_i (1-\hat{p}*i) x*{ij}
]

#### Paso 3: Derivar la función de pérdida

[
J(\beta) = - \frac{1}{N} \sum_{i=1}^N \big[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \big]
]

[
\frac{\partial J}{\partial \beta_j} = - \frac{1}{N} \sum_{i=1}^N \Big[ \frac{y_i}{\hat{p}_i} \frac{\partial \hat{p}_i}{\partial \beta_j} - \frac{1-y_i}{1-\hat{p}_i} \frac{\partial \hat{p}_i}{\partial \beta_j} \Big]
]

Sustituimos (\frac{\partial \hat{p}_i}{\partial \beta_j} = \hat{p}_i(1-\hat{p}*i)x*{ij}):

[
\frac{\partial J}{\partial \beta_j} = - \frac{1}{N} \sum_{i=1}^N \Big[ y_i (1-\hat{p}_i) - (1-y_i) \hat{p}*i \Big] x*{ij}
]

Simplificando:

[
y_i (1-\hat{p}_i) - (1-y_i) \hat{p}_i = y_i - y_i \hat{p}_i - \hat{p}_i + y_i \hat{p}_i = y_i - \hat{p}_i
]

Por lo tanto:

[
\boxed{\frac{\partial J}{\partial \beta_j} = \frac{1}{N} \sum_{i=1}^N (\hat{p}*i - y_i) x*{ij}}
]

✅ Este es el **gradiente estándar** usado en todas las implementaciones.

---

### 5. Actualización de parámetros (gradiente descendente)

Sea (\eta) el learning rate:

[
\beta_j \leftarrow \beta_j - \eta \frac{\partial J}{\partial \beta_j} = \beta_j - \eta \frac{1}{N} \sum_{i=1}^N (\hat{p}*i - y_i) x*{ij}
]

* Se repite hasta convergencia.
* Puede usarse **batch**, **mini-batch** o **stochastic gradient descent**.
* Se pueden usar optimizadores adaptativos: Adam, RMSProp, Adagrad.

---

### 6. Regularización

#### L2 (Ridge)

[
J_{reg}(\beta) = J(\beta) + \frac{\lambda}{2} \sum_{j=1}^n \beta_j^2
]

Gradiente:

[
\frac{\partial J_{reg}}{\partial \beta_j} = \frac{1}{N} \sum_{i=1}^N (\hat{p}*i - y_i)x*{ij} + \lambda \beta_j
]

#### L1 (Lasso)

[
J_{reg}(\beta) = J(\beta) + \lambda \sum_{j=1}^n |\beta_j|
]

Gradiente (subgradiente):

[
\frac{\partial J_{reg}}{\partial \beta_j} = \frac{1}{N} \sum_{i=1}^N (\hat{p}*i - y_i)x*{ij} + \lambda , \text{sign}(\beta_j)
]

---

### 7. Extensión a multiclase (Softmax)

* Para (K) clases, probabilidad de la clase (k):

[
\hat{p}*{ik} = \frac{e^{z*{ik}}}{\sum_{l=1}^{K} e^{z_{il}}}, \quad z_{ik} = \beta_{0k} + \sum_j \beta_{jk} x_{ij}
]

* Función de pérdida: **categorical cross-entropy**

[
J(\beta) = - \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}
]

* Gradiente respecto a (\beta_{jk}):

[
\frac{\partial J}{\partial \beta_{jk}} = \frac{1}{N} \sum_{i=1}^N (\hat{p}*{ik} - y*{ik}) x_{ij}
]

---

### 8. Complejidad computacional

| Operación                       | Complejidad               |
| ------------------------------- | ------------------------- |
| Evaluación sigmoide             | O(N·n)                    |
| Gradiente (binaria)             | O(N·n·T)                  |
| Gradiente (multiclase K clases) | O(N·n·K·T)                |
| Optimización con SGD/mini-batch | Reducida según batch size |

**Recomendaciones computacionales:**

* Vectorización en NumPy/Pandas para acelerar cálculos.
* Uso de GPUs para grandes datasets o muchas clases.
* Ajuste de batch size y learning rate para convergencia estable.

