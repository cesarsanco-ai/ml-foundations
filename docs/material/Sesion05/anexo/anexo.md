---
layout: default
---

# Fundamento Matemático y Computacional de k-NN, Naive Bayes y SVM
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-05](../../../sesiones/sesion-05.md)



---


## Anexo Matemático y Computacional: K‑Nearest Neighbors (KNN)

## 1. Planteamiento del Problema

Tenemos un conjunto de entrenamiento $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, donde  
- $\mathbf{x}_i \in \mathbb{R}^d$ es el vector de características (atributos) de la $i$-ésima muestra,  
- $y_i \in \{1,2,\dots,K\}$ es la clase a la que pertenece.

**Objetivo:** Dado un nuevo punto $\mathbf{x}$ (sin etiqueta), predecir su clase $\hat{y}(\mathbf{x})$.

KNN aborda este problema **sin asumir una forma paramétrica** para la distribución de los datos. Simplemente almacena $\mathcal{D}$ y, para cada $\mathbf{x}$, busca los $k$ puntos de $\mathcal{D}$ más cercanos según una **métrica de distancia** $d(\cdot,\cdot)$. La predicción se basa en la clase mayoritaria entre esos vecinos.

---

## 2. Definición Matemática del Modelo

### 2.1 Conjunto de vecinos
Para un punto $\mathbf{x}$, definimos $\mathcal{N}_k(\mathbf{x})$ como el conjunto de los $k$ puntos de $\mathcal{D}$ con menor distancia $d(\mathbf{x}, \mathbf{x}_i)$ (en caso de empates, se resuelven arbitrariamente).

### 2.2 Regla de decisión (clasificación)
La predicción se obtiene mediante **voto mayoritario**:

$$
\hat{y}(\mathbf{x}) = \arg\max_{c \in \{1,\dots,K\}} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}(y_i = c)
$$

donde $\mathbf{1}(\text{condición})$ es la función indicadora (vale 1 si la condición es verdadera, 0 en caso contrario).

---

## 3. Función de Pérdida y Relación con el Clasificador de Bayes

### 3.1 Pérdida 0‑1
En clasificación, la pérdida típica es la **pérdida 0‑1**:

$$
L(y, \hat{y}) = \begin{cases} 
0 & \text{si } y = \hat{y} \\
1 & \text{si } y \neq \hat{y}
\end{cases}
$$

El **riesgo** (error esperado) de un clasificador $h$ es:

$$
R(h) = \mathbb{E}_{(\mathbf{x},y)}[L(y, h(\mathbf{x}))]
$$

### 3.2 Clasificador de Bayes
El clasificador que minimiza el riesgo puntualmente es el **clasificador de Bayes**:

$$
h^*(\mathbf{x}) = \arg\max_{c} P(y = c \mid \mathbf{x})
$$

donde $P(y=c \mid \mathbf{x})$ es la probabilidad a posteriori de la clase $c$ dado $\mathbf{x}$.

### 3.3 KNN como estimador de la probabilidad a posteriori
KNN estima esta probabilidad como la proporción de vecinos de cada clase:

$$
\hat{P}(y = c \mid \mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}(y_i = c)
$$

Entonces la regla de KNN es simplemente:

$$
\hat{y}(\mathbf{x}) = \arg\max_c \hat{P}(y=c \mid \mathbf{x})
$$

**Propiedad de consistencia:** Bajo condiciones suaves (densidad continua, $k \to \infty$ y $k/n \to 0$), se cumple que $\hat{P}(y=c \mid \mathbf{x}) \xrightarrow{\text{p}} P(y=c \mid \mathbf{x})$ y, por tanto, el riesgo de KNN converge al riesgo de Bayes. Esta es la base teórica que justifica su uso.

---

## 4. Métricas de Distancia – Definición y Propiedades

### 4.1 Distancia Euclidiana (norma $L_2$)
La más común:

$$
d_2(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{j=1}^{d} (p_j - q_j)^2}
$$

**Origen:** Proviene de la norma euclidiana en $\mathbb{R}^d$; satisface:
- $d(\mathbf{p},\mathbf{q}) \ge 0$, igualdad si y solo si $\mathbf{p}=\mathbf{q}$.
- Simetría: $d(\mathbf{p},\mathbf{q}) = d(\mathbf{q},\mathbf{p})$.
- Desigualdad triangular: $d(\mathbf{p},\mathbf{q}) \le d(\mathbf{p},\mathbf{r}) + d(\mathbf{r},\mathbf{q})$.

### 4.2 Otras métricas
- **Manhattan** ($L_1$): $d_1(\mathbf{p},\mathbf{q}) = \sum_{j=1}^{d} |p_j - q_j|$  
  *Más robusta a outliers porque no eleva al cuadrado las diferencias.*
- **Minkowski** ($L_p$): $d_p(\mathbf{p},\mathbf{q}) = \left( \sum_{j=1}^{d} |p_j - q_j|^p \right)^{1/p}$  
  *Generalización; Euclidiana ($p=2$), Manhattan ($p=1$).*
- **Coseno**: $\text{cos}(\mathbf{p},\mathbf{q}) = \frac{\mathbf{p}\cdot\mathbf{q}}{\|\mathbf{p}\|\|\mathbf{q}\|}$  
  *Útil para vectores dispersos (texto), mide similitud angular.*

---

## 5. Optimización de Hiperparámetros

KNN no tiene una fase de entrenamiento iterativo (no hay gradientes), pero el parámetro $k$ (y la métrica) se eligen minimizando el error de clasificación estimado mediante **validación cruzada**.

Para cada $k$ candidato, se calcula:

$$
\text{Error}(k) = \frac{1}{n} \sum_{i=1}^{n} L\left(y_i, \hat{y}_{(-i)}(k)\right)
$$

donde $\hat{y}_{(-i)}(k)$ es la predicción sobre $i$ usando KNN con $k$ vecinos **sin incluir** el punto $i$ (validación leave‑one‑out). Se elige $k$ que minimiza $\text{Error}(k)$.

---

## 6. Complejidad Computacional

### 6.1 Almacenamiento
- **Espacio:** $O(n \cdot d)$ (guardar todas las muestras).

### 6.2 Predicción (sin estructuras de indexación)
- Calcular $n$ distancias: $O(n \cdot d)$.
- Seleccionar los $k$ valores más pequeños: $O(n \log k)$ con heap o $O(n + k \log n)$.
- **Total:** $O(n \cdot d)$ dominante.

### 6.3 Con estructuras de indexación (KD‑Tree, Ball‑Tree)
- En **baja dimensionalidad** ($d$ pequeño): búsqueda $O(\log n \cdot d)$.
- En **alta dimensionalidad**: la eficiencia se degrada exponencialmente porque la estructura no puede podar eficazmente; en el límite, vuelve a $O(n \cdot d)$.

---

## 7. Comportamiento con el Tamaño del Dataset – Demostraciones Matemáticas

### 7.1 Efecto del número de filas $n$
- **$n$ pequeño:** La vecindad contiene pocos puntos, la estimación $\hat{P}(y=c|\mathbf{x})$ tiene alta varianza (ruido).
- **$n$ grande:** El coste $O(n d)$ hace inviable la predicción en tiempo real.  
  *Solución:* técnicas de búsqueda aproximada de vecinos (ANN) que sacrifican exactitud por velocidad.

### 7.2 Maldición de la dimensionalidad – demostración
Supongamos que los puntos $\mathbf{x}_i$ se distribuyen uniformemente en el hipercubo $[0,1]^d$. Para un punto fijo $\mathbf{x}$, sea $R_{\min}$ la distancia al vecino más cercano.

La probabilidad de que ningún punto caiga en una bola de radio $r$ alrededor de $\mathbf{x}$ es:

$$
P(\text{ningún punto en } B(\mathbf{x},r)) = \left(1 - \frac{\text{Vol}(B(\mathbf{x},r))}{\text{Vol}([0,1]^d)}\right)^n
$$

El volumen de una bola de radio $r$ en $\mathbb{R}^d$ es $V_d r^d$, con $V_d$ constante que depende de $d$. Por tanto:

$$
P(\text{ningún punto en } B(\mathbf{x},r)) = \left(1 - V_d r^d\right)^n
$$

Para que el vecino más cercano esté aproximadamente a distancia $r$, igualamos esta probabilidad a $1/2$ (mediana):

$$
\left(1 - V_d r^d\right)^n \approx \frac{1}{2}
$$

Tomando logaritmos y usando $\log(1 - x) \approx -x$ para $x$ pequeño:

$$
- n V_d r^d \approx -\log 2 \quad \Rightarrow \quad r^d \approx \frac{\log 2}{n V_d} \quad \Rightarrow \quad r \approx \left(\frac{\log 2}{n V_d}\right)^{1/d}
$$

Ignorando constantes, se obtiene la conocida relación:

$$
\mathbb{E}[R_{\min}] \propto n^{-1/d}
$$

**Interpretación:**  
- Para $d$ fijo, al aumentar $n$, $r$ disminuye (los vecinos se vuelven más cercanos).  
- Para $n$ fijo, al aumentar $d$, $r$ tiende a 1 (el vecino más cercano se aleja hasta casi el tamaño del dominio).  
- Cuando $d$ es grande, **todos los puntos están aproximadamente a la misma distancia**, por lo que el concepto de “vecino cercano” pierde sentido.

**Consecuencia práctica:** KNN funciona bien cuando $d$ es pequeño ($<20$) o cuando $n$ es exponencial en $d$ (lo cual es inviable en la mayoría de los casos). Por ello es imprescindible **reducir la dimensionalidad** (PCA, selección de características) antes de aplicar KNN si $d$ es alta.

---

## 8. Implicaciones Prácticas Derivadas de la Matemática

### 8.1 Escalado de variables
La distancia euclidiana es sensible a la escala de cada variable. Supongamos dos variables $x^{(1)}$ y $x^{(2)}$ con rangos muy diferentes (ej. $x^{(1)} \in [0,1]$, $x^{(2)} \in [0,10^6]$). Entonces:

$$
d(\mathbf{p},\mathbf{q}) = \sqrt{(p_1-q_1)^2 + (p_2-q_2)^2}
$$

El término $(p_2-q_2)^2$ domina completamente, haciendo que $x^{(1)}$ sea irrelevante. Para equilibrar, se aplica **estandarización** (Z‑score):

$$
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$

donde $\mu_j$ y $\sigma_j$ son la media y desviación estándar de la variable $j$. Así, cada variable contribuye con varianza 1 a la distancia.

### 8.2 Tratamiento de variables categóricas
Las variables categóricas deben convertirse a numéricas (One‑Hot Encoding) y luego escalarse. En algunos casos se usa la distancia de Hamming para datos binarios.

### 8.3 Balanceo de clases
La regla de mayoría es sensible al desbalance: la clase mayoritaria dominará la vecindad. Matemáticamente, si la clase $A$ tiene probabilidad a priori mucho mayor, entonces incluso en regiones donde $B$ es más probable, los vecinos tenderán a ser de $A$ si $k$ no es suficientemente pequeño.  
*Soluciones:* sobremuestreo (SMOTE), submuestreo, o usar pesos por distancia.

### 8.4 Outliers
Un outlier puede convertirse en vecino de muchos puntos si está aislado, distorsionando la predicción. Esto se refleja en la sensibilidad de la distancia euclidiana a valores extremos. Se mitiga con limpieza de datos o usando métricas robustas (Manhattan).

---

## 9. Variantes del Algoritmo – Formulaciones Matemáticas

### 9.1 Weighted KNN (ponderado por distancia)
Cada vecino vota con un peso inversamente proporcional a la distancia:

$$
\hat{y}(\mathbf{x}) = \arg\max_c \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \cdot \mathbf{1}(y_i = c), \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}
$$

(Se añade $\epsilon$ para evitar división por cero.) Esto equivale a un estimador de densidad **kernel** con kernel triangular o gaussiano truncado.

### 9.2 Radius‑based neighbors
En lugar de fijar $k$, se elige un radio $r$ y se toman **todos** los puntos dentro de esa bola:

$$
\hat{y}(\mathbf{x}) = \arg\max_c \sum_{i: d(\mathbf{x}, \mathbf{x}_i) \le r} \mathbf{1}(y_i = c)
$$

Útil cuando la densidad de puntos varía, pero la elección de $r$ es crítica.

### 9.3 KNN para regresión
Cuando $y_i \in \mathbb{R}$, la predicción es el promedio de los valores de los vecinos:

$$
\hat{y}(\mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i
$$

Minimiza el error cuadrático medio local.

### 9.4 Búsqueda aproximada de vecinos (ANN)
Algoritmos como HNSW, Faiss, Annoy implementan estructuras que permiten encontrar vecinos aproximados en tiempo sublineal, sacrificando exactitud. Son esenciales para escalar KNN a conjuntos de millones de puntos.

---

## 10. Resumen de Ecuaciones Clave

| Concepto | Expresión Matemática |
|----------|----------------------|
| Distancia Euclidiana | $d(\mathbf{p},\mathbf{q}) = \sqrt{\sum_{j=1}^{d} (p_j - q_j)^2}$ |
| Probabilidad a posteriori estimada | $\hat{P}(y=c \mid \mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}(y_i = c)$ |
| Regla de decisión | $\hat{y}(\mathbf{x}) = \arg\max_c \hat{P}(y=c \mid \mathbf{x})$ |
| Radio esperado del vecino más cercano | $\mathbb{E}[R_{\min}] \propto n^{-1/d}$ |
| Pérdida 0‑1 | $L(y,\hat{y}) = \mathbf{1}(y \neq \hat{y})$ |
| Validación cruzada para $k$ | $\text{Error}(k) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}(y_i \neq \hat{y}_{(-i)}(k))$ |

---

## 11. Conclusiones y Aplicabilidad

- **Fundamento matemático:** KNN estima la probabilidad a posteriori mediante promedio local; es consistente (converge al clasificador de Bayes).
- **Limitación principal:** la maldición de la dimensionalidad hace que la noción de vecindad se desvanezca cuando $d$ es grande.
- **Exigencias computacionales:** predicción $O(n d)$, por lo que no escala a grandes conjuntos sin estructuras de indexación o búsqueda aproximada.
- **Requisitos de preprocesamiento:** escalado obligatorio, manejo de categóricas, balanceo de clases y limpieza de outliers se derivan directamente de la sensibilidad de la distancia euclidiana.

Esta estructura (problema → definición matemática → pérdida y optimalidad → métricas → optimización → complejidad → análisis asintótico → implicaciones prácticas → variantes) es **replicable** para cualquier modelo de ML, adaptando cada sección a las particularidades del algoritmo (por ejemplo, para SVM se hablará de margen, kernel, dualidad; para Naive Bayes, de independencia condicional y suavizado).

***

## Anexo Matemático y Computacional: Naive Bayes


### 1. Planteamiento del Problema

Tenemos un conjunto de entrenamiento $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, donde  
- $\mathbf{x}_i \in \mathbb{R}^d$ es el vector de características (atributos) de la $i$-ésima muestra,  
- $y_i \in \{1,2,\dots,K\}$ es la clase a la que pertenece.

**Objetivo:** Dado un nuevo punto $\mathbf{x}$ (sin etiqueta), predecir su clase $\hat{y}(\mathbf{x})$ utilizando un modelo probabilístico que estime la **probabilidad a posteriori** $P(y = c \mid \mathbf{x})$.

---

### 2. Definición Matemática del Modelo

#### 2.1 Teorema de Bayes
La probabilidad a posteriori se expresa mediante el teorema de Bayes:

$$
P(y = c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c) \, P(y = c)}{P(\mathbf{x})}
$$

donde:
- $P(y = c)$ es la **probabilidad a priori** de la clase $c$,
- $P(\mathbf{x} \mid y = c)$ es la **verosimilitud** de observar $\mathbf{x}$ dado que la clase es $c$,
- $P(\mathbf{x}) = \sum_{c'} P(\mathbf{x} \mid y = c') P(y = c')$ es un factor de normalización (independiente de $c$).

#### 2.2 Supuesto de independencia condicional (Naive)
Para hacer el problema tratable, Naive Bayes asume que **dada la clase, las características son condicionalmente independientes**:

$$
P(\mathbf{x} \mid y = c) = \prod_{j=1}^{d} P(x_j \mid y = c)
$$

Esta es la suposición *naive* (ingenua). Aunque raramente cierta, simplifica enormemente la estimación.

#### 2.3 Regla de decisión
Sustituyendo en Bayes y omitiendo el denominador (constante para todas las clases), la predicción es:

$$
\hat{y}(\mathbf{x}) = \arg\max_{c} \; P(y = c) \prod_{j=1}^{d} P(x_j \mid y = c)
$$

---

### 3. Función de Pérdida y Relación con el Clasificador de Bayes

#### 3.1 Pérdida 0‑1
En clasificación, la pérdida típica es la **pérdida 0‑1**:

$$
L(y, \hat{y}) = \begin{cases} 
0 & \text{si } y = \hat{y} \\
1 & \text{si } y \neq \hat{y}
\end{cases}
$$

El **riesgo** (error esperado) se minimiza eligiendo la clase con mayor probabilidad a posteriori:

$$
h^*(\mathbf{x}) = \arg\max_{c} P(y = c \mid \mathbf{x})
$$

Naive Bayes es una aproximación a este clasificador óptimo, con la única diferencia de que usa la independencia condicional para estimar $P(\mathbf{x} \mid y = c)$.

#### 3.2 Justificación
Si el supuesto de independencia se cumple, Naive Bayes es exactamente el clasificador de Bayes. En la práctica, incluso cuando no se cumple, a menudo produce buenos resultados porque la regla de decisión puede ser robusta ante dependencias moderadas.

---

### 4. Estimación de Parámetros – Maximización de Verosimilitud

Los parámetros del modelo son las probabilidades a priori $P(y = c)$ y los parámetros de las distribuciones condicionales $P(x_j \mid y = c)$. Se estiman a partir de los datos de entrenamiento mediante **máxima verosimilitud**.

#### 4.1 Probabilidades a priori
La estimación natural es la frecuencia relativa:

$$
\hat{P}(y = c) = \frac{n_c}{n}, \quad \text{donde } n_c = \sum_{i=1}^n \mathbf{1}(y_i = c)
$$

#### 4.2 Distribuciones condicionales – según tipo de dato
La forma de $P(x_j \mid y = c)$ depende de la naturaleza de la característica $x_j$.

- **GaussianNB** (características continuas): se asume que $x_j$ sigue una distribución normal dentro de cada clase:
  $$
  P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)
  $$
  Los parámetros $\mu_{jc}$ y $\sigma_{jc}^2$ se estiman como la media y varianza muestral de los puntos de la clase $c$ en la característica $j$:
  $$
  \hat{\mu}_{jc} = \frac{1}{n_c} \sum_{i: y_i = c} x_{ij}, \quad \hat{\sigma}_{jc}^2 = \frac{1}{n_c} \sum_{i: y_i = c} (x_{ij} - \hat{\mu}_{jc})^2
  $$

- **MultinomialNB** (características de conteo, típicamente texto): cada característica representa la frecuencia de una palabra en un documento. Se asume una distribución multinomial condicionada a la clase:
  $$
  P(\mathbf{x} \mid y = c) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_j \theta_{jc}^{x_j}
  $$
  donde $\theta_{jc}$ es la probabilidad de que la palabra $j$ aparezca en un documento de clase $c$. La estimación de máxima verosimilitud (con suavizado Laplace) es:
  $$
  \hat{\theta}_{jc} = \frac{n_{jc} + \alpha}{\sum_{j'} n_{j'c} + \alpha d}
  $$
  siendo $n_{jc}$ la suma de conteos de la característica $j$ en los documentos de clase $c$, $\alpha$ el parámetro de suavizado (por defecto 1).

- **BernoulliNB** (características binarias): cada característica indica presencia/ausencia. La verosimilitud es un producto de Bernoulli:
  $$
  P(x_j \mid y = c) = \theta_{jc}^{x_j} (1 - \theta_{jc})^{1 - x_j}
  $$
  con estimación suavizada:
  $$
  \hat{\theta}_{jc} = \frac{n_{jc} + \alpha}{n_c + 2\alpha}
  $$
  donde $n_{jc}$ es el número de documentos de clase $c$ que contienen la característica $j$.

#### 4.3 Suavizado (Laplace / Lidstone)
El parámetro $\alpha$ (a menudo 1 para Laplace) evita probabilidades cero cuando una característica no aparece en una clase, lo que de otro modo anularía todo el producto. Matemáticamente, el suavizado corresponde a una **regularización** que introduce un sesgo pero reduce la varianza.

---

### 5. Optimización / Entrenamiento

A diferencia de modelos iterativos (como SVM o redes neuronales), **Naive Bayes tiene una solución de forma cerrada**: basta con calcular frecuencias y promedios a partir de los datos. Esto lo hace extremadamente rápido.

**Complejidad del entrenamiento**:
- Calcular $P(y=c)$: $O(n)$ (contar clases).
- Para cada característica y clase, calcular parámetros: $O(n \cdot d)$ (una pasada por todos los datos).
- **Total:** $O(n \cdot d)$, pero con una constante muy pequeña (solo sumas y conteos).

---

### 6. Hiperparámetros y su Efecto Matemático

| Hiperparámetro | Variante | Efecto matemático |
|----------------|----------|-------------------|
| `alpha` | Multinomial, Bernoulli | Suavizado de Laplace. $alpha=1$ (default) evita probabilidades cero; valores grandes aumentan el sesgo (uniformizan las distribuciones). |
| `var_smoothing` | GaussianNB | Pequeña cantidad añadida a la varianza para evitar divisiones por cero y estabilizar estimaciones. |
| `fit_prior` | Todas | Si `False`, se fuerza $P(y=c)=1/K$ (prior uniforme), lo que puede mitigar desbalance de clases. |

**Optimización:** Se eligen mediante validación cruzada, normalmente con búsqueda en rejilla sobre $\alpha$ (escala logarítmica) o `var_smoothing`.

---

### 7. Complejidad Computacional

#### 7.1 Entrenamiento
- **Tiempo:** $O(n \cdot d)$ (una sola pasada).
- **Espacio:** $O(d \cdot K)$ (almacena matrices de parámetros: medias/varianzas para Gaussian, $\theta_{jc}$ para Multinomial/Bernoulli). Es independiente de $n$, a diferencia de KNN.

#### 7.2 Predicción (inferencia)
Para un nuevo punto $\mathbf{x}$, se evalúa:
$$
\log P(y=c) + \sum_{j=1}^d \log P(x_j \mid y=c)
$$
- **Tiempo:** $O(d \cdot K)$ (producto sobre características y suma sobre clases).  
- **Espacio:** $O(K)$ (probabilidades temporales).

**Conclusión:** La inferencia es extremadamente rápida, incluso para millones de características (alta dimensionalidad). Esto hace a Naive Bayes ideal para aplicaciones en tiempo real y texto.

---

### 8. Comportamiento con el Tamaño del Dataset – Demostraciones

#### 8.1 Número de muestras $n$
- **Pequeño $n$:** La estimación de parámetros tiene alta varianza, pero el suavizado ayuda a evitar overfitting extremo. Naive Bayes suele funcionar razonablemente bien con pocos datos.
- **Grande $n$:** La estimación converge a los parámetros verdaderos (consistencia). No hay cuellos de botella computacional porque el entrenamiento es $O(n d)$.

#### 8.2 Número de características $d$ (dimensionalidad)
Naive Bayes **no sufre la maldición de la dimensionalidad** como KNN. La razón matemática: el modelo factoriza la probabilidad conjunta en un producto de univariantes, por lo que la complejidad crece linealmente con $d$ y la estimación de cada $P(x_j \mid y=c)$ se realiza de forma independiente. Incluso con $d$ enorme (por ejemplo, millones de palabras en texto), el entrenamiento e inferencia siguen siendo lineales en $d$.

**Demostración informal:**  
El número de parámetros es $O(d \cdot K)$ (cada característica-clase tiene sus parámetros). No hay interacciones entre características, por lo que no se requiere una muestra exponencial en $d$ para estimar la distribución conjunta. Esta es la ventaja fundamental del supuesto de independencia.

---

### 9. Implicaciones Prácticas Derivadas de la Matemática

#### 9.1 Escalado de variables
- **GaussianNB:** El escalado no es necesario, pero puede ayudar a la interpretación. La razón es que las medias y varianzas se estiman directamente; la escala afecta ambos parámetros por igual, pero no cambia las probabilidades relativas (si se usa la misma transformación para todas las variables).
- **Multinomial/Bernoulli:** El escalado no aplica porque trabajan con conteos o binarios.

#### 9.2 Manejo de variables categóricas
Las categóricas pueden codificarse como binarias (One‑Hot Encoding) y usarse con BernoulliNB, o transformarse a frecuencias para MultinomialNB.

#### 9.3 Desbalance de clases
Naive Bayes es sensible al desbalance porque usa las prioridades empíricas. Matemáticamente, si una clase es muy minoritaria, $P(y=c)$ pequeño puede dominar el producto. Se puede mitigar:
- Usando `fit_prior=False` (prior uniforme).
- Balanceando el dataset antes de entrenar.
- Usando ComplementNB (variante que ajusta las probabilidades para clases minoritarias).

#### 9.4 Outliers
- **GaussianNB:** los outliers afectan la media y varianza, lo que puede distorsionar la estimación. Es recomendable limpiar datos o usar técnicas robustas.
- **Multinomial/Bernoulli:** son más robustos porque trabajan con conteos acumulados.

---

### 10. Variantes del Algoritmo

- **ComplementNB:** Modifica MultinomialNB para datos desbalanceados; utiliza probabilidades complementarias (1 - θ) para mejorar el rendimiento en la clase minoritaria.
- **CategoricalNB:** Diseñado para variables categóricas con distribución multinomial.
- **GaussianNB, MultinomialNB, BernoulliNB** ya descritas.

---

### 11. Resumen de Ecuaciones Clave

| Concepto | Expresión Matemática |
|----------|----------------------|
| Regla de decisión (log‑probabilidades) | $\hat{y}(\mathbf{x}) = \arg\max_c \left[ \log P(y=c) + \sum_{j=1}^d \log P(x_j \mid y=c) \right]$ |
| GaussianNB: densidad | $P(x_j \mid y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left(-\frac{(x_j-\mu_{jc})^2}{2\sigma_{jc}^2}\right)$ |
| MultinomialNB: parámetros suavizados | $\hat{\theta}_{jc} = \frac{n_{jc} + \alpha}{\sum_{j'} n_{j'c} + \alpha d}$ |
| BernoulliNB: parámetros suavizados | $\hat{\theta}_{jc} = \frac{n_{jc} + \alpha}{n_c + 2\alpha}$ |
| Estimación de prior | $\hat{P}(y=c) = \frac{n_c}{n}$ |
| Complejidad entrenamiento | $O(n \cdot d)$ |
| Complejidad predicción | $O(d \cdot K)$ |

---

### 12. Conclusiones y Aplicabilidad

- **Fundamento matemático:** Naive Bayes es un clasificador probabilístico basado en el teorema de Bayes con independencia condicional. Aunque el supuesto es fuerte, la simplicidad permite estimaciones de parámetros rápidas y escalables.
- **Ventaja clave:** Escala lineal con la dimensionalidad y el número de muestras; predicción instantánea. Ideal para **texto** (spam, sentimiento) y **alta dimensionalidad**.
- **Limitaciones:** La independencia condicional rara vez se cumple, lo que puede degradar el rendimiento si las características están fuertemente correlacionadas. Las probabilidades predichas suelen estar mal calibradas.
- **Requisitos:** Para GaussianNB, es útil escalar (aunque no obligatorio); para Multinomial/Bernoulli, los datos deben ser conteos o binarios. El suavizado es esencial para evitar probabilidades cero.

***


## Anexo Matemático y Computacional: Support Vector Machines (SVM)


### 1. Planteamiento del Problema

Tenemos un conjunto de entrenamiento $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, donde  
- $\mathbf{x}_i \in \mathbb{R}^d$ es el vector de características,  
- $y_i \in \{-1, +1\}$ es la clase (binaria).  

**Objetivo:** Encontrar un clasificador lineal de la forma $h(\mathbf{x}) = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$ que no solo separe correctamente las clases, sino que además maximice el **margen** (distancia entre el hiperplano y los puntos más cercanos de cada clase). Esta propiedad confiere robustez y generalización.

---

### 2. Definición Matemática del Modelo

#### 2.1 Hiperplano de separación
Un hiperplano está definido por $(\mathbf{w}, b)$ con $\mathbf{w} \in \mathbb{R}^d$, $b \in \mathbb{R}$:

$$
\mathbf{w}^\top \mathbf{x} + b = 0
$$

La distancia de un punto $\mathbf{x}$ al hiperplano es $\frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|}$.

#### 2.2 Clasificador lineal
Clasificamos según:

$$
\hat{y}(\mathbf{x}) = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)
$$

#### 2.3 Margen y restricciones
Para que el hiperplano separe los datos con margen $\gamma$, exigimos:

$$
y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge \gamma, \quad \forall i
$$

Sin pérdida de generalidad, fijamos $\gamma = 1$ (escalando $\mathbf{w}, b$ adecuadamente). Así, el problema se convierte en:

$$
y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \quad \forall i
$$

#### 2.4 Máximo margen
El margen geométrico es $1/\|\mathbf{w}\|$. Maximizar el margen equivale a minimizar $\|\mathbf{w}\|^2$. Por lo tanto, el **problema primal** (caso separable) es:

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{sujeto a} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \quad i=1,\dots,n
$$

---

### 3. Función de Pérdida y Relación con el Riesgo Regularizado

Para datos no separables, se introducen **variables de holgura** $\xi_i \ge 0$ que permiten violar la restricción de margen, penalizando esas violaciones. El problema primal **SVM de margen blando** (C‑SVM) es:

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{sujeto a} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
$$

donde $C > 0$ es el parámetro de regularización.

Esta formulación equivale a minimizar el riesgo empírico regularizado con **pérdida hinge**:

$$
L_{\text{hinge}}(y, f(\mathbf{x})) = \max(0, 1 - y f(\mathbf{x})), \quad \text{donde } f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b
$$

El problema primal es equivalente a:

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))
$$

**Interpretación:** La pérdida hinge es convexa, suave a trozos, y penaliza las violaciones del margen de manera lineal. La regularización $\frac{1}{2}\|\mathbf{w}\|^2$ controla la complejidad.

---

### 4. Optimización / Entrenamiento – Dualidad y Kernels

#### 4.1 Formulación dual
El problema primal es convexo y puede resolverse mediante la teoría de **dualidad de Lagrange**. La función lagrangiana es:

$$
L(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\beta}) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i - \sum_i \alpha_i \big[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i \big] - \sum_i \beta_i \xi_i
$$

con $\alpha_i \ge 0$, $\beta_i \ge 0$.  
Derivando con respecto a $\mathbf{w}$, $b$, $\xi_i$ e igualando a cero:

- $\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = 0 \;\Rightarrow\; \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$
- $\frac{\partial L}{\partial b} = -\sum_i \alpha_i y_i = 0 \;\Rightarrow\; \sum_i \alpha_i y_i = 0$
- $\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \beta_i = 0 \;\Rightarrow\; 0 \le \alpha_i \le C$

Sustituyendo en la lagrangiana, se obtiene el **problema dual**:

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^\top \mathbf{x}_j)
$$

sujeto a $0 \le \alpha_i \le C$ y $\sum_i \alpha_i y_i = 0$.

#### 4.2 Kernel trick
Si se mapean los datos a un espacio de características de mayor dimensión mediante una función $\phi(\mathbf{x})$, el producto punto en ese espacio puede calcularse mediante un **kernel** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. La formulación dual solo requiere productos punto, por lo que podemos trabajar con kernels directamente:

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

Kernels comunes:
- **Lineal:** $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{x}_j$
- **Polinomial:** $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^\top \mathbf{x}_j + r)^p$
- **RBF (gaussiano):** $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$
- **Sigmoide:** $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^\top \mathbf{x}_j + r)$

#### 4.3 Vectores de soporte
La solución $\boldsymbol{\alpha}^*$ es dispersa: la mayoría de $\alpha_i$ son 0. Los puntos con $\alpha_i > 0$ son los **vectores de soporte**. La predicción para un nuevo punto $\mathbf{x}$ es:

$$
\hat{y}(\mathbf{x}) = \text{sign}\left( \sum_{i: \alpha_i > 0} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)
$$

donde $b$ se obtiene de cualquier vector de soporte con $0 < \alpha_i < C$: $b = y_i - \sum_{j} \alpha_j y_j K(\mathbf{x}_j, \mathbf{x}_i)$.

---

### 5. Complejidad Computacional

#### 5.1 Entrenamiento
- **Algoritmo SMO (Sequential Minimal Optimization):** La implementación estándar (libsvm) tiene complejidad entre $O(n^2)$ y $O(n^3)$ en el peor caso, dependiendo de la dificultad del problema. En la práctica, suele ser $O(n^2 \cdot n_{\text{sv}})$ o mejor.
- **Memoria:** La matriz de kernel (o su almacenamiento) requiere $O(n^2)$ en memoria si se almacena explícitamente. Para grandes $n$, se utilizan aproximaciones (kernel caching, etc.).

#### 5.2 Predicción (inferencia)
Para un punto $\mathbf{x}$, se evalúa:

$$
\sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b
$$

- **Kernel lineal:** $O(d \cdot n_{\text{sv}})$ (producto punto con cada vector de soporte).
- **Kernel RBF:** $O(d \cdot n_{\text{sv}})$ (hay que calcular distancia Euclidiana por cada vector de soporte).
- **En la práctica,** si $n_{\text{sv}}$ es grande, la predicción puede ser costosa. Sin embargo, $n_{\text{sv}}$ suele ser menor que $n$.

---

### 6. Comportamiento con el Tamaño del Dataset – Demostraciones

#### 6.1 Número de muestras $n$
- **Pequeño $n$:** SVM puede sobreajustar si $C$ es grande. La regularización controla este efecto.
- **Grande $n$:** El entrenamiento se vuelve costoso debido a la complejidad cuadrática/cúbica. Para $n > 50,000$, se recomienda usar **LinearSVC** (kernel lineal con optimización de gradiente) o **SGDClassifier** con pérdida hinge.

#### 6.2 Número de características $d$
- **Lineal:** La complejidad de entrenamiento depende de $n$ y no de $d$ explícitamente (SMO opera sobre pares de puntos). Sin embargo, el almacenamiento de $\mathbf{w}$ es $O(d)$.
- **No lineal:** La complejidad depende del número de vectores de soporte, que puede crecer con $n$ y la dificultad del problema. En alta dimensión, el kernel lineal suele ser más eficiente.

**Derivación del límite de generalización:** SVM tiene una cota de error de generalización basada en el número de vectores de soporte y el margen. En particular, la **cota de Vapnik‑Chervonenkis (VC)** muestra que con margen grande, la capacidad del modelo es menor, lo que justifica la búsqueda de máximo margen.

---

### 7. Hiperparámetros y su Efecto Matemático

| Hiperparámetro | Valores típicos | Efecto matemático |
|----------------|-----------------|-------------------|
| **$C$** | $10^{-3}, 0.1, 1, 10, 100$ | Controla el trade‑off entre margen y violaciones. $C$ grande → margen estrecho, menor sesgo, mayor varianza. $C$ pequeño → margen amplio, mayor sesgo. |
| **$\gamma$** (RBF) | $10^{-3}, 0.01, 0.1, 1$ | Inverso de la escala del kernel. $\gamma$ grande → cada punto influye solo localmente (alta varianza). $\gamma$ pequeño → influencia global (sesgo alto). |
| **$p$** (polinomial) | 2, 3 | Grado del polinomio; mayor $p$ aumenta la flexibilidad. |
| **$r$** (polinomial) | 0, 1 | Término independiente. |

**Optimización:** Búsqueda en rejilla (GridSearchCV) sobre $C$ y $\gamma$ (escala logarítmica) con validación cruzada.

---

### 8. Implicaciones Prácticas Derivadas de la Matemática

#### 8.1 Escalado de variables
**Obligatorio.** La función objetivo depende de $\|\mathbf{w}\|^2$ y las restricciones contienen productos punto. Si las variables tienen escalas diferentes, el margen se distorsiona. La estandarización (Z‑score) es estándar.

#### 8.2 Tratamiento de variables categóricas
Deben codificarse numéricamente (One‑Hot Encoding) y luego escalarse. El kernel lineal funcionará con ellas; los kernels no lineales pueden explotar la estructura.

#### 8.3 Desbalance de clases
El SVM estándar es sensible al desbalance porque la pérdida hinge trata todas las violaciones igualmente. Solución: usar `class_weight='balanced'` que ajusta el peso de cada clase inversamente a su frecuencia, o muestrear.

#### 8.4 Outliers
Los outliers pueden convertirse en vectores de soporte y distorsionar el margen. Un $C$ pequeño mitiga su influencia. Limpiar outliers antes de entrenar es recomendable.

#### 8.5 Interpretabilidad
- **Kernel lineal:** los coeficientes $\mathbf{w}$ pueden interpretarse como la importancia de cada característica (similar a regresión logística).
- **Kernel RBF/polinomial:** la decisión se basa en similitudes con vectores de soporte; se necesitan métodos post‑hoc (SHAP, LIME) para explicar.

---

### 9. Variantes del Algoritmo

- **SVM para regresión (SVR):** busca una banda de tolerancia $\epsilon$ en lugar de margen de separación.
- **Nu‑SVM:** parametriza el número de vectores de soporte y el margen.
- **One‑Class SVM:** para detección de anomalías (aprende una frontera que envuelve los datos normales).
- **LinearSVC:** implementación eficiente para kernel lineal usando optimización dual o primal (liblinear), con complejidad casi lineal en $n$.

---

### 10. Resumen de Ecuaciones Clave

| Concepto | Expresión Matemática |
|----------|----------------------|
| Problema primal (margen blando) | $\displaystyle \min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i,\; \text{s.a. } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0$ |
| Pérdida hinge | $\displaystyle L_{\text{hinge}}(y, f) = \max(0, 1 - y f)$ |
| Problema dual | $\displaystyle \max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j),\; \text{s.a. } 0 \le \alpha_i \le C,\; \sum_i \alpha_i y_i = 0$ |
| Predicción | $\displaystyle \hat{y}(\mathbf{x}) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$ |
| Condiciones KKT | $\alpha_i [y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i] = 0$; $(C - \alpha_i)\xi_i = 0$ |
| Kernel RBF | $\displaystyle K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ |
| Complejidad entrenamiento (SMO) | $O(n^2)$ a $O(n^3)$ en la práctica |
| Complejidad predicción | $O(n_{\text{sv}} \cdot d)$ |

---

### 11. Conclusiones y Aplicabilidad

- **Fundamento matemático:** SVM resuelve un problema de optimización convexa que maximiza el margen, garantizando una solución única y una cota de generalización.
- **Kernel trick:** permite manejar no linealidad sin explotar el espacio de características, manteniendo la complejidad dependiente del número de muestras y no de la dimensión del espacio transformado.
- **Limitaciones:** El entrenamiento es cuadrático en $n$, lo que limita su uso a datasets de tamaño moderado (hasta decenas de miles). Para datasets grandes, se recurre a SVM lineal o aproximaciones.
- **Requisitos:** Escalado obligatorio; ajuste fino de hiperparámetros (C, gamma) mediante validación cruzada; manejo de desbalance mediante pesos.

SVM es una elección robusta para problemas de clasificación con relaciones no lineales, especialmente cuando la dimensionalidad es moderada y el número de muestras no es excesivamente grande. Su base matemática sólida y su capacidad de generalización lo convierten en un algoritmo fundamental en el arsenal de ML.