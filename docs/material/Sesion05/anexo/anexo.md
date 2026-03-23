---
layout: default
---

# Fundamento Matemático y Computacional de k-NN, Naive Bayes y SVM
#### Autor: Carlos César Sánchez Coronel

[⬅️ Volver a la Sesión-05](../../../sesiones/sesion-05.md)



---

## 1. Planteamiento General del Problema



He revisado el anexo y efectivamente contiene la información clave, pero la estructura puede afinarse para que sea **replicable** a cualquier modelo de ML y tenga una **secuencia lógica** que parta del problema, avance por la matemática subyacente, demuestre propiedades y culmine en las implicaciones computacionales y prácticas. A continuación presento una versión reorganizada y complementada, con explicaciones paso a paso, leyendas y un formato que sirva de plantilla para futuros anexos (Naive Bayes, SVM, etc.).

---

# Anexo Matemático y Computacional: K‑Nearest Neighbors (KNN)

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