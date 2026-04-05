

## Datos de ejemplo 

| Muestra | $x$ (única característica numérica) | $y$ (etiqueta real: 0 o 1) |
|--------|--------------------------------------|----------------------------|
| 1      | 1.0                                  | 0                          |
| 2      | 2.0                                  | 0                          |
| 3      | 3.0                                  | 1                          |
| 4      | 4.0                                  | 1                          |
| 5      | 5.0                                  | 1                          |

Número de muestras $N = 5$.  
Suma de $y$ = $0+0+1+1+1 = 3$.

---

## Paso 0: Inicialización del modelo (¿por qué empezamos con una constante?)

### ¿Qué significa “inicializar” un modelo de boosting?
Boosting construye un conjunto de modelos (árboles) de forma secuencial. Necesitamos un **punto de partida** antes del primer árbol. Ese punto de partida es un modelo muy simple: una constante $F_0(x) = c$ (el mismo valor para todas las muestras).  

### ¿Cómo elegimos esa constante $c$?
Elegimos $c$ que **minimice la función de pérdida** sobre los datos de entrenamiento. Para **clasificación binaria** usamos la **log-loss** (también llamada entropía cruzada):

$$
L(y, F) = - \big[ y \cdot \ln(p) + (1-y) \cdot \ln(1-p) \big]
$$

donde $p = \sigma(F) = \frac{1}{1+e^{-F}}$ (función sigmoide, convierte un número real $F$ en una probabilidad entre 0 y 1).  

Queremos encontrar $c$ que minimice:

$$
\sum_{i=1}^N L\big(y_i, c\big) = - \sum_{i=1}^N \big[ y_i \ln(\sigma(c)) + (1-y_i)\ln(1-\sigma(c)) \big]
$$

Derivamos respecto a $c$ e igualamos a cero. Se puede demostrar (lo omito aquí por brevedad, pero es un resultado conocido) que el óptimo es:

$$
\sigma(c) = \frac{\sum_{i=1}^N y_i}{N} = \text{proporción de unos}
$$

Es decir, la mejor constante inicial es **la media de las etiquetas**.

### Aplicación a nuestros datos
Media de $y$ = $3/5 = 0.6$.  
Por tanto $\sigma(c) = 0.6$.  

Ahora necesitamos $c$ tal que $\sigma(c)=0.6$. Despejamos:

$$
\frac{1}{1+e^{-c}} = 0.6 \quad\Rightarrow\quad 1+e^{-c} = \frac{1}{0.6} = 1.6666\ldots \quad\Rightarrow\quad e^{-c} = 0.6666\ldots \quad\Rightarrow\quad -c = \ln(0.6666\ldots) \quad\Rightarrow\quad c = -\ln(0.6666\ldots) = \ln(1.5) \approx 0.405465
$$

A este valor lo llamamos $F_0$ (predicción inicial en escala lineal).  

Para cada muestra $i$, la **probabilidad inicial** es $\hat{p}_i^{(0)} = \sigma(F_0) = 0.6$ (constante para todos).  

**Resumen del Paso 0**:
- $F_0 = 0.405465$ (log-odds inicial)
- $\hat{p}^{(0)} = 0.6$ (probabilidad inicial)

No hay nada inventado: sale de minimizar la pérdida sobre los datos.

---

## Paso 1: Calcular el gradiente y la hessiana de la pérdida (¿por qué estos conceptos?)

### ¿Qué es el gradiente en este contexto?
En cada iteración del boosting, queremos construir un nuevo árbol que **separe bien** las muestras donde el modelo actual se equivoca. La forma más general de medir “error” es mediante la **derivada de la pérdida** con respecto a la predicción actual.  

Para una muestra $(x_i, y_i)$, con predicción actual $F(x_i)$ (en escala log-odds), definimos:

- **Gradiente** $g_i = \frac{\partial L(y_i, F)}{\partial F} \bigg|_{F=F(x_i)}$  
  Indica la dirección y magnitud en la que debemos cambiar $F$ para reducir la pérdida. Un gradiente positivo significa que aumentar $F$ aumenta la pérdida (luego debemos disminuir $F$).

- **Hessiana** $h_i = \frac{\partial^2 L}{\partial F^2}$  
  Indica la curvatura de la pérdida. Nos servirá para hacer una aproximación de segundo orden (más rápida que el gradiente simple).

### Calcular el gradiente y la hessiana para la log-loss
Recordemos: $L = -[y \ln(\sigma(F)) + (1-y)\ln(1-\sigma(F))]$.  
Primera derivada (ya la derivamos antes):

$$
g_i = \frac{\partial L}{\partial F} = \sigma(F) - y_i
$$

Segunda derivada:

$$
h_i = \frac{\partial^2 L}{\partial F^2} = \sigma(F) \cdot (1 - \sigma(F))
$$

*(No voy a derivar ahora para no alargar, pero es un resultado estándar que puedes verificar con la regla de la cadena sabiendo que $\sigma'(F) = \sigma(F)(1-\sigma(F))$.)*

### Aplicación a nuestros datos (después del Paso 0)
Para todas las muestras, $\sigma(F_0)=0.6$.  
Entonces:

- $g_i = 0.6 - y_i$
- $h_i = 0.6 \times (1-0.6) = 0.6 \times 0.4 = 0.24$ (constante para todos, porque la probabilidad es la misma).

Calculamos muestra por muestra:

| i | $y_i$ | $g_i = 0.6 - y_i$ | $h_i$ |
|---|-------|-------------------|-------|
| 1 | 0     | 0.6 - 0 = 0.6     | 0.24  |
| 2 | 0     | 0.6 - 0 = 0.6     | 0.24  |
| 3 | 1     | 0.6 - 1 = -0.4    | 0.24  |
| 4 | 1     | 0.6 - 1 = -0.4    | 0.24  |
| 5 | 1     | 0.6 - 1 = -0.4    | 0.24  |

**Interpretación**:
- Para $y=0$, $g_i=0.6$ positivo → aumentar $F$ aumenta la pérdida → necesitamos **disminuir** $F$ (lo hará el árbol).
- Para $y=1$, $g_i=-0.4$ negativo → aumentar $F$ **reduce** la pérdida → necesitamos **aumentar** $F$.

El gradiente nos da la dirección deseada. El árbol se ajustará a estos $g_i$ (con signo negativo, como veremos).

---

## Paso 2: Construir el primer árbol (un stump de profundidad 1)

### ¿Por qué un árbol y no otra función?
Porque los árboles de decisión pueden modelar interacciones simples y son rápidos. En este ejemplo usamos **un solo split** (profundidad 1) para que sea fácil de calcular.

### ¿Cómo decide XGBoost dónde dividir?
XGBoost usa una **función de ganancia** basada en la reducción de la pérdida aproximada. Para cada posible punto de corte, calcula la suma de gradientes y hessianas en las hojas izquierda y derecha, y luego la ganancia.  

**Simplificación para nuestro cálculo manual**: Probaremos el corte $x \le 2.5$ (dejando las muestras 1-2 a la izquierda, 3-5 a la derecha). Es un corte razonable porque separa los ceros de los unos.

### ¿Qué valores debe devolver el árbol en cada hoja?
Llamemos $w_L$ al valor (en escala log-odds) que devuelve el árbol para muestras en la hoja izquierda, y $w_R$ para la derecha. Queremos elegir $w_L$ y $w_R$ que minimicen la pérdida **de forma local** (dado el modelo actual $F_0$).

En lugar de minimizar la pérdida exacta, XGBoost minimiza una **aproximación de segundo orden** (expansión de Taylor) alrededor de $F_0$:

Para una muestra $i$ en la hoja $j$, si añadimos $w_j$ al modelo actual, la nueva predicción es $F_0 + w_j$. La pérdida aproximada es:

$$
L(y_i, F_0 + w_j) \approx L(y_i, F_0) + g_i w_j + \frac{1}{2} h_i w_j^2
$$

Sumando sobre todas las muestras de la hoja $j$:

$$
\sum_{i \in R_j} L(y_i, F_0) + w_j \sum_{i \in R_j} g_i + \frac{1}{2} w_j^2 \sum_{i \in R_j} h_i
$$

El primer término no depende de $w_j$. Minimizamos la suma de los dos últimos respecto a $w_j$ (derivando e igualando a cero):

$$
\frac{d}{dw_j} \left[ w_j \sum g_i + \frac{1}{2} w_j^2 \sum h_i \right] = \sum g_i + w_j \sum h_i = 0
$$

Despejamos:

$$
w_j = - \frac{\sum_{i \in R_j} g_i}{\sum_{i \in R_j} h_i}
$$

**¿Por qué el signo negativo?** Porque el gradiente $g_i$ nos indica la dirección de *máximo crecimiento* de la pérdida. Para *reducir* la pérdida, debemos movernos en dirección opuesta: $-\frac{g}{h}$ (es un paso de Newton).

**Nota**: En XGBoost real hay un término de regularización $\lambda$ en el denominador: $w_j = - \frac{\sum g_i}{\sum h_i + \lambda}$, pero aquí lo ponemos a cero para simplicidad.

### Aplicación a nuestras hojas

**Hoja izquierda** $R_L$ (muestras 1,2):
- $\sum g = 0.6 + 0.6 = 1.2$
- $\sum h = 0.24 + 0.24 = 0.48$
- $w_L = -\frac{1.2}{0.48} = -2.5$

**Hoja derecha** $R_R$ (muestras 3,4,5):
- $\sum g = (-0.4) + (-0.4) + (-0.4) = -1.2$
- $\sum h = 0.24 \times 3 = 0.72$
- $w_R = -\frac{-1.2}{0.72} = \frac{1.2}{0.72} = 1.666666\ldots$

Por tanto, el árbol $h_1(x)$ (lo llamamos así, el primer árbol) devuelve:

$$
h_1(x) = \begin{cases}
-2.5 & \text{si } x \le 2.5 \\
1.6667 & \text{si } x > 2.5
\end{cases}
$$

**Interpretación**: Para muestras con $x$ pequeña (ceros), el árbol propone **restar** 2.5 (en log-odds) para corregir la sobreestimación inicial (porque $F_0=0.405$ era demasiado alto para esos casos). Para muestras con $x$ grande (unos), el árbol propone **sumar** 1.6667 para corregir la subestimación.

---

## Paso 3: Actualizar el modelo con un factor de aprendizaje (learning rate)

### ¿Por qué un learning rate $\eta$ menor que 1?
Si añadiéramos el árbol completo ($w_L$ y $w_R$ sin escalar), podríamos sobreajustar rápidamente. El **learning rate** $\eta$ (también llamado *shrinkage*) reduce la contribución de cada árbol, lo que obliga al modelo a tomar muchos pasos pequeños y mejora la generalización.

Elegimos $\eta = 0.5$ (un valor común, aunque a menudo se usa 0.1 o 0.3).

La nueva predicción (en escala log-odds) es:

$$
F_1(x) = F_0(x) + \eta \cdot h_1(x)
$$

Aplicamos muestra por muestra:

| i | $x$ | $F_0$ | $h_1(x)$ | $\eta \cdot h_1$ | $F_1 = F_0 + \eta h_1$ |
|---|-----|-------|----------|------------------|------------------------|
| 1 | 1.0 | 0.405465 | -2.5 | -1.25 | -0.844535 |
| 2 | 2.0 | 0.405465 | -2.5 | -1.25 | -0.844535 |
| 3 | 3.0 | 0.405465 | 1.666667 | 0.8333335 | 1.2387985 |
| 4 | 4.0 | 0.405465 | 1.666667 | 0.8333335 | 1.2387985 |
| 5 | 5.0 | 0.405465 | 1.666667 | 0.8333335 | 1.2387985 |

Finalmente, podemos convertir estos log-odds a probabilidades usando la sigmoide:

$$
\hat{p}_1 = \sigma(F_1) = \frac{1}{1+e^{-F_1}}
$$

- Para $F_1 = -0.844535$: $e^{-(-0.844535)} = e^{0.844535} \approx 2.326$ → $\hat{p} = 1/(1+2.326) = 0.3006$
- Para $F_1 = 1.2387985$: $e^{-1.2388} \approx 0.2896$ → $\hat{p} = 1/(1+0.2896) = 0.7753$

**Resultado**: Después de un solo árbol (con $\eta=0.5$), las predicciones pasaron de ser todas 0.6 a:
- Muestras con $y=0$ → 0.3006 (más cerca de 0)
- Muestras con $y=1$ → 0.7753 (más cerca de 1)

La pérdida logarítmica ha disminuido (no lo calculamos, pero es evidente).

---

## ¿Por qué este proceso funciona? (explicación conceptual)

- **Gradiente** indica dirección de corrección.
- **Hessiana** pondera la magnitud (paso de Newton) para converger más rápido que el gradiente simple.
- **Árbol** agrupa muestras con gradientes similares y asigna una corrección conjunta.
- **Learning rate** suaviza el progreso.

Para **regresión** con pérdida MSE, el gradiente es $\hat{y} - y$ (el residuo clásico) y la hessiana es constante $=1$, por lo que $w_j$ es simplemente el promedio de los residuos en la hoja (sin regularización). El proceso es idéntico.

