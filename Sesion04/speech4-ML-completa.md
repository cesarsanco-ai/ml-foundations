
---

### **[Sección 1.1 y 1.2: Logro de la sesión e Introducción: de la regresión a la clasificación]**

(Duración estimada: 7-8 minutos)

Bienvenidos a la Sesión 4. Hoy vamos a dar un salto cualitativo. Hasta ahora, con la regresión lineal, hemos aprendido a predecir números, cantidades, valores continuos. Pero el mundo no solo se trata de "cuánto", sino también de "qué tipo". ¿Es este email spam o no lo es? ¿Este paciente tiene o no tiene la enfermedad? ¿Esta transacción es fraude o es legítima?

El logro de esta sesión, como bien indica el material, es que ustedes sean capaces de **implementar modelos de clasificación binaria y evaluarlos correctamente con las métricas adecuadas**. Y quiero que se fijen en la segunda parte de la frase: "comprendiendo el impacto del desbalance de clases y las técnicas para mitigarlo". Eso es clave. No solo vamos a construir modelos, vamos a entender sus debilidades y a aprender a hacerlos útiles en el mundo real, que suele ser un lugar muy desequilibrado.

Vamos con la introducción. ¿Por qué no podemos usar la regresión lineal para clasificar? Imaginemos que somos un banco y queremos predecir si una persona va a devolver un préstamo (Sí/No). Con regresión lineal, intentaríamos dibujar una recta que ajuste los puntos, donde los "Sí" valen 1 y los "No" valen 0. ¿Qué pasa? Pues que para valores altos de ingresos, la recta nos dará predicciones muy por encima de 1, y para valores bajos, predicciones negativas. Una probabilidad del 150% o del -20% no tiene sentido. Esa es la primera razón.

La segunda razón es más técnica, pero igual de importante: los errores que comete la regresión lineal no se comportan bien con datos de tipo sí/no. En regresión lineal asumimos que los errores son como un ruido blanco, con una distribución normal y con la misma variabilidad para todos los valores. Pero cuando tu variable solo puede ser 0 o 1, los errores son inevitablemente grandes cerca de los extremos y no siguen una campana de Gauss. Es como intentar ponerle falda a un elefante: no es la ropa adecuada.

Por eso necesitamos una nueva herramienta, un modelo hecho a medida para tomar decisiones binarias. Y esa herramienta se llama **Regresión Logística**. Vamos a conocerla.

---

### **[Sección 1.3.1 y 1.3.2: Función sigmoide y Modelo probabilístico]**

(Duración estimada: 7-8 minutos)

Muy bien, el primer ingrediente de nuestra nueva receta es una función matemática con una forma muy especial: la **función sigmoide**. Fíjense en la fórmula que aparece en el material:

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

No se asusten por la pinta. Lo importante es lo que hace. Piensen que tenemos un valor 'z', que puede ser cualquier número, desde -infinito hasta +infinito. La sigmoide lo agarra, lo procesa, y lo convierte en un número entre 0 y 1. Es como una máquina de embutidos que convierte cualquier cosa en una salchicha de tamaño estandarizado.

¿Y qué propiedades tiene esta máquina? Es suave, no tiene picos (es diferenciable, lo cual es genial para el cálculo). Es creciente: a mayor 'z', mayor salida. Y tiene dos asíntotas: cuando 'z' es muy negativo, la salida tiende a 0; cuando 'z' es muy positivo, la salida tiende a 1. Justo lo que necesitamos para hablar de probabilidades.

Ahora, ¿cómo construimos el modelo probabilístico? Pues muy sencillo. En lugar de usar directamente una combinación lineal de las variables (β₀ + β₁x₁ + ...) como predicción, como hacíamos en regresión lineal, ahora metemos esa combinación lineal dentro de la sigmoide. Es decir:

\[ P(y=1|x) = \sigma(\beta_0 + \beta_1x_1 + \cdots + \beta_p x_p) \]

Esto se lee como: "La probabilidad de que la variable 'y' sea igual a 1 (por ejemplo, que el email sea spam), dadas las características 'x' del email, es igual a la sigmoide aplicada a una combinación lineal de esas características".

Y por supuesto, la probabilidad de que sea 0 es el complemento: \( P(y=0|x) = 1 - P(y=1|x) \). Así de simple. Ya tenemos un modelo que nos da probabilidades. Pero, ¿cómo interpretamos esos coeficientes beta? Ahí viene la parte interesante.

---

### **[Sección 1.3.3: Interpretación de coeficientes]**

(Duración estimada: 7-8 minutos)

Esta es una de las partes que más dolores de cabeza causa, pero vamos a desmenuzarla con calma. En la regresión lineal, si el coeficiente de "años de experiencia" es 0.5, decimos: "por cada año más de experiencia, el salario aumenta en 0.5 unidades". Era directo.

En regresión logística, la interpretación es más indirecta, porque estamos modelando una probabilidad a través de un intermediario. Ese intermediario se llama **log-odds** o **logit**.

Vamos por partes. Primero, ¿qué son los **odds**? Si la probabilidad de que llueva hoy es del 75%, la probabilidad de que no llueva es del 25%. Los odds se definen como el cociente entre la probabilidad de que ocurra y la de que no ocurra: 0.75 / 0.25 = 3. Esto se interpreta como "es 3 veces más probable que llueva a que no llueva". Los odds van de 0 a infinito.

El **log-odds** o **logit** es, simplemente, el logaritmo natural de esos odds. ¿Por qué hacemos esto? Porque el logaritmo transforma un valor de 0 a infinito en un valor de -∞ a +∞. Y resulta que la combinación lineal de nuestras variables (β₀ + βx) puede tomar cualquier valor real. Por lo tanto, podemos igualar:

\[ \text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta^T x \]

Es decir, el modelo es lineal, pero en la escala de los log-odds, no en la escala de la probabilidad.

Ahora, ¿cómo interpretamos un coeficiente βⱼ? Si aumentamos xⱼ en una unidad, manteniendo todo lo demás constante, el log-odds aumenta en βⱼ unidades. Pero el log-odds no es una unidad intuitiva. Por eso usamos el **odds ratio**, que es e^{βⱼ}.

Si βⱼ = 0.5, entonces e^{0.5} ≈ 1.65. Esto significa que **las odds de pertenecer a la clase 1 se multiplican por 1.65** por cada incremento unitario en xⱼ. Si βⱼ es negativo, el odds ratio será menor que 1, indicando una disminución en las odds.

Es como si cada variable tuviera un factor multiplicativo sobre la apuesta a favor de que ocurra el evento. Es una forma muy elegante de hablar de riesgos y oportunidades. Y ahora que entendemos cómo funciona el modelo, pasemos a cómo se entrena.

---

### **[Sección 1.3.4 y 1.3.5: Función de pérdida: entropía cruzada y Optimización: gradiente descendente]**

(Duración estimada: 8-9 minutos)

Bien, tenemos un modelo que nos da probabilidades. Necesitamos una forma de medir qué tan buenas son esas probabilidades, es decir, una función de pérdida. En regresión lineal usábamos el error cuadrático medio. ¿Podemos usarlo aquí? La respuesta es no, porque la naturaleza de la salida es diferente. Nuestra salida es una probabilidad, no un valor real cualquiera.

La función de pérdida que se usa en clasificación binaria se llama **entropía cruzada** o **log-loss**. Miren la fórmula para una observación:

\[
\mathcal{L} = -[y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i)]
\]

Vamos a interpretarla con un ejemplo. Supongamos que el valor real y_i es 1 (por ejemplo, es spam). Entonces el segundo término se anula (porque (1-y_i)=0) y nos queda: -[1 * ln(ˆp_i)]. Si nuestro modelo predice una probabilidad alta, digamos 0.95, entonces ln(0.95) es un número pequeño negativo, y con el signo menos se convierte en un número pequeño positivo. La pérdida es baja. Si el modelo predice una probabilidad baja, digamos 0.10, entonces ln(0.10) es un número negativo grande, y la pérdida se dispara. Castiga fuertemente estar muy seguro y equivocarse.

Análogamente, si el valor real es 0, la pérdida es -ln(1-ˆp_i). Si predecimos 0.05, pérdida baja; si predecimos 0.90, pérdida enorme.

Esta función tiene una propiedad maravillosa: es **convexa**. Eso significa que tiene un único mínimo global, como un valle en el paisaje. Así que, si usamos un algoritmo de optimización adecuado, estamos seguros de encontrar el mejor conjunto de coeficientes.

Y ese algoritmo es el **gradiente descendente**. La fórmula del gradiente, sorprendentemente, es muy parecida a la de regresión lineal:

\[
\nabla J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i) x_i
\]

La única diferencia es que aquí ˆp_i se calcula mediante la sigmoide, no es simplemente la combinación lineal. Pero la idea es la misma: el gradiente nos dice la dirección en la que debemos ajustar los coeficientes para reducir el error. Y así, iteración tras iteración, el modelo va aprendiendo.

---

### **[Sección 1.3.6: Límite de decisión]**

(Duración estimada: 5-6 minutos)

Una vez que tenemos el modelo entrenado, necesitamos tomar decisiones. En algún momento, tenemos que pasar de una probabilidad (un número entre 0 y 1) a una etiqueta concreta (Spam / No spam). Lo habitual es usar un umbral, típicamente 0.5. Si la probabilidad es mayor o igual a 0.5, clasificamos como 1; si es menor, como 0.

Ahora, el **límite de decisión** es el lugar geométrico de los puntos donde la probabilidad es exactamente 0.5. ¿Cuándo ocurre eso? Cuando la combinación lineal es cero:

\[
\beta_0 + \beta^T x = 0
\]

¿Y qué forma tiene eso? Es una ecuación lineal. En dos dimensiones, es una recta. En tres dimensiones, es un plano. En más dimensiones, es un hiperplano. Por eso decimos que la regresión logística es un **clasificador lineal**. La frontera que separa las clases es una línea recta (o su equivalente en dimensiones superiores). Esto es importante porque nos dice que si la relación entre las variables y la clase es muy compleja, no lineal, la regresión logística simple no será suficiente. Pero para muchos problemas, es un excelente punto de partida.

---

### **[Sección 1.3.7 y 1.3.8: Regularización y Caso de uso: detección de spam]**

(Duración estimada: 6-7 minutos)

Al igual que en regresión lineal, podemos tener problemas de sobreajuste, especialmente si tenemos muchas variables. La solución es la **regularización**. Las mismas técnicas que vimos antes se aplican aquí: Ridge (L2), Lasso (L1) y Elastic Net. La regularización añade una penalización a la función de pérdida para que los coeficientes no se disparen. Lasso, además, puede forzar coeficientes a cero, haciendo selección de variables. Es especialmente útil cuando tenemos cientos de palabras como características en un problema de detección de spam.

Y hablando de spam, el material nos propone un caso de uso clásico. Imaginen que tenemos un conjunto de correos electrónicos. Como características, podemos usar: frecuencia de palabras como "gane", "dinero", "urgente", presencia de enlaces, longitud del mensaje, etc. La regresión logística modela la probabilidad de que un correo sea spam.

Lo interesante es que, al entrenar el modelo, podemos ver los coeficientes. Un coeficiente positivo grande para la palabra "gane" indica que su presencia aumenta las odds de que sea spam. Esto nos da una herramienta interpretable: podemos saber qué palabras son las más sospechosas.

Además, como la salida es probabilística, podemos ajustar el umbral. Si estamos en una empresa donde es peor marcar un correo legítimo como spam (falso positivo) que dejar pasar un spam (falso negativo), podemos subir el umbral a 0.8 o 0.9, para estar muy seguros antes de enviar algo a la carpeta de spam. Esta flexibilidad es una de las grandes ventajas de los modelos probabilísticos.

---

### **[Sección 1.4.1 y 1.4.2: Matriz de confusión y Métricas derivadas]**

(Duración estimada: 8-9 minutos)

Ya tenemos nuestro modelo. Ahora, ¿cómo sabemos si es bueno? No basta con decir "acertó el 95% de las veces". Necesitamos una herramienta más fina, que nos desglose los aciertos y errores por tipo. Esa herramienta es la **matriz de confusión**.

Es una tablita de 2x2:

- **VP (Verdaderos Positivos)**: Dijimos que sí, y era sí.
- **VN (Verdaderos Negativos)**: Dijimos que no, y era no.
- **FP (Falsos Positivos)**: Dijimos que sí, pero era no. (Error tipo I)
- **FN (Falsos Negativos)**: Dijimos que no, pero era sí. (Error tipo II)

Cada celda tiene un significado en el mundo real. En diagnóstico médico, un falso positivo puede generar ansiedad y pruebas innecesarias. Un falso negativo puede ser mortal. Por eso, no podemos quedarnos con una sola cifra.

A partir de esta matriz, derivamos varias métricas:

- **Exactitud (Accuracy)**: (VP+VN)/total. Es la más intuitiva, pero ojo, puede engañar si las clases están desbalanceadas. Si el 99% de los casos son normales, un modelo que siempre predice "normal" tendrá un 99% de exactitud, pero será inútil para detectar lo raro.

- **Precisión (Precision)**: VP / (VP+FP). De todo lo que marcamos como positivo, ¿cuánto acertamos? Mide la calidad de las alarmas. Es crucial cuando los falsos positivos son caros. Por ejemplo, en un filtro de spam, queremos alta precisión para no enviar correos importantes a la basura.

- **Recall (Sensibilidad)**: VP / (VP+FN). De todos los positivos reales, ¿cuántos capturamos? Mide la cobertura. Es clave cuando los falsos negativos son caros, como en detección de fraudes o enfermedades graves. Queremos no dejar escapar a ningún positivo, aunque eso signifique tener más falsas alarmas.

- **Especificidad**: VN / (VN+FP). De todos los negativos reales, ¿cuántos identificamos correctamente? Es el complemento de la tasa de falsos positivos. Útil cuando los falsos positivos son costosos.

Y luego tenemos el **F1-Score**, que es la media armónica de precisión y recall. ¿Por qué armónica y no aritmética? Porque la media armónica penaliza los desequilibrios. Si tienes precisión 1 y recall 0, la media aritmética da 0.5, pero la armónica da 0. Es mucho más exigente. El F1-Score es ideal cuando buscas un balance entre las dos y cuando las clases están desbalanceadas.

---

### **[Sección 1.4.3: Comparación de métricas con ejemplos simulados]**

(Duración estimada: 6-7 minutos)

Para que esto quede más claro, el material nos presenta un ejemplo numérico de detección de fraudes. Tenemos 1000 transacciones: 20 fraudulentas (positivas) y 980 normales (negativas). Comparamos dos modelos, A y B.

- **Modelo A**: VP=18, FP=10, VN=970, FN=2.
- **Modelo B**: VP=20, FP=100, VN=880, FN=0.

Calculemos:

- Exactitud: A = 98.8%, B = 90.0%. Según esto, A es mejor.
- Precisión: A = 64.3%, B = 16.7%. A es mucho mejor.
- Recall: A = 90%, B = 100%. B es mejor.
- F1: A = 0.75, B = 0.286. A es mejor.

¿Cuál es mejor entonces? Depende. Si el costo de un fraude no detectado es altísimo (por ejemplo, millones de euros), preferimos el modelo B, que captura todos los fraudes, aunque genere muchos falsos positivos que luego habrá que revisar manualmente. Si el costo de revisar falsos positivos es muy alto (por ejemplo, requiere mucho trabajo manual), preferimos el modelo A, que es más preciso.

Esta es la lección: **la métrica principal no es una decisión técnica, es una decisión de negocio**. Depende de lo que más nos duela.

---

### **[Sección 1.4.4 y 1.4.5: Curva ROC y AUC, y Cuándo usar cada métrica]**

(Duración estimada: 7-8 minutos)

Hemos visto que las métricas dependen del umbral que elijamos. Si movemos el umbral de 0.5 a 0.3, obtendremos más positivos, aumentará el recall pero bajará la precisión. Para evaluar la calidad global del modelo, independientemente del umbral, usamos la **curva ROC** y el **AUC**.

La curva ROC representa, para todos los posibles umbrales, la tasa de verdaderos positivos (TPR, que es el recall) en el eje Y, frente a la tasa de falsos positivos (FPR) en el eje X.

- Un modelo perfecto tendría TPR=1 y FPR=0 en algún punto.
- Un modelo aleatorio (como lanzar una moneda) se representa con la diagonal.
- Cuanto más se acerque la curva a la esquina superior izquierda, mejor.

El **AUC** es el área bajo esa curva. Va de 0.5 (modelo aleatorio) a 1 (modelo perfecto). Tiene una interpretación probabilística muy bonita: es la probabilidad de que el modelo asigne una puntuación más alta a una instancia positiva aleatoria que a una negativa aleatoria. Es una métrica muy usada para comparar modelos globalmente.

Ahora, el material nos da una guía práctica de cuándo usar cada métrica:
- **Accuracy**: clases balanceadas y costos simétricos.
- **Precision**: minimizar falsos positivos es prioritario.
- **Recall**: minimizar falsos negativos es prioritario.
- **F1-score**: balance entre precisión y recall, útil con desbalance.
- **AUC-ROC**: comparar modelos independientemente del umbral.

Pero atención, el AUC-ROC puede ser optimista en problemas con mucho desbalance, porque la tasa de falsos positivos puede ser muy baja simplemente porque hay muchos negativos. En esos casos, se recomienda usar la curva Precision-Recall y su área (PR-AUC), que se centra en la clase minoritaria.

---

### **[Sección 1.5.1, 1.5.2 y 1.5.3: Manejo de clases desbalanceadas]**

(Duración estimada: 8-9 minutos)

Llegamos al corazón del problema real: las clases desbalanceadas. Ocurre cuando una clase (la que nos interesa, como el fraude) es muy rara comparada con la otra. Los modelos tienden a ignorar la clase minoritaria porque así cometen menos error global. Tenemos que forzarles a prestarle atención.

El material nos presenta dos grandes familias de técnicas: a nivel de datos y a nivel de algoritmo.

**A nivel de datos:**
- **Submuestreo (Undersampling)**: Quitar aleatoriamente instancias de la clase mayoritaria para igualar frecuencias. Ventaja: reduce el dataset. Desventaja: puedes perder información valiosa.
- **Sobremuestreo (Oversampling)**: Duplicar aleatoriamente instancias de la clase minoritaria. Desventaja: puedes caer en sobreajuste por repetición exacta.
- **SMOTE**: Es la técnica más inteligente. Genera instancias sintéticas de la clase minoritaria interpolando entre vecinos cercanos. Para una muestra de la clase minoritaria, se elige un vecino y se crea una nueva muestra en el segmento que los une. Así se introduce variedad y se evita la simple copia.

**A nivel de algoritmo:**
- **Pesos en la función de pérdida**: Asignar un peso mayor a los errores en la clase minoritaria. En la fórmula de la entropía cruzada, podemos multiplicar el término de los positivos por un peso w₁ y el de los negativos por w₀. Típicamente, w₁ es inversamente proporcional a la frecuencia de la clase positiva. Así, equivocarse con un fraude "duele" más en la función de pérdida.
- **Ajuste del umbral de decisión**: En lugar de usar 0.5, podemos bajar el umbral para favorecer la detección de la clase minoritaria. Esto aumentará el recall, pero también los falsos positivos. Es un trade-off que debemos gestionar.

**Evaluación en problemas desbalanceados**: Olvídense de la exactitud. Usen matriz de confusión, precisión, recall, F1 (especialmente el F1 de la clase minoritaria), y curvas Precision-Recall. El AUC-ROC puede ser demasiado optimista; el PR-AUC es más fiable.

---

### **[Sección 1.6: Caso de estudio integrado: Diagnóstico médico]**

(Duración estimada: 7-8 minutos)

Vamos a cerrar con un caso práctico que integra todo lo aprendido. Imaginen que trabajamos en un hospital y tenemos datos de pacientes: edad, presión arterial, niveles de glucosa, etc. Queremos predecir si tienen una determinada enfermedad. La clase positiva (enfermos) es solo el 10% de los datos. Es un problema desbalanceado.

El flujo de trabajo sería:

1. **Preprocesamiento**: Estandarizamos las variables numéricas (importante para la regularización), manejamos valores faltantes y dividimos en entrenamiento y prueba, asegurándonos de que la proporción de clases se mantenga en ambos conjuntos (muestreo estratificado).

2. **Modelado**: Entrenamos una regresión logística con regularización L2. Probamos tres enfoques:
   - Modelo base: sin tratar el desbalance.
   - Modelo con pesos de clase (inversamente proporcionales a la frecuencia).
   - Modelo con SMOTE aplicado al conjunto de entrenamiento.

3. **Evaluación**: Calculamos en el conjunto de prueba la matriz de confusión, precisión, recall, F1 para la clase positiva, AUC-ROC y AUC-PR.

¿Qué esperamos encontrar?
- El modelo sin tratamiento tendrá una exactitud alta (porque acierta muchos sanos), pero un recall muy bajo. Será un desastre: no detectará a los enfermos.
- Con pesos o SMOTE, el recall mejorará sustancialmente. La precisión puede bajar, porque al intentar capturar más enfermos, también marcaremos más sanos como enfermos (falsos positivos).
- La elección final depende del contexto clínico. Si el costo de un falso negativo es altísimo (dejar a un enfermo sin tratamiento), priorizaremos el recall, aunque eso signifique más falsos positivos que requerirán pruebas adicionales. Si el costo de las pruebas adicionales es inaceptable, quizás busquemos un equilibrio.

Este caso muestra cómo la teoría se convierte en decisiones con impacto real.

---

### **[Sección 1.7: Conexión con el resto del curso]**

(Duración estimada: 4-5 minutos)

Para terminar, quiero que vean que esto no es un tema aislado. La regresión logística es la puerta de entrada a modelos mucho más complejos.

- La función sigmoide que usamos hoy es la misma que se usa en la capa de salida de muchas redes neuronales para clasificación binaria. Es la base.
- La entropía cruzada es la función de pérdida estándar en clasificación, y la verán una y otra vez, desde árboles de decisión hasta redes profundas.
- Las métricas de evaluación y el manejo de desbalance son transversales. No importa si usamos regresión logística, SVM, random forest o una red neuronal; siempre tendremos que evaluar con matriz de confusión, precisión, recall, F1, y lidiar con el desbalance.

En la próxima sesión exploraremos otros clasificadores: k-vecinos más cercanos, árboles de decisión, máquinas de soporte vectorial... y los compararemos con la regresión logística. Veremos que cada uno tiene sus fortalezas y debilidades, y que la elección del modelo también es parte del arte de la ciencia de datos.

Por hoy, hemos cubierto mucho. Recuerden: la regresión logística es simple, interpretable y poderosa. Pero sobre todo, nos enseña a pensar en términos de probabilidades, a medir el rendimiento con métricas adecuadas y a tomar decisiones informadas por el contexto del negocio. Ahora, a practicar con código. ¡Nos vemos en la siguiente clase!