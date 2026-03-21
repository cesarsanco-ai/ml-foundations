
## SLIDE 4 - NTB-LM
---

### **[Lámina 2: El límite de la regresión lineal para predecir categorías]**

(Duración estimada: 5-7 minutos)

Muy bien, observen esta primera lámina. Es crucial. Nos plantea un problema fundamental: **"El límite de la regresión lineal para predecir categorías"**.

Imagínense que somos médicos en un servicio de urgencias. Llega un paciente y, basándonos en una serie de síntomas y analíticas, tenemos que dar un veredicto: ¿Tiene o no tiene una enfermedad grave? La respuesta es un "sí" o un "no". Es una categoría, una etiqueta. En el lenguaje de Machine Learning, decimos que nuestra variable a predecir, la que llamamos 'y', es **discreta** y pertenece al conjunto {0, 1}. Cero, no tiene la enfermedad; uno, sí la tiene.

Ahora, piensen en la regresión lineal que ya conocen. Su función es predecir valores continuos. Si yo les pido que predigan el precio de una casa (que puede ser 150.000, 250.000...), la regresión lineal es perfecta. Pero, ¿cómo usaríamos una línea recta para predecir si un tumor es benigno o maligno? Vamos a hacer el experimento mental.

Dibujemos un eje. En el eje horizontal (X) ponemos, por ejemplo, el nivel de una proteína en sangre. En el vertical (Y) ponemos nuestra variable a predecir, que solo puede ser 0 o 1. Ajustamos una recta de regresión lineal a esos puntos. 

Ahora viene el problema. La recta, inevitablemente, para valores muy altos de la proteína nos dará predicciones muy por encima de 1, y para valores muy bajos, predicciones negativas. ¿Qué significa una probabilidad de 1.5? ¿O una probabilidad de -0.2? ¡No tiene ningún sentido en nuestro contexto! La probabilidad de tener una enfermedad no puede ser negativa ni superar el 100%.

---

### **[Lámina 3: La solución sigmoide]**

(Duración estimada: 5-7 minutos)

Fíjense bien en la imagen. Es una curva monótona creciente (siempre va hacia arriba), es diferenciable (suave, sin picos, lo cual es genial para el cálculo que usaremos para optimizarla) y, lo más importante, por mucho que el eje horizontal (X) se extienda hacia el infinito positivo, la curva nunca supera el 1. Y por mucho que se extienda hacia el infinito negativo, nunca baja del 0.

Piensen en que La regresión lineal le tira un número, cualquier número (desde -100 hasta +1000), y la función sigmoide lo procesa y lo convierte en una probabilidad limpia y ordenada, un valor entre 0 y 1. Es la puerta de entrada al reino de la clasificación probabilística.

¿Y esto por qué es tan potente? Porque ahora, en lugar de una respuesta seca de "sí" o "no", nuestro modelo nos da un matiz. Nos dice: "Para este paciente, con esos niveles de proteína, la probabilidad de que tenga la enfermedad es del 0.85 (es decir, un 85%)". Eso es muchísimo más informativo. Luego, nosotros, como analistas, o el propio sistema, podemos establecer un punto de corte: "Si la probabilidad es mayor o igual a 0.5, lo clasificamos como 'Sí'. Si es menor, como 'No'". Este punto de corte es nuestro umbral de decisión, y podemos moverlo según lo conservadores que queramos ser. Ya hablaremos de eso.

Así que, resumiendo: la regresión logística no es más que una regresión lineal cuyo resultado, en lugar de darlo directamente, lo pasamos por esta función mágica en forma de 'S' que lo transforma en una probabilidad. 

---

### **[Lámina 4: El motor probabilístico y la interpretación de coeficientes]**

(Duración estimada: 7-8 minutos)

Esta es una de las láminas donde más gente se pierde, así que presten mucha atención. Vamos a hablar de **"El motor probabilístico y la interpretación de coeficientes"**.

Cuando estudiamos regresión lineal, un coeficiente (β) nos decía: "por cada unidad que aumente la variable X, la variable Y aumenta, en promedio, β unidades". Era una relación directa, lineal, fácil de entender.

En regresión logística, es más sutil. Ya no estamos prediciendo "Y" directamente, estamos prediciendo una probabilidad a través de un "intermediario". Ese intermediario se llama **log-odds** o **logit**. 

¿Qué son los *odds*? Es un concepto muy usado en las apuestas. Si la probabilidad de que un evento ocurra es del 75%, la probabilidad de que no ocurra es del 25%. Los *odds* se definen como el cociente entre la probabilidad de que ocurra y la de que no ocurra: 0.75 / 0.25 = 3. Esto se interpreta como "es 3 veces más probable que ocurra a que no ocurra". Los *odds* van de 0 a infinito.

El *log-odds* o *logit* es, simplemente, el logaritmo natural de esos *odds*. ¿Y qué consigue esto? Transforma un valor que iba de 0 a infinito en un valor que va de -∞ a +∞. ¡Y este es el dominio en el que trabaja la regresión lineal! Por eso, la regresión logística, internamente, es un modelo lineal, pero no en el espacio de las probabilidades, sino en el espacio de los *log-odds*.

Ahora sí, entendamos el coeficiente. El coeficiente β ya no me dice "cuánto aumenta la probabilidad", sino **"cuánto aumentan los log-odds por cada incremento unitario en la variable X"**. Es un trabalenguas, lo sé. Por eso, para hacerlo más tangible, usamos el **Odds Ratio**.

Miren el ejemplo que nos da la lámina: **Riesgo Crediticio**. La variable es "Ingresos mensuales" y el coeficiente β es -0.05. ¿Qué significa esto?
1.  El signo es negativo: A más ingresos, menor riesgo de falla. Tiene lógica.
2.  Para interpretar la magnitud, calculamos el Odds Ratio, que es simplemente elevar el número *e* al coeficiente: *e*^(-0.05) ≈ 0.951.

Y aquí viene la magia de la interpretación. El Odds Ratio nos dice: **"Por cada incremento unitario en los ingresos (por cada euro o cada mil euros, según la escala), las odds de que la persona entre en default (no pague su deuda) se multiplican por 0.951"**.

Como es menor que 1, es una disminución. ¿De cuánto? Pues 1 - 0.951 = 0.049, es decir, un 4.9%. Por lo tanto, podemos decir: "Por cada unidad que aumentan los ingresos mensuales, las odds de default disminuyen un 4.9%".

¿Ven la diferencia? No decimos que la probabilidad de default baja un 4.9% (eso sería incorrecto). Decimos que las *odds* bajan un 4.9%. Es un matiz importante, pero lo crucial es que podemos dar una interpretación de negocio: el dinero protege del riesgo. Es una herramienta muy poderosa para explicar el modelo a los directivos de un banco. Ahora que ya sabemos interpretar el modelo, pasemos a ver cómo lo "entrenamos".

---

### **[Lámina 5: Optimización y el límite de decisión]**

(Duración estimada: 5-6 minutos)

Bien, ya tenemos nuestro modelo y sabemos cómo interpretar sus coeficientes. La pregunta del millón es: ¿cómo aprende este modelo? ¿Cómo encuentra los valores mágicos de esos coeficientes β para que la curva sigmoide se ajuste lo mejor posible a nuestros datos?

La lámina nos dice: **"El aprendizaje del modelo se basa en minimizar los errores de probabilidad mediante una función de pérdida convexa."**

Vamos a verlo con una analogía simple. Imagínense que nuestro modelo es un coche y los coeficientes son los mandos: el volante y los pedales. Nuestro objetivo es conducir desde Lima hasta Piura, y queremos llegar con el mínimo error posible. El "error" es la distancia que nos desviamos de la carretera correcta.

En regresión lineal, usábamos el "Error Cuadrático Medio". Era como un GPS que nos decía: "Te has desviado 10 kilómetros, corrige". Pero para la regresión logística, ese GPS no funciona bien. ¿Por qué? Porque nuestras salidas ya no son puntos en una línea, sino probabilidades entre 0 y 1.

Necesitamos un GPS especial, que entienda de probabilidades. Ese GPS se llama **función de pérdida de log-verosimilitud negativa**, o más coloquialmente, **pérdida de entropía cruzada** (cross-entropy). No se asusten con el nombre.

La idea es brillante: nuestro GPS castiga mucho más los errores cuando estamos muy seguros de una predicción equivocada. Piénsenlo: si el modelo predice con un 99% de probabilidad que un email es "No Spam", pero en realidad es Spam, ese error es mucho más grave que si lo hubiera predicho con un 51% de probabilidad. El primer caso demuestra una confianza absoluta en lo erróneo. La función de pérdida de entropía cruzada le pone una multa enorme a ese tipo de errores, forzando al modelo a ser más humilde y preciso.

Además, como ven en la imagen es una función "convexa". Esto es una gran noticia. Una función convexa tiene forma de valle. ¿Y qué significa eso? Que cuando nuestro coche (el modelo) esté buscando el camino a Piura (el mínimo error), no se va a quedar atascado en un lago seco o un agujero en medio del campo (mínimos locales). Va a encontrar, sí o sí, el punto más bajo del valle, la mejor solución posible. Es un viaje sin engaños. Una vez que entendemos cómo optimizar, el siguiente paso es controlar que nuestro modelo no se vuelva loco y memorice los datos en lugar de aprender. Hablemos de regularización.

---

### **[Lámina 6: Controlando la complejidad del modelo]**

(Duración estimada: 7-8 minutos)

Perfecto. Ya tenemos un modelo que aprende minimizando errores. Pero, ¿y si nuestro modelo es demasiado listo? ¿Y si, en su afán por minimizar el error en los datos que le hemos dado (los de entrenamiento), se los aprende de memoria? Esto se llama **sobreajuste** u **overfitting**.

Imaginen que estamos estudiando para un examen y, en lugar de entender los conceptos, nos memorizamos las respuestas del temario. El día del examen, si las preguntas son exactamente las mismas, sacaremos un 10. Pero si el profesor cambia un poco el enunciado, estaremos perdidos. Eso es el sobreajuste: el modelo es increíble con los datos que ya vio, pero un desastre con datos nuevos.

La lámina nos presenta dos recetas de regularización: **L1 (Lasso)** y **L2 (Ridge)**.

1.  **L2, o Regularización Ridge**: Es el chef conservador. Añade un poquito de todos los condimentos, pero con mesura. Su lema es "menos es más, pero sin eliminar nada". Matemáticamente, añade una penalización que es la suma de los coeficientes al cuadrado (||β||²). Esto hace que los coeficientes de las variables se "encojan", se acerquen a cero, pero nunca lleguen a ser cero exactamente. Es ideal para controlar la varianza general y cuando tenemos variables que están muy correlacionadas entre sí (multicolinealidad).

2.  **L1, o Regularización Lasso**: Este es el chef radical. Su lema es "lo que no sirve, que no estorbe". Además de encoger los coeficientes, tiene la capacidad de forzarlos a ser exactamente cero. Es como si cogiera algunos ingredientes y los tirara a la basura porque no los necesita. Matemáticamente, penaliza con la suma de los valores absolutos de los coeficientes (||β||₁). Esta propiedad es fantástica para la **selección de variables (feature selection)**. Si tenemos un dataset masivo con cientos de variables, Lasso nos ayudará a quedarnos solo con las realmente importantes, haciendo el modelo más simple, rápido y fácil de interpretar.

¿Cuál es mejor? No hay una respuesta única. Depende del problema. Si sospechamos que muchas variables no sirven, Lasso es genial. Si queremos un control más suave, usamos Ridge. Incluso existe una combinación de ambas, llamada Elastic Net.

---

### **[Lámina 7: La Matriz de Confusión como tablero de impacto]**

(Duración estimada: 5-6 minutos)

Aquí es donde dejamos de hablar de matemáticas y empezamos a hablar de negocio. Lámina 7: **"La Matriz de Confusión como tablero de impacto"**.

Imagínense que son el director de un banco y les presento un modelo de detección de fraudes. El vendedor de turno les dice: "¡Es fantástico, acierta el 99% de las veces!". Ustedes, felices, lo implementan. Al mes siguiente, descubren que el banco ha perdido millones porque el modelo, en ese 1% de error, dejó pasar todos los fraudes. La "exactitud" les mintió.

Necesitamos una herramienta más fina, que nos muestre no solo los aciertos, sino, sobre todo, los errores y su tipo. Esa herramienta es la **Matriz de Confusión**. Es una tabla de 2x2 que enfrenta lo que el modelo predijo con lo que realmente ocurrió. Es el tablero de mandos del negocio.

Vamos a dibujarla con el ejemplo de los correos electrónicos.
- **Verdaderos Positivos (VP)**: Correos que son Spam y el modelo dijo que eran Spam. ¡Bien!
- **Verdaderos Negativos (VN)**: Correos que No son Spam (los queremos) y el modelo dijo que No eran Spam. ¡Bien!
- **Falsos Positivos (FP)**: Correos que No son Spam (como la factura de la luz) pero el modelo dijo que eran Spam y los mandó a la carpeta de correo no deseado. ¡Error! En estadística esto se llama Error Tipo I.
- **Falsos Negativos (FN)**: Correos que Sí son Spam (una estafa) pero el modelo dijo que no lo eran y los dejó en nuestra bandeja de entrada principal. ¡Error! Error Tipo II.

¿Ven? No todos los errores son iguales. Tener un Falso Positivo (un correo bueno en spam) es molesto. Pero tener un Falso Negativo (un correo malo en la bandeja de entrada) puede ser un peligro. La matriz de confusión nos obliga a mirar directamente a estos errores y a ponerles una etiqueta de costo. A partir de esta matriz, podemos construir las métricas que realmente importan.

---

Excelente observación. Vamos a mejorar esa explicación. La clave no es solo dar la definición, sino hacer que el alumno *sienta* por qué cada métrica es importante en contextos diferentes. Vamos a usar el "tren del pensamiento" para que cada concepto se convierta en una herramienta mental que puedan aplicar instintivamente.

Aquí tienes la versión mejorada de la explicación para la Lámina 8, con los nuevos ejemplos integrados.

---

### **[Lámina 8: Radiografía de las métricas base]**

(Duración estimada: 8-9 minutos)

Muy bien, clase. Ya tenemos nuestra matriz de confusión, que es como el tablero de un coche. Ahora vamos a aprender a leer cada uno de sus indicadores. Esta lámina es la **"Radiografía de las métricas base"**. Y esto no es una radiografía cualquiera, es como cuando vas al médico y te toman las constantes vitales. El médico no mira solo una cosa; mira la tensión, el pulso, la temperatura y la frecuencia respiratoria. Cada una le da una pista diferente sobre tu salud. Con los modelos de clasificación pasa exactamente lo mismo.

Vamos a ver cada una de estas "constantes vitales" de nuestro clasificador. Y para que lo entiendan de verdad, vamos a ponerle contexto a cada una con ejemplos de la vida real, de esos que os podríais encontrar trabajando en una empresa tecnológica, un banco o una fábrica.

**1. Exactitud (Accuracy)**

La primera es la **Exactitud**. Es la más intuitiva de todas, la que todo el mundo quiere ver en un titular. La fórmula es sencilla: es el total de aciertos (Verdaderos Positivos más Verdaderos Negativos) dividido entre el total de casos. La pregunta que responde es: **"De cada 100 predicciones que hace mi modelo, ¿cuántas acierta en total?"**

Piensen en un **sistema de autenticación de doble factor** en una aplicación bancaria. El modelo intenta decidir si un intento de acceso es legítimo (positivo) o es un intento de fraude (negativo). Si el modelo tiene una exactitud del 99%, suena genial, ¿verdad?

Pues cuidado. La exactitud es la más peligrosa de las métricas, sobre todo cuando los datos están desbalanceados. ¿Por qué? Porque si el 99% de los intentos de acceso son legítimos, un modelo "vago" que simplemente diga "todo es legítimo" tendrá una exactitud del 99%. ¡Pero será pésimo! Porque el 100% de los intentos de fraude (el 1% restante) le estarán colando. La exactitud nos da una visión global, pero nos puede estar mintiendo descaradamente si no miramos las demás constantes.

**2. Precisión (Precision)**

Vamos con la segunda métrica: la **Precisión**. Esta ya es más fina. Se centra únicamente en las veces que nuestro modelo se ha mojado y ha dicho "Sí, esto es un caso positivo". La fórmula es VP dividido entre (VP + FP). Y la pregunta clave que responde es: **"De todo lo que mi modelo ha marcado como positivo (alarma roja), ¿qué porcentaje era realmente positivo?"**. Es la métrica de la confianza, de la calidad de la alarma.

Imagina ahora el sistema de **autenticación de doble factor**. Cuando el modelo dice "esto es un intento de fraude", bloquea el acceso y envía un SMS al usuario para que confirme. Si el modelo tiene una precisión baja, significa que tiene muchos Falsos Positivos. Es decir, está bloqueando constantemente a usuarios legítimos, enviándoles SMS de verificación que no tocan. ¿Qué pasa? El usuario se frustra, pierde confianza en el banco y, lo que es peor, puede empezar a ignorar las alertas de seguridad auténticas. El costo de un Falso Positivo aquí es la fricción con el cliente y la pérdida de confianza.

Ahora llevémoslo a otro campo: la **robótica**. Imaginad un robot en una línea de montaje que debe detectar piezas defectuosas. Su brazo mecánico tiene una pinza que, cuando detecta un defecto (positivo), aparta la pieza. Si la **Precisión** de su sistema de visión es baja, el robot estará apartando muchas piezas correctas (Falsos Positivos). Esto paraliza la producción, genera desperdicio y reduce la eficiencia de toda la fábrica. En este caso, una alta precisión es vital para no interrumpir el flujo de trabajo.

**3. Recall (Sensibilidad)**

La tercera métrica es el **Recall**, también conocido como Sensibilidad. Esta métrica mira el problema desde el otro lado. Se fija en todos los casos positivos que existían realmente en el mundo. La fórmula es VP dividido entre (VP + FN). Y su pregunta es: **"De todos los casos positivos que había ahí fuera, ¿cuántos fue capaz de capturar mi modelo?"**. Es la métrica de la cobertura, de la eficacia para no dejar escapar a los malos.

Pensemos en el ejemplo del **phishing**. Un modelo que analiza los correos de una gran empresa. De todos los correos de phishing verdaderamente peligrosos que llegan, ¿cuántos consigue el modelo meter en la carpeta de spam antes de que un empleado los abra? Un Falso Negativo aquí (un correo de phishing que el modelo deja pasar) puede ser un desastre. Un empleado incauto hace clic y pone en riesgo toda la red de la empresa. El costo es altísimo. Por eso, en ciberseguridad, muchas veces priorizamos un Recall altísimo, aunque eso signifique que algún correo legítimo acabe en spam (un Falso Positivo). Preferimos tener que revisar la carpeta de spam a que nos hackeen.

Otro ejemplo, ahora en **robótica**. Imaginad un robot de exploración o un vehículo autónomo. Su sensor debe detectar obstáculos (peatones, otros coches). Si el **Recall** es bajo, significa que el robot no está detectando muchos de los obstáculos reales (Falsos Negativos). Es como si el coche autónomo "no viera" a un peatón. Las consecuencias pueden ser catastróficas. En este caso, queremos un Recall lo más cercano al 100% posible. Preferimos que el coche frene ante una sombra (un Falso Positivo) a que atropelle a alguien (un Falso Negativo).

**4. Especificidad (Specificity)**

Y llegamos a la cuarta métrica, la **Especificidad**. Es la hermana gemela del Recall, pero centrada en la clase negativa. Su fórmula es VN dividido entre (VN + FP). Y responde a la pregunta: **"De todos los casos negativos reales (los que no eran fraude, no eran spam, no eran defectuosos), ¿cuántos identifiqué correctamente como tales?"**. Mide la capacidad de no molestar a los negativos, de no generar falsas alarmas.

Volvamos al ejemplo del **phishing**. Una alta especificidad significa que el modelo es muy bueno dejando en la bandeja de entrada los correos legítimos (los negativos). No los molesta. Es la otra cara de la moneda del Recall. Si priorizamos mucho el Recall, la especificidad puede caer, y empezaremos a tener más Falsos Positivos (correos buenos en spam).

En el caso del **robot de fábrica**, la especificidad mide la capacidad de dejar pasar las piezas correctas sin apartarlas. Es decir, de no entorpecer la producción. Una alta especificidad es el objetivo de eficiencia.

Para que lo recuerden siempre, les dejo esta comparación final con los nuevos ejemplos:
- La **Precisión** es: "De los accesos que bloqueé por ser sospechosos, ¿cuántos eran realmente fraudes?" (Calidad de la alarma en el banco).
- El **Recall** es: "De todos los fraudes que intentaron colarse, ¿cuántos bloqueé?" (Cobertura de seguridad en el banco).

Como ven, cada métrica ilumina una faceta diferente. Una optimiza una cosa a costa de la otra. Aumentar el Recall (querer atrapar a todos los malos) suele hacer que baje la Precisión (porque meteremos a más inocentes en la lista de sospechosos). Es el equilibrio natural, el *trade-off* del que hablaremos después. Y para manejar este equilibrio, necesitamos una métrica que las combine, y de eso hablaremos en la siguiente lámina con el F1-Score.

---

### **[Lámina 9: El equilibrio matemático: F1-Score]**

(Duración aproximada: 5-6 minutos)

Lámina 9: **"El equilibrio matemático: F1-Score"**.

Imaginen a un estudiante que tiene dos exámenes, uno de Matemáticas y uno de Lenguaje. Su nota final no es la media aritmética de las dos notas. ¿Por qué? Porque si saca un 20 en Matemáticas y un 0 en Lenguaje, la media aritmética es 10, aprobado. Pero todos sabemos que ese estudiante tiene un grave problema, es un desastre en Lenguaje. La media aritmética disfraza la debilidad.

Para notas que representan capacidades diferentes, a veces usamos la **media armónica**. La media armónica es mucho más severa. Si una de las dos notas es muy baja, la media armónica también será muy baja. En nuestro ejemplo, la media armónica de 20 y 0 es 0. ¡Suspenso directo!

Pues el **F1-Score** es exactamente eso: la media armónica entre la Precisión y el Recall. Es una métrica que castiga severamente a los modelos que son muy buenos en una métrica pero pésimos en la otra. Obliga al modelo a tener un buen equilibrio entre no equivocarse cuando dice "sí" (Precisión) y no dejar escapar a los "síes" reales (Recall).

Y aquí viene la joya de la corona, el ejemplo que lo explica todo: **"¿Por qué usar F1-Score en lugar de Exactitud?"**

Imaginen un problema de detección de fraudes con tarjetas de crédito. De cada 10,000 transacciones, solo 100 son fraudulentas (1%). Es un problema **desbalanceado**.
Ahora imaginemos un modelo inútil, un modelo "vago", que aprende una sola regla: "clasificar todo como Normal". ¿Qué exactitud tendría este modelo?
Acertaría con las 9,900 transacciones normales, y fallaría en las 100 fraudulentas. Por lo tanto, su Exactitud sería del 99% (9,900 / 10,000). ¡Suena espectacular! Pero es un modelo completamente inútil para el negocio, porque no ha detectado ni un solo fraude.

Ahora calculemos su F1-Score:
- Precisión: Como nunca dice que algo es fraude (nunca predice positivo), VP = 0, por lo tanto, Precisión = 0 / (0+0) = 0 (indefinido, pero lo tomamos como 0).
- Recall: De los 100 fraudes reales, capturó 0. Recall = 0 / 100 = 0.
- F1-Score = Media armónica de 0 y 0 = 0.

El F1-Score le pone un 0 rotundo, una nota que refleja la realidad: el modelo no sirve para nada. Por eso, en problemas con clases desbalanceadas, el F1-Score es mucho más fiable que la Exactitud. Nos fuerza a buscar un modelo que realmente tenga poder predictivo sobre la clase que nos interesa. Y eso nos lleva directamente a la siguiente decisión.

---

### **[Lámina 10: El Trade-off del negocio: ¿Qué error duele más?]**

(Duración estimada: 6-7 minutos)

Lámina 10. Esta es, probablemente, la lámina más importante de toda la presentación. **"El Trade-off del negocio: ¿Qué error duele más?"**.

Y quiero que recuerden esto: **La elección de la métrica principal no es una decisión técnica, es una decisión financiera y operativa**. No es el ingeniero el que decide si queremos maximizar la Precisión o el Recall; es el negocio, el contexto, el costo del error.

Vamos a ver los dos extremos, los dos dolores de cabeza.

**Caso 1: Prioridad: RECALL (Minimizar Falsos Negativos).**

Nuestro objetivo es no dejar escapar a ningún positivo. Preferimos tener falsas alarmas a que se nos cuele un caso malo.
- **Ejemplo 1: Diagnóstico médico grave (Cáncer).** Si una persona tiene un tumor, queremos detectarlo. Un Falso Negativo (decirle a alguien que no tiene cáncer cuando sí lo tiene) es catastrófico, puede costar una vida. Preferimos mil veces un Falso Positivo (mandar a una persona sana a hacerse más pruebas, a una biopsia). El costo de una alarma innecesaria es mucho menor que el costo de no actuar. Por eso, en estos casos, priorizamos el Recall.
- **Ejemplo 2: Detección de Fraude millonario.** Si hay una transferencia fraudulenta de un millón de soles, queremos bloquearla. Un Falso Negativo (dejarla pasar) es una pérdida millonaria. Preferimos bloquear una operación legítima por error (Falso Positivo) y luego pedir disculpas, a perder el dinero.

**Caso 2: Prioridad: PRECISIÓN (Minimizar Falsos Positivos).**

Nuestro objetivo es estar completamente seguros antes de actuar. Preferimos no decir nada a decir algo y equivocarnos.
- **Ejemplo 1: Filtro de Spam.** Si nuestro filtro manda un correo de la universidad con la beca a la carpeta de Spam (Falso Positivo), el estudiante puede perder la oportunidad de su vida. El costo de fricción, de perder un correo importante, es inaceptable. Preferimos que se cuele algún spam (Falso Negativo) en la bandeja de entrada, a que se pierda un correo crucial. Aquí la prioridad es tener una Precisión altísima: cuando el modelo diga que algo es spam, que estemos casi seguros de que lo es.
- **Ejemplo 2: Pruebas médicas invasivas.** Imaginen una prueba para una enfermedad que requiere una biopsia o un tratamiento muy agresivo. No podemos permitirnos someter a alguien a eso si no es estrictamente necesario. Un Falso Positivo (decir que tiene la enfermedad cuando no es así) causaría un daño físico y psicológico enorme. Por lo tanto, la precisión debe ser máxima. Preferimos dejar a algún enfermo sin diagnosticar (Falso Negativo) a tratar a un sano.

Como ven, el mismo modelo, con la misma capacidad, puede ser un éxito o un desastre según lo que prioricemos. Por eso, antes de construir nada, hay que sentarse con el área de negocio y preguntar: "En este escenario, ¿qué error nos duele más?". Esa respuesta definirá nuestra métrica objetivo y cómo ajustaremos el modelo. Ahora, ¿cómo evaluamos la capacidad del modelo más allá de este punto de corte?

---

### **[Lámina 11: Visión global independiente del umbral: Curva ROC y AUC]**

(Duración estimada: 7-8 minutos)

Muy bien. Hasta ahora hemos estado hablando de métricas que dependen de un punto de corte. Si decido que el umbral para clasificar como "Positivo" es 0.5, obtengo una matriz de confusión, y de ahí una Precisión y un Recall. Pero si cambio el umbral a 0.8, la matriz cambia, y las métricas también.

Esto plantea una pregunta: ¿podemos evaluar la calidad "innata" de nuestro modelo, independientemente del umbral que luego elijamos? La respuesta es sí, y las herramientas son la **Curva ROC** y el **AUC**.

Piensen en un test de aptitud. Queremos saber si una persona es apta para un puesto, y tenemos una prueba que da una puntuación de 0 a 100. Podríamos poner el corte en 70, o en 80, o en 50. Según donde lo pongamos, contrataremos a más o menos gente, y cometeremos más o menos errores.

La **Curva ROC (Receiver Operating Characteristic)** es una gráfica que representa, para todos los posibles puntos de corte, la relación entre la **Tasa de Verdaderos Positivos (Recall/Sensibilidad)** en el eje Y, y la **Tasa de Falsos Positivos (1 - Especificidad)** en el eje X.

- Si movemos el corte a un valor muy bajo (por ejemplo, 10), clasificaremos a casi todos como "aptos". Atraparemos a casi todos los verdaderos aptos (TVP alta), pero también meteremos a un montón de no aptos (TFP alta). Nos moveremos en la parte superior derecha de la curva.
- Si movemos el corte a un valor muy alto (por ejemplo, 95), seremos muy exigentes. Solo los mejores serán "aptos". Tendremos muy pocos falsos positivos (TFP baja), pero también capturaremos a muy pocos de los verdaderos aptos (TVP baja). Nos moveremos en la parte inferior izquierda de la curva.

Un modelo perfecto sería aquel que, para algún punto de corte, logra una TVP del 100% con una TFP del 0%. Su curva ROC pasaría por la esquina superior izquierda.

Un modelo que no sirve para nada, que es aleatorio, sería una línea diagonal de 45 grados (como lanzar una moneda al aire).

Ahora, ¿cómo medimos numéricamente la bondad de nuestro modelo a partir de esta curva? Usamos el **AUC (Área Bajo la Curva)**. Es, literalmente, el área bajo esa curva ROC.
- Un AUC de 1 significa un modelo perfecto.
- Un AUC de 0.5 significa un modelo aleatorio, que no distingue nada.

El AUC es una métrica fantástica para comparar modelos de forma global. Nos dice: si cojo al azar un caso positivo y uno negativo, ¿con qué frecuencia mi modelo asignará una probabilidad más alta al positivo que al negativo? Un AUC alto significa que mi modelo tiene una gran capacidad para separar las dos clases, independientemente de dónde ponga el corte después. Sin embargo, cuidado, que ni el AUC es infalible, como veremos en la siguiente lámina.

---

### **[Lámina 12: El desafío del mundo real: La trampa de la Exactitud]**

(Duración estimada: 6-7 minutos)

Lámina 12: **"El desafío del mundo real: La trampa de la Exactitud"**. Vamos a profundizar en el problema que ya anticipamos con el F1-Score, pero ahora con más detalle.

**"Cuando la clase de interés es una anomalía, las métricas tradicionales mienten."** Esta frase lo resume todo. En el mundo real, los problemas más interesantes suelen ser los de las anomalías: el fraude, la enfermedad rara, la avería de la máquina, el cliente que se va a la competencia. Son eventos poco frecuentes, pero de alto impacto.

Miren el ejemplo del **Espejismo del 99%** que nos da la lámina. Un modelo que clasifica todas las transacciones como normales, en un entorno con un 1% de fraude, tiene una exactitud del 99%. Es un espejismo en el desierto. Parece un oasis de rendimiento, pero cuando te acercas, no hay agua, no hay utilidad. Su Recall es 0%. No ha detectado ni un solo fraude. Para el negocio, es un modelo nulo, o incluso dañino porque da una falsa sensación de seguridad.

Y aquí viene una advertencia importante: incluso el AUC-ROC, nuestra métrica estrella de la lámina anterior, puede caer en esta trampa. ¿Por qué? Porque la Curva ROC, al graficar Falsos Positivos, es muy sensible a los cambios en la clase mayoritaria (la normal). Si la clase negativa es enorme, puedo tener una tasa de falsos positivos muy baja (porque el denominador es gigantesco) y, sin embargo, tener un montón de falsos positivos en términos absolutos.

Para estos casos de desbalance extremo, los expertos recomiendan migrar a otra curva: el **área bajo la curva Precision-Recall (PR-AUC)**. La curva Precision-Recall se centra únicamente en la clase positiva (la minoritaria). Ignora los Verdaderos Negativos y se fija en la relación entre los aciertos dentro de los positivos (Precisión) y la capacidad de capturarlos (Recall). Es una prueba mucho más dura y realista para modelos que trabajan con eventos raros. Si un modelo tiene un PR-AUC alto, significa que es realmente bueno identificando esas agujas en el pajar. Ante la dificultad, tenemos que preparar bien nuestros datos.

---

### **[Lámina 13: Estrategias de muestreo a nivel de datos]**

(Duración estimada: 6-7 minutos)

Lámina 13: **"Estrategias de muestreo a nivel de datos"**. Hemos visto que los datos desbalanceados son un problema. Una forma de atajarlo es modificar el terreno de juego, es decir, los propios datos, antes de que el modelo empiece a jugar.

Imaginen que queremos enseñar a un niño a distinguir entre manzanas y peras, pero le mostramos 99 manzanas por cada pera. El niño aprenderá muy bien qué es una manzana, pero tendrá una idea muy pobre de lo que es una pera. Para solucionarlo, podemos hacer dos cosas principalmente:

1.  **Submuestreo (Undersampling) de la clase mayoritaria**: Consiste en quitar manzanas de la muestra. Nos quedamos con, por ejemplo, 99 manzanas y 99 peras, igualando las clases. El riesgo aquí es que podemos estar tirando información valiosa. Puede que las manzanas que quitamos fueran muy importantes para distinguir casos límite. Es como si, para que el niño aprenda, le quitamos muchos ejemplos de manzanas, perdiendo la riqueza de su variedad.

2.  **Sobremuestreo (Oversampling) de la clase minoritaria**: Consiste en poner más peras. Pero no tenemos más peras reales, así que las "clonamos". La técnica más famosa es **SMOTE (Synthetic Minority Over-sampling Technique)**. No se limita a duplicar las peras que ya tenemos, sino que crea peras nuevas, sintéticas, que son "híbridas" entre las peras reales. Es como si el niño, a partir de las pocas peras que ha visto, fuera capaz de imaginar nuevas peras ligeramente diferentes, pero igualmente válidas. Es una técnica muy potente.

Estas estrategias a nivel de datos son como preparar un campo de fútbol con el césped perfecto para que juegue tu equipo. Pero también podemos cambiar las reglas del partido, que es lo que veremos en la siguiente lámina.

---

### **[Lámina 14: Estrategias algorítmicas]**

(Duración estimada: 5-6 minutos)

Lámina 14: **"Estrategias algorítmicas"**. Si no queremos tocar los datos, podemos modificar la maquinaria matemática del modelo para que preste más atención a la clase minoritaria.

Recuerdan la función de pérdida de la que hablamos? La que calculaba el error. Podemos manipularla para que los errores en la clase minoritaria duelan mucho más que los errores en la mayoritaria.

Es como un profesor que, para compensar que en clase hay pocos estudiantes fuera de Lima, decide que sus errores en los exámenes cuenten el doble. Así, el modelo (el alumno) pondrá mucho más empeño en no equivocarse con esos casos raros pero importantes.

La técnica se llama **ponderación de clases (class weighting)**. Algoritmos como la regresión logística, las SVM o los árboles de decisión suelen tener un parámetro que permite asignar un peso mayor a la clase minoritaria. Internamente, cuando el modelo calcula el error, multiplica el error de cada instancia de la clase minoritaria por un factor (por ejemplo, 99, si la clase mayoritaria es 99 veces más grande). De esta forma, el modelo aprende que equivocarse con un fraude es 99 veces más grave que equivocarse con una transacción normal, y ajusta sus coeficientes para minimizar esos errores "caros".

No hay una estrategia mejor que otra. Muchas veces, la combinación de ambas (un poco de muestreo y un poco de ponderación) da los mejores resultados. Y con todas estas herramientas en la mochila, ya estamos listos para enfrentarnos a un caso real.

---

### **[Lámina 15: Caso Integrado: Pipeline de Diagnóstico Médico]**

(Duración estimada: 5-6 minutos, como cierre y motivación)

Lámina 15: **"Caso Integrado: Pipeline de Diagnóstico Médico"**. Miren, todo lo que hemos hablado hoy no es teoría abstracta. Es una hoja de ruta para resolver problemas de verdad.

Aquí tenemos el resumen de un ciclo de vida completo. Un hospital quiere diagnosticar una enfermedad rara con una prevalencia del 10%. Es decir, de cada 100 pacientes, solo 10 tienen la enfermedad. Es nuestro problema desbalanceado.

¿Qué harían ustedes ahora?
1.  **Entenderían el negocio**: Se sentarían con los médicos y les preguntarían: "¿Qué error duele más?" ¿Es peor decirle a un sano que está enfermo (Falso Positivo) y someterlo a un tratamiento innecesario? ¿O es peor decirle a un enfermo que está sano (Falso Negativo) y mandarlo a casa? En medicina, como vimos, suele ser peor el Falso Negativo. Priorizaríamos el **Recall**.

2.  **Prepararían los datos**: Como hay desbalance, aplicarían **SMOTE** para generar más ejemplos sintéticos de pacientes con la enfermedad, o usarían pesos en el algoritmo.

3.  **Construirían y evaluarían el modelo**: No se fiarían de la Exactitud. Mirarían la matriz de confusión, y su métrica principal sería el **Recall**. También mirarían el **F1-Score** para asegurarse de que, al tener un recall alto, no han destrozado la precisión (no están diciendo que todo el mundo está enfermo). Usarían el **PR-AUC** para evaluar la capacidad global del modelo.

4.  **Ajustarían el umbral**: Con los médicos, decidirían el punto de corte óptimo. Quizás no usan el 0.5 por defecto. Quizás, para garantizar un recall altísimo, bajan el umbral a 0.3. Esto significa que si el modelo dice que hay un 30% o más de probabilidad de tener la enfermedad, se activa una alerta para pruebas más profundas. Esto generará más falsos positivos, pero es el costo que están dispuestos a asumir para no dejar escapar a ningún enfermo.

Este es el poder de la Regresión Logística bien entendida. No es solo una fórmula, es una forma de pensar, de traducir un problema del mundo real a matemáticas y luego volver a traducir los resultados a una decisión de negocio con impacto real.

Hemos cubierto mucho terreno hoy. Desde por qué una línea recta no sirve para clasificar, hasta cómo tomar decisiones millonarias basadas en el costo de los errores. Espero que se hayan llevado una idea clara: el Machine Learning no va solo de algoritmos, va de entender el problema, medir lo que importa y tomar decisiones informadas. 