

## **Caso 3: Mantenimiento Industrial de Motores (Fallas)**

### **Contexto (recordatorio)**
Una fábrica utiliza sensores para predecir fallas inminentes en motores. Las fallas ocurren en el 5% de las mediciones (500 de 10,000). Un **falso negativo** (no predecir una falla) causa paradas no planificadas, costosas reparaciones y accidentes. Un **falso positivo** (predecir falla cuando no la hay) implica detener la máquina innecesariamente, con costo de producción perdida, pero es mucho menor que una falla real.

**Matriz de confusión:**
|                 | Real Falla | Real No Falla |
|-----------------|------------|---------------|
| **Pred. Falla**     | VP = 450   | FP = 300      |
| **Pred. No Falla**  | FN = 50    | VN = 9200     |

---

### **Preguntas y Respuestas**

**Pregunta 1:**  
Calcula la exactitud del modelo. ¿Por qué esta métrica no es fiable en este contexto, a pesar de su alto valor?

**Respuesta:**  
La exactitud es (VP + VN) / Total = (450 + 9200) / 10000 = 9650/10000 = **96.5%**.  
No es fiable porque las clases están desbalanceadas (95% no falla, 5% falla). Un modelo que siempre predijera "no falla" tendría una exactitud del 95%, solo un 1.5% menos que nuestro modelo. La exactitud no nos dice nada sobre cómo el modelo maneja la clase minoritaria (las fallas), que es la que realmente nos importa para evitar paradas.

---

**Pregunta 2:**  
Interpreta el valor de la **precisión (60%)** en términos de negocio. ¿Qué implica para los operarios de la fábrica?

**Respuesta:**  
Precisión = VP / (VP + FP) = 450 / 750 = **0.6 = 60%**.  
Esto significa que, de cada 10 veces que el modelo activa una alarma de falla y se para la máquina, solo 6 veces la falla era real. Las otras 4 veces (40%) son falsas alarmas, lo que implica detener la producción innecesariamente. Para los operarios, esto puede generar desconfianza en el sistema ("el modelo siempre se equivoca") y posiblemente empiece a ser ignorado, lo que sería peligroso.

---

**Pregunta 3:**  
El **recall** es del 90%. ¿Es un valor aceptable? Justifica desde el punto de vista de seguridad y costos.

**Respuesta:**  
Recall = VP / (VP + FN) = 450 / 500 = **0.9 = 90%**.  
Esto significa que el modelo detecta el 90% de las fallas, pero deja escapar el 10% (50 fallas no detectadas). Desde el punto de vista de seguridad y costos, **no es aceptable**. Cada falla no detectada puede causar una parada no planificada, daños mayores al motor, accidentes laborales y pérdidas de producción muy superiores al costo de una falsa alarma. Idealmente, en mantenimiento predictivo se busca un recall cercano al 100%, aunque la precisión baje.

---

**Pregunta 4:**  
Si el equipo de mantenimiento te dice que pueden asumir más falsas alarmas porque tienen personal para revisarlas, pero no pueden permitirse fallas no detectadas, ¿qué métrica priorizarías y cómo modificarías el modelo?

**Respuesta:**  
Priorizaría el **recall** por encima de todo. Para mejorarlo, se puede **bajar el umbral de decisión** del modelo. Por ejemplo, en lugar de clasificar como "falla" cuando la probabilidad es ≥ 0.5, hacerlo cuando sea ≥ 0.3. Esto hará que el modelo sea más sensible: detectará más fallas reales (aumentará el recall), pero también aumentarán los falsos positivos (bajará la precisión). Si el equipo puede gestionar más falsas alarmas, es un intercambio aceptable. También se podrían aplicar técnicas de sobremuestreo (SMOTE) para que el modelo aprenda mejor las señales de falla.

---

**Pregunta 5:**  
El F1-Score es 0.72. ¿Es una métrica útil aquí para tomar decisiones? Explica por qué sí o por qué no.

**Respuesta:**  
El F1-Score es 2 * (0.6 * 0.9) / (0.6 + 0.9) = 0.72.  
En este contexto, el F1-Score **no es la métrica más útil** para la decisión final, porque presupone que queremos un equilibrio entre precisión y recall. Pero aquí el negocio ha dicho claramente que el recall es más importante (evitar fallas). El F1-Score puede servir como referencia para comparar modelos, pero la decisión de ajuste debe guiarse por el recall. Un modelo con F1=0.72 pero recall=95% sería mejor que otro con F1=0.80 pero recall=85%, aunque el primero tenga peor F1.

---

**Pregunta 6:**  
Supón que el AUC-ROC del modelo es 0.95. ¿Qué nos dice este valor? ¿Es suficiente para confiar en el modelo?

**Respuesta:**  
Un AUC-ROC de 0.95 indica que el modelo tiene una capacidad excelente para distinguir entre falla y no falla, independientemente del umbral. Es decir, si tomamos una muestra aleatoria de una falla real y una de no falla, hay un 95% de probabilidad de que el modelo asigne una puntuación más alta a la falla. Esto es muy bueno. Sin embargo, el AUC-ROC puede ser optimista en problemas desbalanceados. Sería más informativo mirar el **AUC-PR (Precision-Recall)**, que se centra en la clase minoritaria. Si el AUC-PR también es alto (por ejemplo, >0.7), la confianza en el modelo es mayor.

---

## **Caso 4: Calidad del Agua Potable (Contaminación)**

### **Contexto (recordatorio)**
Una empresa de suministro de agua analiza muestras para detectar contaminación. La contaminación es extremadamente rara: 0.1% de las muestras (10 de 10,000). Un **falso negativo** puede causar una crisis de salud pública. Un **falso positivo** implica pruebas adicionales, con costo económico pero mucho menor.

**Matriz de confusión:**
|                 | Real Contam | Real No Contam |
|-----------------|-------------|----------------|
| **Pred. Contam**    | VP = 8      | FP = 50        |
| **Pred. No Contam** | FN = 2      | VN = 9940      |

---

### **Preguntas y Respuestas**

**Pregunta 1:**  
Calcula la exactitud y explica por qué, en este caso, es una métrica completamente engañosa y peligrosa.

**Respuesta:**  
Exactitud = (VP + VN) / Total = (8 + 9940) / 10000 = 9948/10000 = **99.48%**.  
Es peligrosa porque un modelo inútil que siempre predijera "no contaminado" tendría una exactitud del 99.9%, incluso mejor que la de nuestro modelo. La exactitud esconde el hecho de que el modelo solo detecta 8 de cada 10 contaminaciones reales, permitiendo que 2 muestras contaminadas pasen desapercibidas. En salud pública, esa "casi perfección" en la exactitud es una trampa mortal.

---

**Pregunta 2:**  
Interpreta la **precisión (13.8%)** y el **recall (80%)** en el contexto del laboratorio de análisis de agua. ¿Qué significa cada uno para los técnicos?

**Respuesta:**  
- **Precisión 13.8%:** De cada 100 veces que el modelo da la alarma de "contaminado", solo 14 son realmente contaminadas. Las otras 86 son falsas alarmas. Para los técnicos, esto significa que la mayoría de las alertas requerirán una verificación (una prueba de laboratorio más precisa), lo que aumenta su carga de trabajo y los costos operativos.  
- **Recall 80%:** De cada 10 muestras realmente contaminadas, el modelo detecta 8. Las otras 2 pasan como limpias. Esto significa que, periódicamente, agua contaminada llegará a la población sin ser detectada, con el consiguiente riesgo sanitario. Para los técnicos, esto es inaceptable.

---

**Pregunta 3:**  
El gerente de la planta está preocupado por los altos costos de las pruebas de verificación (falsos positivos) y propone subir el umbral de decisión para tener menos falsas alarmas. ¿Qué efecto tendría esto sobre el recall? ¿Es una buena idea?

**Respuesta:**  
Subir el umbral (por ejemplo, de 0.5 a 0.8) hará que el modelo sea más exigente para declarar una muestra como contaminada. Esto **reducirá los falsos positivos** (mejorará la precisión), pero también **reducirá el recall**, porque algunas muestras realmente contaminadas con probabilidad moderada (ej. 0.7) dejarán de ser detectadas. En un contexto de salud pública, **no es una buena idea**. El recall ya es bajo (80%) y empeoraría, aumentando el riesgo de crisis sanitaria. Los costos de las falsas alarmas son asumibles comparados con el desastre de un falso negativo.

---

**Pregunta 4:**  
¿Qué métrica debería ser la prioridad absoluta para este modelo y por qué? ¿Qué valor mínimo consideras aceptable?

**Respuesta:**  
La prioridad absoluta es el **recall**. Idealmente, debe ser **100%** , es decir, detectar todas las muestras contaminadas. No se puede permitir ningún falso negativo en salud pública. Si el recall es inferior al 100%, significa que periódicamente estamos enviando agua contaminada a la población. El valor mínimo aceptable, en un contexto real, sería 99.9% (es decir, 1 falso negativo cada 1000 contaminaciones), pero dada la rareza del evento, se debe luchar por el 100%.

---

**Pregunta 5:**  
El F1-Score es 0.235. ¿Debería usarse esta métrica para evaluar el modelo? Explica por qué es tan bajo y si eso invalida al modelo.

**Respuesta:**  
El F1-Score es 2 * (0.138 * 0.8) / (0.138 + 0.8) = 0.235. Es tan bajo porque la precisión es muy pobre (13.8%), y el F1, al ser una media armónica, castiga severamente ese desequilibrio. Sin embargo, **no invalida al modelo** en este contexto, porque el F1-Score presupone que precisión y recall tienen igual importancia, y aquí no es así. El modelo puede ser útil si se prioriza el recall, aunque tenga un F1 bajo. La métrica adecuada aquí es el recall, no el F1.

---

**Pregunta 6:**  
El AUC-ROC del modelo es 0.85. ¿Es un valor aceptable? ¿Qué limitación tiene esta métrica en un problema con desbalance extremo como este?

**Respuesta:**  
Un AUC-ROC de 0.85 es aceptable, pero no excelente. Indica que el modelo distingue razonablemente bien entre contaminado y no contaminado. Sin embargo, en problemas con desbalance extremo, el AUC-ROC puede ser **optimista** porque la tasa de falsos positivos (FPR) se calcula sobre la clase mayoritaria (miles de muestras limpias), por lo que un pequeño aumento en FPR apenas se nota. Es más fiable usar el **AUC-PR (Precision-Recall)**, que se centra en la clase minoritaria. Si el AUC-PR es bajo, el modelo puede no ser tan bueno como sugiere el AUC-ROC.

---

**Pregunta 7:**  
Propón una estrategia concreta para mejorar el recall de este modelo, asumiendo que podemos aumentar los recursos de laboratorio para verificar más muestras.

**Respuesta:**  
Para mejorar el recall (y buscar el 100%), se pueden aplicar varias estrategias:  
1. **Bajar el umbral de decisión** drásticamente, por ejemplo, a 0.1. Así, cualquier muestra con una mínima sospecha se marcará como contaminada y se enviará a verificación. Esto generará muchos más falsos positivos, pero capturará casi todas las contaminaciones reales.  
2. **Aplicar sobremuestreo con SMOTE** para que el modelo tenga más ejemplos sintéticos de contaminación y aprenda mejor sus patrones.  
3. **Usar un modelo de detección de anomalías (one-class SVM)** que se entrene solo con muestras limpias y marque como anomalía cualquier desviación, en lugar de un clasificador binario tradicional.  
4. **Incorporar más sensores o variables** que puedan correlacionarse con la contaminación, mejorando la señal de la clase minoritaria.