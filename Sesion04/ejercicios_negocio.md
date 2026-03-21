
---

## **Caso 1: Detección de Phishing en Correos Electrónicos**

### **Contexto**
Una empresa de tecnología quiere proteger a sus empleados de correos de phishing. De cada 10,000 correos, 200 son realmente phishing (2% del total). El costo de un **falso negativo** (un phishing que llega a la bandeja de entrada) es altísimo: puede comprometer la seguridad de toda la empresa. El costo de un **falso positivo** (un correo legítimo marcado como phishing) también es relevante, porque puede hacer que se pierdan comunicaciones importantes.

### **Matriz de confusión del modelo**
|                 | Real Phishing | Real Legítimo |
|-----------------|---------------|---------------|
| **Pred. Phishing**  | VP = 180      | FP = 50       |
| **Pred. Legítimo**  | FN = 20       | VN = 9750     |

**Total de casos:** 180 + 50 + 20 + 9750 = 10,000

### **Cálculo de métricas paso a paso**

1. **Exactitud (Accuracy)**  
   \[
   \text{Accuracy} = \frac{VP + VN}{Total} = \frac{180 + 9750}{10000} = \frac{9930}{10000} = 0.993 \quad \Rightarrow \quad 99.3\%
   \]
   *Interpretación:* El modelo acierta el 99.3% de las veces. Pero ojo, si hubiera clasificado todo como legítimo, habría tenido un 98% de acierto. Esta métrica es engañosa por el desbalance.

2. **Precisión (Precision)**  
   \[
   \text{Precision} = \frac{VP}{VP + FP} = \frac{180}{180 + 50} = \frac{180}{230} \approx 0.7826 \quad \Rightarrow \quad 78.3\%
   \]
   *Interpretación:* De cada 100 correos que el modelo marca como phishing, 78 son realmente phishing y 22 son legítimos (falsos positivos). Es la **calidad de la alarma**.

3. **Recall (Sensibilidad)**  
   \[
   \text{Recall} = \frac{VP}{VP + FN} = \frac{180}{180 + 20} = \frac{180}{200} = 0.9 \quad \Rightarrow \quad 90\%
   \]
   *Interpretación:* El modelo captura el 90% de los correos de phishing reales. Se escapan 20 (falsos negativos) que son un riesgo de seguridad.

4. **Especificidad (Specificity)**  
   \[
   \text{Specificity} = \frac{VN}{VN + FP} = \frac{9750}{9750 + 50} = \frac{9750}{9800} \approx 0.9949 \quad \Rightarrow \quad 99.5\%
   \]
   *Interpretación:* De los correos legítimos, el 99.5% son correctamente identificados. Solo el 0.5% (50 correos) son erróneamente marcados como phishing.

5. **F1-Score**  
   \[
   F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.7826 \times 0.9}{0.7826 + 0.9} = 2 \cdot \frac{0.70434}{1.6826} \approx 0.837 \quad \Rightarrow \quad 83.7\%
   \]
   *Interpretación:* Es un equilibrio razonable, pero no excelente. La media armónica castiga el hecho de que la precisión no es tan alta como el recall.

6. **Curva ROC y AUC**  
   Supongamos que el AUC obtenido es **0.98**. Esto indica una capacidad excelente para distinguir entre phishing y legítimo, independientemente del umbral.

### **Análisis de negocio y recomendación**
- **¿Qué duele más?** En ciberseguridad, los falsos negativos (phishing no detectado) son críticos. Un recall del 90% significa que 1 de cada 10 ataques pasa, lo cual puede ser inaceptable.
- **Estrategia:** Se debe priorizar aumentar el recall, incluso si la precisión baja (más falsos positivos). Por ejemplo, bajando el umbral de decisión para que el modelo marque más correos como phishing. Luego, los falsos positivos pueden gestionarse con una cuarentena revisable por el usuario o un sistema de doble verificación.
- **Métrica principal:** **Recall**. Monitorear también la precisión para no saturar de falsas alarmas.

---

## **Caso 2: Detección de Fraude en Compras de E-commerce**

### **Contexto**
Una plataforma de e-commerce quiere detectar transacciones fraudulentas. El fraude representa el 1% de las compras (1,000 de 100,000). Un **falso negativo** (fraude no detectado) causa pérdida económica y chargebacks. Un **falso positivo** (compra legítima bloqueada) genera insatisfacción del cliente y pérdida de venta. Se busca un equilibrio.

### **Matriz de confusión**
|                 | Real Fraude | Real Legítima |
|-----------------|-------------|---------------|
| **Pred. Fraude**    | VP = 800    | FP = 2000     |
| **Pred. Legítima**  | FN = 200    | VN = 97000    |

**Total:** 800 + 2000 + 200 + 97000 = 100,000

### **Cálculo de métricas**

1. **Exactitud**  
   \[
   \frac{800 + 97000}{100000} = \frac{97800}{100000} = 0.978 \quad (97.8\%)
   \]
   *Comentario:* Alta, pero engañosa por desbalance.

2. **Precisión**  
   \[
   \frac{800}{800 + 2000} = \frac{800}{2800} \approx 0.2857 \quad (28.6\%)
   \]
   *Interpretación:* De cada 100 transacciones marcadas como fraude, solo 29 lo son. 71 son clientes legítimos bloqueados. ¡Muy malo para la experiencia de usuario!

3. **Recall**  
   \[
   \frac{800}{800 + 200} = \frac{800}{1000} = 0.8 \quad (80\%)
   \]
   *Interpretación:* Se detecta el 80% de los fraudes, pero se escapan 200.

4. **Especificidad**  
   \[
   \frac{97000}{97000 + 2000} = \frac{97000}{99000} \approx 0.9798 \quad (98.0\%)
   \]
   *Interpretación:* El 98% de las compras legítimas son correctamente aceptadas, pero en valor absoluto son 2000 falsos positivos.

5. **F1-Score**  
   \[
   F1 = 2 \cdot \frac{0.2857 \times 0.8}{0.2857 + 0.8} = 2 \cdot \frac{0.22856}{1.0857} \approx 0.421 \quad (42.1\%)
   \]
   *Interpretación:* Muy bajo, refleja el pobre equilibrio.

6. **AUC-ROC** ≈ 0.92 (bueno, pero no excelente).

### **Análisis y recomendación**
- **Problema:** La precisión es demasiado baja (28.6%), lo que enfadará a muchos clientes. El recall del 80% puede ser insuficiente si las pérdidas por fraude son altas.
- **Estrategia:** Se necesita un equilibrio. Aquí el **F1-Score** es la métrica clave, y con 42% hay mucho margen de mejora. Se puede ajustar el umbral para aumentar la precisión (aunque baje el recall) o implementar un sistema híbrido: las transacciones marcadas como fraude pasan por una verificación adicional (ej. SMS) en lugar de bloquearse directamente.
- **Objetivo:** Mejorar la precisión a >50% sin que el recall baje de 75%. También se podrían usar técnicas de balanceo (SMOTE) o modelos más complejos.

---

## **Caso 3: Falla de Mantenimiento Industrial de Motores**

### **Contexto**
Una fábrica utiliza sensores para predecir fallas inminentes en motores. Las fallas ocurren en el 5% de las mediciones (500 de 10,000). Un **falso negativo** (no predecir una falla) puede causar una parada no planificada, reparaciones costosas y accidentes. Un **falso positivo** (predecir falla cuando no la hay) implica detener la máquina innecesariamente, con costo de producción perdida, pero es mucho menor que una falla real.

### **Matriz de confusión**
|                 | Real Falla | Real No Falla |
|-----------------|------------|---------------|
| **Pred. Falla**     | VP = 450   | FP = 300      |
| **Pred. No Falla**  | FN = 50    | VN = 9200     |

**Total:** 450 + 300 + 50 + 9200 = 10,000

### **Cálculo de métricas**

1. **Exactitud**  
   \[
   \frac{450 + 9200}{10000} = \frac{9650}{10000} = 0.965 \quad (96.5\%)
   \]

2. **Precisión**  
   \[
   \frac{450}{450 + 300} = \frac{450}{750} = 0.6 \quad (60\%)
   \]
   *Interpretación:* De cada 10 alertas de falla, 6 son correctas y 4 son falsas alarmas. Aceptable si el costo de parada no es altísimo.

3. **Recall**  
   \[
   \frac{450}{450 + 50} = \frac{450}{500} = 0.9 \quad (90\%)
   \]
   *Interpretación:* Se detecta el 90% de las fallas, pero aún se escapan 50 que causarán paradas no planificadas.

4. **Especificidad**  
   \[
   \frac{9200}{9200 + 300} = \frac{9200}{9500} \approx 0.9684 \quad (96.8\%)
   \]

5. **F1-Score**  
   \[
   F1 = 2 \cdot \frac{0.6 \times 0.9}{0.6 + 0.9} = 2 \cdot \frac{0.54}{1.5} = \frac{1.08}{1.5} = 0.72 \quad (72\%)
   \]

6. **AUC-ROC** ≈ 0.95 (excelente).

### **Análisis y recomendación**
- **Prioridad:** Evitar falsos negativos es crucial. El recall del 90% es bueno, pero aún mejorable. Se debe intentar aumentar el recall al 95% o más, aunque la precisión baje (más falsos positivos). El costo de una parada no planizada es mucho mayor que el de una falsa alarma.
- **Estrategia:** Ajustar el umbral de decisión para ser más sensible. También se podría aplicar sobremuestreo de la clase minoritaria para que el modelo aprenda mejor las señales de falla.
- **Métrica principal:** **Recall**. El F1-Score sirve como referencia secundaria.

---

## **Caso 4: Calidad del Agua Potable (Detección de Contaminación)**

### **Contexto**
Una empresa de suministro de agua analiza muestras para detectar contaminación. La contaminación es extremadamente rara: 0.1% de las muestras (10 de 10,000). Un **falso negativo** (no detectar contaminación) puede causar una crisis de salud pública, demandas y cierre del suministro. Un **falso positivo** (declarar contaminada agua limpia) implica realizar pruebas adicionales y posiblemente paradas innecesarias, con costo económico pero mucho menor que un FN.

### **Matriz de confusión**
|                 | Real Contam | Real No Contam |
|-----------------|-------------|----------------|
| **Pred. Contam**    | VP = 8      | FP = 50        |
| **Pred. No Contam** | FN = 2      | VN = 9940      |

**Total:** 8 + 50 + 2 + 9940 = 10,000

### **Cálculo de métricas**

1. **Exactitud**  
   \[
   \frac{8 + 9940}{10000} = \frac{9948}{10000} = 0.9948 \quad (99.48\%)
   \]
   *Nota:* Un modelo que siempre predijera "no contaminado" tendría 99.9% de exactitud. Es completamente engañosa.

2. **Precisión**  
   \[
   \frac{8}{8 + 50} = \frac{8}{58} \approx 0.1379 \quad (13.8\%)
   \]
   *Interpretación:* De cada 100 alertas de contaminación, solo 14 son reales. ¡Muchas falsas alarmas!

3. **Recall**  
   \[
   \frac{8}{8 + 2} = \frac{8}{10} = 0.8 \quad (80\%)
   \]
   *Interpretación:* Se detecta el 80% de las contaminaciones, pero se escapan 2 muestras peligrosas. En salud pública, es inaceptable.

4. **Especificidad**  
   \[
   \frac{9940}{9940 + 50} = \frac{9940}{9990} \approx 0.995 \quad (99.5\%)
   \]
   *Interpretación:* El 99.5% de las muestras limpias se identifican correctamente, pero en números absolutos son 50 falsos positivos.

5. **F1-Score**  
   \[
   F1 = 2 \cdot \frac{0.1379 \times 0.8}{0.1379 + 0.8} = 2 \cdot \frac{0.11032}{0.9379} \approx 0.235 \quad (23.5\%)
   \]
   *Interpretación:* Muy bajo, refleja la dificultad del problema.

6. **AUC-ROC** ≈ 0.85 (aceptable, pero podría mejorar).

### **Análisis y recomendación**
- **Prioridad absoluta:** Maximizar el recall, incluso si la precisión cae aún más. La salud pública no admite falsos negativos. Idealmente, se debe buscar un recall del 100% (detectar todas las contaminaciones).
- **Estrategia:** Ajustar el umbral de decisión para que el modelo sea extremadamente sensible. Esto generará más falsos positivos (más de 50), pero cada falso positivo puede ser verificado con una prueba de laboratorio adicional, lo cual es un costo asumible frente a una catástrofe sanitaria.
- **Métrica principal:** **Recall**. El F1-Score no es adecuado aquí porque penaliza la baja precisión, que en este contexto es secundaria.
- **Técnicas complementarias:** Usar sobremuestreo extremo (SMOTE) o modelos de detección de anomalías (one-class SVM) para mejorar la sensibilidad.

---

## **Tabla Resumen de Métricas por Caso**

| Caso                 | Exactitud | Precisión | Recall | Especificidad | F1-Score | Métrica Prioritaria |
|----------------------|-----------|-----------|--------|---------------|----------|---------------------|
| **Phishing**         | 99.3%     | 78.3%     | 90%    | 99.5%         | 83.7%    | Recall (seguridad)  |
| **Fraude e-commerce**| 97.8%     | 28.6%     | 80%    | 98.0%         | 42.1%    | F1-Score (equilibrio) |
| **Mantenimiento**    | 96.5%     | 60%       | 90%    | 96.8%         | 72%      | Recall (evitar fallas) |
| **Calidad del agua** | 99.5%     | 13.8%     | 80%    | 99.5%         | 23.5%    | Recall (salud pública) |

---

## **Conclusión General**

Hemos visto que la misma métrica puede tener interpretaciones opuestas según el negocio. La **exactitud** es una trampa en problemas desbalanceados. La **precisión** y el **recall** reflejan los dos tipos de error, y su importancia relativa viene dada por el costo de los falsos positivos y falsos negativos. El **F1-Score** es un buen resumen cuando se busca un equilibrio, pero en contextos extremos (como salud pública) se debe optimizar directamente la métrica crítica. La **curva ROC** y el **AUC** nos ayudan a evaluar la capacidad global del modelo, pero en desbalance severo es preferible usar la **curva Precision-Recall** (PR-AUC).

Con estos ejercicios, ya están listos para enfrentarse a problemas reales y tomar decisiones basadas en datos y en el impacto de negocio. ¡Manos a la obra!









