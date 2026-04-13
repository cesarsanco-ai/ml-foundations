---
layout: default
---
# Sesión 2: Análisis Exploratorio y Feature Engineering

### 1. Logro de la sesión

Realizar un análisis exploratorio de datos (EDA) completo y aplicar técnicas de feature engineering que transformen datos crudos en conjuntos listos para modelado, con criterio para detectar problemas de calidad, formular hipótesis de negocio y diseñar variables predictivas que maximicen el rendimiento sin sacrificar interpretabilidad ni reproducibilidad.

---

### 2. Fundamentos conceptuales

#### Definición y papel del EDA

**Marco teórico:** El *Exploratory Data Analysis* (EDA) formaliza la idea de **explorar antes de modelar**. John Tukey lo planteó como un enfoque para **“escuchar a los datos”** mediante estadísticos y visualizaciones antes de imponer supuestos fuertes (Tukey, 1977). En el ciclo de vida de un proyecto (p. ej. CRISP-DM), el EDA se concentra en las fases de **comprensión y preparación de datos**.

El EDA permite:

- Detectar errores, inconsistencias y valores atípicos.
- Generar hipótesis iniciales sobre el comportamiento de los datos.
- Orientar la elección de modelos y de transformaciones (feature engineering).
- Comunicar hallazgos preliminares a las partes interesadas.

#### Casos de estudio: éxito y fricción en la industria

| Contexto | Ejemplo | Lectura |
|----------|---------|---------|
| Éxito | **Netflix:** EDA sobre hábitos de visualización como base del sistema de recomendación (gran parte del consumo vía recomendación). | Datos de uso + EDA riguroso → señales predictivas estables. |
| Éxito | **Amazon:** Patrones de compra y navegación alimentan recomendación y precios dinámicos. | Comportamiento agregado bien caracterizado antes del modelado. |
| Fracaso | **Google Flu Trends:** No modeló adecuadamente estacionalidad y cambios en el comportamiento de búsqueda. | Predicciones erróneas; falta de validación frente a cambios de *query*. |
| Fracaso | **COMPAS (reincidencia):** EDA insuficiente respecto a sesgos en datos y etiquetas. | Modelos perpetúan sesgos; impacto ético y de confianza. |

**Referentes:** Tukey (1977); Lazer et al. (2014) sobre limitaciones de Google Flu Trends; investigación sobre equidad algorítmica en sistemas de scoring (p. ej. Angwin et al., 2016, sobre COMPAS).

---

### 3. Pipeline de análisis exploratorio

**Marco teórico:** Un pipeline de EDA es una **secuencia reproducible** de pasos alineada con la preparación de datos en CRISP-DM. No es lineal al 100%: los hallazgos suelen obligar a **volver atrás** (p. ej. nueva limpieza tras un análisis bivariado).

| Etapa | Contenido típico |
|-------|------------------|
| 1. Comprensión del problema y de los datos | Fuentes, diccionario de datos, tipos de variables (numéricas, categóricas, temporales), tamaño y granularidad. |
| 2. Evaluación de calidad | Valores faltantes, duplicados, reglas de negocio y consistencia lógica. |
| 3. Limpieza y preprocesamiento | Tipos correctos, imputación o exclusión de faltantes, tratamiento de outliers según criterio. |
| 4. Análisis univariado | Distribución, dispersión y forma de cada variable. |
| 5. Análisis bivariado | Relaciones entre pares (correlación, tablas de contingencia, comparación de grupos). |
| 6. Análisis multivariado | Redundancia, multicolinealidad, estructura conjunta, outliers multivariantes, reducción de dimensión exploratoria. |
| 7. Documentación | Visualizaciones, conclusiones accionables para feature engineering y modelado. |

---

### 4. Calidad de datos y limpieza

**Marco teórico:** La calidad condiciona el **techo de rendimiento** del modelo: los algoritmos aprenden patrones de $X$ e $y$ tal como se observan (incluidos errores). La limpieza busca **fiabilidad** sin destruir señal legítima (p. ej. outliers que son casos reales raros).

#### Valores faltantes: mecanismos

**Marco teórico:** Clasificación de Little & Rubin (2002) para el mecanismo de ausencia:

- **MCAR (Missing Completely At Random):** la ausencia es independiente de observados y no observados.
- **MAR (Missing At Random):** depende de variables observadas, no del valor faltante condicional a lo observado.
- **MNAR (Missing Not At Random):** depende del propio valor no observado u otros no observados.

#### Estrategias de imputación

**Imputación simple (media / mediana):**

$$ x_{\text{imp}} = \begin{cases} \bar{x} & \text{media} \\ \text{mediana}(x) & \text{mediana} \end{cases} $$

*Limitación:* puede subestimar varianza y correlaciones.

**Imputación por regresión:**

$$ x_{\text{faltante}} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots $$

**k-Nearest Neighbors:** imputar con la media (u otra agregación) de los $k$ vecinos más cercanos en el espacio de predictores.

**Imputación múltiple (MICE / encadenada):** generar varios conjuntos imputados y combinar inferencias para reflejar **incertidumbre**. En la práctica iterativa:

$$ x_j = f(x_{-j}) $$

repitiendo hasta convergencia (Iterative Imputer / encadenamiento por variables).

**Referente:** Van Buuren & Groothuis-Oudshoorn (2011) sobre MICE; sklearn documenta `IterativeImputer` como aproximación práctica.

#### Cuándo usar cada tipo de imputación (criterios prácticos)

La elección depende del **tipo de variable**, del **mecanismo de ausencia** (Sección 4), del **coste del error** y de si el **hecho de faltar** aporta información.

| Estrategia | Casos de uso típicos | Precauciones |
|------------|----------------------|--------------|
| **Media** | Variables **numéricas** continuas; distribución **aproximadamente simétrica**; ausencia ligera y poco sesgada (p. ej. MCAR cercano). | Muy sensible a **outliers** y colas pesadas; **reduce la varianza** y puede **atenuar correlaciones** con otras columnas. |
| **Mediana** | Numéricas con **asimetría**, **outliers** o colas largas; mismo rol que la media pero más **robusta**. | Sigue siendo imputación univariada: no usa otras variables salvo implícitamente vía el percentil. |
| **Moda** | Variables **categóricas** (nominales); **ordinales** solo si la moda es representativa del nivel “típico”. | Si hay **empates** o muchas categorías con frecuencias similares, la moda es **arbitraria**; valor “desconocido” o categoría **“missing”** explícita suele ser mejor. |
| **Constante o categoría “missing”** | Cuando el **missingness** puede ser **informativo** (MNAR o señal de proceso: no respondió, sensor apagado). | Combinar con **variable indicadora** $m_i=\mathbf{1}(\text{faltante})$ para no perder esa señal. |
| **Imputación por grupos** (media/mediana/modal **por estrato**) | Ausencia **MAR**: el valor típico depende de otra variable (p. ej. ciudad, segmento). | Los estratos deben tener **suficiente $n$**; evitar grupos demasiado finos (varianza alta / sobreajuste al train). |
| **Forward-fill / backward-fill** | **Series temporales** ordenadas donde es razonable arrastrar el **último valor observado** (sensores, precios intradía). | No usar si el orden temporal no es el de generación del dato o si los huecos son largos y no intercambiables. |
| **Interpolación lineal / splines** | **Series temporales** con huecos **cortos** entre puntos fiables. | Puede inventar dinámicas inexistentes en huecos largos o en saltos estructurales. |
| **Regresión / k-NN / MICE** | Relaciones **multivariantes**; MAR con buenos predictores; necesidad de respetar **covarianzas** aproximadas entre columnas. | Más **complejidad** y riesgo de **leakage** si se usan estadísticas del test; **ajuste solo en train** y mismo transformador en validación/test. |

**Regla práctica:** para variables numéricas “limpias” y simétricas → media o mediana (mediana ante duda); para categóricas → moda o categoría explícita de ausencia; cuando el faltante **no es aleatorio** o es parte del fenómeno → **indicador** + imputación conservadora o modelo que explícitamente modele la ausencia.

#### Duplicados

Detección por coincidencia exacta de filas o por **claves parciales**. Eliminar solo cuando el duplicado **no** representa observaciones distintas (regla de negocio).

#### Errores de tipo de dato

| Problema | Corrección típica |
|----------|-------------------|
| Fechas como texto | `datetime` coherente con zona/formato |
| Categorías codificadas como enteros | Tipo categórico u ordinal explícito |
| Números con símbolos o separadores | Normalización previa a numérico |

#### Tratamiento de outliers (univariado)

**Rango intercuartílico (IQR):**

$$ IQR = Q_3 - Q_1 $$

$$ \text{Inferior} = Q_1 - 1.5 \cdot IQR, \quad \text{Superior} = Q_3 + 1.5 \cdot IQR $$

**Z-score clásico:**

$$ z = \frac{x - \mu}{\sigma} $$

Umbral habitual: $|z| > 3$ (bajo supuestos y con cautela).

**Z-score robusto (MAD):**

$$ MAD = \text{mediana}(|x_i - \text{mediana}(x)|) $$

$$ M_i = \frac{0.6745\,(x_i - \text{mediana}(x))}{MAD} $$

Más robusto a colas pesadas que $\mu,\sigma$ clásicos.

---

### 5. Análisis exploratorio: univariado, bivariado y multivariado

#### Univariado — variables numéricas

**Tendencia central:** media $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$, mediana, moda.

**Dispersión:**

$$ R = x_{\max} - x_{\min}, \quad \sigma^2 = \frac{1}{n-1}\sum (x_i-\bar{x})^2, \quad \sigma = \sqrt{\sigma^2}, \quad IQR = Q_3-Q_1 $$

**Forma:** asimetría (skewness), curtosis (colas).

**Visualización y densidad:**

- Histogramas, KDE:

$$ \hat f(x)=\frac{1}{nh}\sum_{i=1}^{n} K\left(\frac{x-x_i}{h}\right) $$

con $K$ kernel (p. ej. gaussiano).

- Boxplots, ECDF:

$$ F_n(x)=\frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(x_i \le x) $$

#### Univariado — variables categóricas

Frecuencias absolutas/relativas, moda; barras, Pareto, tablas de contingencia para cruces posteriores.

#### Univariado — variables temporales

Descomposición conceptual: **tendencia**, **estacionalidad**, **ruido**.

Modelo aditivo: $Y_t = T_t + S_t + R_t$.  
Modelo multiplicativo: $Y_t = T_t \cdot S_t \cdot R_t$.

#### Bivariado — numérica vs numérica

**Pearson:**

$$ r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}} $$

| $|r|$ (orden de magnitud) | Interpretación orientativa |
|-------------------------|----------------------------|
| $\approx 0$ | Nula lineal |
| $\approx 0.3$ | Débil |
| $\approx 0.5$ | Moderada |
| $\geq 0.7$ | Fuerte |

**Spearman:** correlación de rangos; útil para relaciones **monótonas** no lineales.

Visualización: scatter, pairplots, heatmaps de correlación.

#### Bivariado — categórica vs categórica

Tablas de contingencia; **chi-cuadrado**:

$$ \chi^2 = \sum \frac{(O - E)^2}{E} $$

#### Bivariado — numérica vs categórica

Comparación de distribuciones por grupo: boxplots por categoría, **t-test**, **ANOVA**:

$$ F = \frac{\text{varianza entre grupos}}{\text{varianza dentro de grupos}} $$

#### Multivariado

**Matriz de correlación:** detectar multicolinealidad / redundancia; regla orientativa $|r| > 0.9$ como señal para revisar eliminación o regularización.

**Pairplots:** todas las parejas numéricas; útil para no linealidad, grupos y outliers conjuntos.

**Outliers multivariantes — distancia de Mahalanobis:**

$$ D_M(x) = \sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)} $$

Incorpora correlaciones entre variables.

**Reducción de dimensionalidad exploratoria — PCA:** nuevas direcciones ortogonales $z_1 = w_1^\top x$ maximizando varianza explicada (base para visualización y detección de estructura; el uso predictivo formal se profundiza más adelante en el curso).

**Síntesis del rol del EDA:** comprender estructura, detectar problemas de calidad y generar hipótesis para el feature engineering. Un EDA profundo acota el **rendimiento alcanzable**, porque los algoritmos dependen de la calidad y la representación de los datos.

---

### 6. Feature engineering: marco formal y pipeline

**Marco teórico:** El *feature engineering* busca una representación $\phi$ tal que el algoritmo capture mejor la estructura relevante. Si $X \in \mathbb{R}^{n \times p}$, se busca:

$$ \phi(X) \rightarrow Z, \quad Z \in \mathbb{R}^{n \times k} $$

donde $k$ puede ser mayor o menor que $p$. En la práctica, **buenas características** a menudo importan más que cambiar marginalmente de algoritmo (Hastie et al.; Domingos, 2012, sobre “mastering the features”).

**Pipeline típico de ingeniería de características:**

1. Creación de nuevas características a partir de las existentes.
2. Transformaciones (log, Box-Cox, polinomios, interacciones).
3. Codificación de categóricas.
4. Escalado o normalización cuando el algoritmo lo exige.
5. Imputación coherente con el pipeline (evitando *data leakage*).
6. Tratamiento de outliers acorde al negocio (no siempre eliminación).
7. Selección de características.
8. Automatización reproducible (`Pipeline` en sklearn u orquestación equivalente).

#### Feature Store (temario del curso)

**En el temario** (Semana 2: *Análisis Exploratorio y Feature Engineering*), el **feature store** se incluye como tema explícito junto al pipeline de características y la imputación.

**Marco teórico:** Un *feature store* es un **repositorio y servicio** que centraliza **definiciones**, **cálculo** y **suministro** (*serving*) de características para entrenamiento e inferencia. Objetivos habituales:

- **Consistencia train / serving:** mismas transformaciones y mismas fuentes para el modelo offline y en producción (evita *training-serving skew*).
- **Reutilización y gobernanza:** equipos comparten *features* versionadas, con linaje y control de acceso.
- **Punto en el tiempo (*point-in-time correctness*):** para datos tabulares con historia, las *features* deben calcularse **solo con información disponible hasta el instante de cada etiqueta** (crítico en validación y en backtesting).

En la práctica industrial suele distinguirse **ingesta por lotes** (entrenamiento, backfill) y **baja latencia** (predicción online). Herramientas de referencia incluyen **Feast** (open source), **Tecton**, **Vertex AI Feature Store**, **Databricks Feature Store**, entre otras, según nube y stack.

**Cuándo plantearlo:** varios modelos consumiendo las mismas señales, equipos de datos/ML que colisionan en definiciones duplicadas, o necesidad fuerte de **trazabilidad** y despliegue recurrente. Para prototipos pequeños suele bastar un *pipeline* bien versionado; el feature store aporta más valor al **escalar** personas, modelos o entornos.

#### Creación y transformación de variables numéricas

**Log:** $x' = \log(x)$ (soportes positivos; reduce asimetría fuerte).

**Box-Cox:**

$$ x' = \begin{cases} \frac{x^\lambda -1}{\lambda} & \lambda \neq 0 \\ \log(x) & \lambda = 0 \end{cases} $$

$\lambda$ suele estimarse por verosimilitud.

**Yeo-Johnson:** extensión que admite valores no positivos (familia análoga a Box-Cox en espíritu).

#### Interacciones y términos polinomiales

$$ x_{ij}^{(\text{int})} = x_i \cdot x_j; \quad x^2, x^3, \ldots $$

#### Binning (discretización)

**Igual ancho:** $\text{bin}_k = [a + k\Delta, a + (k+1)\Delta]$.  
**Igual frecuencia:** bins con conteos similares.  
También **puntos de corte** guiados por árboles (segmentación supervisada informal).

#### Variables temporales: rezagos, ventanas y ciclicidad

**Rezagos:** $y_{t-1}, y_{t-2}, \ldots$

**Media móvil:**

$$ MA_t = \frac{1}{k} \sum_{i=0}^{k-1} y_{t-i} $$

**Desviación móvil:**

$$ SD_t = \sqrt{\frac{1}{k-1}\sum_{i=0}^{k-1}(y_{t-i}-MA_t)^2} $$

**Codificación cíclica** (hora, día de la semana):

$$ \sin\left(\frac{2\pi t}{T}\right), \quad \cos\left(\frac{2\pi t}{T}\right) $$

#### Agregaciones por grupo (datos jerárquicos)

Por cliente, tienda, etc.: medias, conteos, desviaciones, máximos/mínimos históricos — resumen del comportamiento longitudinal.

#### Texto: BoW, TF-IDF y embeddings

**Bag of Words:** vectores de conteos por término.

**TF-IDF:**

$$ \mathrm{TF\text{-}IDF}(t,d) = \mathrm{TF}(t,d) \times \log\left(\frac{N}{\mathrm{DF}(t)}\right) $$

**Embeddings densos:** Word2Vec, contextual (BERT, etc.) según necesidad y presupuesto computacional.

---

### 7. Codificación, escalado, clases desbalanceadas e imputación en pipeline

#### Codificación de categóricas

| Método | Idea | Notas |
|--------|------|-------|
| One-Hot | Una binaria por categoría | Explota dimensionalidad si cardinalidad alta |
| Label encoding | Enteros por categoría | Adecuado para ordinales; riesgo de orden falso en nominales |
| Target encoding | Media de $y$ por categoría | **Riesgo de leakage**; usar validación cruzada anidada u smoothing |
| Frequency encoding | $n_{\text{cat}}/n$ | Compacto; puede perder interacción fina |
| Binary encoding | Categoría → binario compacto | Reduce ancho frente a OHE |
| Hash encoding | $h(x) \in \{0,\ldots,k-1\}$ | Alta cardinalidad; colisiones controladas |

Target encoding (formalización útil):

$$ x_{\text{cat}} = \frac{1}{n_{\text{cat}}} \sum_{i \in \text{cat}} y_i $$

#### Escalado

**Estandarización (Z-score):** $x' = \frac{x-\mu}{\sigma}$ (media 0, varianza 1). Relevante para p. ej. SVM, PCA, muchos solvers de regresión penalizada.

**Min-Max:** $x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}$ a $[0,1]$.

**Robusto:** $x' = \frac{x - \text{mediana}(x)}{IQR}$ — menos sensible a outliers extremos.

#### Clases desbalanceadas

**Oversampling (p. ej. SMOTE):** sintéticos por interpolación con vecinos:

$$ x_{\text{new}} = x_i + \lambda (x_{\text{NN}} - x_i) $$

**Undersampling:** reduce mayoritaria; puede descartar información.

**Métricas:** precisión, recall, F1, ROC-AUC — la elección depende del coste de errores (véase Sesión 1):

$$ F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} $$

#### Imputación avanzada dentro del flujo (recordatorio operativo)

Además de la Sección 4: **KNN Imputer**, **Iterative Imputer (MICE)** como encadenamiento $x_j = f(x_{-j})$. Lo crítico es **ajustar imputadores solo sobre train** (o CV) y aplicar transformación a validación/test para evitar fuga de información.

#### Outliers en feature engineering

No siempre errores: pueden ser señal. Estrategias:

- **Indicador:** $\text{outlier}_i \in \{0,1\}$.
- **Winsorización:** cap al percentil (p. ej. 1 % / 99 %).
- **Transformaciones** (log, raíz, Box-Cox) para acotar colas.

---

### 8. Selección de características, dominio y comunicación

#### Selección de características

**Marco teórico:** Objetivo: menor dimensionalidad, menor varianza y mejor generalización, sin perder señal (Guyon & Elisseeff, 2003).

**Filtros:** correlación, **información mutua**

$$ I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} $$

**Wrappers:** evalúan subconjuntos con el modelo (p. ej. RFE).

**Embebidos:** Lasso penaliza $L_1$:

$$ \min_{\beta} \sum (y - X\beta)^2 + \lambda \sum |\beta_j| $$

promoviendo esparcidad en $\beta$.

#### Variables derivadas del negocio

**Marco teórico:** Incorporar **conocimiento de dominio** suele ser la fuente de mayor valor predictivo interpretable.

Ejemplos:

- **Finanzas:** ratios de liquidez, endeudamiento, márgenes.
- **Temporal:** ventas últimos 7 días, medias móviles, crecimiento mes a mes.
- **Comportamiento (RFM y afines):** recencia, frecuencia, valor monetario — churn, fraude, recomendación, scoring.

#### Comunicación de hallazgos

**Storytelling e insights**

- Patrones: gráficos comparativos, heatmaps, scatter con color por clase.
- Impacto: importancia de variables (p. ej. bosques), **SHAP** para contribuciones locales/globales.
- Claridad visual: simplicidad, anotaciones, accesibilidad.
- Audiencia: técnicos (metodología, métricas, supuestos) vs negocio (impacto, acciones).

**Elevator pitch (2–3 minutos)**

- **Técnico:** métodos, métricas ($R^2$, AUC, F1), hipótesis validadas, limitaciones y deuda técnica.
- **No técnico:** valor, recomendaciones, riesgos y siguientes pasos.

**Estructura útil:** Contexto → Hallazgos → Acción.

---

### 9. Apéndice de estudio: línea temporal EDA / *feature engineering* y lecturas

| Año | Hito | Por qué importa |
|-----|------|-----------------|
| **1960s–1977** | Tukey formaliza el EDA como contrapeso a tests hipotéticos rígidos | Legitima exploración visual y robusta |
| **1987** | Rousseeuw define la **silueta** para clustering | Métrica interna aún estándar |
| **2002** | Little & Rubin sistematizan datos faltantes (MCAR/MAR/MNAR) | Base de imputación seria |
| **2002–2011** | **MICE** / encadenamiento en R (`mice`) y equivalentes en Python | Imputación múltiple en la práctica |
| **2010s** | **Tidy data** (Wickham) y cultura `pandas` | Estandariza transformaciones reproducibles |
| **2020s** | **Feature stores** (Feast, Tecton, cloud) | Consistencia train/serving a escala |

#### 9.1 Pandas: patrones mínimos reproducibles

```python
import pandas as pd

df = pd.read_parquet("datos.parquet")
df = df.assign(
    month=df["fecha"].dt.month,
    dow=df["fecha"].dt.dayofweek,
)
summary = df.describe(include="all").T
```

#### 9.2 Documentar transformaciones

Cada paso de imputación o codificación debe poder **repetirse en inferencia** (mismos parámetros aprendidos en train). Por ello los **`Pipeline`** de sklearn y los feature stores versionan **definiciones** y **valores** de features.

#### 9.3 Comunicación: checklist antes de presentar

1. ¿La figura tiene **título** y **ejes** con unidades?  
2. ¿Se declara el **tamaño muestral** y el periodo temporal?  
3. ¿Los hallazgos se conectan con **acciones** concretas para negocio?

---

## Referencias bibliográficas principales

1. Angwin, J., et al. (2016). *Machine Bias* (investigación sobre COMPAS; ProPublica).  
2. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *JMLR*, 3, 1157–1182.  
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.  
4. Lazer, D., et al. (2014). The parable of Google Flu: traps in big data analysis. *Science*, 343(6176), 1203–1205.  
5. Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data* (2nd ed.). Wiley.  
6. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.  
7. Van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67.  
8. Huyen, C. (2022). *Designing Machine Learning Systems*. O’Reilly. (datos en producción, consistencia train/serving y feature stores).  
