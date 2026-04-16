---
layout: default
---
# Sesión 13: Interpretabilidad de modelos

### 1. Logro de la sesión

Aplicar **interpretabilidad global y local** a modelos de ML: **importancia por permutación**, **PDP/ICE**, **SHAP** (valores de Shapley y *TreeSHAP*), y **LIME** para explicaciones locales aproximadas. El alumno debe reconocer **limitaciones matemáticas y éticas**: inestabilidad bajo correlación fuerte, riesgo de **confundir explicación con causalidad**, y el hecho de que explicar un modelo **sesgado** puede **legitimar** decisiones injustas.

---

### 2. Historia y contexto

| Periodo | Hito |
|---------|------|
| **1980s–90s** | Modelos lineales y árboles como “interpretables” por construcción |
| **2000s** | *Partial dependence* en *random forests* (Friedman) |
| **2010s** | Auge de cajas negras (boosting profundo, redes); demanda regulatoria (GDPR “derecho a explicación” discutido) |
| **2016** | **LIME**: explicaciones locales modelo-agnósticas |
| **2017** | **SHAP**: unificación con valores de Shapley y aproximaciones eficientes para árboles |
| **2020s** | Auditorías de fairness, *model cards*, tensiones entre interpretabilidad y rendimiento |

**Lectura:** “interpretable” no es una propiedad binaria: depende del **público** (auditor vs usuario final), del **coste del error** y del **nivel de abstracción** (peso de coeficiente vs contrafactual).

---

### 3. Taxonomía de interpretabilidad

| Dimensión | Pregunta típica |
|-----------|-----------------|
| **Alcance global** | ¿Qué variables impulsan el modelo en todo el dataset? |
| **Alcance local** | ¿Por qué el modelo predijo $\hat{y}$ para **esta** instancia $\mathbf{x}$? |
| **Intrínseca vs post-hoc** | ¿La estructura del modelo ya es legible (regresión corta) o necesitamos herramientas externas (SHAP sobre XGBoost)? |
| **Fidelidad vs comprensibilidad** | ¿La explicación **reproduce** $f$ o solo aproxima con modelo simple? |

**Modelos intrínsecamente interpretables (relativos):** regresión lineal con pocos términos, árbol de poca profundidad, reglas lista.  
**Post-hoc:** aplica a cualquier $f$ con API de predicción.

---

### 4. Importancia por permutación

#### 4.1 Procedimiento

1. Entrenar $f$ y medir métrica $S$ en un conjunto de validación.  
2. Para cada feature $j$, **permutar** aleatoriamente la columna $X_j$ rompiendo su asociación con $y$ (y con otras features solo indirectamente).  
3. Recalcular $S$; la **caída** $\Delta S_j$ mide importancia.

#### 4.2 Ventajas y límites

- **Modelo-agnóstica**; implementación simple (`sklearn.inspection.permutation_importance`).  
- **Coste:** $O(p \times n_{\mathrm{rep}} \times \text{coste inferencia})$.  
- **Correlación:** permutar una variable de un grupo colineal puede que **no** destruya la señal porque otra variable la reemplaza → importancias **inestables** o repartidas de forma engañosa.

---

### 5. Dependencia parcial (PDP) y ICE

#### 5.1 PDP

Para feature $x_j$, la **dependencia parcial** marginaliza sobre la distribución empírica de las demás variables:

$$f_j^{\mathrm{PDP}}(x_j) = \mathbb{E}_{\mathbf{x}_{\setminus j}}[f(x_j, \mathbf{x}_{\setminus j})] \approx \frac{1}{n}\sum_{i=1}^n f(x_j, \mathbf{x}_{\setminus j}^{(i)})$$

Se promedia la predicción del modelo fijando $x_j$ y dejando el resto como en las muestras observadas.

#### 5.2 ICE

Las curvas **ICE** (*Individual Conditional Expectation*) grafican $f(x_j, \mathbf{x}_{\setminus j}^{(i)})$ por cada $i$ sin promediar. Revelan **heterogeneidad**: el efecto de $x_j$ puede ser positivo para unos y negativo para otros.

#### 5.3 Cuidado con correlación

Si $x_j$ y $x_k$ están fuertemente correlacionados, evaluar $x_j$ en valores **imposibles** conjuntamente con $\mathbf{x}_{\setminus j}$ de muestras reales produce **extrapolaciones** fuera del soporte de datos → curvas PDP/ICE engañosas.

---

### 6. SHAP (SHapley Additive exPlanations)

#### 6.1 Valores de Shapley (juego cooperativo)

Se define un juego donde el “valor” de un subconjunto $S$ de features es la expectativa del modelo cuando solo se conocen esas features (formalización vía integración sobre complemento con distribución de referencia). Los **valores de Shapley** $\phi_j$ reparten la predicción total sobre la predicción base cumpliendo **axiomas** (eficiencia, simetría, *dummy*, aditividad en modelos lineales locales).

#### 6.2 Explicación aditiva local

Para una instancia $\mathbf{x}$, SHAP construye una descomposición:

$$f(\mathbf{x}) \approx g(\mathbf{x}') = \phi_0 + \sum_{j=1}^p \phi_j$$

donde $\mathbf{x}'$ es una codificación de presencia/ausencia de features y $\phi_0$ es el valor base (expectativa de $f$).

#### 6.3 TreeSHAP

Para ensembles de árboles, **TreeSHAP** explota la estructura del árbol para calcular contribuciones **exactas** (bajo la definición de SHAP para árboles) con complejidad polinómica razonable en muchos casos — mucho más rápido que **KernelSHAP**, que muestrea coaliciones.

#### 6.4 KernelSHAP (modelo-agnóstico)

Aproxima Shapley muestreando subconjuntos de features y ajustando una regresión ponderada. **Costoso** en muchas features; sensible a la elección de **fondo** (*background dataset*).

---

### 7. LIME (Local Interpretable Model-agnostic Explanations)

#### 7.1 Idea

Alrededor de $\mathbf{x}$, se generan perturbaciones $\mathbf{x}'$, se obtienen predicciones $f(\mathbf{x}')$ y se ajusta un modelo **lineal simple** ponderado por proximidad a $\mathbf{x}$:

$$\xi(\mathbf{x}) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_{\mathbf{x}}) + \Omega(g)$$

con $\pi_{\mathbf{x}}$ kernel de proximidad y $\Omega$ penalización de complejidad.

#### 7.2 Riesgos

- **Inestabilidad:** pequeños cambios en $\mathbf{x}$ o en el muestreo alteran coeficientes locales.  
- **Vecindario:** si el radio es grande, la aproximación lineal es mala; si es pequeño, el ajuste es ruidoso.  
- **Datos tabulares vs texto/imagen:** la perturbación debe ser semánticamente plausible.

---

### 8. Comparación práctica de métodos

| Método | Alcance | Modelo | Coste típico | Correlación |
|--------|---------|--------|--------------|-------------|
| Coef. lineales | global (lineal) | lineal | bajo | problemática VIF |
| Permutación | global | cualquiera | medio–alto | inestable |
| PDP/ICE | global de efecto de $x_j$ | cualquiera | medio | extrapolación si correlación |
| TreeSHAP | local (+ agregación) | árboles/GBDT | medio | reparte entre colineales |
| KernelSHAP | local | cualquiera | alto | sensible a fondo |
| LIME | local | cualquiera | medio | muy sensible a vecindario |

---

### 9. Visualización

- **Summary plot SHAP** (*beeswarm*): cada punto es una instancia; color = valor de feature; eje = SHAP value.  
- **Dependence plot:** SHAP de $x_j$ vs $x_j$; revela no linealidades e **interacciones** (color con otra feature).  
- **Waterfall** / **force**: descomposición aditiva para una predicción concreta.

---

### 10. Plantillas Python

**Permutación:**

```python
from sklearn.inspection import permutation_importance

r = permutation_importance(
    model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
)
# r.importances_mean, r.importances_std
```

**SHAP con modelo de árboles:**

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)  # o shap_values(X_sample) en API reciente
shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns)
```

**LIME tabular:**

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=list(X_train.columns),
    class_names=["neg", "pos"],
    mode="classification",
)
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10,
)
```

---

### 11. Laboratorio (según sílabo)

- **NTB 1 —** Importancia global + SHAP summary.  
- **NTB 2 —** Explicaciones locales SHAP/LIME y sesgos.

---

### 12. Sesgos y malas interpretaciones

- **Correlación fuerte:** SHAP y permutación pueden **repartir** importancia entre variables redundantes de modo que ninguna “gana” claramente — no implica que sean irrelevantes.  
- **Datos de entrenamiento sesgados:** el modelo aprende el sesgo; una explicación fiel **describe** el sesgo, no lo corrige.  
- **Causalidad:** “$x_j$ tiene alto SHAP” no prueba que intervenir en $x_j$ en el mundo real cambie $y$ (confusores).  
- **Seguridad:** explicaciones pueden usarse para **ingeniería adversaria** del input.

---

### 13. Profundización: axiomas de Shapley (intuición)

- **Eficiencia:** las contribuciones suman la diferencia entre predicción y baseline.  
- **Simetría:** dos features que aportan igual en todas las coaliciones reciben el mismo $\phi$.  
- ***Dummy***: una feature que no cambia nunca la predicción recibe $\phi=0$.

Estas propiedades hacen atractivo el marco, pero **no** eliminan el problema de **definir el baseline** y la **distribución de fondo** — elecciones distintas → valores SHAP distintos.

---

### 14. Profundización: interacciones en SHAP

`shap.TreeExplainer` puede calcular **valores de interacción** para pares $(j,k)$, útiles cuando el efecto marginal de $x_j$ depende fuertemente de $x_k$ (p. ej. descuento solo relevante para clientes premium).

---

### 15. Profundización: explicaciones y cumplimiento

El art. 22 del GDPR y debates legales no otorgan un derecho técnico único a “código fuente del modelo”. Las organizaciones combinan **documentación**, **auditorías**, **pruebas de equidad** y **supervisión humana**. Las herramientas de XAI son **apoyo**, no sustituto de gobernanza.

---

### 16. PDP con sklearn (referencia)

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 1], kind="average"
)
```

Para ICE, `kind="individual"` o `kind="both"` según versión.

---

### 17. Errores frecuentes

| Error | Consecuencia |
|-------|--------------|
| Interpretar SHAP como causal | Decisiones de política incorrectas |
| Usar muestra de fondo de 3 filas | SHAP/KSHAP inestable |
| Ignorar correlación al leer PDP | Curvas en regiones sin datos |
| Mostrar LIME sin sensibilidad al vecindario | Explicaciones contradictorias entre runs |

---

### 18. Checklist antes de presentar explicaciones a negocio

1. ¿Se aclara que es **asociación**, no **causalidad**?  
2. ¿Baseline y datos de fondo documentados?  
3. ¿Se contrastó estabilidad (repetir con semillas / submuestras)?  
4. ¿Hay variables **protegidas** que no deberían usarse pero aparecen por proxy?  
5. ¿La métrica de negocio alineada con la función objetivo explicada?  

---

### 19. Lecturas complementarias orientadas a práctica

- Capítulos sobre **interpretable models** vs **post-hoc** en Molnar.  
- Guías de `shap` para evitar API obsoletas: en versiones recientes, `Explainer(model)` unifica *TreeExplainer* cuando es posible.

---

### 20. Relación con fairness

Métricas de equidad (paridad demográfica, igualdad de oportunidades) son **ortogonales** a la interpretabilidad: un modelo puede ser “explicable” y **injusto**. La auditoría requiere **subgrupos** explícitos y tests estadísticos o bayesianos, no solo gráficos globales.

---

### 21. Escalera de causalidad (Pearl) y XAI

Los niveles **asociación** → **intervención** → **contrafactuales** exigen supuestos crecientes. SHAP y LIME operan típicamente en el nivel **asociativo** respecto al modelo entrenado: describen dependencia de $f$ sobre entradas, no el efecto de una **política** en el mundo. Para decisiones de alto impacto (crédito, medicina), la interpretación ML debe integrarse con **diseño causal** cuando sea posible.

---

### 22. Elección del conjunto de fondo en SHAP

Los valores SHAP dependen de la **expectativa condicional** respecto a una distribución de referencia (muestras de fondo). Opciones:

- **Muestra aleatoria** del train: simple pero puede incluir puntos poco representativos.  
- **k-medias** sobre train para prototipos: reduce tamaño y coste.  
- **Valor esperado del modelo** en train como $\phi_0$.

Comparar explicaciones con **fondos distintos** para sensibilidad; si cambian radicalmente, la narrativa para negocio debe ser cautelosa.

---

### 23. Explicaciones contrastivas

Además de “por qué $\hat{y}$”, a veces se pregunta “**por qué no** la clase alternativa”. Algunas implementaciones construyen contrastes variando el mínimo número de features para cruzar un umbral de probabilidad. Útil en servicio al cliente y auditoría, pero el espacio de contrastes puede ser **enorme**.

---

### 24. Estabilidad: bootstrap de explicaciones

Para un punto $\mathbf{x}$, repetir LIME o KernelSHAP con semillas distintas y medir varianza de coeficientes locales. **Alta varianza** sugiere que la explicación no es confiable para comunicación externa — aunque la predicción de $f$ sea estable.

---

### 25. Cuándo un modelo “interpretable” sigue siendo inadecuado

Incluso con coeficientes visibles, **interacciones no modeladas** (en un modelo lineal mal especificado) o **variables proxy** de atributos sensibles pueden producir decisiones éticamente inaceptables. La interpretabilidad **no sustituye** gobernanza de datos ni revisiones de atributos prohibidos.

---

### 26. Integración con producto: qué mostrar al usuario final

Reguladores y UX difieren: a veces basta un **motivo corto** (“precio competitivo”, “compras similares”); otras se requiere trazabilidad técnica completa para auditores. Diseñar **dos capas** de explicación — resumen humano + detalle técnico — reduce fricción.

---

### 27. ALE (*Accumulated Local Effects*) — motivación

Cuando PDP mezcla regiones del espacio de covariables incompatibles por correlación, **ALE** estima el efecto local de $x_j$ promediando **diferencias** de predicción al mover $x_j$ en pequeños pasos, manteniendo el resto de $\mathbf{x}$ fijo en cada observación — evitando algunas extrapolaciones de PDP. Implementaciones en librerías especializadas; conceptualmente es el refinamiento más honesto para efectos **marginales** en presencia de dependencia fuerte entre features.

---

### 28. Sobrecarga del término “explicación”

En literatura y producto se mezclan: (a) **atribución** de contribución a entradas, (b) **contraste** con casos cercanos, (c) **reglas** aproximadas, (d) **causa** en sentido intervencionista. Al leer papers o documentación comercial, identificar cuál de los cuatro se ofrece evita malentendidos con compliance o con equipos de ciencia causal.

---

## Referencias bibliográficas principales

1. Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). christophm.github.io/interpretable-ml-book/  
2. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.  
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?” Explaining the predictions of any classifier. *KDD*.  
4. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics* (incluye marco de dependencia parcial en contexto de boosting).  
5. Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. *Knowledge and Information Systems*, 41(3), 647–665.  
