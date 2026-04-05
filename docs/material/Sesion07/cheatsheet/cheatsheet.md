---
layout: default
---

# Cheatsheet: Gradient Boosting
**Autor:** Carlos César Sánchez Coronel  

[⬅️ Volver a la Sesión-07](../../../sesiones/sesion-07.md)

---



## 1. Introducción
- **Tipo**: Supervisado (clasificación/regresión).
- **Objetivo**: Ensemble secuencial de árboles corrigiendo residuos. Escalable, robusto, preciso.

## 2. Evolución rápida
- **2001** GBM (Friedman) → **2014** XGBoost (regularización, paralelo) → **2016** LightGBM (histogramas, leaf-wise) → **2017** CatBoost (categóricas nativas).

## 3. Algoritmo base
1. Inicializar $F_0(x)$ constante (media o log-odds).
2. Para $m=1..M$:
   - Residuos: $r_{im} = -\frac{\partial L}{\partial F}\big|_{F=F_{m-1}}$  
   - Ajustar árbol $h_m$ a $(x_i, r_{im})$
   - $F_m = F_{m-1} + \eta \cdot \gamma_m h_m$

**Pérdida común (clasificación)**:  
$L(y,\hat{y}) = -[y\log\hat{y}+(1-y)\log(1-\hat{y})]$, $\hat{y}=\sigma(F)$

## 4. Cuándo usarlos
- Datos tabulares >100k filas, mix numérico/categórico.
- Necesitas velocidad (LightGBM), precisión con categóricas (CatBoost), o madurez (XGBoost).

**Trade-offs**:
| | XGBoost | LightGBM | CatBoost |
|-|---------|----------|----------|
| Velocidad | Media | **Muy alta** | Media |
| Memoria | Media | **Baja** | Media-alta |
| Categóricas | manual | manual | **nativo** |

## 5. Plantilla mínima (LightGBM binario)
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.05,'verbose':-1}
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data],
                  callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
print(f"AUC: {roc_auc_score(y_test, y_pred):.4f} | LogLoss: {log_loss(y_test, y_pred):.4f}")
```

## 6. Métricas clave
- **AUC** = P(positivo aleatorio > negativo aleatorio). Rango [0.5,1]. >0.75 aceptable.
- **LogLoss** = $-\frac{1}{N}\sum [y\log\hat{y}+(1-y)\log(1-\hat{y})]$. Mínimo 0.
- **RMSE** = $\sqrt{\frac{1}{N}\sum (y_i-\hat{y}_i)^2}$ (regresión).

## 7. Interpretabilidad
- Importancia de features (`feature_importances_`).
- **SHAP**: $f(x)=\phi_0+\sum \phi_j(x)$ (contribución de cada variable).
- LIME, PDP.

## 8. Fórmula clave (gradiente de log-loss)
$$\frac{\partial L}{\partial F} = \hat{y} - y \quad\Rightarrow\quad \text{pseudo-residuo } r_i = y_i - \hat{y}_i$$

**XGBoost usa Newton**: $w_j = -\frac{\sum g_i}{\sum h_i + \lambda}$ con $g_i=\hat{y}_i-y_i$, $h_i=\hat{y}_i(1-\hat{y}_i)$.