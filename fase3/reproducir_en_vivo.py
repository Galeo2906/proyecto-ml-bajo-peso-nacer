# %% [markdown]
# # Reproducción rápida en vivo (≈ 1 minuto)
# **Proyecto Integrador · Machine Learning I · Daniel Felipe Galeano Peña**
#
# Reproduce el **modelo final** (logística sin penalizar) y sus resultados en la prueba **sin** la validación cruzada anidada de los 9 modelos (esa está en `fase3_final.ipynb` y tarda 15–40 minutos).
# Usa exactamente el mismo código, la misma semilla (42) y la misma partición, así que las cifras deben coincidir con el informe: PR-AUC 0,266 · ROC-AUC 0,663 · recall 70,2 % · precisión 14,1 % · 51,7 % marcados · quintil superior 20,3 % de bajo peso y 39 % de los casos.
#
# En Colab: subir este cuaderno y `nacimientos_dane_2022_muestra.csv`, y ejecutar todo.

# %%
import os, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, brier_score_loss, confusion_matrix
warnings.filterwarnings("ignore")
SEED = 42

# %% [markdown]
# ## 1. Datos: códigos del DANE → categorías (regla fija) y variable objetivo

# %%
ruta = "data/nacimientos_dane_2022_muestra.csv"
if not os.path.exists(ruta):
    ruta = "nacimientos_dane_2022_muestra.csv"          # por si se sube suelto (Colab)
d = pd.read_csv(ruta, dtype=str)

REGION = {"11": "Bogota"}
REGION.update({c: "Caribe" for c in ["08", "13", "20", "23", "44", "47", "70", "88"]})
REGION.update({c: "Pacifica" for c in ["19", "27", "52", "76"]})
REGION.update({c: "Andina" for c in ["05", "15", "17", "25", "41", "54", "63", "66", "68", "73"]})
REGION.update({c: "Orinoquia_Amazonia" for c in ["18", "50", "81", "85", "86", "91", "94", "95", "97", "99"]})

def a_entero(s):
    return pd.to_numeric(s, errors="coerce")

def construir_X(d):
    nc, hv, ne = a_entero(d.NUMCONSUL), a_entero(d.N_HIJOSV), a_entero(d.N_EMB)
    hv_ok, ne_ok = hv.between(1, 30), ne.between(1, 30)
    return pd.DataFrame({
        "edad_madre": d.EDAD_MADRE.map({"1": "10-14", "2": "15-19", "3": "20-24", "4": "25-29", "5": "30-34",
                                        "6": "35-39", "7": "40-44", "8": "45+", "9": "45+"}).fillna("sin_info"),
        "educacion": d.NIV_EDUM.map({"13": "ninguno", "01": "primaria_o_menos", "02": "primaria_o_menos",
                                     "03": "secundaria", "04": "media", "05": "media", "06": "media",
                                     "07": "tecnica_tecnologica", "08": "tecnica_tecnologica",
                                     "09": "profesional_posgrado", "10": "profesional_posgrado",
                                     "11": "profesional_posgrado", "12": "profesional_posgrado"}).fillna("sin_info"),
        "zona": d.AREA_RES.map({"1": "cabecera", "2": "centro_poblado", "3": "rural_disperso"}).fillna("sin_info"),
        "regimen": d.SEG_SOCIAL.map({"1": "contributivo", "2": "subsidiado", "3": "excepcion_especial",
                                     "4": "excepcion_especial", "5": "no_asegurado"}).fillna("sin_info"),
        "estado_conyugal": d.EST_CIVM.map({"1": "union_libre_2mas", "2": "union_libre_menos2", "3": "separada_viuda",
                                           "4": "separada_viuda", "5": "soltera", "6": "casada"}).fillna("sin_info"),
        "etnia": d.IDPERTET.map({"1": "indigena", "5": "afrodescendiente", "6": "ninguna"}).fillna("otra_o_sin_info"),
        "region": d.CODPTORE.map(REGION).fillna("sin_info"),
        "embarazo_multiple": d.MUL_PARTO.map({"1": "simple", "2": "multiple", "3": "multiple",
                                              "4": "multiple"}).fillna("sin_info"),
        "paridad": np.select([~hv_ok, hv == 1, hv == 2], ["sin_info", "primer_hijo", "segundo"], "tercero_o_mas"),
        "perdidas_previas": np.select([~(hv_ok & ne_ok), ne > hv], ["sin_info", "si"], "no"),
        "controles": np.select([~nc.between(0, 60), nc < 4, nc <= 6], ["sin_info", "menos_4", "4_a_6"], "7_o_mas"),
    })

X = construir_X(d)
y = d.PESO_NAC.isin(list("1234")).astype(int).values
validos = d.PESO_NAC != "9"                         # peso sin información: no se puede etiquetar
X, y = X[validos.values].reset_index(drop=True), y[validos.values]
print(f"Registros con peso conocido: {len(y):,} | BPN: {y.sum():,} ({y.mean():.2%}) | Archivo completo DANE 2022: 10,48 % (567.766 con peso conocido)")


# %% [markdown]
# ## 2. Partición 80/20 estratificada y modelo final (un solo Pipeline)

# %%
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
modelo = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                   ("escala", StandardScaler()),
                   ("modelo", LogisticRegression(penalty=None, max_iter=5000))])
print(f"Entrenamiento: {len(ytr):,} | Prueba: {len(yte):,}")

# %% [markdown]
# ## 3. Umbral F2 con predicciones fuera de muestra del entrenamiento (nunca la prueba)

# %%
def umbral_f2(yv, s):
    p, r, u = precision_recall_curve(yv, s)
    f2 = 5 * p[:-1] * r[:-1] / np.maximum(4 * p[:-1] + r[:-1], 1e-12)
    return u[int(np.argmax(f2))]

oof = cross_val_predict(modelo, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
UMBRAL = umbral_f2(ytr, oof)
modelo.fit(Xtr, ytr)
print(f"Umbral F2: {UMBRAL:.4f}")

# %% [markdown]
# ## 4. Evaluación única en la prueba

# %%
s = modelo.predict_proba(Xte)[:, 1]
pred = (s >= UMBRAL).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
print(f"PR-AUC: {average_precision_score(yte, s):.4f}  (azar: {yte.mean():.4f})")
print(f"ROC-AUC: {roc_auc_score(yte, s):.4f}")
print(f"Brier: {brier_score_loss(yte, s):.4f}")
print(f"Recall: {tp/(tp+fn):.1%} ({tp} de {tp+fn}) | Precisión: {tp/(tp+fp):.1%} | Marcados en riesgo: {pred.mean():.1%}")

# %% [markdown]
# ## 5. Uso para focalizar: quintiles de riesgo

# %%
q = pd.qcut(pd.Series(s).rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
t = pd.DataFrame({"q": q, "y": yte}).groupby("q", observed=True).y.agg(["count", "sum", "mean"])
t["% de todos los casos"] = 100 * t["sum"] / t["sum"].sum()
t.columns = ["nacimientos", "casos", "tasa de bajo peso", "% de todos los casos"]
print(t.round(3).to_string())
