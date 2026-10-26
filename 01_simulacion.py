"""
01_simulacion.py
Simulación de una cohorte de gestantes colombianas para predecir bajo peso al nacer (BPN < 2500 g).

Artículo base: Feyisa et al. (2023), Front. Public Health 11:1213291. Es un estudio de casos y
controles (81 casos / 243 controles), así que su prevalencia de BPN (25 %) la fija el diseño.
Aquí se simula una COHORTE con tres tipos de parámetros (ver tabla de fuentes del informe):

  A. DEL ARTÍCULO: proporción de cada exposición (controles, su Tabla 2) y efectos de las cuatro
     exposiciones significativas (AOR, su Tabla 3).
  B. DE FUENTES COLOMBIANAS: prevalencia de BPN, edad, nivel educativo y controles prenatales de la
     madre (DANE, Estadísticas Vitales 2021), zona rural (Censo 2018) y régimen subsidiado
     (Boletín 17 del Externado, oct.-dic. 2025).
  C. SUPUESTOS DEL EQUIPO: medias/desviaciones de variables continuas sin cifra publicada, correlaciones
     entre exposiciones (vía una desventaja latente), efectos leves de las variables no significativas,
     faltantes y errores de digitación.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import norm

SEED = 42
N = 6000
PREV_OBJETIVO = 0.097          # DANE EEVV 2021: 9.7 % (rango 2018-2022: 9.1 % a 10.4 %)
P_RURAL = 0.229                # Censo 2018: 22.9 % de la población en centros poblados y rural disperso
P_SUBSIDIADO = 0.525           # Boletín 17 (Externado), oct.-dic. 2025: 52.5 % de afiliados

# DANE EEVV 2021, distribución de nacimientos según la madre
EDAD_GRUPOS = [(12, 19, 0.183), (20, 24, 0.284), (25, 29, 0.250), (30, 34, 0.168), (35, 39, 0.089), (40, 45, 0.026)]
EDUC_NIVELES = ["ninguna", "primaria", "secundaria", "superior"]
EDUC_BRUTO = np.array([1.1, 12.1, 22.3 + 34.2 + 2.7, 23.9])   # superior = técnica + tecnológica + universitaria
EDUC_P = EDUC_BRUTO / EDUC_BRUTO.sum()                          # se excluye "sin información"
ANC_GRUPOS = [(0, 0, 5.3), (1, 3, 13.0), (4, 7, 54.1), (8, 12, 25.5), (13, 15, 1.7)]   # % (0.4 % sin dato excluido)

OUT = Path(__file__).parent
(OUT / "data").mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- utilidades
z = rng.standard_normal(N)  # desventaja socioeconómica latente (induce correlación entre exposiciones)


def std(x):
    return (x - x.mean()) / x.std()


def mix(rho):
    """Variable latente con correlación rho con la desventaja z."""
    return rho * z + np.sqrt(1 - rho**2) * rng.standard_normal(N)


def top(latente, p):
    """Indicador del p superior de la latente (prevalencia exacta p)."""
    return latente > np.quantile(latente, 1 - p)


def grupo_por_cuantil(latente, probs):
    """Índice de grupo (0..k-1) según el cuantil de la latente; los grupos tienen proporciones probs."""
    cortes = np.quantile(latente, np.cumsum(np.asarray(probs) / np.sum(probs))[:-1])
    return np.searchsorted(cortes, latente)


def entero_en_grupo(idx, rangos):
    lo = np.array([r[0] for r in rangos])[idx]
    hi = np.array([r[1] for r in rangos])[idx]
    return lo + (rng.random(len(idx)) * (hi - lo + 1)).astype(int)


# ------------------------------------------------------- variables de exposición
# Estrés percibido (PSS-10, 0-40). Distribución Beta escalada a [0, 40]: acotada y asimétrica, calibrada
# para reproducir las categorías del artículo (controles): bajo 45.3 %, moderado 39.9 %, alto 14.8 %.
def _ajuste_beta(par):
    a, b = np.exp(par)
    return [beta_dist.cdf(13.5 / 40, a, b) - 0.453, beta_dist.cdf(26.5 / 40, a, b) - 0.852]


A_PSS, B_PSS = np.exp(fsolve(_ajuste_beta, [np.log(1.2), np.log(2.0)]))
l_s = std(mix(0.35))
pss = np.clip(np.round(40 * beta_dist.ppf(norm.cdf(l_s), A_PSS, B_PSS)), 0, 40)

# Perímetro braquial (MUAC, cm). Controles: 26.3 % < 23 cm  (media/DE: supuesto del equipo)
l_m = std(mix(-0.30))
muac = np.round(24.6 + 2.5 * l_m, 1)

# Controles prenatales: distribución colombiana (DANE 2021); <4 controles = 18.3 %
l_a = std(mix(-0.40))
anc = entero_en_grupo(grupo_por_cuantil(l_a, [g[2] for g in ANC_GRUPOS]), [(g[0], g[1]) for g in ANC_GRUPOS]).astype(float)

# Consejería nutricional (60.1 % sí en controles); más probable con más controles prenatales
l_c = std(mix(-0.30) + 0.5 * l_a)
consejeria = top(l_c, 0.601)

# Hierro: 91.4 % toma; tabletas recibidas (< 100 en ~45 %)
toma_fe = top(std(mix(-0.20)), 0.914)
tabs = np.where(toma_fe, np.clip(np.round(rng.normal(110 + 15 * l_a, 50)), 1, 300), 0)

# Comidas adicionales (56.4 % sí)
comidas = top(std(mix(-0.25)), 0.564)

# Hemoglobina (g/dL); anemia = Hb < 11 (≈ 7 % en controles del artículo)
l_h = std(mix(-0.20) + 0.25 * l_m)
hb = np.round(12.48 + 1.0 * l_h, 1)

# Ganancia de peso gestacional (kg); < 12 kg en ~39.5 %
gwg = np.round(13.0 + 3.5 * std(0.3 * l_m + 0.2 * l_a + rng.standard_normal(N)), 1)

# Hipertensión inducida por el embarazo (5.8 %) y otros problemas de salud (13.2 %)
hta = top(std(mix(0.10)), 0.058)
otro_prob = top(std(mix(0.15)), 0.132)

# Embarazo planeado (76.5 %)
planeado = top(std(mix(-0.20)), 0.765)

# Edad materna: distribución por grupos de edad colombiana (DANE 2021); mayor desventaja -> más jóvenes
edad = entero_en_grupo(grupo_por_cuantil(mix(-0.15), [g[2] for g in EDAD_GRUPOS]), [(g[0], g[1]) for g in EDAD_GRUPOS]).astype(float)

# Variables socio-demográficas (efecto solo indirecto, vía z)
rural = top(mix(0.35), P_RURAL)
subsidiado = top(mix(0.50), P_SUBSIDIADO)
l_e = mix(-0.50)
educ = np.array(EDUC_NIVELES, dtype=object)[grupo_por_cuantil(l_e, EDUC_P)]

# ---------------------------------------------------------------- variable objetivo
# Efectos generadores (OR). Significativos del artículo (AOR): sin consejería 2.4, MUAC<23 3.1,
# estrés alto 2.8, <4 controles 2.8. Efectos leves (supuesto) para los no significativos del artículo
# (anemia, hierro<100, HTA, ganancia<12 kg) y para edades extremas (<20 o >=35).
# Educación, zona, régimen, embarazo planeado, comidas extra y otros problemas: sin efecto directo.
flags = {
    "sin_consejeria": (~consejeria, 2.4),
    "muac_lt23": (muac < 23, 3.1),
    "estres_alto": (pss >= 27, 2.8),
    "anc_lt4": (anc < 4, 2.8),
    "anemia": (hb < 11, 1.5),
    "hierro_lt100": (tabs < 100, 1.3),
    "hta": (hta, 1.2),
    "gwg_lt12": (gwg < 12, 1.3),
    "edad_extrema": ((edad < 20) | (edad >= 35), 1.3),
}
xb = sum(np.log(orv) * f.astype(float) for f, orv in flags.values())
b0 = brentq(lambda b: expit(b + xb).mean() - PREV_OBJETIVO, -10, 2)
p_bpn = expit(b0 + xb)
bpn = rng.binomial(1, p_bpn)

# ---------------------------------------------------------------- ensamblar
yn = lambda b: np.where(b, "si", "no").astype(object)
df = pd.DataFrame(
    {
        "edad": edad,
        "nivel_educativo": educ.astype(object),
        "zona_rural": yn(rural),
        "regimen_subsidiado": yn(subsidiado),
        "embarazo_planeado": yn(planeado),
        "consejeria_nutricional": yn(consejeria),
        "comidas_adicionales": yn(comidas),
        "hta_embarazo": yn(hta),
        "otro_problema_salud": yn(otro_prob),
        "muac_cm": muac,
        "puntaje_pss": pss,
        "n_controles_prenatales": anc,
        "n_tabletas_hierro": tabs,
        "hemoglobina_gdl": hb,
        "ganancia_peso_kg": gwg,
        "bpn": bpn,
    }
)

# Copia limpia (antes de inyectar problemas) para verificar la simulación y dibujar distribuciones
limpio = df.copy()
limpio.to_csv(OUT / "data" / "gestantes_simulado_limpio.csv", index=False)

# ---------------------------------------------------------------- problemas de calidad
# (1) Valores faltantes (dependen de covariables observadas, no del desenlace -> MAR)
def poner_nan(col, p):
    m = rng.random(N) < p
    df.loc[m, col] = np.nan


p_pss = 0.04 + 0.06 * rural
df.loc[rng.random(N) < p_pss, "puntaje_pss"] = np.nan
p_gwg = 0.06 + 0.10 * (anc < 4)
df.loc[rng.random(N) < p_gwg, "ganancia_peso_kg"] = np.nan
poner_nan("hemoglobina_gdl", 0.08)
poner_nan("muac_cm", 0.03)
poner_nan("n_tabletas_hierro", 0.07)
poner_nan("nivel_educativo", 0.02)
poner_nan("consejeria_nutricional", 0.03)
poner_nan("comidas_adicionales", 0.02)

# (2) Errores de digitación (valores fuera de rango clínico)
idx = rng.choice(N, 12, replace=False)
df.loc[idx[:6], "muac_cm"] = df.loc[idx[:6], "muac_cm"] * 10  # registrado en mm
df.loc[idx[6:9], "hemoglobina_gdl"] = 99.0
df.loc[idx[9:12], "n_controles_prenatales"] = -1

df.to_csv(OUT / "data" / "gestantes_simulado.csv", index=False)
# Probabilidad verdadera de BPN (solo existe en simulación; NO se usa para entrenar): da el "techo".
pd.DataFrame({"p_verdadera": p_bpn}).to_csv(OUT / "data" / "prob_verdadera_simulacion.csv", index=False)

# ---------------------------------------------------------------- verificación vs. fuentes
ART, DANE, CENSO, EXT = "Artículo (controles)", "DANE EEVV 2021", "Censo 2018", "Boletín 17 Externado"
ref = [
    ("sin consejería nutricional", flags["sin_consejeria"][0].mean(), 0.399, ART),
    ("MUAC < 23 cm", flags["muac_lt23"][0].mean(), 0.263, ART),
    ("estrés alto (PSS >= 27)", flags["estres_alto"][0].mean(), 0.148, ART),
    ("estrés moderado (14-26)", ((pss >= 14) & (pss < 27)).mean(), 0.399, ART),
    ("< 4 controles prenatales", flags["anc_lt4"][0].mean(), (5.3 + 13.0) / 99.6, DANE),
    ("anemia (Hb < 11)", flags["anemia"][0].mean(), 0.07, ART),
    ("hierro < 100 tabletas", flags["hierro_lt100"][0].mean(), 0.455, ART),
    ("HTA del embarazo", hta.mean(), 0.058, ART),
    ("ganancia de peso < 12 kg", flags["gwg_lt12"][0].mean(), 0.395, ART),
    ("comidas adicionales", comidas.mean(), 0.564, ART),
    ("embarazo planeado", planeado.mean(), 0.765, ART),
    ("edad < 20 años", (edad < 20).mean(), 0.183, DANE),
    ("edad >= 35 años", (edad >= 35).mean(), 0.115, DANE),
    ("educación: ninguna", (educ == "ninguna").mean(), EDUC_P[0], DANE),
    ("educación: primaria", (educ == "primaria").mean(), EDUC_P[1], DANE),
    ("educación: secundaria o media", (educ == "secundaria").mean(), EDUC_P[2], DANE),
    ("educación: superior", (educ == "superior").mean(), EDUC_P[3], DANE),
    ("zona rural", rural.mean(), P_RURAL, CENSO),
    ("régimen subsidiado", subsidiado.mean(), P_SUBSIDIADO, EXT),
    ("bajo peso al nacer", bpn.mean(), PREV_OBJETIVO, DANE),
]
ver = pd.DataFrame([(k, round(float(s), 3), round(float(r), 3), f) for k, s, r, f in ref],
                   columns=["variable", "simulado", "referencia", "fuente"])
ver.to_csv(OUT / "data" / "verificacion_proporciones.csv", index=False)

# Tasa de BPN por exposición (cohorte completa, antes de introducir errores)
tasas = []
for nombre in ["sin_consejeria", "muac_lt23", "estres_alto", "anc_lt4", "anemia"]:
    f = flags[nombre][0]
    tasas.append((nombre, round(bpn[f].mean(), 3), round(bpn[~f].mean(), 3)))
tasas = pd.DataFrame(tasas, columns=["exposicion", "tasa_bpn_expuestas", "tasa_bpn_no_expuestas"])
tasas.to_csv(OUT / "data" / "tasas_bpn_por_exposicion.csv", index=False)

meta = {
    "semilla": SEED,
    "n": N,
    "prevalencia_bpn": float(bpn.mean()),
    "n_bpn": int(bpn.sum()),
    "intercepto": float(b0),
    "or_generadores": {k: v[1] for k, v in flags.items()},
    "beta_pss": [float(A_PSS), float(B_PSS)],
}
(OUT / "data" / "metadatos_simulacion.json").write_text(json.dumps(meta, indent=2))

print(f"N={N}  BPN={bpn.sum()} ({bpn.mean():.3%})  intercepto={b0:.3f}  Beta PSS=({A_PSS:.3f}, {B_PSS:.3f})")
print("\nProporciones simuladas vs. referencia")
print(ver.to_string(index=False))
print("\nTasa de BPN según exposición")
print(tasas.to_string(index=False))
print("\nPSS: % en 0 =", round((pss == 0).mean() * 100, 1), "| % en 40 =", round((pss == 40).mean() * 100, 1),
      "| media =", round(pss.mean(), 1), "| DE =", round(pss.std(), 1))
print("\nFaltantes (%):")
print((df.isna().mean() * 100).round(1)[lambda s: s > 0].to_string())
