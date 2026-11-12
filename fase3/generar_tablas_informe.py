"""Convierte las tablas de resultados/ (salidas del cuaderno) en tablas LaTeX para el informe.
Así las cifras del informe son exactamente las que produce el código (corrección de la segunda entrega)."""
import os
import pandas as pd

RES, OUT = "resultados/", "informe/tablas/"
os.makedirs(OUT, exist_ok=True)

def m(n):
    """Entero con punto de miles."""
    return f"{int(n):,}".replace(",", ".")

def c(x, d=3):
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")

# Tabla: comparación en validación cruzada
t = pd.read_csv(RES + "tabla2_comparacion_cv.csv").sort_values("PR_AUC_media", ascending=False)
def hp(s):
    if s == "-":
        return "--"
    k, v = s.strip("{}").split(":")
    nombre = {"'modelo__n_neighbors'": "$k$", "'modelo__alpha'": r"$\alpha$", "'modelo__C'": "$C$",
              "'modelo__reg_param'": "reg"}[k.strip()]
    return f"{nombre} = {v.strip().replace('.', ',')}"
filas = [f"{r.modelo} & {c(r.PR_AUC_media)} $\\pm$ {c(r.PR_AUC_de)} & {c(r.ROC_AUC_media)} $\\pm$ {c(r.ROC_AUC_de)} & "
         f"{c(r.Brier_media, 4)} & {hp(r['hiperparámetro más elegido'])} \\\\" for _, r in t.iterrows()]
open(OUT + "tab_cv.tex", "w").write("\n".join(filas).replace("α=0,5", r"$\alpha$ = 0,5"))

# Tabla: prueba final
p = pd.read_csv(RES + "tabla3_prueba_final.csv").set_index("métrica")
def ic(m):
    return f"{c(p.loc[m,'valor'])} ({c(p.loc[m,'IC95 inf'])}--{c(p.loc[m,'IC95 sup'])})"
mc = pd.read_csv(RES + "matriz_confusion.csv", index_col=0)
tn, fp = mc.iloc[0]; fn, tp = mc.iloc[1]
filas = [
    f"PR-AUC (IC 95 \\%) & {ic('PR-AUC')} & azar: {c(p.loc['PR-AUC del azar (prevalencia)','valor'])} \\\\",
    f"ROC-AUC (IC 95 \\%) & {ic('ROC-AUC')} & azar: 0,500 \\\\",
    f"Brier & {c(p.loc['Brier','valor'], 4)} & línea base: {c(p.loc['Brier de la línea base','valor'], 4)} \\\\",
    r"\midrule",
    f"Umbral F2 (elegido fuera de muestra) & {c(p.loc['Umbral (F2, fuera de muestra)','valor'])} & \\\\",
    f"Recall (casos detectados) & {c(100*p.loc['Recall (sensibilidad)','valor'],1)} \\% & {m(tp)} de {m(tp+fn)} \\\\",
    f"Precisión (VPP) & {c(100*p.loc['Precisión (VPP)','valor'],1)} \\% & azar: {c(100*p.loc['PR-AUC del azar (prevalencia)','valor'],1)} \\% \\\\",
    f"Especificidad & {c(100*p.loc['Especificidad','valor'],1)} \\% & {m(tn)} de {m(tn+fp)} \\\\",
    f"F2 & {c(p.loc['F2','valor'])} & \\\\",
    f"Nacimientos marcados en riesgo & {c(100*p.loc['% de nacimientos marcados en riesgo','valor'],1)} \\% & {m(tp+fp)} de {m(tn+fp+fn+tp)} \\\\",
]
open(OUT + "tab_prueba.tex", "w").write("\n".join(filas))

# Tabla: odds ratios (todas las significativas + controles) y contraste con el artículo
o = pd.read_csv(RES + "tabla6_odds_ratios.csv", index_col=0)
NOM = {
    "embarazo_multiple_multiple": "Embarazo múltiple (vs. simple)",
    "controles_menos_4": "Menos de 4 controles prenatales (vs. 7 o más)",
    "controles_4_a_6": "4 a 6 controles prenatales (vs. 7 o más)",
    "edad_madre_40-44": "Madre de 40--44 años (vs. 20--24)",
    "region_Bogota": "Residencia en Bogotá (vs. región Andina)",
    "edad_madre_10-14": "Madre de 10--14 años (vs. 20--24)",
    "edad_madre_35-39": "Madre de 35--39 años (vs. 20--24)",
    "perdidas_previas_si": "Embarazos previos sin nacido vivo",
    "edad_madre_30-34": "Madre de 30--34 años (vs. 20--24)",
    "estado_conyugal_soltera": "Madre soltera (vs. unión libre $\\geq$ 2 años)",
    "region_Caribe": "Región Caribe (vs. Andina)",
    "zona_centro_poblado": "Centro poblado (vs. cabecera)",
    "educacion_profesional_posgrado": "Educación profesional o posgrado (vs. media)",
    "paridad_segundo": "Segundo hijo (vs. primer hijo)",
    "regimen_no_asegurado": "Madre no asegurada (vs. subsidiado)",
    "region_Orinoquia_Amazonia": "Orinoquía-Amazonía (vs. Andina)",
    "paridad_tercero_o_mas": "Tercer hijo o más (vs. primer hijo)",
}
sig = o[o.p < 0.05].sort_values("OR", ascending=False)
faltan = [i for i in sig.index if i not in NOM]
assert not faltan, f"Falta nombre para {faltan}"
art = {"controles_menos_4": "2,8 (1,12--6,82)"}
filas = []
for i, r in sig.iterrows():
    pv = "$<$0,001" if r.p < 0.001 else c(r.p)
    filas.append(f"{NOM[i]} & {c(r.OR,2)} & {c(r['IC95 inf'],2)}--{c(r['IC95 sup'],2)} & {pv} & {art.get(i, '--')} \\\\")
open(OUT + "tab_or.tex", "w").write("\n".join(filas))

# Datos sueltos que cita el texto (para verificar contra el informe)
q = pd.read_csv(RES + "tabla4_quintiles_riesgo.csv", index_col=0)
s = pd.read_csv(RES + "tabla8_sensibilidad.csv")
e = pd.read_csv(RES + "tabla5_perfil_errores.csv", index_col=0)
v = pd.read_csv(RES + "verificacion_L1_L2.csv")
print("Quintiles:\n", q.to_string())
print("Sensibilidad:\n", s.to_string())
print("Perfil errores:\n", e.T.to_string())
print("Verificación L1/L2:\n", v.to_string())
print("OR significativos:", len(sig))
