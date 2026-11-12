"""Prepara la muestra de trabajo a partir de los microdatos públicos del DANE (Estadísticas Vitales, nacimientos 2022).
Uso:  python 00_preparar_muestra.py  ruta/a/nac2022.csv
Genera data/nacimientos_dane_2022_muestra.csv (50.000 nacimientos al azar, semilla 42; solo columnas usadas)."""
import sys
import pandas as pd

ruta = sys.argv[1] if len(sys.argv) > 1 else "nac2022.csv"
COLS = ["PESO_NAC", "EDAD_MADRE", "NIV_EDUM", "AREA_RES", "SEG_SOCIAL", "NUMCONSUL", "N_HIJOSV", "N_EMB",
        "EST_CIVM", "MUL_PARTO", "IDPERTET", "CODPTORE"]
d = pd.read_csv(ruta, dtype=str, encoding="latin-1", usecols=COLS)
print("Nacimientos en el archivo:", len(d))
conocido = d[d.PESO_NAC != "9"]
print(f"Con peso conocido: {len(conocido):,} | % con bajo peso (< 2.500 g): {conocido.PESO_NAC.isin(list('1234')).mean():.2%}")
m = d.sample(n=50_000, random_state=42).reset_index(drop=True)
m.to_csv("data/nacimientos_dane_2022_muestra.csv", index=False)
print("Muestra guardada:", m.shape)
