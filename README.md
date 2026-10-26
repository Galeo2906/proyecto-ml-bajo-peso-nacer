# Proyecto Integrador – Fase 2: Predicción de bajo peso al nacer

Machine Learning I · Universidad Externado de Colombia · Docente: Diana Pulido
Autor: Daniel Felipe Galeano Peña

Artículo base: Feyisa et al. (2023), *Nutrition, stress, and healthcare use during pregnancy are associated with low birth weight: evidence from a case–control study in West Ethiopia*, Frontiers in Public Health, 11, 1213291.

## Contenido

| Archivo / carpeta | Descripción |
|---|---|
| `fase2_notebook.ipynb` | Notebook principal (Python + scikit-learn): flujo sin fuga, comparación de modelos, validación e interpretación. |
| `01_simulacion.py` | Genera los datos simulados (6 000 gestantes, semilla 42). |
| `data/` | Datos simulados y archivos de verificación de la simulación. |
| `resultados/` | Tablas y figuras producidas por el notebook. |
| `requirements.txt` | Dependencias. |

## Cómo reproducir

1. Instalar dependencias: `pip install -r requirements.txt` (se requiere scikit-learn ≥ 1.8).
2. (Opcional) Regenerar los datos: `python 01_simulacion.py`.
3. Ejecutar `fase2_notebook.ipynb` completo (`Restart & Run All`). Toda la aleatoriedad usa la semilla 42.

El notebook lee los datos de la carpeta `data/`; debe ejecutarse desde la carpeta que contiene `data/`.

## Nota sobre los datos

Los datos son **simulados**: combinan parámetros del artículo (proporciones y *odds ratios*), fuentes colombianas (DANE, Censo 2018) y supuestos del autor. Los resultados describen el comportamiento del flujo de análisis, no efectos verificados en población colombiana.
