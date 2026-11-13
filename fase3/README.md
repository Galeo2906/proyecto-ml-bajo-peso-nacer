# Predicción del bajo peso al nacer en Colombia (DANE, nacimientos 2022)

**Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia**
Autor: Daniel Felipe Galeano Peña · Docente: Diana Pulido

Artículo base: Feyisa BR, Mulatu Y, Fentahun F, Biru B, Atlantis E (2023). *Nutrition, stress, and healthcare use during pregnancy are associated with low birth weight: evidence from a case–control study in West Ethiopia.* Frontiers in Public Health, 11, 1213291. https://doi.org/10.3389/fpubh.2023.1213291

## Qué hay en el repositorio

| Archivo | Contenido |
|---|---|
| `fase3_final.ipynb` | Cuaderno final (Fase 3), ejecutado de principio a fin. |
| `fase3_final.py` | El mismo cuaderno en formato script (percent). |
| `reproducir_en_vivo.ipynb` | Reproducción rápida (≈ 1 min) del modelo final y sus cifras en prueba, sin la comparación de los 9 modelos. |
| `00_preparar_muestra.py` | Genera la muestra de 50.000 nacimientos desde el archivo completo del DANE. |
| `construir_notebook.py` | Convierte `fase3_final.py` en el `.ipynb`. |
| `data/nacimientos_dane_2022_muestra.csv` | Muestra aleatoria (semilla 42) con solo las 12 columnas usadas. |
| `resultados/` | Todas las tablas (`tabla*.csv`) y figuras (`fig*.png`) del informe, generadas por el cuaderno. |
| `informe/` | Informe técnico final (PDF y fuente LaTeX). |

## Datos

Estadísticas Vitales (EEVV) del DANE, nacimientos 2022, microdatos anonimizados (573.625 registros), descargados del catálogo de microdatos del DANE. Bajo peso al nacer = códigos 1–4 de `PESO_NAC` (< 2.500 g). Los códigos de cada variable se tomaron del diccionario del DANE y del formato de certificado de nacido vivo.

## Cómo reproducir

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib
python 00_preparar_muestra.py ruta/a/nac2022.csv   # opcional: la muestra ya está en data/
jupyter nbconvert --to notebook --execute fase3_final.ipynb
```

En Colab: subir `fase3_final.ipynb` y `nacimientos_dane_2022_muestra.csv` (en `data/` o junto al cuaderno) y ejecutar todo. Tarda entre 15 y 40 minutos (validación cruzada anidada de 9 modelos). Para comprobar solo el modelo final y sus cifras en prueba, `reproducir_en_vivo.ipynb` corre en cerca de un minuto. Semilla 42 en todo.

## Modelos (solo los vistos en clase)

Línea base (prevalencia), KNN, Naive Bayes (Bernoulli), LDA, QDA, regresión logística sin penalizar, y logística regularizada L2, L1 y Elastic Net (`penalty` fijado explícitamente, como en la Sesión 37b).
