"""Convierte fase3_final.py (formato percent) en fase3_final.ipynb."""
import re
import nbformat as nbf

src = open("fase3_final.py", encoding="utf-8").read()
partes = re.split(r"(?m)^# %%(.*)\n", src)
nb = nbf.v4.new_notebook()
for marca, cuerpo in zip(partes[1::2], partes[2::2]):
    cuerpo = cuerpo.strip("\n")
    if "[markdown]" in marca:
        txt = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in cuerpo.splitlines())
        nb.cells.append(nbf.v4.new_markdown_cell(txt))
    else:
        nb.cells.append(nbf.v4.new_code_cell(cuerpo))
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "fase3_final.ipynb")
print(len(nb.cells), "celdas")
