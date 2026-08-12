"""Suite de tests de `khausdorff`.

Este archivo no es decorativo.  Los tests se importan entre sí como
`from tests.helpers import ...`, y sin un `__init__.py` la carpeta es solo una
*namespace portion* (PEP 420), que pierde la resolución de nombres frente a
cualquier paquete `tests` regular de nivel superior instalado en el entorno.
`greedypermutation` instalaba exactamente uno de esos; volver este paquete
regular hace que la suite se resuelva a sí misma sin importar qué más haya en
el path.

`pyproject.toml` restringe el empaquetado a `khausdorff*`, así que este paquete
nunca se distribuye: agregarlo aquí no repite el error del que protege.
"""
