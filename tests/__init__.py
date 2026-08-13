"""Suite de tests de `khausdorff`.

Este archivo no es decorativo.  Los tests se importan entre sí como
`from tests.helpers import ...`, y sin un `__init__.py` la carpeta es solo una
*namespace portion* (PEP 420), que pierde la resolución de nombres frente a
cualquier paquete `tests` regular de nivel superior instalado en el entorno.
`greedypermutation` instala exactamente uno de esos; volver este paquete regular
hace que la suite se resuelva a sí misma sin importar qué más haya en el path.

Se corre desde la raíz del repositorio:

    python3 -m pytest tests/ -q

El `-m` mete el directorio actual en `sys.path`, así que `import khausdorff`
encuentra `khausdorff.py` sin que haya nada instalado.
"""
