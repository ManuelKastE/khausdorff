"""
Lectura de conjuntos de puntos desde archivos de texto.

El formato es el mismo que usa el CLI de `greedypermutation` -- un punto por
línea -- pero admitiendo también comas, que es como salen los datos de casi
cualquier herramienta:

    # las líneas que empiezan con # se ignoran
    x,y
    0.1,0.4
    0.7,0.2

Se ignoran las líneas vacías, las que empiezan con `#`, y una primera línea no
numérica (un encabezado como `x,y`).  Como en `MetricSpace.fromstrings`, todo lo
que venga después de un `;` se descarta.

Las validaciones no son decorativas.  `Point.dist` calcula la distancia
"proyectando al subespacio común" cuando las dimensiones no coinciden, así que
una fila a la que le falte una coordenada no produce ningún error: produce un
resultado silenciosamente equivocado.  Por eso `load_points` exige que todas las
filas tengan la misma cantidad de coordenadas y dice en qué línea se rompe.
"""

from greedypermutation.point import Point


def _coordenadas(texto, separador):
    """Parte una línea en coordenadas, por comas o por espacios."""
    if separador == ",":
        partes = texto.split(",")
    else:
        partes = texto.split()
    return [p.strip() for p in partes if p.strip()]


def _es_numerica(partes):
    try:
        [float(p) for p in partes]
        return True
    except ValueError:
        return False


def load_points(source, dim=None):
    """
    Lee un conjunto de puntos y lo devuelve como lista de `Point`.

    `source` puede ser una ruta o un archivo ya abierto.  El separador se
    autodetecta: si la primera línea con contenido tiene una coma, se usan
    comas; si no, espacios.

    `dim` fija la cantidad de coordenadas esperada.  Si no se da, se toma la de
    la primera línea de datos y se exige que las demás coincidan.

    Lanza `ValueError` indicando el número de línea si una fila no es numérica o
    tiene una cantidad distinta de coordenadas, y si el archivo no contiene
    ningún punto.
    """
    if hasattr(source, "read"):
        lineas = source.readlines()
        nombre = getattr(source, "name", "<archivo>")
    else:
        nombre = str(source)
        with open(source) as handle:
            lineas = handle.readlines()

    utiles = []  # (numero_de_linea, texto)
    for numero, linea in enumerate(lineas, start=1):
        texto = linea.split(";")[0].strip()
        if not texto or texto.startswith("#"):
            continue
        utiles.append((numero, texto))

    if not utiles:
        raise ValueError(f"{nombre}: no contiene ningún punto.")

    separador = "," if "," in utiles[0][1] else " "

    # Un encabezado como `x,y` es la primera línea si no es numérica.
    if not _es_numerica(_coordenadas(utiles[0][1], separador)):
        utiles = utiles[1:]
        if not utiles:
            raise ValueError(
                f"{nombre}: solo contiene un encabezado, ningún punto."
            )

    puntos = []
    for numero, texto in utiles:
        partes = _coordenadas(texto, separador)
        try:
            coords = [float(p) for p in partes]
        except ValueError as error:
            raise ValueError(
                f"{nombre}, línea {numero}: no se pudo leer como número -> {texto!r}"
            ) from error
        if dim is None:
            dim = len(coords)
        if len(coords) != dim:
            raise ValueError(
                f"{nombre}, línea {numero}: tiene {len(coords)} coordenadas y se "
                f"esperaban {dim}.  Las dimensiones desparejas no dan error al "
                f"calcular: `Point.dist` proyecta al subespacio común y devuelve "
                f"un resultado equivocado en silencio."
            )
        puntos.append(Point(coords))

    return puntos


def duplicates(points):
    """
    Devuelve los puntos que aparecen más de una vez, sin repetirlos.

    Importa porque `greedy_tree` no admite puntos exactamente duplicados: la
    permutación greedy se queda sin centros distintos y le entrega `None` a la
    métrica, lo que sale como `TypeError: 'NoneType' object is not iterable`, un
    error que no dice nada sobre la causa.  Conviene detectarlos antes.
    """
    vistos, repetidos = set(), {}
    for punto in points:
        clave = tuple(punto)
        if clave in vistos:
            repetidos[clave] = punto
        else:
            vistos.add(clave)
    return list(repetidos.values())


def deduplicate(points):
    """Devuelve los puntos sin repetidos, conservando el orden de aparición."""
    vistos, salida = set(), []
    for punto in points:
        clave = tuple(punto)
        if clave not in vistos:
            vistos.add(clave)
            salida.append(punto)
    return salida
