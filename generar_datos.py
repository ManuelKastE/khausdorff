"""
Genera dos archivos de puntos al azar para probar `khausdorff.py`.

    python3 generar_datos.py --tamano low --salida datos/

El tamaño de cada conjunto se sortea dentro del rango de la categoría, así que
|A| y |B| salen distintos entre sí y distintos en cada corrida.  Los puntos
nunca se repiten: `greedy_tree` no admite duplicados.
"""

import argparse
import random
from pathlib import Path

# Cuántos puntos por conjunto, según la categoría.
RANGOS = {"low": (20, 100), "mid": (100, 1000), "max": (1000, 5000)}


def nube(cantidad, rng):
    """`cantidad` puntos distintos del cuadrado unitario, con 6 decimales."""
    puntos = set()
    while len(puntos) < cantidad:
        puntos.add((round(rng.random(), 6), round(rng.random(), 6)))
    return sorted(puntos)


def escribir(ruta, puntos):
    ruta.write_text("x,y\n" + "".join(f"{x},{y}\n" for x, y in puntos))
    print(f"{ruta}: {len(puntos)} puntos")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salida", default="datos", help="carpeta donde escribir A.csv y B.csv"
    )
    parser.add_argument(
        "--tamano",
        choices=tuple(RANGOS),
        default="mid",
        help="cuántos puntos por conjunto: "
        + ", ".join(f"{k} = {a}-{b}" for k, (a, b) in RANGOS.items()),
    )
    args = parser.parse_args()

    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = random.Random()
    for nombre in ("A", "B"):
        escribir(carpeta / f"{nombre}.csv", nube(rng.randint(*RANGOS[args.tamano]), rng))


if __name__ == "__main__":
    main()
