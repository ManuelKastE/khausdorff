"""
Genera dos archivos de puntos de ejemplo, en el formato que lee `khausdorff`.

    python examples/generar_datos.py --salida datos/
    python examples/generar_datos.py --n 1000 --m 600 --atipicos 20 --seed 7

Escribe `A.csv` y `B.csv`: dos nubes uniformes parcialmente superpuestas, más un
puñado de valores atípicos lejanos en A.  Los atípicos son a propósito: hacen
visible la diferencia entre pedir el percentil 100 (que los mide) y el 95 (que
los descarta), que es justamente lo que las distancias parciales resuelven.

Con `--seed` fijo los archivos son reproducibles.
"""

import argparse
import random
from pathlib import Path


def nube(n, rng, offset=0.0, dim=2):
    return [
        [rng.random() + (offset if d == 0 else 0.0) for d in range(dim)]
        for _ in range(n)
    ]


def atipicos(cuantos, rng, dim=2, lejania=5.0):
    return [[lejania + rng.random() for _ in range(dim)] for _ in range(cuantos)]


def escribir(ruta, puntos, encabezado=True):
    dim = len(puntos[0])
    with open(ruta, "w") as handle:
        if encabezado:
            handle.write(",".join(["x", "y", "z"][:dim] or [f"c{i}" for i in range(dim)]))
            handle.write("\n")
        for punto in puntos:
            handle.write(",".join(f"{c:.6f}" for c in punto) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=500, help="puntos en A")
    parser.add_argument("--m", type=int, default=300, help="puntos en B")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument(
        "--atipicos", type=int, default=10, help="valores atípicos lejanos a añadir en A"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.3,
        help="cuánto se desplaza B respecto de A sobre el primer eje",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--salida", default=".", help="carpeta donde escribir A.csv y B.csv")
    parser.add_argument(
        "--sin-encabezado",
        dest="encabezado",
        action="store_false",
        help="no escribir la primera línea con los nombres de las columnas",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    A = nube(args.n, rng, dim=args.dim) + atipicos(args.atipicos, rng, dim=args.dim)
    B = nube(args.m, rng, offset=args.offset, dim=args.dim)

    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    escribir(carpeta / "A.csv", A, args.encabezado)
    escribir(carpeta / "B.csv", B, args.encabezado)

    print(f"escritos {carpeta / 'A.csv'} ({len(A)} puntos, {args.atipicos} atípicos)")
    print(f"         {carpeta / 'B.csv'} ({len(B)} puntos)")
    print()
    print("Para probar la diferencia que hacen los atípicos:")
    print(f"  khausdorff {carpeta / 'A.csv'} {carpeta / 'B.csv'} --percentile 100")
    print(f"  khausdorff {carpeta / 'A.csv'} {carpeta / 'B.csv'} --percentile 95")


if __name__ == "__main__":
    main()
