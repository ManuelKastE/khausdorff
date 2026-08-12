"""
Calcula distancias de Hausdorff dirigidas parciales sobre dos archivos de puntos.

    khausdorff A.csv B.csv --percentile 95
    khausdorff A.csv B.csv --k 5 --epsilon 0.5 --variant bucket
    khausdorff A.csv B.csv --todos --csv salida.csv

Un punto por línea, coordenadas separadas por comas o espacios; ver
`khausdorff.io`.  Por omisión calcula además la respuesta exacta por fuerza
bruta para poder contrastarla, lo que cuesta O(|A|·|B|): con entradas grandes
conviene `--sin-exacto`.

La distancia es dirigida: el primer archivo es A y el segundo es B, y
intercambiarlos da otra respuesta.
"""

import argparse
import csv
import sys
from time import perf_counter

from greedypermutation.balltree import greedy_tree
from metricspaces import MetricSpace

from khausdorff.bucketkhausdorff import all_k_hausdorff_bucket
from khausdorff.io import deduplicate, duplicates, load_points
from khausdorff.khausdorff import all_k_hausdorff
from khausdorff.naive import all_partial_hausdorff
from khausdorff.percentile import k_for_percentile


def _cronometrar(fn):
    inicio = perf_counter()
    resultado = fn()
    return resultado, perf_counter() - inicio


def _preparar(ruta, deduplicar):
    """Lee un archivo y aborta con un mensaje útil si tiene puntos repetidos."""
    puntos = load_points(ruta)
    repetidos = duplicates(puntos)
    if repetidos:
        if not deduplicar:
            raise SystemExit(
                f"error: {ruta} tiene {len(repetidos)} punto(s) repetido(s), y "
                f"`greedy_tree` no admite puntos exactamente duplicados (falla con "
                f"un TypeError que no explica la causa).\n"
                f"       El primero es {repetidos[0]}.\n"
                f"       Usa --deduplicar para quitarlos y continuar."
            )
        antes = len(puntos)
        puntos = deduplicate(puntos)
        print(f"  aviso: {ruta}: se quitaron {antes - len(puntos)} punto(s) repetido(s)")
    return puntos


def _arboles(A, B):
    """Un par de árboles nuevos.  Cada búsqueda consume los que recibe."""
    return greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("A", help="archivo con los puntos de A")
    parser.add_argument("B", help="archivo con los puntos de B")

    cual = parser.add_mutually_exclusive_group()
    cual.add_argument(
        "--percentile",
        type=float,
        help="percentil a calcular: 95 descarta el 5%% de los puntos de A más "
        "lejanos de B (100 es la distancia de Hausdorff ordinaria)",
    )
    cual.add_argument(
        "--k", type=int, help="cantidad de valores atípicos a descartar (0 = ninguno)"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="parámetro de aproximación; 0 (por omisión) da la respuesta exacta",
    )
    parser.add_argument("--variant", choices=("heap", "bucket"), default="heap")
    parser.add_argument(
        "--todos",
        action="store_true",
        help="mostrar la secuencia completa en vez de un solo valor",
    )
    parser.add_argument("--csv", help="volcar la secuencia completa a este archivo")
    parser.add_argument(
        "--sin-exacto",
        dest="exacto",
        action="store_false",
        help="no calcular la referencia exacta por fuerza bruta, que cuesta O(|A|·|B|)",
    )
    parser.add_argument(
        "--deduplicar",
        action="store_true",
        help="quitar puntos repetidos en vez de abortar",
    )
    parser.add_argument("--filas", type=int, default=15, help="filas a mostrar con --todos")
    args = parser.parse_args(argv)

    if args.percentile is None and args.k is None:
        args.percentile = 100.0

    A = _preparar(args.A, args.deduplicar)
    B = _preparar(args.B, args.deduplicar)
    n, dim = len(A), len(A[0])
    if len(B[0]) != dim:
        raise SystemExit(
            f"error: los puntos de A tienen {dim} coordenadas y los de B "
            f"{len(B[0])}.  Deben coincidir."
        )

    k = args.k if args.k is not None else k_for_percentile(n, args.percentile)
    if not 0 <= k <= n:
        raise SystemExit(f"error: k debe estar en [0, {n}], y es {k}.")

    if args.variant == "bucket" and args.epsilon <= 0:
        raise SystemExit(
            "error: la variante de buckets necesita --epsilon > 0.  "
            "Usa --variant heap para respuestas exactas."
        )

    print(f"|A| = {n}, |B| = {len(B)}, dimensión = {dim}")
    etiqueta = (
        f"percentil {args.percentile:g}" if args.k is None else f"k = {k}"
    )
    print(f"{etiqueta}  ->  k = {k} de {n}   (epsilon = {args.epsilon:g}, {args.variant})")
    print()

    resolver = all_k_hausdorff if args.variant == "heap" else all_k_hausdorff_bucket
    corte = None if (args.todos or args.csv) else k

    (G_A, G_B), t_construir = _cronometrar(lambda: _arboles(A, B))
    aprox, t_consulta = _cronometrar(
        lambda: resolver(G_A, G_B, args.epsilon, stop_after=corte)
    )

    exacto = None
    if args.exacto:
        exacto, t_exacto = _cronometrar(lambda: all_partial_hausdorff(A, B))

    print(f"  construcción de los árboles : {t_construir:8.4f} s")
    print(f"  consulta                    : {t_consulta:8.4f} s", end="")
    print("  (recorrido completo)" if corte is None else "  (con terminación temprana)")
    if exacto is not None:
        print(f"  referencia exacta           : {t_exacto:8.4f} s")
    print()

    if args.todos:
        filas = min(args.filas, len(aprox))
        print(f"{'k':>6}  {'aproximado':>14}", end="")
        print(f"  {'exacto':>14}  {'razón':>7}" if exacto else "")
        print("-" * (24 if exacto is None else 48))
        for i in range(filas):
            linea = f"{i:>6}  {aprox[i]:>14.6f}"
            if exacto:
                razon = exacto[i] / aprox[i] if aprox[i] > 0 else 1.0
                linea += f"  {exacto[i]:>14.6f}  {razon:>7.4f}"
            print(linea)
        if filas < len(aprox):
            print(f"{'...':>6}  ({len(aprox) - filas} más)")
        print()

    print(f"  valor aproximado : {aprox[k]:.6f}")
    if exacto is not None:
        print(f"  valor exacto     : {exacto[k]:.6f}")
        if aprox[k] > 0:
            print(f"  razón            : {exacto[k] / aprox[k]:.4f}")
        cota_inferior = aprox[k] <= exacto[k] + 1e-9
        cota_superior = exacto[k] <= (1 + args.epsilon) * aprox[k] + 1e-9
        if cota_inferior and cota_superior:
            print(f"  ambas cotas se respetan (garantía: razón <= {1 + args.epsilon:g})")
        else:
            print("  ATENCIÓN: alguna cota no se respeta", file=sys.stderr)
            return 1

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            escritor = csv.writer(handle)
            if exacto is not None:
                escritor.writerow(["k", "aproximado", "exacto"])
                escritor.writerows(zip(range(len(aprox)), aprox, exacto))
            else:
                escritor.writerow(["k", "aproximado"])
                escritor.writerows(zip(range(len(aprox)), aprox))
        print(f"\n  secuencia completa escrita en {args.csv} ({len(aprox)} filas)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
