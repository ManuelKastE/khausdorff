"""
Banco de pruebas de tiempos para k-HAUSDORFF.

Reporta, por tamaño de entrada, el costo del preprocesamiento (construir los dos
árboles greedy) y el costo del recorrido en sí, opcionalmente junto a la
referencia ingenua O(|A|*|B|).

    # comparar contra la referencia exacta en tamaños que aún tolera
    python benchmarks/bench_khausdorff.py --sizes 100,500,1000 --compare naive

    # solo cronometrar k-Hausdorff, donde la referencia sería inviable
    python benchmarks/bench_khausdorff.py --sizes 5000,20000 --compare none

    # barrer epsilon, ambas variantes, guardar la tabla
    python benchmarks/bench_khausdorff.py --epsilon 0.1,0.5,1.0 --variant both --csv out.csv
"""

import argparse
import csv
import random
import sys
from time import perf_counter

from greedypermutation.balltree import greedy_tree
from greedypermutation.point import Point
from metricspaces import MetricSpace

from khausdorff.bucketkhausdorff import all_k_hausdorff_bucket
from khausdorff.khausdorff import all_k_hausdorff
from khausdorff.naive import all_partial_hausdorff
from khausdorff.percentile import hausdorff_percentile, hausdorff_percentile_bucket

# Por encima de este tamaño la referencia O(n*m) es demasiado lenta para
# ejecutarla por omisión.
NAIVE_AUTO_LIMIT = 2000


def cloud(n, rng, offset=0.0, dim=2):
    return [
        Point([rng.random() + (offset if d == 0 else 0.0) for d in range(dim)])
        for _ in range(n)
    ]


def timed(fn):
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def number_list(text, cast):
    return [cast(part) for part in text.split(",") if part.strip()]


def run_case(A, B, epsilon, variant, compare, repeat, percentile=None):
    """Cronometra una celda (tamaño, epsilon, variante) y devuelve una fila."""
    build_times, run_times, early_times, results = [], [], [], None
    for _ in range(repeat):
        (trees, build) = timed(
            lambda: (greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B)))
        )
        G_A, G_B = trees
        solver = all_k_hausdorff if variant == "heap" else all_k_hausdorff_bucket
        results, run = timed(lambda: solver(G_A, G_B, epsilon))
        build_times.append(build)
        run_times.append(run)

        if percentile is not None:
            # Arboles nuevos: la busqueda anterior ya consumio los otros.
            G_A, G_B = greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B))
            query = (
                hausdorff_percentile
                if variant == "heap"
                else hausdorff_percentile_bucket
            )
            _, early = timed(lambda: query(G_A, G_B, percentile, epsilon))
            early_times.append(early)

    row = {
        "n": len(A),
        "m": len(B),
        "epsilon": epsilon,
        "variant": variant,
        "build_s": min(build_times),
        "khausdorff_s": min(run_times),
        "early_s": min(early_times) if early_times else None,
        "early_speedup": None,
        "naive_s": None,
        "speedup": None,
        "worst_ratio": None,
    }
    if row["early_s"]:
        row["early_speedup"] = row["khausdorff_s"] / row["early_s"]

    if compare:
        exact, naive_time = timed(lambda: all_partial_hausdorff(A, B))
        row["naive_s"] = naive_time
        row["speedup"] = naive_time / row["khausdorff_s"] if row["khausdorff_s"] else None
        ratios = [e / a for a, e in zip(results, exact) if a > 0]
        row["worst_ratio"] = max(ratios) if ratios else 1.0
        violated = any(a > e + 1e-9 for a, e in zip(results, exact)) or (
            row["worst_ratio"] > 1 + epsilon + 1e-9
        )
        if violated:
            print(
                f"  WARNING: approximation bounds violated at n={len(A)}, "
                f"epsilon={epsilon}, variant={variant}",
                file=sys.stderr,
            )
    return row


HEADER = (
    f"{'n':>7} {'m':>7} {'eps':>5} {'variant':>7} {'build s':>9} "
    f"{'kH s':>9} {'corte s':>9} {'x corte':>8} {'naive s':>9} {'speedup':>8} "
    f"{'ratio':>7}"
)


def format_row(row):
    def fmt(value, width, decimals):
        text = "-" if value is None else f"{value:.{decimals}f}"
        return text.rjust(width)

    return (
        f"{row['n']:>7} {row['m']:>7} {row['epsilon']:>5} {row['variant']:>7} "
        f"{row['build_s']:>9.4f} {row['khausdorff_s']:>9.4f} "
        f"{fmt(row['early_s'], 9, 4)} {fmt(row['early_speedup'], 8, 2)} "
        f"{fmt(row['naive_s'], 9, 4)} {fmt(row['speedup'], 8, 1)} "
        f"{fmt(row['worst_ratio'], 7, 4)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,500,1000,2000", help="valores de |A|")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.6,
        help="|B| como fracción de |A|",
    )
    parser.add_argument("--epsilon", default="0.1", help="un valor o una lista separada por comas")
    parser.add_argument("--variant", choices=("heap", "bucket", "both"), default="heap")
    parser.add_argument(
        "--compare",
        choices=("naive", "none", "auto"),
        default="auto",
        help="'naive' cronometra además la referencia exacta O(n*m) y reporta "
        "la razón de aproximación observada; 'none' cronometra solo k-Hausdorff; "
        f"'auto' usa naive hasta n={NAIVE_AUTO_LIMIT} y none por encima",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="además del recorrido completo, cronometrar la consulta de este "
        "percentil con terminación temprana, y reportar la aceleración",
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--offset", type=float, default=0.4)
    parser.add_argument("--repeat", type=int, default=1, help="el mejor de N intentos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="escribir además la tabla en este archivo")
    args = parser.parse_args()

    sizes = number_list(args.sizes, int)
    epsilons = number_list(args.epsilon, float)
    variants = ("heap", "bucket") if args.variant == "both" else (args.variant,)

    print(HEADER)
    print("-" * len(HEADER))
    rows = []
    for n in sizes:
        rng = random.Random(args.seed)
        A = cloud(n, rng, dim=args.dim)
        B = cloud(max(1, int(n * args.ratio)), rng, offset=args.offset, dim=args.dim)
        compare = (
            args.compare == "naive"
            or (args.compare == "auto" and n <= NAIVE_AUTO_LIMIT)
        )
        if args.compare == "auto" and not compare:
            print(f"  (n={n} above {NAIVE_AUTO_LIMIT}: skipping the naive reference)")
        for epsilon in epsilons:
            for variant in variants:
                if variant == "bucket" and epsilon <= 0:
                    print("  (bucket variant needs epsilon > 0: skipped)")
                    continue
                row = run_case(
                    A, B, epsilon, variant, compare, args.repeat, args.percentile
                )
                rows.append(row)
                print(format_row(row))

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
