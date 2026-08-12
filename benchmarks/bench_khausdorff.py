"""
Timing harness for k-HAUSDORFF.

Reports, per input size, the preprocessing cost (building the two greedy trees)
and the cost of the traversal itself, optionally next to the O(|A|*|B|) naive
reference.

    # compare against the exact reference on sizes it can still handle
    python benchmarks/bench_khausdorff.py --sizes 100,500,1000 --compare naive

    # just time k-Hausdorff, where the reference would be hopeless
    python benchmarks/bench_khausdorff.py --sizes 5000,20000 --compare none

    # sweep epsilon, both variants, save the table
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

# Above this size the O(n*m) reference is too slow to run by default.
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


def run_case(A, B, epsilon, variant, compare, repeat):
    """Time one (size, epsilon, variant) cell and return a result row."""
    build_times, run_times, results = [], [], None
    for _ in range(repeat):
        (trees, build) = timed(
            lambda: (greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B)))
        )
        G_A, G_B = trees
        solver = all_k_hausdorff if variant == "heap" else all_k_hausdorff_bucket
        results, run = timed(lambda: solver(G_A, G_B, epsilon))
        build_times.append(build)
        run_times.append(run)

    row = {
        "n": len(A),
        "m": len(B),
        "epsilon": epsilon,
        "variant": variant,
        "build_s": min(build_times),
        "khausdorff_s": min(run_times),
        "naive_s": None,
        "speedup": None,
        "worst_ratio": None,
    }

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
    f"{'kH s':>9} {'naive s':>9} {'speedup':>8} {'ratio':>7}"
)


def format_row(row):
    def fmt(value, width, decimals):
        text = "-" if value is None else f"{value:.{decimals}f}"
        return text.rjust(width)

    return (
        f"{row['n']:>7} {row['m']:>7} {row['epsilon']:>5} {row['variant']:>7} "
        f"{row['build_s']:>9.4f} {row['khausdorff_s']:>9.4f} "
        f"{fmt(row['naive_s'], 9, 4)} {fmt(row['speedup'], 8, 1)} "
        f"{fmt(row['worst_ratio'], 7, 4)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,500,1000,2000", help="values of |A|")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.6,
        help="|B| as a fraction of |A|",
    )
    parser.add_argument("--epsilon", default="0.1", help="one value or a comma list")
    parser.add_argument("--variant", choices=("heap", "bucket", "both"), default="heap")
    parser.add_argument(
        "--compare",
        choices=("naive", "none", "auto"),
        default="auto",
        help="'naive' also times the exact O(n*m) reference and reports the "
        "observed approximation ratio; 'none' times k-Hausdorff only; 'auto' "
        f"uses naive up to n={NAIVE_AUTO_LIMIT} and none above",
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--offset", type=float, default=0.4)
    parser.add_argument("--repeat", type=int, default=1, help="best of N runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="also write the table to this file")
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
                row = run_case(A, B, epsilon, variant, compare, args.repeat)
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
