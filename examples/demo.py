"""
End-to-end demo: approximate every partial directed Hausdorff distance between
two point clouds and compare against the exact answer.

    python examples/demo.py
    python examples/demo.py --n 200 --m 120 --epsilon 0.3 --variant bucket
"""

import argparse
import random

from greedypermutation.balltree import greedy_tree
from greedypermutation.point import Point
from metricspaces import MetricSpace

from khausdorff.bucketkhausdorff import all_k_hausdorff_bucket
from khausdorff.khausdorff import all_k_hausdorff
from khausdorff.naive import all_partial_hausdorff


def cloud(n, rng, offset=0.0, dim=2):
    return [
        Point([rng.random() + (offset if d == 0 else 0.0) for d in range(dim)])
        for _ in range(n)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="points in A")
    parser.add_argument("--m", type=int, default=25, help="points in B")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--offset",
        type=float,
        default=0.4,
        help="how far B is shifted from A along the first axis",
    )
    parser.add_argument("--variant", choices=("heap", "bucket"), default="heap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=12, help="rows to print")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    A = cloud(args.n, rng, dim=args.dim)
    B = cloud(args.m, rng, offset=args.offset, dim=args.dim)

    # Preprocessing: one greedy tree per set, reusable across queries.
    G_A = greedy_tree(MetricSpace(A))
    G_B = greedy_tree(MetricSpace(B))

    if args.variant == "heap":
        approx = all_k_hausdorff(G_A, G_B, args.epsilon)
    else:
        approx = all_k_hausdorff_bucket(G_A, G_B, args.epsilon)
    exact = all_partial_hausdorff(A, B)

    print(
        f"|A| = {args.n}, |B| = {args.m}, dim = {args.dim}, "
        f"epsilon = {args.epsilon}, variant = {args.variant}\n"
    )
    print(f"{'k':>4}  {'delta_k (approx)':>17}  {'d_h^(k) (exact)':>16}  {'ratio':>7}")
    print("-" * 52)
    shown = min(args.rows, len(exact))
    for k in range(shown):
        ratio = exact[k] / approx[k] if approx[k] > 0 else 1.0
        print(f"{k:>4}  {approx[k]:>17.6f}  {exact[k]:>16.6f}  {ratio:>7.4f}")
    if shown < len(exact):
        print(f"{'...':>4}  ({len(exact) - shown} more)")

    ratios = [e / a for a, e in zip(approx, exact) if a > 0]
    worst = max(ratios) if ratios else 1.0
    print(
        f"\nd_h(A -> B) ~= {approx[0]:.6f} (exact {exact[0]:.6f})"
        f"\nworst ratio d_h^(k) / delta_k = {worst:.4f}, guaranteed <= {1 + args.epsilon}"
    )
    assert all(a <= e + 1e-9 for a, e in zip(approx, exact)), "lower bound violated"
    assert worst <= 1 + args.epsilon + 1e-9, "upper bound violated"
    print("both approximation bounds hold.")


if __name__ == "__main__":
    main()
