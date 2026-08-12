"""
Exact, brute-force reference for the partial directed Hausdorff distance.

This is the ground truth the approximation algorithms are tested against.
It costs O(|A| * |B|) distance computations, so it is only usable on small
inputs, but it needs no preprocessing and no approximation parameter.
"""


def nearest_distances(A: list, B: list) -> list:
    """
    Return the list of d(a, B) = min_{b in B} d(a, b) for every a in A.

    `A` and `B` are iterables of points (anything with a `dist` method, such as
    `greedypermutation.point.Point` or `metricspaces.R1`).
    """
    B = list(B)
    if not B:
        raise ValueError("B must contain at least one point.")
    return [min(a.dist(b) for b in B) for a in A]


def all_partial_hausdorff(A: list, B: list) -> list:
    """
    Return the exact sequence (delta_0, ..., delta_{n-1}) of partial directed
    Hausdorff distances, where delta_k = d_h^(k)(A, B).

    The k-th partial directed Hausdorff distance is

        d_h^(k)(A, B) = min_{S in A^(k)} d_h(S, B),

    where A^(k) is the family of subsets of A with k points removed.  The
    minimum is attained by removing the k points of A that are farthest from B,
    so d_h^(k)(A, B) is simply the (k+1)-th largest value of d(a, B).

    In particular delta_0 is the ordinary directed Hausdorff distance d_h(A, B),
    and the sequence is non-increasing.
    """
    return sorted(nearest_distances(A, B), reverse=True)


def partial_hausdorff(A: list, B: list, k: int) -> float:
    """Return the exact k-th partial directed Hausdorff distance d_h^(k)(A, B)."""
    distances = all_partial_hausdorff(A, B)
    if not 0 <= k < len(distances):
        raise IndexError(f"k must be in [0, {len(distances)}), got {k}.")
    return distances[k]
