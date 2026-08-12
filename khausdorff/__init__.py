"""
khausdorff: distancias de Hausdorff dirigidas parciales aproximadas.

Implementa el algoritmo k-HAUSDORFF de la Sección 5 de Chubet, Parikh, Sheehy y
Sheth, "Approximating the Directed Hausdorff Distance" (2023), sobre los árboles
greedy y el recorrido de árbol dual del paquete `greedypermutation`.
"""

from khausdorff.khausdorff import KHausdorff, all_k_hausdorff, k_hausdorff
from khausdorff.bucketkhausdorff import KHausdorffBucket, all_k_hausdorff_bucket
from khausdorff.betabucketqueue import BetaBucketQueue
from khausdorff.naive import (
    all_partial_hausdorff,
    nearest_distances,
    partial_hausdorff,
)
from khausdorff.percentile import (
    hausdorff_percentile,
    hausdorff_percentile_bucket,
    k_for_percentile,
    partial_hausdorff_percentile,
)

__all__ = [
    "KHausdorff",
    "all_k_hausdorff",
    "k_hausdorff",
    "KHausdorffBucket",
    "all_k_hausdorff_bucket",
    "BetaBucketQueue",
    "all_partial_hausdorff",
    "partial_hausdorff",
    "nearest_distances",
    "hausdorff_percentile",
    "hausdorff_percentile_bucket",
    "partial_hausdorff_percentile",
    "k_for_percentile",
]

__version__ = "0.2.0"
