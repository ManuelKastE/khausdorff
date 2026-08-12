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
]

__version__ = "0.1.0"
