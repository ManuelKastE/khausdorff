"""
khausdorff: approximate partial directed Hausdorff distances.

Implements the k-HAUSDORFF algorithm of Section 5 of Chubet, Parikh, Sheehy and
Sheth, "Approximating the Directed Hausdorff Distance" (2023), on top of the
greedy trees and dual-tree traversal of the `greedypermutation` package.
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
