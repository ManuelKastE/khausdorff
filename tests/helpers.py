"""Shared fixtures for the test suite."""

import random

from greedypermutation.balltree import greedy_tree
from greedypermutation.point import Point
from metricspaces import MetricSpace, R1

TOL = 1e-9


def line(values):
    """A point set on the real line."""
    return [R1(v) for v in values]


def cloud(n, rng, offset=0.0, dim=2):
    """`n` uniform points in the unit cube, shifted along the first axis."""
    return [
        Point([rng.random() + (offset if d == 0 else 0.0) for d in range(dim)])
        for _ in range(n)
    ]


def trees(A, B):
    """Fresh greedy trees for `A` and `B`.

    Always build a new pair: the search mutates the viability graph it is given,
    so a tree must not be reused across two runs.
    """
    return greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B))


def random_pair(rng, max_a=60, max_b=40, dim=2):
    """A random pair of partially overlapping point clouds."""
    A = cloud(rng.randint(1, max_a), rng, dim=dim)
    B = cloud(rng.randint(1, max_b), rng, offset=0.4, dim=dim)
    return A, B


def seeded(seed):
    return random.Random(seed)
