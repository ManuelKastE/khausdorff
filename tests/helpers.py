"""Utilidades compartidas por los tests."""

import random

from greedypermutation.point import Point
from metricspaces import R1

TOL = 1e-9


def line(values):
    """Un conjunto de puntos sobre la recta real."""
    return [R1(v) for v in values]


def cloud(n, rng, offset=0.0, dim=2):
    """`n` puntos uniformes en el cubo unitario, desplazados sobre el primer eje."""
    return [
        Point([rng.random() + (offset if d == 0 else 0.0) for d in range(dim)])
        for _ in range(n)
    ]


def random_pair(rng, max_a=60, max_b=40, dim=2):
    """Un par aleatorio de nubes de puntos parcialmente superpuestas."""
    return (
        cloud(rng.randint(1, max_a), rng, dim=dim),
        cloud(rng.randint(1, max_b), rng, offset=0.4, dim=dim),
    )


def seeded(seed):
    return random.Random(seed)
