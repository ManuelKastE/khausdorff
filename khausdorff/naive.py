"""
Referencia exacta, por fuerza bruta, de la distancia de Hausdorff dirigida
parcial.

Esta es la verdad de referencia contra la que se contrastan los algoritmos de
aproximación.  Cuesta O(|A| * |B|) cálculos de distancia, así que solo es
utilizable en entradas pequeñas, pero no necesita preprocesamiento ni parámetro
de aproximación.
"""


def nearest_distances(A: list, B: list) -> list:
    """
    Devuelve la lista de d(a, B) = min_{b en B} d(a, b) para cada a en A.

    `A` y `B` son iterables de puntos (cualquier cosa con un método `dist`, como
    `greedypermutation.point.Point` o `metricspaces.R1`).
    """
    B = list(B)
    if not B:
        raise ValueError("B must contain at least one point.")
    return [min(a.dist(b) for b in B) for a in A]


def all_partial_hausdorff(A: list, B: list) -> list:
    """
    Devuelve la secuencia exacta (delta_0, ..., delta_n) de distancias de
    Hausdorff dirigidas parciales, donde delta_k = d_h^(k)(A, B).

    La k-ésima distancia de Hausdorff dirigida parcial es

        d_h^(k)(A, B) = min_{S en A^(k)} d_h(S, B),

    donde A^(k) es la familia de subconjuntos de A a los que se les quitaron k
    puntos.  El mínimo se alcanza quitando los k puntos de A más lejanos de B,
    de modo que d_h^(k)(A, B) es simplemente el (k+1)-ésimo valor más grande de
    d(a, B).

    En particular delta_0 es la distancia de Hausdorff dirigida ordinaria
    d_h(A, B), y la secuencia es no creciente.

    La lista tiene n + 1 elementos, con k recorriendo {0, ..., n} igual que en
    el artículo.  El último es 0: A^(n) = {conjunto vacío} y d_h(vacío, B) = 0
    por convención.
    """
    return sorted(nearest_distances(A, B), reverse=True) + [0.0]


def partial_hausdorff(A: list, B: list, k: int) -> float:
    """Devuelve la k-ésima distancia de Hausdorff dirigida parcial exacta d_h^(k)(A, B)."""
    distances = all_partial_hausdorff(A, B)
    if not 0 <= k < len(distances):
        raise IndexError(f"k must be in [0, {len(distances)}), got {k}.")
    return distances[k]
