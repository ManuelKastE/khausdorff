"""
Indexar las distancias parciales por percentil en vez de por cantidad de
descartes.

El artículo indexa por conteo: d_h^(k) descarta k valores atípicos.  En la
práctica suele ser más natural pedir un porcentaje -- "el percentil 95 de las
distancias", es decir descartar el 5% de los puntos de A más lejanos de B.  Son
la misma cantidad con distinto índice, porque, como dice la Sección 5 del
artículo, d_h^(k)(A, B) es el (k+1)-ésimo valor más grande de d(a, B) sobre
a en A: un estadístico de orden.

Se usa la convención *nearest-rank*: el percentil q es el valor que ocupa la
posición ceil(q*n/100) al ordenar las n distancias de forma ascendente.  Es la
definición clásica de percentil por estadístico de orden, y tiene una propiedad
que aquí importa mucho: el resultado es siempre uno de los delta_k que el
algoritmo ya calculó, así que hereda intacta la garantía

    delta <= d_h^(percentil q) <= (1 + epsilon) * delta

demostrada en el artículo.  Interpolar entre dos delta_k -- como hace
`numpy.percentile` por omisión -- devolvería un valor que el algoritmo nunca
computó, y la demostración de la cota dejaría de aplicarse tal cual.
"""

from math import ceil

from khausdorff.bucketkhausdorff import all_k_hausdorff_bucket
from khausdorff.khausdorff import all_k_hausdorff
from khausdorff.naive import all_partial_hausdorff


def k_for_percentile(n, q):
    """
    El k tal que delta_k es el percentil `q` de las `n` distancias d(a, B).

    Con las distancias ordenadas de forma ascendente en v[0..n-1], el percentil
    nearest-rank es v[ceil(q*n/100) - 1].  Como delta_k = v[n-1-k], eso da

        k = n - ceil(q*n/100).

    Los extremos salen solos: q = 100 da k = 0, la distancia de Hausdorff
    dirigida ordinaria, y q = 0 da k = n, el 0 final de la secuencia.
    """
    if not 0 <= q <= 100:
        raise ValueError(f"q must be a percentile in [0, 100], got {q}.")
    return n - ceil(q * n / 100)


def _at_percentile(deltas, q):
    """Indexa por percentil una secuencia (delta_0, ..., delta_n) ya calculada."""
    return deltas[k_for_percentile(len(deltas) - 1, q)]


def hausdorff_percentile(G_A, G_B, q, epsilon=0, monotone=True):
    """
    El percentil `q` de las distancias d(a, B), aproximado.

    Equivale a `k_hausdorff` con k = n - ceil(q*n/100).  `G_A` y `G_B` son
    árboles greedy; con `epsilon=0` el resultado es exacto.

        hausdorff_percentile(G_A, G_B, 100)  # la distancia de Hausdorff dirigida
        hausdorff_percentile(G_A, G_B, 95)   # descartando el 5% de los puntos

    El recorrido se abandona apenas se conoce ese delta_k.  Como `len(G_A)` da
    |A| sin recorrer nada, el k se conoce de antemano y el corte se aprovecha
    entero: pedir el percentil 95 cuesta casi lo mismo que pedir el máximo.
    """
    k = k_for_percentile(len(G_A), q)
    return all_k_hausdorff(G_A, G_B, epsilon, monotone, stop_after=k)[k]


def hausdorff_percentile_bucket(G_A, G_B, q, epsilon, monotone=True):
    """
    El percentil `q` de las distancias d(a, B), con la variante de buckets.

    Requiere `epsilon > 0`, igual que `all_k_hausdorff_bucket`.  Corta el
    recorrido igual que `hausdorff_percentile`.
    """
    k = k_for_percentile(len(G_A), q)
    return all_k_hausdorff_bucket(G_A, G_B, epsilon, monotone, stop_after=k)[k]


def partial_hausdorff_percentile(A: list, B: list, q) -> float:
    """
    El percentil `q` de las distancias d(a, B), exacto y por fuerza bruta.

    Recibe **puntos**, no árboles, igual que el resto de `khausdorff.naive`.
    """
    return _at_percentile(all_partial_hausdorff(A, B), q)
