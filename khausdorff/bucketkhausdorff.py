"""
La variante de k-HAUSDORFF de la Sección 5.2: una cola de buckets beta como
heap de cotas inferiores.

`KHausdorff` mantiene un max-heap exacto de cotas inferiores locales, que cuesta
O(log n) por actualización.  La Sección 5.2 observa que ya se está haciendo una
aproximación, así que las cotas inferiores pueden agruparse geométricamente en
buckets, bajando las operaciones del heap a tiempo constante y el algoritmo
completo a

    (2 + 1/eps)^O(d) n + O(log_beta Delta),      beta = 1 + eps/2.

Dos cosas cambian respecto de la variante exacta.  Los nodos se terminan de a un
bucket completo en vez de uno por uno, y una actualización de cota inferior que
empujaría a un nodo por encima del umbral actual termina el nodo en lugar de
moverlo.  Ambas siguen la Observación 7 del artículo, que es lo que garantiza
que el barrido descendente visite cada bucket una sola vez.
"""

from math import ceil, inf, log

from khausdorff.betabucketqueue import BetaBucketQueue
from khausdorff.khausdorff import KHausdorff


class KHausdorffBucket(KHausdorff):
    """
    k-HAUSDORFF con una cola de buckets beta como heap de cotas inferiores.

    Requiere `epsilon > 0`: con epsilon = 0 la razón de los buckets beta sería 1
    y los buckets colapsarían.  Usar `KHausdorff` para respuestas exactas.
    """

    def __init__(self, G_A, G_B, epsilon, monotone=True):
        if epsilon <= 0:
            raise ValueError(
                "The bucket variant needs epsilon > 0 (beta = 1 + epsilon/2 must "
                "exceed 1).  Use KHausdorff for exact answers."
            )
        # Se fija antes de super().__init__, que construye el heap y registra G_A.
        self.beta = 1 + epsilon / 2
        # El nivel umbral de terminación.  Infinito hasta el primer barrido,
        # para que nada se termine antes de que el recorrido haya empezado.
        self.threshold_level = inf
        super().__init__(G_A, G_B, epsilon, monotone)

    # -- heap de cotas inferiores -------------------------------------------

    def _make_lb_heap(self):
        return BetaBucketQueue(self.beta, key=lambda x: self.lb[x])

    def _reprioritize(self, node):
        self.L.changepriority(node, self.lb[node])

    def _pop_max_lb(self):
        return self.L.findmax()

    def _record(self, node):
        """
        Recalcula l(node) y lo reubica, salvo que la nueva cota haya trepado por
        encima del umbral actual, en cuyo caso el nodo se termina en el acto en
        lugar de cambiarle la clave.
        """
        new_lb = self.G.lower_bound(node)
        self.lb[node] = new_lb
        level = self.L.level_of(new_lb)
        if level >= self.threshold_level:
            if node in self.L:
                self.L.remove(node)
            self._retire(node, self.L.value(level))
            return False
        if node in self.L:
            self._reprioritize(node)
        else:
            self.L.insert(node, new_lb)
        return True

    # -- terminación ---------------------------------------------------------

    def _threshold_for(self, r):
        """
        El nivel s a partir del cual (inclusive) un bucket puede terminarse.

        El artículo usa s = ceil(log_beta(2 r beta / (beta - 1))), que va de la
        mano con reportar beta**j directamente.  Esta implementación reporta
        beta**j - rad(x) para que el valor sea una cota inferior de todo punto
        del nodo y no solo de su centro (ver `KHausdorff._retire`), y s debe
        ajustarse para conservar la cota superior.

        Barriendo de arriba hacia abajo, al llegar al bucket j todo nodo vivo
        cumple l <= beta**(j+1), así que el Lema 4 da d_h^(i) <= beta**(j+1) + 2r,
        mientras que el valor reportado es delta >= beta**j - r.  Exigiendo
        d_h^(i) <= (1 + eps) delta y usando beta = 1 + eps/2:

            beta**(j+1) + 2r <= (1 + eps)(beta**j - r)
            r (3 + eps)      <= (1 + eps - beta) beta**j = (eps/2) beta**j

        de modo que beta**j >= 2 r (3 + eps) / eps basta, para todo j >= s.
        """
        if r <= 0:
            return -inf
        return ceil(log(2 * r * (3 + self.epsilon) / self.epsilon) / log(self.beta))

    def _sweep(self, threshold):
        """Termina todo bucket ocupado desde `threshold` hacia arriba, de arriba abajo."""
        for level in self.L.levels_at_or_above(threshold):
            value = self.L.value(level)
            for node in self.L.pop_level(level):
                self._retire(node, value)
            # El corte se consulta *entre* buckets y nunca dentro de uno:
            # `pop_level` los saca todos de golpe, así que abandonar a mitad
            # dejaría nodos fuera de la cola sin retirar.
            if self._enough():
                return

    def finish_pending(self, r):
        """
        La condición de terminación de la Sección 5.2: recalcula el umbral para
        el radio actual y termina buckets completos en vez de nodos sueltos.
        """
        self.threshold_level = self._threshold_for(r)
        self._sweep(self.threshold_level)

    def cleanup_all(self):
        """Barre todos los buckets restantes, incluido el centinela."""
        self.threshold_level = -inf
        self._sweep(-inf)


def all_k_hausdorff_bucket(G_A, G_B, epsilon, monotone=True, stop_after=None):
    """
    Devuelve (delta_0, ..., delta_n) con delta_i <= d_h^(i)(A,B) <= (1+eps) delta_i,
    calculado con la variante de cola de buckets beta.  Requiere `epsilon > 0`.

    Con `stop_after=k` corta apenas conoce delta_k; ver `all_k_hausdorff`.
    """
    return KHausdorffBucket(G_A, G_B, epsilon, monotone)(stop_after)


def k_hausdorff_bucket(G_A, G_B, k, epsilon, monotone=True):
    """
    Devuelve la k-ésima distancia de Hausdorff dirigida parcial aproximada.

    Corta el recorrido apenas conoce delta_k.
    """
    if k < 0:
        raise IndexError(f"k must be non-negative, got {k}.")
    distances = all_k_hausdorff_bucket(G_A, G_B, epsilon, monotone, stop_after=k)
    if k >= len(distances):
        raise IndexError(f"k must be in [0, {len(distances)}), got {k}.")
    return distances[k]
