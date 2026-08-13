"""
La variante de la Sección 5.2: una cola de buckets beta como heap de cotas
inferiores.

**Nadie importa este archivo.**  `khausdorff.py` no lo menciona y el CLI tampoco.
Está acá porque es la variante que alcanza el tiempo de ejecución que afirma el
artículo, y hace falta para medirlo:

    from buckets import KHausdorffBucket, all_distances_bucket

    all_distances_bucket(A, B, epsilon=0.5)

`KHausdorff` mantiene un max-heap exacto de cotas inferiores, que cuesta
O(log n) por actualización.  La Sección 5.2 observa que, ya que se está
aproximando de todos modos, las cotas pueden agruparse en buckets geométricos:
las operaciones bajan a tiempo constante y el algoritmo completo a

    (2 + 1/eps)^O(d) n + O(log_beta Delta),      beta = 1 + eps/2.

Los valores que devuelve están dentro de la misma garantía (1 + epsilon) que los
de `KHausdorff`, pero no son idénticos a ellos, y ni siquiera son reproducibles
entre corridas: el orden dentro de un bucket es arbitrario y sale del hash de
los objetos, o sea de las direcciones de memoria.  Eso es deliberado -- no
distinguir dos nodos cuyas cotas difieren en menos de un factor beta es
justamente lo que compra el O(1).  **No compares dos corridas por igualdad
exacta; compará contra las cotas.**
"""

from math import ceil, floor, inf, log

from khausdorff import KHausdorff, _as_points, _trees


class BetaBucketQueue:
    """
    Una cola de prioridad máxima aproximada sobre buckets geométricos de razón
    `beta`: el bucket `m` contiene todo elemento cuya prioridad cae en
    (beta**m, beta**(m+1)].  Solo hay O(log_beta Delta) buckets no vacíos.

    Las prioridades no positivas son válidas: comparten un bucket centinela en
    el nivel -inf cuyo valor representativo es 0.0.  Queda por debajo de todos
    los numerados, así que esos elementos se terminan al final.

    Es una implementación nueva y no un uso de
    `greedypermutation.fvm.bucketqueue.BucketQueue`, que acá no sirve: `insert`
    sobre una cola vacía lanza `ValueError: max() arg is an empty sequence`, y
    su índice de bucket toma `log2` de la prioridad, lo que falla de plano con
    las prioridades no positivas que las cotas inferiores toman rutinariamente.
    """

    SENTINEL = -inf

    def __init__(self, beta, items=(), key=lambda x: x):
        if beta <= 1:
            raise ValueError(f"beta debe ser mayor que 1, y es {beta}.")
        self.beta = beta
        self._log_beta = log(beta)
        self.key = key
        self.buckets = {}  # nivel -> conjunto de elementos; los vacíos se borran
        self.levels = {}  # elemento -> nivel
        for item in items:
            self.insert(item, key(item))

    def level_of(self, priority):
        """El nivel m con beta**m < priority <= beta**(m+1), o el centinela."""
        if priority <= 0:
            return self.SENTINEL
        # floor da el m con beta**m <= priority < beta**(m+1); restar uno en una
        # potencia exacta de beta lo lleva al intervalo semiabierto del artículo.
        m = floor(log(priority) / self._log_beta)
        if self.beta**m >= priority:
            m -= 1
        return m

    def value(self, level):
        """
        El valor representativo de un bucket: su extremo inferior.

        Reportar el extremo inferior es lo que mantiene la salida como una cota
        inferior válida, ya que todo elemento del bucket está por encima de él.
        """
        return 0.0 if level == self.SENTINEL else self.beta**level

    def insert(self, item, priority=None):
        if item in self.levels:
            raise RuntimeError(f"{item!r} ya está en la cola.")
        if priority is None:
            priority = self.key(item)
        level = self.level_of(priority)
        self.levels[item] = level
        self.buckets.setdefault(level, set()).add(item)

    def remove(self, item):
        if item not in self.levels:
            raise RuntimeError(f"{item!r} no está en la cola.")
        level = self.levels.pop(item)
        self.buckets[level].discard(item)
        if not self.buckets[level]:
            del self.buckets[level]

    def changepriority(self, item, priority=None):
        if priority is None:
            priority = self.key(item)
        level = self.level_of(priority)
        if self.levels.get(item) == level:
            return
        self.remove(item)
        self.insert(item, priority)

    def maxlevel(self):
        return max(self.buckets) if self.buckets else None

    def findmax(self):
        """
        Un elemento arbitrario del bucket ocupado más alto.  Solo
        aproximadamente el máximo: cualquier elemento dentro de un factor beta
        del máximo verdadero es una respuesta válida.
        """
        if not self.buckets:
            raise RuntimeError("La cola está vacía.")
        return next(iter(self.buckets[self.maxlevel()]))

    def removemax(self):
        item = self.findmax()
        self.remove(item)
        return item

    def levels_at_or_above(self, threshold):
        """
        Los niveles ocupados j >= threshold, en orden decreciente.  Se
        materializa como lista para que quien llame pueda vaciar buckets
        mientras itera.  Pasar -inf barre la cola entera.
        """
        return sorted((j for j in self.buckets if j >= threshold), reverse=True)

    def pop_level(self, level):
        """Quita y devuelve todos los elementos del bucket `level`."""
        items = self.buckets.pop(level, set())
        for item in items:
            del self.levels[item]
        return items

    def __contains__(self, item):
        return item in self.levels

    def __len__(self):
        return len(self.levels)


class KHausdorffBucket(KHausdorff):
    """
    k-HAUSDORFF con una cola de buckets beta en vez del max-heap exacto.

    Dos cosas cambian respecto de `KHausdorff`, ambas siguiendo la Observación 7
    del artículo, que es lo que garantiza que el barrido descendente visite cada
    bucket una sola vez: los nodos se terminan de a un bucket completo en vez de
    uno por uno, y una actualización de cota que empujaría a un nodo por encima
    del umbral actual lo termina en lugar de moverlo.

    Requiere `epsilon > 0`: con epsilon = 0 la razón beta sería 1 y los buckets
    colapsarían.  Para respuestas exactas, `KHausdorff`.
    """

    def __init__(self, G_A, G_B, epsilon):
        if epsilon <= 0:
            raise ValueError(
                "la variante de buckets necesita epsilon > 0 (beta = 1 + epsilon/2 "
                "debe superar a 1).  Usá KHausdorff para respuestas exactas."
            )
        # Se fijan antes de super().__init__, que construye el heap y registra G_A.
        self.beta = 1 + epsilon / 2
        # El umbral es infinito hasta el primer barrido, para que nada se
        # termine antes de que el recorrido haya empezado.
        self.threshold_level = inf
        super().__init__(G_A, G_B, epsilon)

    def _make_lb_heap(self):
        return BetaBucketQueue(self.beta, key=lambda x: self.lb[x])

    def _reprioritize(self, node):
        self.L.changepriority(node, self.lb[node])

    def _pop_max_lb(self):
        return self.L.findmax()

    def _record(self, node):
        """
        Recalcula l(node) y lo reubica, salvo que la nueva cota haya trepado por
        encima del umbral actual, en cuyo caso el nodo se termina en el acto.
        """
        nueva = self.G.lower_bound(node)
        self.lb[node] = nueva
        level = self.L.level_of(nueva)
        if level >= self.threshold_level:
            if node in self.L:
                self.L.remove(node)
            self._retire(node, self.L.value(level))
            return False
        if node in self.L:
            self._reprioritize(node)
        else:
            self.L.insert(node, nueva)
        return True

    def _threshold_for(self, r):
        """
        El nivel s a partir del cual (inclusive) un bucket puede terminarse.

        El artículo usa s = ceil(log_beta(2 r beta / (beta - 1))), que va con
        reportar beta**j directamente.  Acá se reporta beta**j - rad(x), para
        que el valor acote a todo punto del nodo y no solo a su centro (ver
        `KHausdorff._retire`), y s debe ajustarse para conservar la cota
        superior.  Barriendo de arriba hacia abajo, al llegar al bucket j todo
        nodo vivo cumple l <= beta**(j+1), así que el Lema 4 da
        d_h^(i) <= beta**(j+1) + 2r, mientras lo reportado es
        delta >= beta**j - r.  Pidiendo d_h^(i) <= (1 + eps) delta y usando
        beta = 1 + eps/2:

            beta**(j+1) + 2r <= (1 + eps)(beta**j - r)
            r (3 + eps)      <= (eps/2) beta**j

        de modo que beta**j >= 2 r (3 + eps) / eps basta, para todo j >= s.
        """
        if r <= 0:
            return -inf
        return ceil(log(2 * r * (3 + self.epsilon) / self.epsilon) / log(self.beta))

    def _sweep(self, threshold):
        """Termina todo bucket ocupado desde `threshold` hacia arriba."""
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
        self.threshold_level = self._threshold_for(r)
        self._sweep(self.threshold_level)

    def cleanup_all(self):
        """Barre todos los buckets restantes, incluido el centinela."""
        self.threshold_level = -inf
        self._sweep(-inf)


def all_distances_bucket(A, B, epsilon):
    """La secuencia completa con la variante de buckets.  Requiere epsilon > 0."""
    G_A, G_B = _trees(_as_points(A), _as_points(B))
    return KHausdorffBucket(G_A, G_B, epsilon)()
