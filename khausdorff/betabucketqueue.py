"""
Una cola de buckets beta: la cola de prioridad máxima aproximada de la Sección
5.2 de Chubet, Parikh, Sheehy y Sheth (2023).

Los elementos se agrupan en buckets geométricos: el bucket `m` contiene todo
elemento cuya prioridad cae en (beta**m, beta**(m+1)].  Solo hay
O(log_beta Delta) buckets no vacíos, y todas las operaciones que k-HAUSDORFF
necesita corren en tiempo constante salvo el barrido descendente, que visita
cada bucket a lo sumo una vez.

El orden dentro de un bucket es arbitrario, y eso es exactamente lo que compra
la velocidad: el algoritmo nunca necesita distinguir dos nodos cuyas cotas
inferiores coinciden salvo por un factor beta.

Esta es una implementación nueva y no una reutilización de
`greedypermutation.fvm.bucketqueue.BucketQueue`, que no puede usarse aquí:
`insert` sobre una cola vacía lanza `ValueError: max() arg is an empty sequence`,
y su índice de bucket toma `log2` de la prioridad, lo que falla de plano con las
prioridades no positivas que las cotas inferiores locales toman rutinariamente.
"""

from math import floor, inf, log


class BetaBucketQueue:
    """
    Una cola de prioridad máxima aproximada sobre buckets geométricos de razón
    `beta`.

    Las prioridades no positivas son válidas: todas comparten un bucket
    centinela en el nivel -inf cuyo valor representativo es 0.0.  Ese bucket
    queda por debajo de todos los numerados, así que esos elementos siempre se
    terminan al final.
    """

    SENTINEL = -inf

    def __init__(self, beta, items=(), key=lambda x: x):
        if beta <= 1:
            raise ValueError(f"beta must be greater than 1, got {beta}.")
        self.beta = beta
        self._log_beta = log(beta)
        self.key = key
        self.buckets = {}  # nivel -> conjunto de elementos; los vacíos se borran
        self.levels = {}  # elemento -> nivel
        for item in items:
            self.insert(item, key(item))

    # -- niveles y valores ---------------------------------------------------

    def level_of(self, priority):
        """
        Devuelve el nivel de bucket de `priority`, es decir el m con
        beta**m < priority <= beta**(m+1), o el centinela si priority <= 0.
        """
        if priority <= 0:
            return self.SENTINEL
        # floor da el m con beta**m <= priority < beta**(m+1); restar uno en una
        # potencia exacta de beta lo lleva al intervalo semiabierto que usa el
        # artículo, (beta**m, beta**(m+1)].
        m = floor(log(priority) / self._log_beta)
        if self.beta**m >= priority:
            m -= 1
        return m

    def value(self, level):
        """
        El valor representativo de un bucket: su extremo inferior.

        Reportar el extremo inferior es lo que mantiene la salida del algoritmo
        como una cota inferior válida, ya que todo elemento del bucket tiene una
        prioridad por encima de él.
        """
        return 0.0 if level == self.SENTINEL else self.beta**level

    # -- operaciones de la cola ----------------------------------------------

    def insert(self, item, priority=None):
        """Inserta `item` en el bucket de `priority` (o de `key(item)`)."""
        if item in self.levels:
            raise RuntimeError(f"{item!r} is already in the queue.")
        if priority is None:
            priority = self.key(item)
        level = self.level_of(priority)
        self.levels[item] = level
        self.buckets.setdefault(level, set()).add(item)

    def remove(self, item):
        """Quita `item` de la cola."""
        if item not in self.levels:
            raise RuntimeError(f"{item!r} is not in the queue.")
        level = self.levels.pop(item)
        bucket = self.buckets[level]
        bucket.discard(item)
        if not bucket:
            del self.buckets[level]

    def changepriority(self, item, priority=None):
        """Mueve `item` al bucket que corresponde a su nueva prioridad."""
        if priority is None:
            priority = self.key(item)
        level = self.level_of(priority)
        if self.levels.get(item) == level:
            return
        self.remove(item)
        self.insert(item, priority)

    def maxlevel(self):
        """El nivel ocupado más alto, o None si la cola está vacía."""
        return max(self.buckets) if self.buckets else None

    def findmax(self):
        """
        Un elemento arbitrario del bucket ocupado más alto.

        Solo aproximadamente el máximo: cualquier elemento que esté dentro de un
        factor beta del máximo verdadero es una respuesta válida.
        """
        if not self.buckets:
            raise RuntimeError("The queue is empty.")
        return next(iter(self.buckets[self.maxlevel()]))

    def removemax(self):
        """Quita y devuelve un elemento del bucket ocupado más alto."""
        item = self.findmax()
        self.remove(item)
        return item

    def levels_at_or_above(self, threshold):
        """
        Los niveles ocupados `j >= threshold`, en orden decreciente.

        Se materializa como lista para que quien llame pueda vaciar buckets
        mientras itera.  Pasar `-inf` para barrer la cola entera.
        """
        return sorted((j for j in self.buckets if j >= threshold), reverse=True)

    def pop_level(self, level):
        """Quita y devuelve, como conjunto, todos los elementos del bucket `level`."""
        items = self.buckets.pop(level, set())
        for item in items:
            del self.levels[item]
        return items

    def __contains__(self, item):
        return item in self.levels

    def __len__(self):
        return len(self.levels)
