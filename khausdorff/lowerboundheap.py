"""
Un max-heap exacto de cotas inferiores locales, con un `remove` que funciona.

k-HAUSDORFF remueve nodos desde el medio del heap de cotas inferiores todo el
tiempo: cada vez que se divide un nodo del lado A se retira a su padre, y cada
vez que se dispara la condición de terminación se recorta un nodo.  El heap del
que hereda no admite eso correctamente, de ahí esta subclase.

El bug está en `ds2`, un nivel por debajo de `greedypermutation`.
`PriorityQueue._remove_at_index` rellena el hueco con la última entrada y luego
solo la hunde hacia *abajo*:

    def _remove_at_index(self, index):
        L = self._entries
        self._swap(index, len(L) - 1)
        del self._itemmap[L[-1].item]
        L.pop()
        self._downheap(index)          # <- una entrada que debe subir no puede

La entrada que se mueve al hueco viene de un subárbol no relacionado, así que
bien puede corresponderle estar *por encima* de él.  Cuando así es, el orden del
heap se rompe silenciosamente y `findmax` empieza a devolver el nodo equivocado.
El fuzzing lo sitúa en torno al 1% de las remociones sobre entradas aleatorias,
más que suficiente para desordenar la salida.

Nada más dentro de `greedypermutation` llama a `remove`, que es presumiblemente
por qué el bug pasó inadvertido; `dist_H` y `DualTreeSearch` solo insertan y
extraen el máximo.
"""

from greedypermutation.maxheap import MaxHeap


class LowerBoundHeap(MaxHeap):
    """Un `MaxHeap` cuyo `remove` preserva el orden del heap."""

    def changepriority(self, item, priority=None):
        """
        Mueve `item` a `priority`, o a `key(item)` si no se da ninguna.

        `MaxHeap` niega la prioridad en `insert` pero hereda `changepriority`
        sin cambios, de modo que en el original ambos discrepan en el signo y
        una prioridad explícita invierte el orden en silencio.  Negar aquí los
        deja consistentes.
        """
        if priority is not None:
            priority = -priority
        super().changepriority(item, priority)

    def _remove_at_index(self, index):
        entries = self._entries
        self._swap(index, len(entries) - 1)
        del self._itemmap[entries[-1].item]
        entries.pop()
        if index < len(entries):
            # A la entrada que se intercambió al hueco puede corresponderle
            # tanto subir como bajar.  Solo una de estas dos llamadas puede
            # tener efecto, igual que hace `changepriority` ante un cambio de
            # prioridad arbitrario.
            self._upheap(index)
            self._downheap(index)

    def __contains__(self, item):
        return item in self._itemmap
