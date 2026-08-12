"""
k-HAUSDORFF: aproxima la k-ésima distancia de Hausdorff dirigida parcial para
todo k a la vez, en un único recorrido de árbol dual.

Implementa la Sección 5 de

    O. A. Chubet, P. M. Parikh, D. R. Sheehy y S. S. Sheth,
    "Approximating the Directed Hausdorff Distance",
    Computing in Geometry and Topology, 4(2):6:1-6:16, 2023.

El algoritmo HAUSDORFF de la Sección 4 se detiene apenas identifica el punto de
A que está (aproximadamente) más lejos de B.  La observación detrás de
k-HAUSDORFF es que si, en lugar de detenerse, se descarta ese punto y se sigue,
la próxima vez que la condición se cumpla habremos encontrado el segundo punto
más lejano, y así sucesivamente.  Llevar el recorrido hasta el final produce por
lo tanto la secuencia completa

    (delta_0, ..., delta_{n-1})    con    delta_i <= d_h^(i)(A,B) <= (1+eps) delta_i,

donde delta_0 es la distancia de Hausdorff dirigida ordinaria.

Este módulo contiene la variante de referencia, que usa un max-heap exacto como
heap de cotas inferiores.  `khausdorff.bucketkhausdorff` contiene la variante de
la Sección 5.2, que lo reemplaza por una cola de buckets beta para alcanzar el
tiempo de ejecución que afirma el artículo.
"""

from greedypermutation.dualtrees.dualtreesearch import DualTreeSearch

from khausdorff.lowerboundheap import LowerBoundHeap


class KHausdorff(DualTreeSearch):
    """
    Aproxima todas las distancias de Hausdorff dirigidas parciales d_h^(k)(A, B).

    La maquinaria del recorrido (grafo de viabilidad `self.G`, heap de radios
    `self.H`) viene de `DualTreeSearch`.  Sobre ella, esta clase mantiene el
    *heap de cotas inferiores* `self.L`: un max-heap que contiene todo nodo de
    G_A actualmente en el grafo de viabilidad, indexado por su cota inferior
    local l(x).
    """

    def __init__(self, G_A, G_B, epsilon=0, monotone=True):
        """
        `G_A` y `G_B` son árboles greedy (`greedypermutation.balltree.Ball`),
        `epsilon >= 0` es el parámetro de aproximación (0 da respuestas exactas)
        y `monotone` fuerza a que la salida sea no creciente.
        """
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
        super().__init__(G_A, G_B)
        self.epsilon = epsilon
        self.monotone = monotone
        self.out = []
        # Cotas inferiores locales l(x), para cada nodo x de G_A en el grafo.
        self.lb = {}
        self.L = self._make_lb_heap()
        # Nodos ya terminados y removidos del grafo de viabilidad.
        self.done = set()
        self._record(G_A)

    # -- heap de cotas inferiores -------------------------------------------

    def _make_lb_heap(self):
        """
        Construye el heap de cotas inferiores.  Su clave lee `self.lb`, que es
        mutable, así que refrescar una prioridad es simplemente
        `changepriority(node)` sin pasar valor.

        Se usa `LowerBoundHeap` y no el `MaxHeap` de la dependencia porque este
        algoritmo remueve nodos desde el medio del heap, algo que la
        implementación original hace mal; ver ese módulo.
        """
        return LowerBoundHeap(key=lambda x: self.lb[x])

    def _record(self, node):
        """
        Calcula l(node) y (re)ubica `node` en el heap de cotas inferiores.

        Devuelve True si `node` sigue en el grafo después de eso.  Aquí siempre
        lo está; la variante de buckets puede terminar un nodo en el acto.
        """
        new_lb = self.G.lower_bound(node)
        if node in self.lb:
            self.lb[node] = new_lb
            self._reprioritize(node)
        else:
            self.lb[node] = new_lb
            self.L.insert(node)
        return True

    def _reprioritize(self, node):
        """Refresca la posición de `node` tras cambiar `self.lb[node]`."""
        self.L.changepriority(node)

    def _pop_max_lb(self):
        """Devuelve el nodo de G_A con la mayor cota inferior local."""
        return self.L.findmax()

    # -- grafo de viabilidad -------------------------------------------------

    def _prune(self, a):
        """
        Elimina las aristas salientes de `a` que no pueden contener al vecino
        más cercano de ningún punto de `a`.

        Esta es la regla de poda de `NNBallGraph.prune` del módulo `hausdorff`
        de la dependencia, que `ViabilityGraph` no ofrece.  El vecino que
        realiza `mindist` siempre sobrevive, de modo que `self.G.A[a]` nunca
        queda vacío y `lower_bound(a)` siempre está bien definido.
        """
        self.G.update_mindist(a)
        threshold = self.G.mindist[a] + 2 * a.radius
        for b in [b for b in self.G.A[a] if a.dist(b.center) - b.radius > threshold]:
            self.G.remove_edge(a, b)

    # -- terminación ---------------------------------------------------------

    def _emit(self, value, count):
        """Agrega `value` a la salida una vez por punto, manteniendo monotonía."""
        if self.monotone and self.out:
            # Recortar hacia abajo siempre es seguro: la garantía es
            # delta_i <= d_h^(i), y la secuencia d_h^(i) es no creciente.
            value = min(value, self.out[-1])
        self.out.extend([value] * count)

    def _finish(self, x, value=None):
        """
        Termina el nodo `x`: emite su cota inferior una vez por cada punto de
        pts(x), y luego lo saca del heap de cotas inferiores y del grafo de
        viabilidad.
        """
        self.L.remove(x)
        self._retire(x, value)

    def _retire(self, x, value=None):
        """
        Termina `x` cuando *ya* fue sacado del heap de cotas inferiores.

        El valor reportado para cada punto de pts(x) es l(x) - rad(x), no l(x).
        El artículo reporta l(x), pero l(x) solo acota inferiormente
        d(ctr(x), B); un punto de pts(x) que quede más cerca de B que el centro
        recibiría entonces una distancia demasiado grande, rompiendo
        delta_i <= d_h^(i).  Restar el radio da una cota válida para *todo*
        punto del nodo, ya que d(p, B) >= d(ctr(x), B) - rad(x) >= l(x) - rad(x).
        Ver `_finish_constant` para cómo lo paga la condición de terminación.
        """
        bound = self.lb[x] if value is None else value
        self._emit(max(0.0, bound - x.radius), len(x))
        del self.lb[x]
        self.G.remove(x)
        self.done.add(x)

    def _finish_constant(self):
        """
        La constante c de la condición de terminación r <= c * l(x).

        El artículo usa c = eps/2, que asegura la cota superior
        d_h^(i) <= (1+eps) delta_i pero no la cota inferior delta_i <= d_h^(i)
        (ver `_retire`).  Reportar l(x) - rad(x) en vez de l(x) arregla la cota
        inferior pero cuesta precisión, así que c debe achicarse para conservar
        la cota superior.  Escribiendo L = l(x) para el tope del heap, el Lema 4
        del artículo da d_h^(i) <= L + 2r, y el valor reportado es
        delta = L - rad(x) >= (1-c)L, de modo que

            d_h^(i) <= (1 + 2c) L <= (1 + 2c)/(1 - c) * delta.

        Exigir (1 + 2c)/(1 - c) <= 1 + eps da c <= eps/(3 + eps).
        """
        return self.epsilon / (3 + self.epsilon)

    def finish_pending(self, r):
        """
        La condición de terminación de la Sección 5.2.

        Con `r` el radio del nodo que está por procesarse, termina el tope del
        heap de cotas inferiores mientras r <= c * l(x).
        """
        c = self._finish_constant()
        while len(self.L):
            x = self._pop_max_lb()
            # Escrito así y no como r / l(x) para que epsilon == 0 y l(x) == 0
            # sean ambos casos válidos.
            if r > c * self.lb[x]:
                return
            self._finish(x)

    def cleanup_all(self):
        """
        Termina todo lo que quede, con r = 0.

        Vaciar el heap de cotas inferiores en orden decreciente de l(x), en vez
        de iterar sobre `self.G.A` como hace `DualTreeSearch.cleanup`, es lo que
        mantiene ordenada la secuencia de salida.
        """
        while len(self.L):
            self._finish(self._pop_max_lb())

    # -- ganchos de DualTreeSearch -------------------------------------------

    def setup_children(self, ball):
        """
        Se llama cuando se divide un `ball` del lado A, después de agregar sus
        hijos como vértices pero *antes* de que existan sus aristas, así que
        aquí no se puede calcular ninguna cota inferior.  Lo único que hacemos
        es retirar al padre.
        """
        self.L.remove(ball)
        del self.lb[ball]

    def update(self, node, ball):
        """Refresca el vértice `node` del lado A afectado tras dividir `ball`."""
        if self._record(node):
            self._prune(node)

    # -- bucle principal -----------------------------------------------------

    def _skip(self, ball):
        """
        True si `ball` ya no forma parte del grafo de viabilidad y no debe
        dividirse.

        Hay dos casos.  Un nodo del lado A ya terminado desapareció de `self.G`,
        y `DualTreeSearch.iteration` lo buscaría en `self.G.B` lanzando un
        KeyError.  Un nodo del lado B cuya vecindad quedó vacía nunca podrá
        recuperar una arista, así que dividirlo es puro desperdicio.
        """
        if ball in self.done:
            return True
        if ball in self.G.B and not self.G.B[ball]:
            del self.G.B[ball]
            return True
        return False

    def __call__(self):
        """
        Ejecuta el recorrido hasta el final y devuelve la lista de distancias
        parciales aproximadas (delta_0, ..., delta_n), de largo n + 1.

        No se reutiliza `DualTreeSearch.__call__` porque la condición de
        terminación debe evaluarse al *inicio* de cada iteración, y porque hay
        que saltarse los nodos ya terminados.
        """
        for ball in self.H:
            self.finish_pending(ball.radius)
            if ball.isleaf():
                # El heap está ordenado por radio decreciente, así que de aquí
                # en adelante todo nodo sobreviviente es una hoja de radio 0.
                break
            if self._skip(ball):
                continue
            self.iteration(ball)

        self.cleanup_all()
        # El caso k = n: A^(n) = {conjunto vacío}, y d_h(vacío, B) = 0 por
        # convención.  El artículo incluye este valor -- su salida es una lista
        # de n + 1 elementos, con k recorriendo {0, ..., n}.
        self.out.append(0.0)
        return self.out


def all_k_hausdorff(G_A, G_B, epsilon=0, monotone=True):
    """
    Devuelve (delta_0, ..., delta_n) con delta_i <= d_h^(i)(A,B) <= (1+eps) delta_i.

    La lista tiene n + 1 elementos, uno por cada k en {0, ..., n} con n = |A|.
    El último, delta_n, es 0: descartar los n puntos de A no deja nada que medir.

    `G_A` y `G_B` son árboles greedy, como los que produce
    `greedypermutation.balltree.greedy_tree`.  Con `epsilon=0` el resultado es
    exacto.  Nótese que, igual que la distancia de Hausdorff subyacente, esto es
    dirigido: intercambiar los argumentos da una respuesta distinta.

    Para indexar por percentil en vez de por cantidad de descartes, ver
    `khausdorff.percentile`.
    """
    return KHausdorff(G_A, G_B, epsilon, monotone)()


def k_hausdorff(G_A, G_B, k, epsilon=0, monotone=True):
    """Devuelve la k-ésima distancia de Hausdorff dirigida parcial aproximada."""
    distances = all_k_hausdorff(G_A, G_B, epsilon, monotone)
    if not 0 <= k < len(distances):
        raise IndexError(f"k must be in [0, {len(distances)}), got {k}.")
    return distances[k]
