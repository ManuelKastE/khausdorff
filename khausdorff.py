"""
Distancia de Hausdorff dirigida parcial: aproximada y exacta.

    python3 khausdorff.py datos/A.csv datos/B.csv --percentil 95

La distancia de Hausdorff dirigida d_h(A, B) mide cuán lejos está el punto de A
más alejado de B.  Un único punto extraviado la domina, así que la versión
robusta descarta los peores valores atípicos antes de medir: el **percentil 95**
es la distancia que queda tras ignorar el 5% de los puntos de A más lejanos.

Este archivo tiene dos formas de calcularla:

  - `hausdorff(A, B, percentil, epsilon)` -- el algoritmo k-HAUSDORFF de la
    Sección 5 de Chubet, Parikh, Sheehy y Sheth, "Approximating the Directed
    Hausdorff Distance" (Computing in Geometry and Topology, 4(2):6:1-6:16,
    2023).  Casi lineal en |A| + |B|, con la garantía

        delta <= respuesta_verdadera <= (1 + epsilon) * delta.

  - `hausdorff_naive(A, B, percentil)` -- fuerza bruta, O(|A| * |B|).  Exacta y
    sin parámetros, para contrastar.

Ambas aceptan rutas a archivos, archivos abiertos o listas de puntos, y ambas
son **dirigidas**: intercambiar A y B da otra respuesta.

La variante de la Sección 5.2 (cola de buckets beta) vive en `buckets.py`, fuera
del camino: da los mismos valores dentro de la garantía y solo cambia el tiempo
de ejecución.
"""

import argparse
import sys
from math import ceil
from os import PathLike
from time import perf_counter

from greedypermutation.balltree import greedy_tree
from greedypermutation.dualtrees.dualtreesearch import DualTreeSearch
from greedypermutation.maxheap import MaxHeap
from greedypermutation.point import Point
from metricspaces import MetricSpace


# --------------------------------------------------------------------------
# Leer puntos de un archivo
# --------------------------------------------------------------------------


def load_points(source):
    """
    Lee un conjunto de puntos y lo devuelve como lista de `Point`.

    `source` es una ruta o un archivo ya abierto.  Un punto por línea, con las
    coordenadas separadas por comas o por espacios -- el separador se
    autodetecta.  Se ignoran las líneas vacías, las que empiezan con `#`, un
    encabezado no numérico como `x,y`, y todo lo que venga después de un `;`.

        # este comentario se ignora
        x,y
        0.1,0.4
        0.7,0.2

    Que todas las filas tengan la misma cantidad de coordenadas no es un
    capricho: `Point.dist` proyecta al subespacio común cuando las dimensiones
    no coinciden, así que a una fila a la que le falte una coordenada no le pasa
    nada visible -- devuelve un resultado equivocado en silencio.  Por eso se
    valida, y el error dice en qué línea.
    """
    if hasattr(source, "read"):
        lineas, nombre = source.readlines(), getattr(source, "name", "<archivo>")
    else:
        nombre = str(source)
        with open(source) as handle:
            lineas = handle.readlines()

    # (numero_de_linea, texto), ya sin comentarios ni líneas vacías.
    utiles = []
    for numero, linea in enumerate(lineas, start=1):
        texto = linea.split(";")[0].strip()
        if texto and not texto.startswith("#"):
            utiles.append((numero, texto))
    if not utiles:
        raise ValueError(f"{nombre}: no contiene ningún punto.")

    separador = "," if "," in utiles[0][1] else None  # None = partir por espacios
    if not _es_numerica(utiles[0][1], separador):
        utiles = utiles[1:]  # era un encabezado
        if not utiles:
            raise ValueError(f"{nombre}: solo contiene un encabezado.")

    puntos, dim = [], None
    for numero, texto in utiles:
        try:
            coords = [float(p) for p in texto.split(separador) if p.strip()]
        except ValueError as error:
            raise ValueError(
                f"{nombre}, línea {numero}: no se pudo leer como número -> {texto!r}"
            ) from error
        if dim is None:
            dim = len(coords)
        if len(coords) != dim:
            raise ValueError(
                f"{nombre}, línea {numero}: tiene {len(coords)} coordenadas y se "
                f"esperaban {dim}.  Las dimensiones desparejas no dan error al "
                f"calcular: devuelven un resultado equivocado en silencio."
            )
        puntos.append(Point(coords))
    return puntos


def _es_numerica(texto, separador):
    try:
        [float(p) for p in texto.split(separador) if p.strip()]
        return True
    except ValueError:
        return False


def _as_points(source):
    """
    Normaliza la entrada a una lista de puntos únicos.

    Acepta una ruta, un archivo abierto, o un iterable de puntos ya construidos.
    Lo de ruta va primero porque un `str` también es iterable, y recorrerlo
    carácter por carácter fallaría mucho más adelante y sin relación con la
    causa.

    Los puntos repetidos se quitan acá y se avisa cuántos.  No es cosmético:
    `greedy_tree` se queda sin centros distintos y le entrega `None` a la
    métrica, lo que sale como `TypeError: 'NoneType' object is not iterable`, un
    error que no dice nada sobre la causa.
    """
    if isinstance(source, (str, PathLike)) or hasattr(source, "read"):
        puntos = load_points(source)
    else:
        puntos = list(source)

    vistos, unicos = set(), []
    for punto in puntos:
        clave = _clave(punto)
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(punto)
    if len(unicos) < len(puntos):
        print(
            f"aviso: se quitaron {len(puntos) - len(unicos)} punto(s) repetido(s).",
            file=sys.stderr,
        )
    return unicos


def _clave(punto):
    """
    Algo hashable que identifique al punto por su valor.

    `Point` se hashea por identidad, así que hay que mirarle las coordenadas.
    Los puntos que no son iterables (`metricspaces.R1` es un `float`) ya se
    comparan por valor y sirven tal cual.
    """
    try:
        return tuple(punto)
    except TypeError:
        return punto


# --------------------------------------------------------------------------
# El heap de cotas inferiores
# --------------------------------------------------------------------------


class LowerBoundHeap(MaxHeap):
    """
    Un max-heap cuyo `remove` preserva el orden del heap.

    Hace falta porque el algoritmo remueve nodos desde el *medio* del heap todo
    el tiempo, y el heap del que hereda no lo admite: en `ds2`, un nivel por
    debajo de `greedypermutation`, `PriorityQueue._remove_at_index` rellena el
    hueco con la última entrada y solo la hunde hacia abajo.  Esa entrada viene
    de un subárbol no relacionado, así que bien puede corresponderle estar por
    encima; cuando así es, el orden se rompe en silencio y `findmax` empieza a
    devolver el nodo equivocado.  El fuzzing lo sitúa en torno al 1% de las
    remociones.
    """

    def changepriority(self, item, priority=None):
        """
        Mueve `item` a `priority`, o a `key(item)` si no se da ninguna.

        `MaxHeap` niega la prioridad en `insert` pero hereda `changepriority`
        sin cambios, de modo que los dos discrepan en el signo y una prioridad
        explícita invierte el orden.  Negar acá los deja consistentes.
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
            # A la entrada que quedó en el hueco puede tocarle subir o bajar.
            # Solo una de las dos llamadas puede tener efecto.
            self._upheap(index)
            self._downheap(index)

    def __contains__(self, item):
        return item in self._itemmap


# --------------------------------------------------------------------------
# k-HAUSDORFF: el algoritmo aproximado
# --------------------------------------------------------------------------


class KHausdorff(DualTreeSearch):
    """
    Aproxima todas las distancias de Hausdorff dirigidas parciales de una vez.

    HAUSDORFF (Sección 4 del artículo) se detiene apenas identifica el punto de
    A aproximadamente más lejano de B.  La idea de k-HAUSDORFF es que si en vez
    de detenerse se descarta ese punto y se sigue, la próxima vez que la
    condición se cumpla aparece el segundo más lejano, y así sucesivamente.

    La maquinaria del recorrido (el grafo de viabilidad `self.G`, el heap de
    radios `self.H`) viene de `DualTreeSearch`.  Sobre ella, esta clase mantiene
    el *heap de cotas inferiores* `self.L`: un max-heap con todo nodo de G_A que
    siga en el grafo, indexado por su cota inferior local l(x).

    Recibe árboles greedy, no puntos.  Para el uso normal están `hausdorff` y
    `all_distances`, que los construyen solos.
    """

    def __init__(self, G_A, G_B, epsilon=0):
        if epsilon < 0:
            raise ValueError(f"epsilon debe ser no negativo, y es {epsilon}.")
        super().__init__(G_A, G_B)
        self.epsilon = epsilon
        self.out = []
        self.stop_after = None  # hasta qué k llegar; lo fija `__call__`
        self.lb = {}  # cotas inferiores locales l(x)
        self.L = self._make_lb_heap()
        self.done = set()  # nodos ya terminados y sacados del grafo
        self._record(G_A)

    # -- heap de cotas inferiores (los ganchos que sobreescribe buckets.py) --

    def _make_lb_heap(self):
        """La clave lee `self.lb`, que es mutable, así que refrescar una
        prioridad es `changepriority(node)` sin pasar valor."""
        return LowerBoundHeap(key=lambda x: self.lb[x])

    def _record(self, node):
        """
        Calcula l(node) y lo (re)ubica en el heap de cotas inferiores.

        Devuelve True si `node` sigue en el grafo después de eso.  Acá siempre
        lo está; la variante de buckets puede terminar un nodo en el acto.
        """
        nueva = self.G.lower_bound(node)
        if node in self.lb:
            self.lb[node] = nueva
            self._reprioritize(node)
        else:
            self.lb[node] = nueva
            self.L.insert(node)
        return True

    def _reprioritize(self, node):
        self.L.changepriority(node)

    def _pop_max_lb(self):
        """El nodo de G_A con la mayor cota inferior local."""
        return self.L.findmax()

    # -- emitir valores ------------------------------------------------------

    def _emit(self, value, count):
        """
        Agrega `value` a la salida una vez por punto, sin dejar que suba.

        Recortar hacia abajo siempre es seguro: la garantía es
        delta_i <= d_h^(i), y la secuencia verdadera es no creciente.
        """
        if self.out:
            value = min(value, self.out[-1])
        self.out.extend([value] * count)

    def _retire(self, x, value=None):
        """
        Termina `x`, que ya fue sacado del heap de cotas inferiores.

        Lo que se reporta para cada punto de pts(x) es l(x) - rad(x), no l(x).
        El artículo reporta l(x), pero l(x) solo acota inferiormente
        d(ctr(x), B): un punto de pts(x) más cercano a B que el centro recibiría
        una distancia demasiado grande, rompiendo delta_i <= d_h^(i).  Restar el
        radio vale para *todo* punto del nodo, ya que
        d(p, B) >= d(ctr(x), B) - rad(x) >= l(x) - rad(x).  Lo que eso cuesta en
        precisión lo paga `_finish_constant`.
        """
        cota = self.lb[x] if value is None else value
        self._emit(max(0.0, cota - x.radius), len(x))
        del self.lb[x]
        self.G.remove(x)
        self.done.add(x)

    def _finish(self, x, value=None):
        """Saca `x` del heap de cotas inferiores y lo termina."""
        self.L.remove(x)
        self._retire(x, value)

    # -- terminación ---------------------------------------------------------

    def _enough(self):
        """
        True si ya se emitió el delta_k pedido y se puede abandonar el recorrido.

        Es correcto porque `_emit` solo agrega al final: nunca reescribe una
        entrada ya emitida, así que out[k] queda fijo apenas se escribe y cortar
        después no puede alterarlo.
        """
        return self.stop_after is not None and len(self.out) > self.stop_after

    def _finish_constant(self):
        """
        La constante c de la condición de terminación r <= c * l(x).

        El artículo usa c = eps/2, que asegura la cota superior pero no la
        inferior (ver `_retire`).  Reportar l(x) - rad(x) arregla la inferior a
        costa de precisión, así que c debe achicarse para conservar la superior.
        Con L = l(x) el tope del heap, el Lema 4 da d_h^(i) <= L + 2r, y lo
        reportado es delta = L - rad(x) >= (1-c)L, de modo que

            d_h^(i) <= (1 + 2c) L <= (1 + 2c)/(1 - c) * delta.

        Pedir (1 + 2c)/(1 - c) <= 1 + eps da c <= eps/(3 + eps).
        """
        return self.epsilon / (3 + self.epsilon)

    def finish_pending(self, r):
        """
        La condición de terminación, con `r` el radio del nodo por procesarse:
        termina el tope del heap de cotas inferiores mientras r <= c * l(x).
        """
        c = self._finish_constant()
        while len(self.L):
            x = self._pop_max_lb()
            # Escrito así y no como r / l(x) para que epsilon == 0 y l(x) == 0
            # sean ambos casos válidos.
            if r > c * self.lb[x]:
                return
            self._finish(x)
            if self._enough():
                return

    def cleanup_all(self):
        """
        Termina todo lo que quede, con r = 0.

        Vaciar el heap en orden decreciente de l(x), en vez de iterar sobre
        `self.G.A` como hace `DualTreeSearch.cleanup`, es lo que mantiene
        ordenada la secuencia de salida.
        """
        while len(self.L):
            self._finish(self._pop_max_lb())
            if self._enough():
                return

    # -- ganchos de DualTreeSearch -------------------------------------------

    def setup_children(self, ball):
        """
        Se llama al dividir un `ball` del lado A, después de agregar sus hijos
        como vértices pero *antes* de que existan sus aristas, así que acá no se
        puede calcular ninguna cota inferior.  Lo único que se hace es retirar
        al padre.
        """
        self.L.remove(ball)
        del self.lb[ball]

    def update(self, node, ball):
        """Refresca el vértice `node` del lado A afectado tras dividir `ball`."""
        if self._record(node):
            self._prune(node)

    def _prune(self, a):
        """
        Elimina las aristas salientes de `a` que no pueden contener al vecino
        más cercano de ningún punto de `a`.

        Es la regla de poda de `NNBallGraph.prune` del módulo `hausdorff` de la
        dependencia, que `ViabilityGraph` no ofrece.  El vecino que realiza
        `mindist` siempre sobrevive, así que `self.G.A[a]` nunca queda vacío y
        `lower_bound(a)` siempre está bien definido.
        """
        self.G.update_mindist(a)
        umbral = self.G.mindist[a] + 2 * a.radius
        for b in [b for b in self.G.A[a] if a.dist(b.center) - b.radius > umbral]:
            self.G.remove_edge(a, b)

    def _skip(self, ball):
        """
        True si `ball` ya no forma parte del grafo de viabilidad.

        Dos casos.  Un nodo del lado A ya terminado desapareció de `self.G`, y
        `DualTreeSearch.iteration` lo buscaría en `self.G.B` lanzando KeyError.
        Un nodo del lado B con vecindad vacía nunca podrá recuperar una arista,
        así que dividirlo es puro desperdicio.
        """
        if ball in self.done:
            return True
        if ball in self.G.B and not self.G.B[ball]:
            del self.G.B[ball]
            return True
        return False

    # -- bucle principal -----------------------------------------------------

    def __call__(self, stop_after=None):
        """
        Corre el recorrido y devuelve (delta_0, ..., delta_n), de largo n + 1.

        Con `stop_after=k` abandona el recorrido apenas conoce delta_k y
        devuelve solo (delta_0, ..., delta_k).  Los valores son idénticos a los
        del recorrido completo; lo único que cambia es cuánto trabajo se hace, y
        cuánto se ahorra depende de epsilon (con epsilon = 0 la condición de
        terminación nunca se cumple y hay que recorrer todo igual).

        No se reutiliza `DualTreeSearch.__call__` porque la condición hay que
        evaluarla al *inicio* de cada iteración y porque hay que saltearse los
        nodos ya terminados.
        """
        self.stop_after = stop_after

        for ball in self.H:
            self.finish_pending(ball.radius)
            if self._enough():
                return self.out[: stop_after + 1]
            if ball.isleaf():
                # El heap está ordenado por radio decreciente: de acá en
                # adelante todo nodo sobreviviente es una hoja de radio 0.
                break
            if self._skip(ball):
                continue
            self.iteration(ball)

        self.cleanup_all()
        if self._enough():
            return self.out[: stop_after + 1]

        # El caso k = n: A^(n) = {conjunto vacío} y d_h(vacío, B) = 0 por
        # convención.  El artículo incluye este valor.
        self.out.append(0.0)
        return self.out if stop_after is None else self.out[: stop_after + 1]


def _k_for_percentile(n, q):
    """
    El k tal que delta_k es el percentil `q` de las `n` distancias d(a, B).

    Se usa la convención *nearest-rank*: con las distancias ordenadas de forma
    ascendente en v[0..n-1], el percentil q es v[ceil(q*n/100) - 1].  Como
    delta_k = v[n-1-k], eso da k = n - ceil(q*n/100).  Los extremos salen solos:
    q = 100 da k = 0, la distancia de Hausdorff dirigida ordinaria, y q = 0 da
    k = n, el 0 final de la secuencia.

    No interpolar entre dos delta_k -- como sí hace `numpy.percentile` por
    omisión -- es lo que importa: el resultado es siempre uno de los valores que
    el algoritmo ya calculó, así que hereda intacta la garantía (1 + epsilon).
    """
    if not 0 <= q <= 100:
        raise ValueError(f"el percentil debe estar en [0, 100], y es {q}.")
    return n - ceil(q * n / 100)


def _trees(A, B):
    """
    Un par de árboles greedy.

    Se pueden reusar entre búsquedas: `Ball` se arma una vez en `greedy_tree` y
    la búsqueda solo lo lee.  Todo lo que se muta -- el grafo de viabilidad, el
    heap de radios, el de cotas inferiores -- lo crea de cero cada `KHausdorff`.
    Construir el árbol es lo más caro de todo, así que para varias consultas
    sobre los mismos puntos conviene guardarlo y pasárselo a `KHausdorff`
    directamente en vez de llamar a `hausdorff`, que lo reconstruye cada vez.
    """
    return greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B))


def hausdorff(A, B, percentile=100, epsilon=0):
    """
    El percentil `percentile` de las distancias d(a, B), aproximado.

        hausdorff("datos/A.csv", "datos/B.csv")          # la distancia dirigida
        hausdorff(A, B, percentile=95, epsilon=0.1)      # descartando el 5%

    `A` y `B` son rutas, archivos abiertos o listas de puntos.  Con
    `epsilon=0` la respuesta es exacta, al costo de recorrer todo; con
    `epsilon>0` vale delta <= verdadera <= (1 + epsilon) * delta, y el recorrido
    se abandona apenas se conoce el valor pedido.
    """
    A, B = _as_points(A), _as_points(B)
    k = _k_for_percentile(len(A), percentile)
    G_A, G_B = _trees(A, B)
    return KHausdorff(G_A, G_B, epsilon)(stop_after=k)[k]


def all_distances(A, B, epsilon=0):
    """
    La secuencia completa (delta_0, ..., delta_n), aproximada.

    delta_k es la distancia tras descartar los k puntos de A más lejanos de B.
    Tiene n + 1 valores y es no creciente; el último es 0, porque descartar los
    n puntos de A no deja nada que medir.
    """
    G_A, G_B = _trees(_as_points(A), _as_points(B))
    return KHausdorff(G_A, G_B, epsilon)()


# --------------------------------------------------------------------------
# Fuerza bruta: la referencia exacta
# --------------------------------------------------------------------------


def nearest_distances(A, B):
    """La lista de d(a, B) = min_{b en B} d(a, b) para cada a en A."""
    A, B = _as_points(A), _as_points(B)
    if not B:
        raise ValueError("B debe contener al menos un punto.")
    return [min(a.dist(b) for b in B) for a in A]


def hausdorff_naive(A, B, percentile=100):
    """
    El percentil `percentile` de las distancias d(a, B), exacto.

    Cuesta O(|A| * |B|), así que solo es usable en entradas chicas, pero no
    necesita preprocesamiento ni parámetro de aproximación.  Es la verdad de
    referencia contra la que se contrasta `hausdorff`.
    """
    distancias = sorted(nearest_distances(A, B), reverse=True) + [0.0]
    return distancias[_k_for_percentile(len(distancias) - 1, percentile)]


# --------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Distancia de Hausdorff dirigida parcial entre dos archivos "
        "de puntos.  Es dirigida: el primer archivo es A y el segundo es B, e "
        "intercambiarlos da otra respuesta.",
        epilog="Un punto por línea, coordenadas separadas por comas o espacios.",
    )
    parser.add_argument("A", help="archivo con los puntos de A")
    parser.add_argument("B", help="archivo con los puntos de B")
    parser.add_argument(
        "--percentil",
        type=float,
        default=100.0,
        help="qué valor pedir: 95 descarta el 5%% de los puntos de A más lejanos "
        "de B; 100 (por omisión) es la distancia de Hausdorff dirigida",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="aproximación del algoritmo rápido; 0 (por omisión) lo hace exacto",
    )
    parser.add_argument(
        "--modo",
        choices=("ambos", "aprox", "naive"),
        default="ambos",
        help="qué calcular: 'ambos' (por omisión) corre el algoritmo rápido y la "
        "fuerza bruta para poder contrastarlos, 'naive' cuesta O(|A|*|B|)",
    )
    args = parser.parse_args(argv)

    if not 0 <= args.percentil <= 100:
        raise SystemExit(f"error: el percentil debe estar en [0, 100], y es {args.percentil}.")

    A, B = _as_points(args.A), _as_points(args.B)
    if len(A[0]) != len(B[0]):
        raise SystemExit(
            f"error: los puntos de A tienen {len(A[0])} coordenadas y los de B "
            f"{len(B[0])}.  Deben coincidir."
        )
    print(
        f"|A| = {len(A)}, |B| = {len(B)}, dimensión = {len(A[0])}, "
        f"percentil {args.percentil:g}, epsilon {args.epsilon:g}\n"
    )

    aprox = exacto = None
    if args.modo in ("ambos", "aprox"):
        inicio = perf_counter()
        aprox = hausdorff(A, B, args.percentil, args.epsilon)
        print(f"  aproximado : {aprox:10.6f}   [{perf_counter() - inicio:6.3f} s]")
    if args.modo in ("ambos", "naive"):
        inicio = perf_counter()
        exacto = hausdorff_naive(A, B, args.percentil)
        print(f"  exacto     : {exacto:10.6f}   [{perf_counter() - inicio:6.3f} s]  fuerza bruta")

    if aprox is not None and exacto is not None:
        razon = exacto / aprox if aprox > 0 else 1.0
        print(f"  razón      : {razon:10.4f}   (la garantía pide <= {1 + args.epsilon:g})")
        if not (aprox <= exacto + 1e-9 and exacto <= (1 + args.epsilon) * aprox + 1e-9):
            print("\n  ATENCIÓN: la garantía no se respeta", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
