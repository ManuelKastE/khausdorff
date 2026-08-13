import random
import unittest

from greedypermutation.hausdorff import dist_H
from greedypermutation.maxheap import MaxHeap
from metricspaces import MetricSpace
from greedypermutation.balltree import greedy_tree

from khausdorff import (
    KHausdorff,
    LowerBoundHeap,
    _k_for_percentile,
    all_distances,
    hausdorff,
    hausdorff_naive,
)
from tests.helpers import TOL, line, random_pair, seeded

EPSILONS = (0.1, 0.5, 1.0)


def naive_sequence(A, B):
    """La secuencia exacta completa, por fuerza bruta."""
    return sorted((min(a.dist(b) for b in B) for a in A), reverse=True) + [0.0]


class Comprobaciones:
    """Lo que toda corrida debe cumplir, sea cual sea epsilon."""

    def assert_valida(self, got, exact, epsilon):
        self.assertEqual(len(got), len(exact), "un valor por punto de A, más el k = n")
        for i in range(len(got) - 1):
            self.assertGreaterEqual(got[i] + TOL, got[i + 1], f"sube en {i}")
        for i, (aprox, true) in enumerate(zip(got, exact)):
            self.assertLessEqual(aprox, true + TOL, f"delta_{i} pasa a d_h^({i})")
            self.assertLessEqual(
                true, (1 + epsilon) * aprox + TOL, f"d_h^({i}) pasa la cota superior"
            )


class TestExacto(unittest.TestCase, Comprobaciones):
    """Con epsilon = 0 hay que reproducir la respuesta de la fuerza bruta."""

    def test_ejemplos_en_la_recta(self):
        for a, b, esperado in [
            ([1, 2, 3, 6, 7, 8, 12], [1, 2, 3, 6, 7, 9], [3, 1, 0, 0, 0, 0, 0, 0]),
            ([1, 2, 3, 6, 7, 8], [1, 2, 3, 6, 7, 9], [1, 0, 0, 0, 0, 0, 0]),
        ]:
            self.assertEqual(all_distances(line(a), line(b)), esperado)

    def test_delta_0_es_la_distancia_de_hausdorff(self):
        rng = seeded(3)
        for _ in range(15):
            A, B = random_pair(rng)
            esperado = dist_H(greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B)))
            self.assertAlmostEqual(all_distances(A, B)[0], esperado)

    def test_coincide_con_la_fuerza_bruta(self):
        rng = seeded(4)
        for _ in range(25):
            A, B = random_pair(rng)
            exacto = naive_sequence(A, B)
            got = all_distances(A, B)
            for aprox, true in zip(got, exacto):
                self.assertAlmostEqual(aprox, true)
            self.assert_valida(got, exacto, 0)

    def test_coincide_en_mas_dimensiones(self):
        rng = seeded(5)
        for dim in (1, 3, 5):
            A, B = random_pair(rng, max_a=30, max_b=20, dim=dim)
            for aprox, true in zip(all_distances(A, B), naive_sequence(A, B)):
                self.assertAlmostEqual(aprox, true)


class TestAproximacion(unittest.TestCase, Comprobaciones):
    """Debe valer delta_i <= d_h^(i) <= (1+eps) delta_i."""

    def test_nubes_al_azar(self):
        rng = seeded(6)
        for _ in range(25):
            A, B = random_pair(rng)
            exacto = naive_sequence(A, B)
            for epsilon in EPSILONS:
                self.assert_valida(all_distances(A, B, epsilon), exacto, epsilon)

    def test_entradas_agrupadas(self):
        # Dos grupos compactos y distantes: una dispersión grande, que es de lo
        # que hablan los términos log(Delta) del tiempo de ejecución.
        rng = seeded(8)
        A = line([rng.uniform(0, 1) for _ in range(40)] + [1e4, 1e4 + 1])
        B = line([rng.uniform(0, 1) for _ in range(30)])
        exacto = naive_sequence(A, B)
        for epsilon in EPSILONS:
            self.assert_valida(all_distances(A, B, epsilon), exacto, epsilon)

    def test_puntos_casi_repetidos(self):
        # Una distancia mínima entre pares muy chica dispara la dispersión
        # Delta, que es la cantidad en la que el tiempo es logarítmico.
        A, B = line([1, 1 + 1e-9, 1 + 2e-9, 7]), line([1, 1 + 1e-9])
        exacto = naive_sequence(A, B)
        self.assertEqual(all_distances(A, B), exacto)
        for epsilon in EPSILONS:
            self.assert_valida(all_distances(A, B, epsilon), exacto, epsilon)


class TestCasosBorde(unittest.TestCase, Comprobaciones):
    def test_un_solo_punto_en_a(self):
        A, B = line([5]), line([1, 2, 3])
        for epsilon in (0,) + EPSILONS:
            got = all_distances(A, B, epsilon)
            self.assertEqual(len(got), 2)  # delta_0 y el delta_n = 0
            self.assertLessEqual(got[0], 2 + TOL)
            self.assertLessEqual(2, (1 + epsilon) * got[0] + TOL)

    def test_un_solo_punto_en_b(self):
        self.assertEqual(all_distances(line([0, 4, 10]), line([0])), [10, 4, 0, 0])

    def test_conjuntos_identicos(self):
        A = line([1, 4, 9, 16])
        for epsilon in (0,) + EPSILONS:
            self.assertEqual(all_distances(A, A, epsilon), [0, 0, 0, 0, 0])

    def test_los_repetidos_se_quitan_solos(self):
        # greedy_tree no puede construir un árbol sobre puntos exactamente
        # repetidos, así que se descartan al entrar.
        self.assertEqual(all_distances(line([1, 1, 1, 7]), line([1, 2])), [5, 0, 0])

    def test_es_dirigida_y_no_simetrica(self):
        A, B = line([0, 10]), line([0])
        self.assertEqual(all_distances(A, B), [10, 0, 0])
        self.assertEqual(all_distances(B, A), [0, 0])

    def test_epsilon_negativo(self):
        with self.assertRaises(ValueError):
            all_distances(line([1, 2]), line([1]), -0.1)


class TestTerminacionTemprana(unittest.TestCase):
    """`hausdorff` corta el recorrido, y eso no puede cambiar el valor."""

    def test_el_valor_cortado_es_el_de_la_secuencia_completa(self):
        rng = seeded(11)
        for _ in range(10):
            A, B = random_pair(rng, max_a=40, max_b=25)
            for epsilon in (0, 0.5):
                completa = all_distances(A, B, epsilon)
                for q in (100, 95, 50, 0):
                    k = _k_for_percentile(len(A), q)
                    self.assertAlmostEqual(
                        hausdorff(A, B, q, epsilon), completa[k], msg=f"q={q}"
                    )

    def test_el_corte_ahorra_trabajo(self):
        # Con epsilon > 0 pedir el percentil 100 debe emitir muchísimos menos
        # valores que la secuencia completa: es la razón de ser del corte.
        A, B = random_pair(seeded(12), max_a=60, max_b=40)
        G_A, G_B = greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B))
        busqueda = KHausdorff(G_A, G_B, 0.5)
        busqueda(stop_after=0)
        self.assertLess(len(busqueda.out), len(A))

    def test_percentil_fuera_de_rango(self):
        A, B = line([1, 2]), line([1])
        for q in (-1, 101):
            with self.assertRaises(ValueError):
                hausdorff(A, B, q)

    def test_coincide_con_la_fuerza_bruta(self):
        rng = seeded(13)
        for _ in range(10):
            A, B = random_pair(rng, max_a=40, max_b=25)
            for q in (100, 95, 50):
                self.assertAlmostEqual(hausdorff(A, B, q), hausdorff_naive(A, B, q))


class TestLowerBoundHeap(unittest.TestCase):
    """El heap con `remove` arreglado, del que depende todo el orden de salida."""

    def vaciar(self, heap):
        return [heap.removemax() for _ in range(len(heap))]

    def test_orden_basico(self):
        heap = LowerBoundHeap()
        for v in [5, 3, 9, 1, 7]:
            heap.insert(v, v)
        self.assertEqual(self.vaciar(heap), [9, 7, 5, 3, 1])

    def test_el_maxheap_original_realmente_esta_roto(self):
        # Fija la razón por la que existe la subclase.  Si una versión futura de
        # ds2 arregla `_remove_at_index`, este test falla y se puede borrar.
        heap = MaxHeap()
        valores = [6, 63, 103, 19, 69, 142, 187]
        for v in valores:
            heap.insert(v, v)
        heap.remove(6)
        self.assertNotEqual(self.vaciar(heap), sorted(set(valores) - {6}, reverse=True))

    def test_fuzz_de_remove(self):
        rng = random.Random(1)
        for _ in range(2000):
            valores = rng.sample(range(1, 500), rng.randint(4, 20))
            heap = LowerBoundHeap()
            for v in valores:
                heap.insert(v, v)
            quitado = rng.choice(valores)
            heap.remove(quitado)
            self.assertEqual(
                self.vaciar(heap), sorted(set(valores) - {quitado}, reverse=True)
            )

    def test_changepriority_en_ambos_sentidos(self):
        heap = LowerBoundHeap()
        for v in [1, 2, 3]:
            heap.insert(v, v)
        heap.changepriority(1, 100)
        self.assertEqual(heap.findmax(), 1)
        heap.changepriority(1, -100)
        self.assertEqual(heap.findmax(), 3)

    def test_clave_mutable(self):
        # Como lo usa KHausdorff: la clave lee un dict que cambia por debajo.
        cotas = {"a": 1.0, "b": 2.0}
        heap = LowerBoundHeap(key=lambda x: cotas[x])
        heap.insert("a")
        heap.insert("b")
        self.assertEqual(heap.findmax(), "b")
        cotas["a"] = 5.0
        heap.changepriority("a")
        self.assertEqual(heap.findmax(), "a")


if __name__ == "__main__":
    unittest.main()
