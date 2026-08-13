"""La variante de la Sección 5.2, que vive fuera del camino en `buckets.py`."""

import random
import unittest
from math import inf

from buckets import BetaBucketQueue, all_distances_bucket
from khausdorff import all_distances
from tests.helpers import TOL, line, random_pair, seeded

EPSILONS = (0.1, 0.5, 1.0)


def naive_sequence(A, B):
    return sorted((min(a.dist(b) for b in B) for a in A), reverse=True) + [0.0]


class TestVariante(unittest.TestCase):
    """Vale la misma garantía delta_i <= d_h^(i) <= (1+eps) delta_i."""

    def assert_valida(self, got, exact, epsilon):
        self.assertEqual(len(got), len(exact))
        for i, (aprox, true) in enumerate(zip(got, exact)):
            self.assertLessEqual(aprox, true + TOL, f"delta_{i} pasa a d_h^({i})")
            self.assertLessEqual(true, (1 + epsilon) * aprox + TOL, f"cota superior en {i}")

    def test_nubes_al_azar(self):
        rng = seeded(7)
        for _ in range(20):
            A, B = random_pair(rng)
            exacto = naive_sequence(A, B)
            for epsilon in EPSILONS:
                self.assert_valida(all_distances_bucket(A, B, epsilon), exacto, epsilon)

    def test_entradas_agrupadas(self):
        rng = seeded(8)
        A = line([rng.uniform(0, 1) for _ in range(40)] + [1e4, 1e4 + 1])
        B = line([rng.uniform(0, 1) for _ in range(30)])
        exacto = naive_sequence(A, B)
        for epsilon in EPSILONS:
            self.assert_valida(all_distances_bucket(A, B, epsilon), exacto, epsilon)

    def test_concuerda_con_la_variante_de_heap(self):
        rng = seeded(9)
        for _ in range(10):
            A, B = random_pair(rng, max_a=40, max_b=25)
            for epsilon in EPSILONS:
                heap = all_distances(A, B, epsilon)
                bucket = all_distances_bucket(A, B, epsilon)
                self.assertEqual(len(heap), len(bucket))
                for h, b in zip(heap, bucket):
                    # Acotan la misma verdad, así que coinciden dentro del
                    # factor combinado.  Nunca por igualdad exacta: el orden
                    # dentro de un bucket es arbitrario.
                    self.assertLessEqual(b, (1 + epsilon) * h + TOL)
                    self.assertLessEqual(h, (1 + epsilon) * b + TOL)

    def test_rechaza_epsilon_cero(self):
        with self.assertRaises(ValueError):
            all_distances_bucket(line([1, 2]), line([1]), 0)


class TestBetaBucketQueue(unittest.TestCase):
    def test_beta_debe_superar_a_uno(self):
        for beta in (1, 0.5, 0):
            with self.assertRaises(ValueError):
                BetaBucketQueue(beta)

    def test_insertar_en_una_cola_vacia(self):
        # La regresión que vuelve inutilizable a greedypermutation.fvm.bucketqueue:
        # su `insert` llama a max() sobre un dict de buckets vacío y falla.
        q = BetaBucketQueue(1.5)
        q.insert("a", 3.0)
        self.assertEqual(q.findmax(), "a")

    def test_invariante_de_los_buckets(self):
        beta = 1.25
        q = BetaBucketQueue(beta)
        for prioridad in (0.001, 0.3, 1.0, 1.25, 2.7, 40.0, 1e6):
            nivel = q.level_of(prioridad)
            self.assertLess(beta**nivel, prioridad)
            self.assertLessEqual(prioridad, beta ** (nivel + 1))

    def test_las_potencias_exactas_caen_en_el_bucket_de_abajo(self):
        # Los buckets son (beta**m, beta**(m+1)], así que beta**m cae en el m-1.
        self.assertEqual(BetaBucketQueue(2.0).level_of(2.0**5), 4)

    def test_las_prioridades_no_positivas_usan_el_centinela(self):
        q = BetaBucketQueue(1.5)
        for prioridad in (0.0, -1.0, -1e9):
            self.assertEqual(q.level_of(prioridad), -inf)
        self.assertEqual(q.value(-inf), 0.0)
        q.insert("cero", 0.0)
        q.insert("chico", 1e-6)
        self.assertEqual(q.findmax(), "chico", "el centinela va debajo de todo")

    def test_el_valor_es_el_extremo_inferior(self):
        # Es lo que mantiene la salida como cota inferior válida.
        q = BetaBucketQueue(1.5)
        for prioridad in (0.4, 2.0, 17.0):
            self.assertLess(q.value(q.level_of(prioridad)), prioridad)

    def test_remove_contains_y_errores(self):
        q = BetaBucketQueue(1.5)
        q.insert("a", 2.0)
        self.assertIn("a", q)
        with self.assertRaises(RuntimeError):
            q.insert("a", 3.0)
        nivel = q.maxlevel()
        q.remove("a")
        self.assertNotIn("a", q)
        self.assertNotIn(nivel, q.buckets, "los buckets vacíos se descartan")
        self.assertIsNone(q.maxlevel())
        with self.assertRaises(RuntimeError):
            q.remove("a")
        with self.assertRaises(RuntimeError):
            q.findmax()

    def test_changepriority_mueve_entre_buckets(self):
        q = BetaBucketQueue(1.5)
        q.insert("a", 1.0)
        q.insert("b", 100.0)
        self.assertEqual(q.findmax(), "b")
        q.changepriority("a", 10000.0)
        self.assertEqual(q.findmax(), "a")
        q.changepriority("a", 0.01)
        self.assertEqual(q.findmax(), "b")

    def test_levels_at_or_above_y_pop_level(self):
        q = BetaBucketQueue(2.0)
        for nombre, prioridad in [("a", 1.5), ("b", 3.0), ("c", 6.0), ("d", 100.0)]:
            q.insert(nombre, prioridad)
        niveles = q.levels_at_or_above(-inf)
        self.assertEqual(niveles, sorted(niveles, reverse=True))
        self.assertEqual(len(niveles), 4)
        self.assertEqual(len(q.levels_at_or_above(q.level_of(6.0))), 2)  # c y d
        self.assertEqual(q.pop_level(q.level_of(3.0)), {"b"})
        self.assertEqual(q.pop_level(q.level_of(3.0)), set())

    def test_removemax_vacia_en_buckets_no_crecientes(self):
        rng = random.Random(0)
        q = BetaBucketQueue(1.3)
        prioridades = {i: rng.uniform(0.01, 1000) for i in range(200)}
        for item, prioridad in prioridades.items():
            q.insert(item, prioridad)
        niveles = [q.level_of(prioridades[q.removemax()]) for _ in range(len(q))]
        self.assertEqual(niveles, sorted(niveles, reverse=True))

    def test_la_funcion_clave_se_usa_sin_prioridad(self):
        self.assertEqual(BetaBucketQueue(1.5, items=[1.0, 8.0, 4.0]).findmax(), 8.0)


if __name__ == "__main__":
    unittest.main()
