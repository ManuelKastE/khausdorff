import unittest
from math import ceil

from greedypermutation.hausdorff import dist_H

from khausdorff.naive import all_partial_hausdorff, nearest_distances
from khausdorff.percentile import (
    hausdorff_percentile,
    hausdorff_percentile_bucket,
    k_for_percentile,
    partial_hausdorff_percentile,
)
from tests.helpers import TOL, line, random_pair, seeded, trees

EPSILONS = (0.1, 0.5, 1.0)


class TestKForPercentile(unittest.TestCase):
    def test_the_worked_example(self):
        # Con n = 100, el percentil 95 descarta exactamente 5 puntos.
        self.assertEqual(k_for_percentile(100, 95), 5)

    def test_extremes(self):
        for n in (1, 2, 37, 100, 1000):
            self.assertEqual(k_for_percentile(n, 100), 0, "q = 100 no descarta nada")
            self.assertEqual(k_for_percentile(n, 0), n, "q = 0 descarta todo")

    def test_non_integral_case_rounds_by_nearest_rank(self):
        # n = 37, q = 95: hay que descartar 1.85 puntos.  Nearest-rank toma el
        # valor en la posicion ceil(0.95 * 37) = 36 de 37 ascendente, o sea
        # descarta 1.
        self.assertEqual(k_for_percentile(37, 95), 1)

    def test_k_stays_in_range(self):
        for n in range(1, 60):
            for q in (0, 0.5, 1, 33.3, 50, 95, 99.9, 100):
                self.assertTrue(0 <= k_for_percentile(n, q) <= n, f"n={n} q={q}")

    def test_q_out_of_range(self):
        for q in (-1, 100.001, 200):
            with self.assertRaises(ValueError):
                k_for_percentile(10, q)


class TestDefiningProperty(unittest.TestCase):
    """
    La prueba que de verdad importa: el valor devuelto tiene que ser un percentil
    segun su definicion, no segun la formula que usamos para calcularlo.
    """

    def test_at_least_q_percent_of_points_are_at_or_below(self):
        rng = seeded(4)
        for _ in range(25):
            A, B = random_pair(rng)
            n = len(A)
            distancias = nearest_distances(A, B)
            for q in (0, 10, 25, 50, 75, 90, 95, 99, 100):
                v = partial_hausdorff_percentile(A, B, q)
                por_debajo = sum(1 for d in distancias if d <= v + TOL)
                self.assertGreaterEqual(
                    por_debajo,
                    ceil(q * n / 100),
                    f"n={n} q={q}: solo {por_debajo} puntos a distancia <= {v}",
                )


class TestAgreesWithTheCountIndex(unittest.TestCase):
    def test_percentile_95_is_delta_5_when_n_is_100(self):
        rng = seeded(0)
        A, B = random_pair(rng, max_a=100, max_b=60)
        while len(A) != 100:
            A, B = random_pair(rng, max_a=100, max_b=60)
        self.assertEqual(
            partial_hausdorff_percentile(A, B, 95), all_partial_hausdorff(A, B)[5]
        )

    def test_percentile_100_is_the_directed_hausdorff_distance(self):
        rng = seeded(7)
        for _ in range(10):
            A, B = random_pair(rng)
            self.assertAlmostEqual(
                partial_hausdorff_percentile(A, B, 100), dist_H(*trees(A, B)), delta=TOL
            )

    def test_percentile_0_is_zero(self):
        A, B = line([1, 5, 20]), line([0])
        self.assertEqual(partial_hausdorff_percentile(A, B, 0), 0)
        self.assertEqual(hausdorff_percentile(*trees(A, B), 0), 0)


class TestApproximationBounds(unittest.TestCase):
    """Ambas variantes deben respetar aprox <= exacto <= (1+eps) * aprox."""

    def test_both_variants_bracket_the_exact_percentile(self):
        rng = seeded(11)
        for _ in range(10):
            A, B = random_pair(rng)
            for q in (25, 50, 90, 95, 100):
                exacto = partial_hausdorff_percentile(A, B, q)
                for epsilon in EPSILONS:
                    for got in (
                        hausdorff_percentile(*trees(A, B), q, epsilon),
                        hausdorff_percentile_bucket(*trees(A, B), q, epsilon),
                    ):
                        self.assertLessEqual(got, exacto + TOL, f"q={q} eps={epsilon}")
                        self.assertLessEqual(
                            exacto, (1 + epsilon) * got + TOL, f"q={q} eps={epsilon}"
                        )

    def test_exact_variant_matches_brute_force(self):
        rng = seeded(12)
        for _ in range(10):
            A, B = random_pair(rng)
            for q in (10, 50, 95, 100):
                self.assertAlmostEqual(
                    hausdorff_percentile(*trees(A, B), q, epsilon=0),
                    partial_hausdorff_percentile(A, B, q),
                    delta=TOL,
                )


class TestEdgeCases(unittest.TestCase):
    def test_single_point_in_a(self):
        A, B = line([5]), line([1, 2, 3])
        self.assertEqual(partial_hausdorff_percentile(A, B, 100), 2)
        self.assertEqual(partial_hausdorff_percentile(A, B, 1), 2)
        self.assertEqual(partial_hausdorff_percentile(A, B, 0), 0)

    def test_percentile_is_non_decreasing_in_q(self):
        rng = seeded(13)
        A, B = random_pair(rng)
        valores = [partial_hausdorff_percentile(A, B, q) for q in range(0, 101, 5)]
        for anterior, siguiente in zip(valores, valores[1:]):
            self.assertLessEqual(anterior, siguiente + TOL)

    def test_q_out_of_range_reaches_the_wrappers(self):
        A, B = line([1, 2]), line([1])
        with self.assertRaises(ValueError):
            partial_hausdorff_percentile(A, B, 101)
        with self.assertRaises(ValueError):
            hausdorff_percentile(*trees(A, B), -5)

    def test_bucket_variant_still_rejects_epsilon_zero(self):
        A, B = line([1, 2]), line([1])
        with self.assertRaises(ValueError):
            hausdorff_percentile_bucket(*trees(A, B), 50, 0)


if __name__ == "__main__":
    unittest.main()
