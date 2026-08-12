import unittest

from greedypermutation.hausdorff import dist_H

from khausdorff.bucketkhausdorff import all_k_hausdorff_bucket, k_hausdorff_bucket
from khausdorff.khausdorff import all_k_hausdorff, k_hausdorff
from khausdorff.naive import all_partial_hausdorff
from tests.helpers import TOL, line, random_pair, seeded, trees

EPSILONS = (0.1, 0.5, 1.0)


class ApproximationChecks:
    """Comprobaciones que toda variante debe cumplir, sea cual sea epsilon."""

    def assert_valid(self, got, exact, epsilon):
        n = len(exact)
        self.assertEqual(len(got), n, "one distance per point of A")
        for i in range(n - 1):
            self.assertGreaterEqual(
                got[i] + TOL, got[i + 1], f"output is not non-increasing at {i}"
            )
        for i, (approx, true) in enumerate(zip(got, exact)):
            self.assertLessEqual(
                approx, true + TOL, f"delta_{i} = {approx} exceeds d_h^({i}) = {true}"
            )
            self.assertLessEqual(
                true,
                (1 + epsilon) * approx + TOL,
                f"d_h^({i}) = {true} exceeds (1+{epsilon}) * delta_{i} = {approx}",
            )


class TestExact(unittest.TestCase, ApproximationChecks):
    """Con epsilon = 0 la variante de heap debe reproducir la respuesta ingenua."""

    def test_line_examples(self):
        for a_values, b_values, expected in [
            ([1, 2, 3, 6, 7, 8, 12], [1, 2, 3, 6, 7, 9], [3, 1, 0, 0, 0, 0, 0]),
            ([1, 2, 3, 6, 7, 8], [1, 2, 3, 6, 7, 9], [1, 0, 0, 0, 0, 0]),
        ]:
            A, B = line(a_values), line(b_values)
            G_A, G_B = trees(A, B)
            self.assertEqual(all_k_hausdorff(G_A, G_B, 0), expected)

    def test_delta_0_matches_dist_H(self):
        rng = seeded(3)
        for _ in range(15):
            A, B = random_pair(rng)
            G_A, G_B = trees(A, B)
            expected = dist_H(*trees(A, B))
            self.assertAlmostEqual(all_k_hausdorff(G_A, G_B, 0)[0], expected)

    def test_matches_naive_on_random_clouds(self):
        rng = seeded(4)
        for _ in range(25):
            A, B = random_pair(rng)
            G_A, G_B = trees(A, B)
            got = all_k_hausdorff(G_A, G_B, 0)
            exact = all_partial_hausdorff(A, B)
            for approx, true in zip(got, exact):
                self.assertAlmostEqual(approx, true)
            self.assert_valid(got, exact, 0)

    def test_matches_naive_in_higher_dimensions(self):
        rng = seeded(5)
        for dim in (1, 3, 5):
            A, B = random_pair(rng, max_a=30, max_b=20, dim=dim)
            G_A, G_B = trees(A, B)
            got = all_k_hausdorff(G_A, G_B, 0)
            for approx, true in zip(got, all_partial_hausdorff(A, B)):
                self.assertAlmostEqual(approx, true)


class TestApproximation(unittest.TestCase, ApproximationChecks):
    """Ambas variantes deben respetar delta_i <= d_h^(i) <= (1+eps) delta_i."""

    def test_heap_variant(self):
        rng = seeded(6)
        for _ in range(25):
            A, B = random_pair(rng)
            exact = all_partial_hausdorff(A, B)
            for epsilon in EPSILONS:
                got = all_k_hausdorff(*trees(A, B), epsilon)
                self.assert_valid(got, exact, epsilon)

    def test_bucket_variant(self):
        rng = seeded(7)
        for _ in range(25):
            A, B = random_pair(rng)
            exact = all_partial_hausdorff(A, B)
            for epsilon in EPSILONS:
                got = all_k_hausdorff_bucket(*trees(A, B), epsilon)
                self.assert_valid(got, exact, epsilon)

    def test_clustered_inputs(self):
        # Dos grupos compactos y distantes: una dispersión grande, que es de lo
        # que hablan los términos log(Delta) del tiempo de ejecución.
        rng = seeded(8)
        A = line([rng.uniform(0, 1) for _ in range(40)] + [1e4, 1e4 + 1])
        B = line([rng.uniform(0, 1) for _ in range(30)])
        exact = all_partial_hausdorff(A, B)
        for epsilon in EPSILONS:
            self.assert_valid(all_k_hausdorff(*trees(A, B), epsilon), exact, epsilon)
            self.assert_valid(
                all_k_hausdorff_bucket(*trees(A, B), epsilon), exact, epsilon
            )

    def test_variants_agree_within_tolerance(self):
        rng = seeded(9)
        for _ in range(10):
            A, B = random_pair(rng, max_a=40, max_b=25)
            for epsilon in EPSILONS:
                heap = all_k_hausdorff(*trees(A, B), epsilon)
                bucket = all_k_hausdorff_bucket(*trees(A, B), epsilon)
                self.assertEqual(len(heap), len(bucket))
                for h, b in zip(heap, bucket):
                    # Ambas acotan la misma verdad, así que coinciden dentro del
                    # factor de aproximación combinado.
                    self.assertLessEqual(b, (1 + epsilon) * h + TOL)
                    self.assertLessEqual(h, (1 + epsilon) * b + TOL)


class TestEdgeCases(unittest.TestCase, ApproximationChecks):
    def test_single_point_in_a(self):
        A, B = line([5]), line([1, 2, 3])
        for epsilon in (0,) + EPSILONS:
            variants = [all_k_hausdorff(*trees(A, B), epsilon)]
            if epsilon:
                variants.append(all_k_hausdorff_bucket(*trees(A, B), epsilon))
            for got in variants:
                self.assertEqual(len(got), 1)
                self.assertLessEqual(got[0], 2 + TOL)
                self.assertLessEqual(2, (1 + epsilon) * got[0] + TOL)

    def test_single_point_in_b(self):
        A, B = line([0, 4, 10]), line([0])
        self.assertEqual(all_k_hausdorff(*trees(A, B), 0), [10, 4, 0])

    def test_identical_sets(self):
        A = line([1, 4, 9, 16])
        for epsilon in (0,) + EPSILONS:
            got = all_k_hausdorff(*trees(A, A), epsilon)
            self.assertEqual(got, [0, 0, 0, 0])

    def test_near_duplicate_points(self):
        # Una distancia mínima entre pares muy pequeña dispara la dispersión
        # Delta, que es la cantidad en la que el tiempo es logarítmico.
        A = line([1, 1 + 1e-9, 1 + 2e-9, 7])
        B = line([1, 1 + 1e-9])
        exact = all_partial_hausdorff(A, B)
        self.assertEqual(all_k_hausdorff(*trees(A, B), 0), exact)
        for epsilon in EPSILONS:
            self.assert_valid(all_k_hausdorff(*trees(A, B), epsilon), exact, epsilon)

    def test_exactly_duplicated_points_are_an_upstream_limitation(self):
        # greedy_tree no puede construir un árbol sobre un conjunto con puntos
        # repetidos: la permutación greedy se queda sin centros distintos y le
        # entrega `None` a la métrica.  Se fija aquí para que la limitación sea
        # visible, y para que el test falle si alguna vez la levantan.
        with self.assertRaises(TypeError):
            trees(line([1, 1, 1, 7]), line([1, 2]))

    def test_directed_not_symmetric(self):
        A, B = line([0, 10]), line([0])
        self.assertEqual(all_k_hausdorff(*trees(A, B), 0), [10, 0])
        self.assertEqual(all_k_hausdorff(*trees(B, A), 0), [0])

    def test_monotone_flag_off_still_bounds_from_below(self):
        rng = seeded(10)
        A, B = random_pair(rng)
        exact = all_partial_hausdorff(A, B)
        got = all_k_hausdorff(*trees(A, B), 0.5, monotone=False)
        for approx, true in zip(sorted(got, reverse=True), exact):
            self.assertLessEqual(approx, true + TOL)


class TestApi(unittest.TestCase):
    def test_k_hausdorff_indexes_the_sequence(self):
        A, B = line([1, 2, 3, 6, 7, 8, 12]), line([1, 2, 3, 6, 7, 9])
        self.assertEqual(k_hausdorff(*trees(A, B), 0, epsilon=0), 3)
        self.assertEqual(k_hausdorff(*trees(A, B), 1, epsilon=0), 1)
        self.assertEqual(k_hausdorff_bucket(*trees(A, B), 6, epsilon=0.5), 0)

    def test_k_out_of_range(self):
        A, B = line([1, 2]), line([1])
        with self.assertRaises(IndexError):
            k_hausdorff(*trees(A, B), 2)

    def test_negative_epsilon_is_rejected(self):
        A, B = line([1, 2]), line([1])
        with self.assertRaises(ValueError):
            all_k_hausdorff(*trees(A, B), -0.1)

    def test_bucket_variant_rejects_epsilon_zero(self):
        A, B = line([1, 2]), line([1])
        with self.assertRaises(ValueError):
            all_k_hausdorff_bucket(*trees(A, B), 0)


if __name__ == "__main__":
    unittest.main()
