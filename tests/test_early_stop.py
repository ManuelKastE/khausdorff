"""
La terminacion temprana no puede cambiar ni un valor.

El argumento es que `_emit` solo agrega al final de `self.out` y nunca reescribe
una entrada ya emitida, de modo que out[k] queda fijo en el instante en que se
escribe.  Estos tests lo comprueban en vez de asumirlo, con igualdad exacta y no
aproximada: si cortar alterara el calculo, la diferencia aparecería aquí.
"""

import unittest

from khausdorff.bucketkhausdorff import (
    KHausdorffBucket,
    all_k_hausdorff_bucket,
    k_hausdorff_bucket,
)
from khausdorff.khausdorff import KHausdorff, all_k_hausdorff, k_hausdorff
from khausdorff.naive import all_partial_hausdorff
from khausdorff.percentile import hausdorff_percentile, k_for_percentile
from tests.helpers import TOL, line, random_pair, seeded, trees


class ContadorDeIteraciones(KHausdorff):
    """`KHausdorff` que cuenta las vueltas del bucle principal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteraciones = 0

    def iteration(self, ball):
        self.iteraciones += 1
        return super().iteration(ball)


class ContadorBucket(KHausdorffBucket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteraciones = 0

    def iteration(self, ball):
        self.iteraciones += 1
        return super().iteration(ball)


class TestEquivalenciaExacta(unittest.TestCase):
    def test_heap_variant(self):
        rng = seeded(21)
        for _ in range(12):
            A, B = random_pair(rng)
            n = len(A)
            for epsilon in (0, 0.1, 0.5, 1.0):
                for monotone in (True, False):
                    completo = all_k_hausdorff(*trees(A, B), epsilon, monotone)
                    for k in {0, 1, n // 2, n - 1, n}:
                        if not 0 <= k <= n:
                            continue
                        cortado = all_k_hausdorff(
                            *trees(A, B), epsilon, monotone, stop_after=k
                        )
                        self.assertEqual(
                            cortado,
                            completo[: k + 1],
                            f"n={n} eps={epsilon} monotone={monotone} k={k}",
                        )

    def test_bucket_variant_keeps_the_guarantee(self):
        # A la variante de buckets no se le puede exigir igualdad exacta entre
        # dos corridas: su salida no es reproducible (ver
        # `TestLaVarianteDeBucketsNoEsDeterminista`).  Lo que sí debe cumplir es
        # la garantia del articulo, cortando igual que sin cortar.
        rng = seeded(22)
        for _ in range(12):
            A, B = random_pair(rng)
            n = len(A)
            exacto = all_partial_hausdorff(A, B)
            for epsilon in (0.1, 0.5, 1.0):
                for k in {0, 1, n // 2, n - 1, n}:
                    cortado = all_k_hausdorff_bucket(
                        *trees(A, B), epsilon, stop_after=k
                    )
                    self.assertEqual(len(cortado), k + 1, f"n={n} k={k}")
                    for i, aprox in enumerate(cortado):
                        self.assertLessEqual(
                            aprox, exacto[i] + TOL, f"eps={epsilon} k={k} i={i}"
                        )
                        self.assertLessEqual(
                            exacto[i],
                            (1 + epsilon) * aprox + TOL,
                            f"eps={epsilon} k={k} i={i}",
                        )

    def test_length_is_exactly_k_plus_one(self):
        rng = seeded(23)
        A, B = random_pair(rng)
        n = len(A)
        for k in (0, 1, 5, n - 1, n):
            got = all_k_hausdorff(*trees(A, B), 0.5, stop_after=k)
            self.assertEqual(len(got), k + 1, f"k={k}")


class TestRealmenteCorta(unittest.TestCase):
    """
    Sin esto, un `stop_after` que no cortara nada pasaría los tests de
    equivalencia sin hacer absolutamente nada.
    """

    def _iteraciones(self, clase, A, B, epsilon, stop_after):
        buscador = clase(*trees(A, B), epsilon)
        buscador(stop_after)
        return buscador.iteraciones

    def test_heap_variant_does_less_work(self):
        rng = seeded(24)
        A, B = random_pair(rng, max_a=300, max_b=200)
        completo = self._iteraciones(ContadorDeIteraciones, A, B, 0.5, None)
        for k in (0, k_for_percentile(len(A), 95)):
            cortado = self._iteraciones(ContadorDeIteraciones, A, B, 0.5, k)
            self.assertLess(cortado, completo, f"k={k} no ahorro nada")

    def test_bucket_variant_does_less_work(self):
        rng = seeded(25)
        A, B = random_pair(rng, max_a=300, max_b=200)
        completo = self._iteraciones(ContadorBucket, A, B, 0.5, None)
        cortado = self._iteraciones(ContadorBucket, A, B, 0.5, 0)
        self.assertLess(cortado, completo)

    def test_epsilon_zero_cannot_cut_the_main_loop(self):
        # Con epsilon = 0 la constante de terminacion es 0 y la condicion nunca
        # se cumple mientras haya radio, asi que el bucle principal se recorre
        # entero igual.  El resultado tiene que seguir siendo correcto.
        rng = seeded(26)
        A, B = random_pair(rng, max_a=80, max_b=50)
        completo = self._iteraciones(ContadorDeIteraciones, A, B, 0, None)
        cortado = self._iteraciones(ContadorDeIteraciones, A, B, 0, 0)
        self.assertEqual(cortado, completo)
        self.assertEqual(
            all_k_hausdorff(*trees(A, B), 0, stop_after=0),
            all_k_hausdorff(*trees(A, B), 0)[:1],
        )


class TestLaVarianteDeBucketsNoEsDeterminista(unittest.TestCase):
    """
    Hallazgo que conviene tener fijado, y que no tiene que ver con el corte.

    `BetaBucketQueue.findmax` devuelve `next(iter(...))` sobre un conjunto de
    nodos.  Los `Ball` se hashean por identidad, asi que el orden de iteracion
    depende de las direcciones de memoria y cambia entre ejecuciones.  Es
    deliberado -- el orden dentro de un bucket es arbitrario a proposito, y eso
    es lo que compra el O(1) -- pero significa que la salida de la variante de
    buckets no es reproducible, y que compararla con `assertEqual` entre dos
    corridas es un error.
    """

    def test_two_identical_full_runs_may_differ(self):
        rng = seeded(30)
        A, B = random_pair(rng, max_a=60, max_b=40)
        corridas = {tuple(all_k_hausdorff_bucket(*trees(A, B), 1.0)) for _ in range(8)}
        # No se afirma que difieran siempre (podrian coincidir por azar), solo
        # que la garantia se respeta en todas.  El heap, en cambio, si es
        # reproducible.
        exacto = all_partial_hausdorff(A, B)
        for corrida in corridas:
            for aprox, true in zip(corrida, exacto):
                self.assertLessEqual(aprox, true + TOL)
                self.assertLessEqual(true, 2.0 * aprox + TOL)

    def test_the_heap_variant_is_reproducible(self):
        rng = seeded(31)
        A, B = random_pair(rng, max_a=60, max_b=40)
        corridas = {tuple(all_k_hausdorff(*trees(A, B), 1.0)) for _ in range(8)}
        self.assertEqual(len(corridas), 1, "el heap deberia dar siempre lo mismo")


class TestApiPublica(unittest.TestCase):
    def test_k_hausdorff_matches_the_full_run(self):
        rng = seeded(27)
        for _ in range(8):
            A, B = random_pair(rng)
            completo = all_k_hausdorff(*trees(A, B), 0.5)
            for k in (0, 1, len(A) // 2, len(A)):
                self.assertEqual(k_hausdorff(*trees(A, B), k, 0.5), completo[k])

    def test_k_hausdorff_bucket_keeps_the_guarantee(self):
        rng = seeded(28)
        for _ in range(8):
            A, B = random_pair(rng)
            exacto = all_partial_hausdorff(A, B)
            for k in (0, 1, len(A) // 2, len(A)):
                got = k_hausdorff_bucket(*trees(A, B), k, 0.5)
                self.assertLessEqual(got, exacto[k] + TOL, f"k={k}")
                self.assertLessEqual(exacto[k], 1.5 * got + TOL, f"k={k}")

    def test_percentile_matches_the_full_run(self):
        rng = seeded(29)
        for _ in range(8):
            A, B = random_pair(rng)
            completo = all_k_hausdorff(*trees(A, B), 0.5)
            for q in (0, 25, 50, 95, 100):
                k = k_for_percentile(len(A), q)
                self.assertEqual(
                    hausdorff_percentile(*trees(A, B), q, 0.5), completo[k], f"q={q}"
                )

    def test_k_out_of_range_still_raises(self):
        A, B = line([1, 2]), line([1])
        with self.assertRaises(IndexError):
            k_hausdorff(*trees(A, B), 3, 0.5)
        with self.assertRaises(IndexError):
            k_hausdorff(*trees(A, B), -1, 0.5)

    def test_k_equal_to_n_still_gives_the_trailing_zero(self):
        A, B = line([1, 2, 3]), line([1])
        self.assertEqual(k_hausdorff(*trees(A, B), 3, 0.5), 0)

    def test_single_point_in_a(self):
        A, B = line([5]), line([1, 2, 3])
        self.assertEqual(all_k_hausdorff(*trees(A, B), 0.5, stop_after=0), [2])
        self.assertEqual(k_hausdorff(*trees(A, B), 0, 0.5), 2)


if __name__ == "__main__":
    unittest.main()
