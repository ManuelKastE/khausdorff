import io
import tempfile
import unittest
from pathlib import Path

from khausdorff import _k_for_percentile, hausdorff_naive, load_points, nearest_distances
from tests.helpers import line, random_pair, seeded


class TestFuerzaBruta(unittest.TestCase):
    def test_nearest_distances(self):
        A, B = line([1, 2, 3, 6, 7, 8, 12]), line([1, 2, 3, 6, 7, 9])
        self.assertEqual(nearest_distances(A, B), [0, 0, 0, 0, 0, 1, 3])

    def test_percentil_100_es_la_distancia_de_hausdorff(self):
        self.assertEqual(
            hausdorff_naive(line([1, 2, 3, 6, 7, 8, 12]), line([1, 2, 3, 6, 7, 9])), 3
        )
        self.assertEqual(
            hausdorff_naive(line([1, 2, 3, 6, 7, 8]), line([1, 2, 3, 6, 7, 9])), 1
        )

    def test_el_percentil_descarta_los_mas_lejanos(self):
        A, B = line([1, 2, 3, 6, 7, 8, 12]), line([1, 2, 3, 6, 7, 9])
        # n = 7: el percentil 80 se queda en la posición ceil(0.8*7) = 6 del
        # orden ascendente, o sea delta_1, sin el punto más lejano.
        self.assertEqual(hausdorff_naive(A, B, 80), 1)
        self.assertEqual(hausdorff_naive(A, B, 70), 0)

    def test_percentil_0_es_el_cero_final(self):
        self.assertEqual(hausdorff_naive(line([1, 2]), line([1]), 0), 0)

    def test_percentil_fuera_de_rango(self):
        for q in (-1, 101):
            with self.assertRaises(ValueError):
                hausdorff_naive(line([1, 2]), line([1]), q)

    def test_es_dirigida(self):
        A, B = line([0, 10]), line([0])
        self.assertEqual(hausdorff_naive(A, B), 10)
        self.assertEqual(hausdorff_naive(B, A), 0)

    def test_b_vacio(self):
        with self.assertRaises(ValueError):
            nearest_distances(line([1]), [])


class TestPercentil(unittest.TestCase):
    """La traducción percentil -> k, convención nearest-rank."""

    def test_el_ejemplo_trabajado(self):
        # Con n = 100, el percentil 95 descarta exactamente 5 puntos.
        self.assertEqual(_k_for_percentile(100, 95), 5)

    def test_los_extremos(self):
        for n in (1, 2, 7, 100):
            self.assertEqual(_k_for_percentile(n, 100), 0, "q = 100 no descarta nada")
            self.assertEqual(_k_for_percentile(n, 0), n, "q = 0 descarta todo")

    def test_siempre_dentro_del_rango(self):
        for n in (1, 3, 37, 100):
            for q in range(0, 101):
                self.assertTrue(0 <= _k_for_percentile(n, q) <= n, f"n={n} q={q}")

    def test_es_un_valor_de_la_secuencia_y_no_una_interpolacion(self):
        # Es lo que hace que la garantía (1+eps) se herede intacta.
        rng = seeded(2)
        A, B = random_pair(rng)
        secuencia = sorted(nearest_distances(A, B), reverse=True) + [0.0]
        for q in range(0, 101, 5):
            self.assertIn(hausdorff_naive(A, B, q), secuencia)

    def test_es_no_creciente_en_q_decreciente(self):
        A, B = random_pair(seeded(3))
        valores = [hausdorff_naive(A, B, q) for q in range(100, -1, -5)]
        self.assertEqual(valores, sorted(valores, reverse=True))


class TestLeerArchivos(unittest.TestCase):
    def setUp(self):
        self.carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(self.carpeta.cleanup)
        self.ruta_a = self.escribir("A.csv", "x,y\n0,0\n1,0\n0,1\n5,5\n")
        self.ruta_b = self.escribir("B.csv", "x,y\n0,0\n1,0\n0,1\n")

    def escribir(self, nombre, texto):
        ruta = Path(self.carpeta.name) / nombre
        ruta.write_text(texto)
        return ruta

    def test_comas_espacios_comentarios_y_encabezado(self):
        for texto in (
            "x,y\n1,2\n3,4\n",
            "1 2\n3 4\n",
            "# un comentario\n\n1,2\n3,4\n",
            "1,2 ; esto se descarta\n3,4\n",
        ):
            puntos = load_points(self.escribir("t.csv", texto))
            self.assertEqual([list(p) for p in puntos], [[1, 2], [3, 4]], texto)

    def test_cualquier_dimension(self):
        puntos = load_points(self.escribir("t.csv", "1 2 3 4\n5 6 7 8\n"))
        self.assertEqual(len(puntos[0]), 4)

    def test_dimensiones_desparejas_dan_error(self):
        # No es cosmético: sin esto el cálculo devuelve un número equivocado en
        # silencio, porque Point.dist proyecta al subespacio común.
        ruta = self.escribir("t.csv", "1,2\n3\n")
        with self.assertRaises(ValueError) as ctx:
            load_points(ruta)
        self.assertIn("línea 2", str(ctx.exception))

    def test_texto_que_no_es_numero(self):
        with self.assertRaises(ValueError):
            load_points(self.escribir("t.csv", "1,2\nhola,3\n"))

    def test_archivo_vacio(self):
        with self.assertRaises(ValueError):
            load_points(self.escribir("t.csv", "# solo comentarios\n\n"))

    def test_solo_encabezado(self):
        with self.assertRaises(ValueError):
            load_points(self.escribir("t.csv", "x,y\n"))

    def test_archivo_abierto(self):
        with open(self.ruta_a) as handle:
            self.assertEqual(len(load_points(handle)), 4)

    def test_objeto_tipo_archivo(self):
        self.assertEqual(len(load_points(io.StringIO("1,2\n3,4\n"))), 2)

    def test_rutas_como_texto_y_como_path(self):
        esperado = [0.0, 0.0, 0.0, 41**0.5]
        self.assertEqual(nearest_distances(str(self.ruta_a), str(self.ruta_b)), esperado)
        self.assertEqual(nearest_distances(self.ruta_a, self.ruta_b), esperado)

    def test_mezclar_ruta_y_lista_de_puntos(self):
        B = load_points(self.ruta_b)
        self.assertAlmostEqual(hausdorff_naive(self.ruta_a, B, 100), 41**0.5)

    def test_el_percentil_sobre_un_archivo(self):
        # n = 4: el percentil 75 se queda en delta_1, sin el (5,5).
        self.assertEqual(hausdorff_naive(self.ruta_a, self.ruta_b, 75), 0)

    def test_da_lo_mismo_que_cargar_a_mano(self):
        A, B = load_points(self.ruta_a), load_points(self.ruta_b)
        self.assertEqual(
            nearest_distances(self.ruta_a, self.ruta_b), nearest_distances(A, B)
        )

    def test_archivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            nearest_distances(self.ruta_a, Path(self.carpeta.name) / "no.csv")

    def test_los_repetidos_se_quitan(self):
        ruta = self.escribir("dup.csv", "1,2\n1,2\n3,4\n")
        self.assertEqual(len(load_points(ruta)), 3, "load_points no los toca")
        self.assertEqual(len(nearest_distances(ruta, self.ruta_b)), 2, "el cálculo sí")


if __name__ == "__main__":
    unittest.main()
