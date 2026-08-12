import io
import tempfile
import unittest
from pathlib import Path

from khausdorff.io import deduplicate, duplicates, load_points


def leer(texto, **kwargs):
    return load_points(io.StringIO(texto), **kwargs)


class TestParseo(unittest.TestCase):
    def test_comas_con_encabezado(self):
        self.assertEqual(
            [tuple(p) for p in leer("x,y\n0.1,0.4\n0.7,0.2\n")],
            [(0.1, 0.4), (0.7, 0.2)],
        )

    def test_espacios_sin_encabezado(self):
        self.assertEqual(
            [tuple(p) for p in leer("0.1 0.4\n0.7 0.2\n")], [(0.1, 0.4), (0.7, 0.2)]
        )

    def test_comentarios_y_lineas_en_blanco(self):
        self.assertEqual(
            [tuple(p) for p in leer("# datos\n\n1,2\n\n3,4\n# fin\n")],
            [(1.0, 2.0), (3.0, 4.0)],
        )

    def test_todo_lo_posterior_a_punto_y_coma_se_ignora(self):
        # La misma convencion que `MetricSpace.fromstrings`.
        self.assertEqual([tuple(p) for p in leer("1,2;basura\n3,4;otra\n")], [(1.0, 2.0), (3.0, 4.0)])

    def test_una_dimension(self):
        self.assertEqual([tuple(p) for p in leer("1\n5\n9\n")], [(1.0,), (5.0,), (9.0,)])

    def test_tres_dimensiones(self):
        self.assertEqual([tuple(p) for p in leer("1 2 3\n4 5 6\n")], [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])

    def test_espacios_de_sobra(self):
        self.assertEqual([tuple(p) for p in leer("  1 ,  2  \n3,4\n")], [(1.0, 2.0), (3.0, 4.0)])

    def test_negativos_y_notacion_cientifica(self):
        self.assertEqual([tuple(p) for p in leer("-1.5,2e3\n")], [(-1.5, 2000.0)])

    def test_desde_una_ruta(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "puntos.csv"
            ruta.write_text("x,y\n1,2\n3,4\n")
            self.assertEqual([tuple(p) for p in load_points(ruta)], [(1.0, 2.0), (3.0, 4.0)])

    def test_dim_explicita_se_respeta(self):
        with self.assertRaises(ValueError):
            leer("1,2,3\n", dim=2)


class TestErroresClaros(unittest.TestCase):
    """
    Los mensajes importan: sin ellos, los datos malos se manifiestan como
    resultados equivocados o como excepciones de la dependencia que no dicen
    nada sobre la causa.
    """

    def test_dimensiones_desparejas(self):
        # `Point.dist` proyecta al subespacio comun en vez de fallar, asi que
        # esto tiene que detectarse al leer o el resultado sale mal en silencio.
        with self.assertRaises(ValueError) as ctx:
            leer("1,2\n3\n")
        self.assertIn("línea 2", str(ctx.exception))

    def test_coordenada_no_numerica(self):
        with self.assertRaises(ValueError) as ctx:
            leer("1,2\nhola,mundo\n")
        self.assertIn("línea 2", str(ctx.exception))

    def test_archivo_vacio(self):
        with self.assertRaises(ValueError):
            leer("\n\n# solo comentarios\n")

    def test_solo_encabezado(self):
        with self.assertRaises(ValueError):
            leer("x,y\n")


class TestDuplicados(unittest.TestCase):
    def test_se_detectan(self):
        puntos = leer("1,2\n1,2\n3,4\n1,2\n")
        self.assertEqual([tuple(p) for p in duplicates(puntos)], [(1.0, 2.0)])

    def test_sin_duplicados_da_lista_vacia(self):
        self.assertEqual(duplicates(leer("1,2\n3,4\n")), [])

    def test_deduplicate_conserva_el_orden(self):
        puntos = leer("3,4\n1,2\n3,4\n5,6\n")
        self.assertEqual(
            [tuple(p) for p in deduplicate(puntos)], [(3.0, 4.0), (1.0, 2.0), (5.0, 6.0)]
        )

    def test_los_duplicados_rompen_greedy_tree(self):
        # Fija la razon por la que `duplicates` existe.  Si upstream alguna vez
        # levanta la limitacion, este test empieza a fallar.
        from greedypermutation.balltree import greedy_tree
        from metricspaces import MetricSpace

        with self.assertRaises(TypeError):
            greedy_tree(MetricSpace(leer("1,2\n1,2\n3,4\n")))
        greedy_tree(MetricSpace(deduplicate(leer("1,2\n1,2\n3,4\n"))))


if __name__ == "__main__":
    unittest.main()
