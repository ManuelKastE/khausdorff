import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from greedypermutation.balltree import greedy_tree
from metricspaces import MetricSpace

from khausdorff.cli import main
from khausdorff.io import load_points
from khausdorff.percentile import hausdorff_percentile, partial_hausdorff_percentile
from tests.helpers import TOL, cloud, seeded


def escribir(ruta, puntos, encabezado=True):
    with open(ruta, "w") as handle:
        if encabezado:
            handle.write("x,y\n")
        for punto in puntos:
            handle.write(",".join(f"{c:.6f}" for c in punto) + "\n")


class ConDatos(unittest.TestCase):
    """Dos archivos con semilla fija, mas 5 atipicos lejanos en A."""

    def setUp(self):
        self.carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(self.carpeta.cleanup)
        rng = seeded(5)
        self.A = [list(p) for p in cloud(120, rng)]
        self.A += [[5 + rng.random(), 5 + rng.random()] for _ in range(5)]
        self.B = [list(p) for p in cloud(80, rng, offset=0.3)]
        self.ruta_a = Path(self.carpeta.name) / "A.csv"
        self.ruta_b = Path(self.carpeta.name) / "B.csv"
        escribir(self.ruta_a, self.A)
        escribir(self.ruta_b, self.B)

    def correr(self, *args):
        salida = io.StringIO()
        with redirect_stdout(salida):
            codigo = main([str(self.ruta_a), str(self.ruta_b), *args])
        return codigo, salida.getvalue()

    def valor(self, texto, etiqueta):
        m = re.search(rf"valor {etiqueta}\s*:\s*([0-9.]+)", texto)
        self.assertIsNotNone(m, f"no se encontro 'valor {etiqueta}' en:\n{texto}")
        return float(m.group(1))


class TestDePuntaAPunta(ConDatos):
    def test_las_cotas_se_respetan(self):
        for epsilon in ("0", "0.1", "0.5"):
            for q in ("100", "95", "50"):
                codigo, texto = self.correr("--percentile", q, "--epsilon", epsilon)
                self.assertEqual(codigo, 0)
                self.assertIn("ambas cotas se respetan", texto)
                aprox, exacto = self.valor(texto, "aproximado"), self.valor(texto, "exacto")
                self.assertLessEqual(aprox, exacto + TOL, f"q={q} eps={epsilon}")

    def test_coincide_con_la_api(self):
        # El numero que imprime el CLI tiene que ser el mismo que devuelve la
        # funcion llamada directamente sobre los mismos puntos.
        A, B = load_points(self.ruta_a), load_points(self.ruta_b)
        G_A, G_B = greedy_tree(MetricSpace(A)), greedy_tree(MetricSpace(B))
        esperado = hausdorff_percentile(G_A, G_B, 95, 0.5)
        _, texto = self.correr("--percentile", "95", "--epsilon", "0.5")
        self.assertAlmostEqual(self.valor(texto, "aproximado"), esperado, places=6)

    def test_los_atipicos_cambian_el_resultado(self):
        # Razon de ser de las distancias parciales: el percentil 100 mide los
        # cinco atipicos lejanos, el 95 los descarta.
        _, cien = self.correr("--percentile", "100")
        _, noventa_y_cinco = self.correr("--percentile", "95")
        self.assertGreater(
            self.valor(cien, "aproximado"), 10 * self.valor(noventa_y_cinco, "aproximado")
        )

    def test_percentil_100_es_el_exacto_con_epsilon_cero(self):
        # `places=5` y no mas: el CLI imprime con 6 decimales, asi que el valor
        # que se lee de la salida esta redondeado.  La precision completa esta
        # en el CSV, no en la pantalla.
        A, B = load_points(self.ruta_a), load_points(self.ruta_b)
        _, texto = self.correr("--percentile", "100", "--epsilon", "0")
        self.assertAlmostEqual(
            self.valor(texto, "aproximado"),
            partial_hausdorff_percentile(A, B, 100),
            places=5,
        )


class TestBanderas(ConDatos):
    def test_k_en_vez_de_percentil(self):
        codigo, texto = self.correr("--k", "5", "--epsilon", "0.5")
        self.assertEqual(codigo, 0)
        self.assertIn("k = 5 de 125", texto)

    def test_todos_muestra_la_secuencia(self):
        _, texto = self.correr("--percentile", "95", "--todos", "--filas", "4")
        self.assertIn("razón", texto)
        self.assertIn("más)", texto)

    def test_csv_tiene_una_fila_por_k(self):
        destino = Path(self.carpeta.name) / "salida.csv"
        self.correr("--percentile", "95", "--csv", str(destino))
        filas = destino.read_text().strip().splitlines()
        self.assertEqual(filas[0], "k,aproximado,exacto")
        self.assertEqual(len(filas), len(self.A) + 2)  # encabezado + n + 1 valores

    def test_sin_exacto_no_calcula_la_referencia(self):
        _, texto = self.correr("--percentile", "95", "--sin-exacto")
        self.assertNotIn("valor exacto", texto)
        self.assertIn("valor aproximado", texto)

    def test_variante_bucket(self):
        codigo, texto = self.correr("--percentile", "95", "--epsilon", "0.5", "--variant", "bucket")
        self.assertEqual(codigo, 0)
        self.assertIn("ambas cotas se respetan", texto)

    def test_por_omision_es_el_percentil_100(self):
        _, texto = self.correr()
        self.assertIn("percentil 100", texto)


class TestErrores(ConDatos):
    def test_duplicados_se_detectan_antes_de_construir_el_arbol(self):
        ruta = Path(self.carpeta.name) / "dup.csv"
        escribir(ruta, [[1, 2], [1, 2], [3, 4]])
        with self.assertRaises(SystemExit) as ctx:
            main([str(ruta), str(self.ruta_b), "--percentile", "100"])
        mensaje = str(ctx.exception)
        self.assertIn("repetido", mensaje)
        self.assertIn("--deduplicar", mensaje)

    def test_deduplicar_permite_continuar(self):
        ruta = Path(self.carpeta.name) / "dup.csv"
        escribir(ruta, [[1, 2], [1, 2], [3, 4], [5, 6]])
        salida = io.StringIO()
        with redirect_stdout(salida):
            codigo = main([str(ruta), str(self.ruta_b), "--deduplicar"])
        self.assertEqual(codigo, 0)
        self.assertIn("se quitaron 1", salida.getvalue())

    def test_dimensiones_distintas_entre_a_y_b(self):
        ruta = Path(self.carpeta.name) / "tres.csv"
        ruta.write_text("1 2 3\n4 5 6\n")
        with self.assertRaises(SystemExit) as ctx:
            main([str(ruta), str(self.ruta_b)])
        self.assertIn("coincidir", str(ctx.exception))

    def test_bucket_rechaza_epsilon_cero(self):
        with self.assertRaises(SystemExit) as ctx:
            main([str(self.ruta_a), str(self.ruta_b), "--variant", "bucket"])
        self.assertIn("epsilon", str(ctx.exception))

    def test_k_fuera_de_rango(self):
        with self.assertRaises(SystemExit):
            main([str(self.ruta_a), str(self.ruta_b), "--k", "9999"])


if __name__ == "__main__":
    unittest.main()
