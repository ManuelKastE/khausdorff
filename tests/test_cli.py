import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from khausdorff import hausdorff, hausdorff_naive, load_points, main
from tests.helpers import cloud, seeded


class ConDatos(unittest.TestCase):
    """Dos archivos con semilla fija, más 5 atípicos lejanos en A."""

    def setUp(self):
        self.carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(self.carpeta.cleanup)
        rng = seeded(5)
        A = [list(p) for p in cloud(120, rng)]
        A += [[5 + rng.random(), 5 + rng.random()] for _ in range(5)]
        B = [list(p) for p in cloud(80, rng, offset=0.3)]
        self.ruta_a, self.ruta_b = self.escribir("A.csv", A), self.escribir("B.csv", B)

    def escribir(self, nombre, puntos):
        ruta = Path(self.carpeta.name) / nombre
        ruta.write_text(
            "x,y\n" + "".join(",".join(f"{c:.6f}" for c in p) + "\n" for p in puntos)
        )
        return ruta

    def correr(self, *args):
        salida = io.StringIO()
        with redirect_stdout(salida):
            codigo = main([str(self.ruta_a), str(self.ruta_b), *args])
        return codigo, salida.getvalue()

    def valor(self, texto, etiqueta):
        m = re.search(rf"{etiqueta}\s*:\s*([0-9.]+)", texto)
        self.assertIsNotNone(m, f"no se encontró '{etiqueta}' en:\n{texto}")
        return float(m.group(1))


class TestDePuntaAPunta(ConDatos):
    def test_la_garantia_se_respeta(self):
        for epsilon in ("0", "0.1", "0.5"):
            for q in ("100", "95", "50"):
                codigo, texto = self.correr("--percentil", q, "--epsilon", epsilon)
                self.assertEqual(codigo, 0)
                self.assertIn("la garantía pide", texto)
                self.assertLessEqual(
                    self.valor(texto, "aproximado"),
                    self.valor(texto, "exacto") + 1e-9,
                    f"q={q} eps={epsilon}",
                )

    def test_coincide_con_la_api(self):
        A, B = load_points(self.ruta_a), load_points(self.ruta_b)
        _, texto = self.correr("--percentil", "95", "--epsilon", "0.5")
        # places=5 y no más: el CLI imprime con 6 decimales.
        self.assertAlmostEqual(
            self.valor(texto, "aproximado"), hausdorff(A, B, 95, 0.5), places=5
        )
        self.assertAlmostEqual(
            self.valor(texto, "exacto"), hausdorff_naive(A, B, 95), places=5
        )

    def test_los_atipicos_cambian_el_resultado(self):
        # Razón de ser de las distancias parciales: el percentil 100 mide los
        # cinco atípicos lejanos y el 95 los descarta.
        _, cien = self.correr("--percentil", "100")
        _, noventa_y_cinco = self.correr("--percentil", "95")
        self.assertGreater(
            self.valor(cien, "exacto"), 10 * self.valor(noventa_y_cinco, "exacto")
        )

    def test_por_omision_es_el_percentil_100(self):
        _, texto = self.correr()
        self.assertIn("percentil 100", texto)


class TestModos(ConDatos):
    def test_modo_aprox(self):
        codigo, texto = self.correr("--modo", "aprox", "--epsilon", "0.5")
        self.assertEqual(codigo, 0)
        self.assertIn("aproximado", texto)
        self.assertNotIn("exacto", texto)

    def test_modo_naive(self):
        codigo, texto = self.correr("--modo", "naive")
        self.assertEqual(codigo, 0)
        self.assertIn("exacto", texto)
        self.assertNotIn("aproximado", texto)

    def test_modo_ambos_es_el_de_omision(self):
        _, texto = self.correr()
        self.assertIn("aproximado", texto)
        self.assertIn("exacto", texto)
        self.assertIn("razón", texto)


class TestErrores(ConDatos):
    def test_dimensiones_distintas_entre_a_y_b(self):
        ruta = Path(self.carpeta.name) / "tres.csv"
        ruta.write_text("1 2 3\n4 5 6\n")
        with self.assertRaises(SystemExit) as ctx:
            main([str(ruta), str(self.ruta_b)])
        self.assertIn("coincidir", str(ctx.exception))

    def test_percentil_fuera_de_rango(self):
        with self.assertRaises(SystemExit):
            main([str(self.ruta_a), str(self.ruta_b), "--percentil", "101"])

    def test_los_repetidos_no_abortan(self):
        ruta = Path(self.carpeta.name) / "dup.csv"
        ruta.write_text("1,2\n1,2\n3,4\n5,6\n")
        salida = io.StringIO()
        with redirect_stdout(salida):
            codigo = main([str(ruta), str(self.ruta_b)])
        self.assertEqual(codigo, 0)
        self.assertIn("|A| = 3", salida.getvalue())


if __name__ == "__main__":
    unittest.main()
