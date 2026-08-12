# khausdorff

Aproxima las **distancias de Hausdorff dirigidas parciales** — todas las `d_h^(k)(A, B)` de una
vez, en un único recorrido de árbol dual.

Implementa la Sección 5 (`k-HAUSDORFF`) de

> O. A. Chubet, P. M. Parikh, D. R. Sheehy y S. S. Sheth,
> *Approximating the Directed Hausdorff Distance*,
> Computing in Geometry and Topology, 4(2):6:1–6:16, 2023.

Las Secciones 3 y 4 de ese artículo — permutaciones greedy, árboles greedy, la distancia de
Hausdorff dirigida ordinaria y el recorrido genérico de árbol dual — ya están implementadas en
[donsheehy/greedypermutation](https://github.com/donsheehy/greedypermutation). Este paquete se
apoya en aquel en lugar de duplicarlo.

## Qué calcula

La **k-ésima distancia de Hausdorff dirigida parcial** descarta los `k` peores valores atípicos
antes de medir:

```
d_h^(k)(A, B) = min sobre S en A^(k) de d_h(S, B)
```

donde `A^(k)` es la familia de subconjuntos de `A` a los que se les quitaron `k` puntos. `k = 0`
es la distancia de Hausdorff dirigida ordinaria. Es la forma estándar de hacer la distancia de
Hausdorff robusta al ruido, ya que de otro modo un único punto extraviado de `A` domina la
respuesta.

La salida es la secuencia completa `(delta_0, ..., delta_n)` — `n + 1` valores, uno por cada `k`
en `{0, …, n}`, tal como en el artículo. El último es `0`: descartar los `n` puntos de `A` no deja
nada que medir. La garantía es

```
delta_i  <=  d_h^(k=i)(A, B)  <=  (1 + epsilon) * delta_i
```

Calcular una sola de esas distancias de forma ingenua cuesta `O(|A|·|B|)`. Esto calcula *las n+1
en conjunto* en tiempo casi lineal en `|A| + |B|`, tras un preprocesamiento que se hace una vez
por conjunto de puntos.

## Instalación

```bash
pip install git+https://github.com/ManuelKastE/khausdorff.git
```

Eso es todo lo que hace falta: `greedypermutation` proviene de GitHub y no de PyPI, pero `pip`
resuelve esa dependencia git anidada por su cuenta — en
[Nota sobre la dependencia](#nota-sobre-la-dependencia) está el motivo por el que no puede venir
de PyPI. Para ejecutar los tests y la demostración, clona el repositorio; ver
[Desarrollo](#desarrollo).

## Uso

```python
from metricspaces import MetricSpace
from greedypermutation.point import Point
from greedypermutation.balltree import greedy_tree
from khausdorff import all_k_hausdorff, k_hausdorff

A = MetricSpace([Point([x, y]) for x, y in [(0, 0), (1, 0), (0, 1), (5, 5)]])
B = MetricSpace([Point([x, y]) for x, y in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]])

# Preprocesamiento, una vez por conjunto de puntos.  Los árboles se reutilizan
# entre consultas.
G_A = greedy_tree(A)
G_B = greedy_tree(B)

deltas = all_k_hausdorff(G_A, G_B, epsilon=0.1)
print(deltas[0])   # la distancia de Hausdorff dirigida ordinaria, dominada por (5,5)
print(deltas[1])   # la misma distancia tras descartar el peor valor atípico

# O pedir un único k.  Árboles nuevos: la búsqueda anterior ya consumió los
# otros (ver la tercera advertencia más abajo).
G_A, G_B = greedy_tree(A), greedy_tree(B)
print(k_hausdorff(G_A, G_B, k=1, epsilon=0.1))
```

Tres advertencias que conviene dejar claras de entrada:

- Las funciones reciben **árboles greedy**, no objetos `MetricSpace`.
- La distancia es **dirigida y asimétrica**: `all_k_hausdorff(G_A, G_B)` no es
  `all_k_hausdorff(G_B, G_A)`. La versión no dirigida es el máximo de ambas.
- Una búsqueda **consume** los árboles que recibe (muta el grafo de viabilidad construido a
  partir de ellos). Hay que construir un par nuevo para cada llamada.

### API

| Función | Devuelve |
|---|---|
| `all_k_hausdorff(G_A, G_B, epsilon=0, monotone=True, stop_after=None)` | la lista completa `(delta_0, …, delta_n)`, variante de heap exacto; con `stop_after=k` solo hasta `delta_k` |
| `k_hausdorff(G_A, G_B, k, epsilon=0)` | un único `delta_k`, cortando el recorrido |
| `all_k_hausdorff_bucket(G_A, G_B, epsilon)` | lo mismo, variante de cola de buckets β (requiere `epsilon > 0`) |
| `all_partial_hausdorff(A, B)` | la respuesta exacta por fuerza bruta, `O(|A|·|B|)`; recibe **puntos**, no árboles |
| `hausdorff_percentile(G_A, G_B, q, epsilon=0)` | el percentil `q` en vez del `k`-ésimo (ver abajo) |
| `hausdorff_percentile_bucket(G_A, G_B, q, epsilon)` | ídem, variante de buckets |
| `partial_hausdorff_percentile(A, B, q)` | ídem, exacto; recibe **puntos** |
| `k_for_percentile(n, q)` | la traducción de índice, `k = n − ceil(q·n/100)` |

`epsilon = 0` da respuestas exactas a través de `all_k_hausdorff`, al costo de hacer el recorrido
completo sin ninguna terminación temprana.

### Percentiles

El artículo indexa por **conteo**: `d_h^(k)` descarta `k` valores atípicos. En la práctica suele
ser más natural pedir un **porcentaje**. Son la misma cantidad con distinto índice, porque, como
dice la Sección 5, `d_h^(k)(A, B)` es el `(k+1)`-ésimo valor más grande de `d(a, B)`: un
estadístico de orden.

```python
from khausdorff import hausdorff_percentile, partial_hausdorff_percentile

# descartando el 5% de los puntos de A más lejanos de B
hausdorff_percentile(G_A, G_B, 95, epsilon=0.1)

# q = 100 es la distancia de Hausdorff dirigida ordinaria
hausdorff_percentile(G_A, G_B, 100)
```

Con `|A| = 100`, el percentil 95 y `delta_5` son literalmente el mismo número:

```python
partial_hausdorff_percentile(A, B, 95) == all_partial_hausdorff(A, B)[5]   # True
```

Se usa la convención **nearest-rank**: el percentil `q` es el valor en la posición
`ceil(q·n/100)` del orden ascendente, de donde `k = n − ceil(q·n/100)`. Eso hace que el resultado
sea siempre uno de los `delta_k` que el algoritmo ya calculó, y por lo tanto **hereda intacta la
garantía `(1 + ε)`** del artículo. Interpolar entre dos `delta_k`, como hace `numpy.percentile`
por omisión, devolvería un valor que el algoritmo nunca computó y la demostración de la cota
dejaría de aplicarse tal cual.

### Terminación temprana

Pedir un solo valor no paga el recorrido completo. El artículo describe **dos** algoritmos: la
*"simple modification of Hausdorff"* que corta la `(k+1)`-ésima vez que se cumple la condición de
parada, y `k-HAUSDORFF`, del que dice *"the loop runs over the whole input list"*. Este paquete
implementa los dos.

```python
all_k_hausdorff(G_A, G_B, epsilon)                  # los n+1 valores
all_k_hausdorff(G_A, G_B, epsilon, stop_after=k)    # solo (delta_0, …, delta_k)
```

`k_hausdorff`, `k_hausdorff_bucket`, `hausdorff_percentile` y `hausdorff_percentile_bucket` cortan
**por omisión**: los valores devueltos son idénticos a los del recorrido completo, solo cambia
cuánto trabajo se hace. Como `len(G_A)` da `|A|` sin recorrer nada, el `k` de un percentil se
conoce de antemano y el corte se aprovecha entero.

Cuánto se gana depende de `epsilon`, porque la condición es `r ≤ c·ℓ(x)` con `c = ε/(3+ε)`.
Medido con `n = 2000`, `|B| = 1200`, percentil 95:

| `epsilon` | recorrido completo | con corte | aceleración |
|---:|---:|---:|---:|
| 0.1 | 1.34 s | 1.20 s | 1.12× |
| 0.25 | 1.25 s | 0.93 s | 1.34× |
| 0.5 | 1.22 s | 0.68 s | 1.78× |
| 1.0 | 1.12 s | 0.43 s | 2.59× |

Tres advertencias honestas:

- **Con `epsilon = 0` no ahorra nada.** La constante de terminación es 0, la condición no se
  cumple nunca y hay que recorrer todo igual.
- **El ahorro en iteraciones no se traduce proporcionalmente en tiempo.** Con `ε = 0.5` se saltan
  el 70% de las iteraciones pero solo el 44% del tiempo: las que se saltan son las del final,
  sobre bolas pequeñas con pocas aristas, mientras que las caras están al principio.
- **La variante de buckets se beneficia mucho menos** (1.02× a 1.63×). Corta más tarde que el
  heap — con `ε = 0.5` le quedan el 56% de las iteraciones frente al 30% del heap — porque el
  umbral es un nivel de bucket y la cuantización vuelve la condición más conservadora.

Reproducible con `python benchmarks/bench_khausdorff.py --percentile 95 --variant both`.

### Dos variantes

| | heap de cotas inferiores | costo por actualización | notas |
|---|---|---|---|
| `KHausdorff` | max-heap exacto | `O(log n)` | implementación de referencia, admite `epsilon = 0` |
| `KHausdorffBucket` | cola de buckets β, `β = 1 + ε/2` | `O(1)` amortizado | Sección 5.2; alcanza la cota `(2 + 1/ε)^O(d) n + O(log_β Δ)` del artículo |

Ambas satisfacen la misma garantía. La variante de buckets reporta extremos de bucket en vez de
cotas inferiores exactas, así que sus respuestas son levemente más gruesas con el mismo
`epsilon`.

**La variante de buckets no es reproducible.** Dos ejecuciones sobre la misma entrada pueden dar
resultados distintos, ambos dentro de la garantía. `BetaBucketQueue.findmax` devuelve un elemento
arbitrario del bucket más alto —`next(iter(...))` sobre un conjunto— y los nodos se hashean por
identidad, así que el orden depende de las direcciones de memoria. Es deliberado: no distinguir
entre nodos cuyas cotas coinciden salvo por un factor `β` es justamente lo que compra el `O(1)`.
Consecuencia práctica: **no compares dos corridas de la variante de buckets con igualdad exacta**;
compara contra las cotas. `KHausdorff` sí es reproducible.

## Usar tus propios datos

Instalar el paquete deja disponible el comando `khausdorff`:

```bash
khausdorff A.csv B.csv --percentile 95 --epsilon 0.5
```

Calcula el valor pedido, lo contrasta contra la respuesta exacta por fuerza bruta, y reporta la
razón entre ambos y los tiempos de construcción y de consulta por separado.

### El formato

Un punto por línea, coordenadas separadas por **comas o espacios** —se autodetecta—. Se ignoran
las líneas vacías, las que empiezan con `#`, un encabezado como `x,y`, y todo lo que venga después
de un `;`. Cualquier dimensión, siempre que todas las filas tengan la misma.

```
x,y
0.237965,0.544229
0.369955,0.603920
# esta línea se ignora
0.625720,0.065529
```

Si aún no tienes datos, hay un generador para practicar:

```bash
python examples/generar_datos.py --n 500 --m 300 --atipicos 10 --salida datos/
khausdorff datos/A.csv datos/B.csv --percentile 100    # los atípicos dominan
khausdorff datos/A.csv datos/B.csv --percentile 95     # los descarta
```

### Opciones

| Bandera | Para qué |
|---|---|
| `--percentile Q` / `--k K` | qué valor pedir (por omisión, el percentil 100) |
| `--epsilon E` | aproximación; `0` da la respuesta exacta |
| `--variant heap\|bucket` | cuál de las dos implementaciones |
| `--todos` | mostrar la secuencia completa en vez de un solo valor |
| `--csv ARCHIVO` | volcar la secuencia completa, con precisión total |
| `--sin-exacto` | omitir la referencia por fuerza bruta, que cuesta `O(\|A\|·\|B\|)` |
| `--deduplicar` | quitar puntos repetidos en vez de abortar |

Desde Python, el mismo lector está disponible como `khausdorff.load_points`, junto con
`duplicates` y `deduplicate`.

### Tres trampas con datos reales

Las tres muerden en silencio o con errores que no explican nada, así que conviene tenerlas
presentes:

1. **Puntos exactamente duplicados rompen `greedy_tree`**, con un
   `TypeError: 'NoneType' object is not iterable` que no menciona la causa. El comando los detecta
   antes y dice cuántos hay; `--deduplicar` los quita.
2. **Filas con distinta cantidad de coordenadas no dan error al calcular.** `Point.dist` proyecta
   al subespacio común y devuelve un número equivocado sin avisar. Por eso `load_points` valida la
   dimensión e indica la línea exacta donde se rompe.
3. **Cada búsqueda consume los árboles** que recibe. El comando los reconstruye solo; si usas la
   API directamente, construye un par nuevo por consulta.

## Desviaciones respecto al artículo

Dos puntos en los que una transcripción literal de la Sección 5 no se sostiene, ambos verificados
contra la referencia de fuerza bruta sobre entradas aleatorias:

**1. Valor reportado.** El artículo agrega `ℓ(x)` a la salida una vez por cada punto de `pts(x)`.
Pero `ℓ(x)` es una cota inferior de `d(ctr(x), B)` únicamente. Un punto de `pts(x)` que quede más
cerca de `B` que el centro recibe entonces una distancia demasiado grande, y `delta_i <= d_h^(i)`
falla — de forma observable, en aproximadamente 1 de cada 20 entradas aleatorias con
`epsilon = 0.5`. Este paquete reporta `ℓ(x) − rad(x)`, que es una cota inferior para *todo* punto
del nodo, ya que `d(p, B) >= d(ctr(x), B) − rad(x) >= ℓ(x) − rad(x)`.

**2. Constante de terminación.** Restar el radio cuesta precisión, así que la condición de
terminación debe ajustarse para conservar la cota superior. Escribiendo `L = ℓ(x)` para el tope
del heap, el Lema 4 da `d_h^(i) <= L + 2r`, y el valor reportado es `delta = L − rad(x) >= (1−c)L`
bajo una condición de terminación `r <= c·L`. Entonces

```
d_h^(i)  <=  (1 + 2c) L  <=  (1 + 2c)/(1 − c) · delta
```

y exigir `(1 + 2c)/(1 − c) <= 1 + eps` da **`c <= eps/(3 + eps)`**, donde el artículo usa
`c = eps/2`. El nivel umbral de la variante de buckets se ajusta del mismo modo, de
`s = ceil(log_β(2rβ/(β−1)))` a `s = ceil(log_β(2r(3+eps)/eps))`.

Ninguno de los dos cambios afecta la complejidad asintótica: `c` sigue siendo `Θ(eps)`.

## Bugs de dependencias sorteados

Encontrados mientras se construía esto, ambos en dependencias y no en este paquete:

- **`ds2.priorityqueue.PriorityQueue.remove` corrompe el heap.** `_remove_at_index` rellena el
  hueco con la última entrada y solo la hunde hacia *abajo*, nunca hacia arriba. El fuzzing lo
  sitúa en torno al 1% de las remociones. Nada dentro de `greedypermutation` llama a `remove`,
  que es presumiblemente por qué pasó inadvertido; este algoritmo lo llama constantemente.
  Corregido en [`khausdorff/lowerboundheap.py`](khausdorff/lowerboundheap.py), con un test que
  fija el comportamiento de la dependencia para poder eliminar la subclase si `ds2` alguna vez lo
  arregla.
- **`greedypermutation.fvm.bucketqueue.BucketQueue` es inutilizable.** `insert` sobre una cola
  vacía llama a `max()` con un diccionario vacío y lanza `ValueError`, y su índice de bucket toma
  `log2` de la prioridad, lo que falla con las prioridades no positivas que las cotas inferiores
  locales toman rutinariamente. Este paquete incluye su propia
  [`BetaBucketQueue`](khausdorff/betabucketqueue.py) en vez de depender de `fvm/`.

También conviene saberlo: **`greedy_tree` no admite puntos exactamente duplicados** (la
permutación greedy se queda sin centros distintos y le entrega `None` a la métrica). Es una
precondición de la dependencia que este paquete hereda.

## Desarrollo

```bash
git clone https://github.com/ManuelKastE/khausdorff.git
cd khausdorff
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m unittest discover -s tests -t .     # o bien: pytest tests/
python examples/demo.py --n 200 --epsilon 0.3
python examples/generar_datos.py --salida datos/ && khausdorff datos/A.csv datos/B.csv --percentile 95
python benchmarks/bench_khausdorff.py --sizes 200,500,1000,2000 --variant both
```

La bandera `--compare` del benchmark decide si se cronometra en paralelo la referencia exacta
`O(n·m)` (`naive`), si se omite (`none`), o si se decide según el tamaño (`auto`, por omisión).
Medido en un portátil, nubes uniformes en 2-D con `|B| = 0.6·|A|`, `epsilon = 0.5`:

| n | construcción (s) | k-Hausdorff (s) | ingenuo (s) | aceleración |
|---:|---:|---:|---:|---:|
| 500 | 0.32 | 0.23 | 0.23 | 1.0× |
| 1000 | 0.80 | 0.52 | 0.92 | 1.7× |
| 2000 | 1.88 | 1.25 | 3.63 | 2.9× |
| 4000 | 4.60 | 2.50 | — | — |
| 8000 | 10.75 | 5.73 | — | — |

Duplicar `n` aproximadamente duplica el tiempo del recorrido, frente al 4× que paga la referencia
cuadrática.

## Nota sobre la dependencia

`greedypermutation` está en PyPI, pero **ninguna versión instalable suya contiene `dualtrees/`**
— ni la publicada en PyPI, ni `pip install git+https://github.com/donsheehy/greedypermutation.git`.
La razón es un bug de empaquetado, no una versión desactualizada: en el repositorio original,
`greedypermutation/dualtrees/` y `greedypermutation/fvm/` **no tienen `__init__.py`**, de modo que
`setuptools.find_packages()` en su `setup.py` nunca los ve y todos los wheels salen sin esos
subpaquetes. (`hausdorff.py` es un módulo de nivel superior, así que ese sí se instala.)

Por eso este paquete depende de un **fork**,
[ManuelKastE/greedypermutation](https://github.com/ManuelKastE/greedypermutation), rama
`packaging-fix`. Es el original más dos commits, ambos de una línea en el empaquetado y ninguno
en los algoritmos:

1. **Agregar los `__init__.py` faltantes** en `greedypermutation/dualtrees/` y
   `greedypermutation/fvm/`, para que `find_packages()` deje de saltárselos.
2. **Dejar de empaquetar el `tests/` del propio repositorio.** `find_packages()` sin `exclude`
   también recoge el paquete `tests` de nivel superior y lo instala en `site-packages`, donde
   tapa la suite de tests de cualquier proyecto instalado junto a él: un `tests/` sin
   `__init__.py` es una *namespace portion*, que pierde frente a ese paquete regular sin importar
   el orden de `sys.path`. Esta suite se topó exactamente con eso. El fork pasa
   `exclude=["tests", "tests.*"]`.

El efecto se ve directamente en lo que `setup.py` descubre:

```console
$ python -c "import setuptools; print(sorted(setuptools.find_packages()))"
['greedypermutation', 'tests']                                                     # original
['greedypermutation', 'greedypermutation.dualtrees', 'greedypermutation.fvm']      # fork
```

`pyproject.toml` fija la punta del fork por SHA en lugar de por rama, de manera que un push
posterior a `packaging-fix` no pueda cambiar lo que resuelve una instalación existente:

```toml
dependencies = [
  "greedypermutation @ git+https://github.com/ManuelKastE/greedypermutation.git@793a33f3ce3716989d9a9496429dc6a71fca8565",
]
```

Esto amerita un pull request al repositorio original: el cambio son dos archivos vacíos y hace
`dualtrees/` instalable para todo el mundo.

Una consecuencia de tener una dependencia por URL directa: este paquete **no puede subirse a
PyPI**. Instalarlo desde GitHub funciona sin problemas, y pip resuelve la dependencia git anidada
por su cuenta.

## Créditos

El algoritmo es de Chubet, Parikh, Sheehy y Sheth (2023), citado más arriba. La permutación
greedy, los árboles greedy y la maquinaria de árbol dual sobre la que esto se construye son de
Donald R. Sheehy, [greedypermutation](https://github.com/donsheehy/greedypermutation), con
licencia MIT.

Este paquete tiene licencia MIT; ver [LICENSE](LICENSE).
