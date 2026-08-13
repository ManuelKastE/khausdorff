# khausdorff

Distancia de Hausdorff dirigida **parcial**: la que queda tras descartar los peores valores
atípicos.

```bash
python3 khausdorff.py datos/A.csv datos/B.csv --percentil 95
```

```
|A| = 510, |B| = 300, dimensión = 2, percentil 95, epsilon 0

  aproximado :   0.295817   [ 0.726 s]
  exacto     :   0.295817   [ 0.248 s]  fuerza bruta
  razón      :     1.0000   (la garantía pide <= 1)
```

## Qué calcula

La distancia de Hausdorff dirigida `d_h(A, B)` mide cuán lejos está el punto de `A` más alejado
de `B`. Un único punto extraviado la domina, así que la versión robusta descarta los `k` peores
antes de medir:

```
d_h^(k)(A, B) = min sobre S en A^(k) de d_h(S, B)
```

donde `A^(k)` son los subconjuntos de `A` a los que se les quitaron `k` puntos. Acá eso se pide
como **percentil**: el percentil 95 es la distancia que queda tras ignorar el 5% de los puntos
de `A` más lejanos de `B`, y el 100 es la distancia de Hausdorff dirigida de siempre.

Hay dos formas de calcularlo:

- **El algoritmo rápido**, casi lineal en `|A| + |B|`, con la garantía
  `delta <= verdadera <= (1 + epsilon) * delta`. Implementa la Sección 5 (`k-HAUSDORFF`) de

  > O. A. Chubet, P. M. Parikh, D. R. Sheehy y S. S. Sheth,
  > *Approximating the Directed Hausdorff Distance*,
  > Computing in Geometry and Topology, 4(2):6:1–6:16, 2023.

- **La fuerza bruta**, exacta y sin parámetros, que cuesta `O(|A|·|B|)`.

Por omisión se corren las dos y se contrastan. Las Secciones 3 y 4 del artículo —permutaciones
greedy, árboles greedy y el recorrido de árbol dual— ya están en
[donsheehy/greedypermutation](https://github.com/donsheehy/greedypermutation); esto se apoya en
aquel en vez de duplicarlo.

Dos cosas que definen para qué sirve esto:

- **La distancia es dirigida:** intercambiar `A` y `B` da otra respuesta.
- **Está pensado para 2 o 3 dimensiones**, y desde unos mil puntos. Funciona en cualquier
  dimensión y las respuestas siempre son correctas, pero el costo del algoritmo rápido es
  exponencial en la dimensión: desde `d = 3` la fuerza bruta ya conviene, y en `d = 8` le gana
  por 141×. Los números están en
  [Rendimiento y complejidad](#rendimiento-y-complejidad).

## Instalación

No hay nada que instalar más que las dependencias:

```bash
pip install -r requirements.txt
```

`greedypermutation` viene de un **fork** y no de PyPI, y tiene que ser así: upstream no incluye
`__init__.py` en `greedypermutation/dualtrees/`, así que `setuptools` se saltea el subpaquete y
ningún build suyo lo trae. El fork solo agrega ese archivo, y está pinneado a un commit para que
no se mueva bajo los pies.

## El formato de los archivos

Un punto por línea, coordenadas separadas por **comas o espacios** —se autodetecta—. Se ignoran
las líneas vacías, las que empiezan con `#`, un encabezado como `x,y`, y todo lo que venga
después de un `;`. Cualquier dimensión, siempre que todas las filas tengan la misma.

```
x,y
0.237965,0.544229
0.369955,0.603920
# esta línea se ignora
0.625720,0.065529
```

Si no tenés datos a mano, hay un generador:

```bash
python3 generar_datos.py --tamano low --salida datos/
```

Escribe `A.csv` y `B.csv` con un número aleatorio de puntos cada uno, sin repetidos.
`--tamano` es `low` (20–100 puntos), `mid` (100–1000) o `max` (1000–5000).

## Opciones

| Bandera | Para qué |
|---|---|
| `--percentil Q` | qué valor pedir; 100 por omisión, la distancia de Hausdorff dirigida |
| `--epsilon E` | aproximación del algoritmo rápido; 0 por omisión, que lo hace exacto |
| `--modo {ambos,aprox,naive}` | qué calcular; `ambos` por omisión, para contrastarlos |

`--epsilon 0` da la respuesta exacta, al costo de recorrer todo el árbol sin terminación
temprana. Con `epsilon > 0` el recorrido se abandona apenas se conoce el valor pedido, así que
el percentil 95 cuesta casi lo mismo que el 100.

Con entradas grandes conviene `--modo aprox`: la fuerza bruta es cuadrática. Antes de elegir,
conviene leer [Rendimiento y complejidad](#rendimiento-y-complejidad): por debajo de unos mil
puntos, o en dimensión alta, la fuerza bruta gana.

## Desde Python

```python
from khausdorff import hausdorff, hausdorff_naive, all_distances

hausdorff("datos/A.csv", "datos/B.csv")                     # la distancia dirigida
hausdorff("datos/A.csv", "datos/B.csv", 95, epsilon=0.1)    # descartando el 5%
hausdorff_naive("datos/A.csv", "datos/B.csv", 95)           # exacto, O(|A|·|B|)
all_distances(A, B, epsilon=0.1)                            # (delta_0, ..., delta_n)
```

Todas aceptan rutas, archivos abiertos o listas de puntos, y construyen solas los árboles greedy
que necesitan. `all_distances` devuelve la secuencia completa: `delta_k` es la distancia tras
descartar los `k` puntos más lejanos, es no creciente, y el último valor es `0`.

Dos cosas que conviene saber:

- **Los puntos repetidos se descartan al entrar**, avisando cuántos. No es cosmético:
  `greedy_tree` se queda sin centros distintos y le entrega `None` a la métrica, lo que sale como
  un `TypeError` que no dice nada sobre la causa.
- El percentil usa la convención **nearest-rank**, sin interpolar entre dos valores. Eso hace que
  el resultado sea siempre uno de los `delta_k` que el algoritmo ya calculó, y por lo tanto que
  **herede intacta la garantía `(1 + ε)`**. `numpy.percentile` interpola por omisión y devolvería
  un valor que el algoritmo nunca computó.

### Reusar el árbol entre consultas

`hausdorff()` construye los árboles en cada llamada, y construirlos es lo más caro de todo. Los
árboles **sí se pueden reusar**: `Ball` se arma una vez y la búsqueda solo lo lee; todo lo que se
muta lo crea de cero cada `KHausdorff`. Para muchas consultas sobre los mismos puntos conviene
bajar a la clase:

```python
from khausdorff import KHausdorff, _as_points, _trees, _k_for_percentile

A, B = _as_points("datos/A.csv"), _as_points("datos/B.csv")
G_A, G_B = _trees(A, B)                       # una sola vez

for q in (100, 99, 95, 90):
    k = _k_for_percentile(len(A), q)
    print(q, KHausdorff(G_A, G_B, 0.5)(stop_after=k)[k])
```

Ahora bien, esto rara vez conviene. Si el par `(A, B)` es fijo, `all_distances(A, B)` te da los
`n + 1` valores de **todos** los percentiles en un solo recorrido: guardas ese arreglo y respondes
cualquier percentil por indexación, lo que es más chico que el árbol y más rápido de consultar.
Guardar el árbol paga solo cuando **no puedes enumerar las consultas de antemano** — una
referencia fija contra nubes que van llegando, o una colección comparada de a pares.

## Rendimiento y complejidad

Con `n = |A|`, `m = |B|`, `d` la dimensión doblante y `Δ` la dispersión (razón entre la distancia
máxima y la mínima entre pares):

| Pieza | Complejidad | Exponente medido |
|---|---|---:|
| `greedy_tree` (construcción, ×2) | `2^O(d) · n log Δ` | `n^1.31` |
| consulta, `ε > 0` | `(2 + 1/ε)^O(d) · n log n + O(log Δ)` | `n^1.05` (ε=0.5), `n^1.00` (ε=1) |
| consulta, `ε = 0` | sin cota — ver abajo | `n^1.27` |
| `hausdorff_naive` | `Θ(n · m · d)` | `n^2.00` |

Los exponentes son la pendiente log-log medida entre 250 y 2000 puntos, en 2D. El `n^2.00` de la
fuerza bruta y el `n^1.00` de la consulta con `ε = 1` son casi exactos.

**Obtener el percentil no cuesta nada extra.** Es el punto central del artículo: calcular
`d_h(A, B)` y calcular las `n + 1` distancias parciales es el mismo recorrido. Si acaso, es al
revés — el corte temprano hace que los percentiles altos salgan más baratos.

### Está pensado para 2 o 3 dimensiones

El factor `(2 + 1/ε)^O(d)` es **exponencial en la dimensión**. Funciona en cualquier dimensión y
las respuestas siguen siendo correctas, pero se degrada rápido. Con `n = m = 700`, percentil 100,
`ε = 0.5`:

| dim | árbol | consulta | total | naive | |
|---:|---:|---:|---:|---:|:---|
| 2 | 0.68 | 0.36 | 1.03 | 0.74 | |
| 3 | 1.73 | 2.81 | 4.54 | 0.83 | naive 5.4× |
| 5 | 10.39 | 32.59 | 42.99 | 1.02 | naive 42× |
| 8 | 54.68 | 129.80 | 184.48 | 1.30 | naive 141× |
| 12 | — | — | no terminó en 200 s | 1.6 | |

La fuerza bruta apenas se mueve (`Θ(n·m·d)`, lineal en `d`): de 0.74 s a 1.30 s. El algoritmo del
árbol se multiplica por 179 entre `d = 2` y `d = 8`, unas 2.4× por dimensión. **Desde `d = 3` en
adelante, la fuerza bruta conviene.**

### En 2D, desde cuándo conviene

Percentil 100 con `ε = 0.5`, `n = m`:

| n=m | árbol | consulta | total | naive | |
|---:|---:|---:|---:|---:|:---|
| 1000 | 1.04 | 0.48 | 1.52 | 1.55 | empate |
| 2000 | 2.80 | 0.95 | 3.75 | 6.16 | **1.6×** |
| 3000 | 4.99 | 1.43 | 6.43 | 13.95 | **2.2×** |
| 4000 | 7.08 | 1.72 | 8.80 | 25.17 | **2.9×** |

El cruce está alrededor de los **1000 puntos**, y la brecha crece como `n^0.74`. El percentil que
pidas lo mueve: pedir el 100 corta el recorrido mucho antes que pedir el 50, así que con
percentiles bajos el cruce se corre a la derecha.

Nota que **la construcción del árbol domina**, no el algoritmo: 7.08 s contra 1.72 s en
`n = 4000`. El cuello de botella está en `greedy_tree`, que es de la dependencia.

### Tres advertencias

- **`ε = 0` no tiene la cota, y es el valor por omisión.** La constante de terminación es
  `c = ε/(3+ε)`, así que con `ε = 0` da `c = 0` y la condición `r ≤ c·l(x)` no se dispara nunca:
  el recorrido baja hasta las hojas y el factor `(2 + 1/ε)^O(d)` diverge. Medido da `n^1.27` —
  mejor que cuadrático, pero no lineal y sin garantía teórica. El régimen lineal empieza recién
  con `ε > 0`.
- **`Δ` es propiedad de los datos, no de `n`.** No rompe la asintótica —`n log Δ` sigue siendo
  esencialmente lineal— pero infla la constante. Puntos casi repetidos, a `1e-9` uno del otro,
  llevan `log₂Δ` de ~11 a ~30 y corren el cruce hacia la derecha.
- **`--modo ambos`, que es el de omisión, cuesta `Θ(n·m)`**: la fuerza bruta domina todo lo demás.
  Sirve para contrastar, no para producción.

## Los archivos

| Archivo | Qué hay |
|---|---|
| `khausdorff.py` | todo lo principal: lectura de archivos, el algoritmo, la fuerza bruta y el CLI |
| `buckets.py` | la variante de la Sección 5.2 (cola de buckets β), que **nadie importa** |
| `generar_datos.py` | el generador de datos de prueba |
| `tests/` | `python3 -m pytest tests/ -q` desde la raíz |

### Por qué `buckets.py` está fuera del camino

La Sección 5.2 reemplaza el max-heap exacto de cotas inferiores por cajones geométricos de razón
`β = 1 + ε/2`: como ya estamos aproximando, dos nodos cuyas cotas difieren en menos de un factor
`β` no hace falta distinguirlos, y las operaciones bajan de `O(log n)` a `O(1)`.

En la práctica es **más lenta**, entre 1.2× y 2.7×, y la brecha no se cierra al crecer `n`.
Contando operaciones con `n = m = 2000` y `ε = 1`:

| | `dist()` | ops de cola | |
|---|---:|---:|:---|
| heap exacto | 383.747 | 6.557 | 1 op cada 59 distancias |
| buckets | 628.295 | 12.817 | 1 op cada 49 distancias |

Las operaciones de cola son el **1.7% del trabajo**, así que no hay nada que ganar ahí; y los
buckets hacen un 65% más de cómputo de distancias, porque su test de terminación es más grueso
(el umbral es un nivel de cajón, hasta un factor `β` más conservador que `r ≤ c·l(x)`, así que
los nodos sobreviven más y se siguen dividiendo). De hecho hacen *más* operaciones de cola, no
menos.

Es estructural: cada operación de cola viene precedida de un `lower_bound(node)`, que es un `min`
sobre toda la vecindad del nodo, así que la razón cola:distancias queda fija en `1 : 2^O(d)` para
cualquier `n`. En dimensión baja ese denominador es grande. El resultado del artículo es correcto
en su modelo de costos; simplemente no es el modelo de costos de esta implementación.

Súmale que **no es reproducible entre corridas**: el orden dentro de un cajón sale del hash de los
objetos, o sea de las direcciones de memoria. Eso es deliberado y es justamente lo que compra el
`O(1)`, pero significa que no hay que compararla por igualdad exacta, sino contra las cotas.

## Licencia

MIT, igual que `greedypermutation`. Ver [LICENSE](LICENSE).
