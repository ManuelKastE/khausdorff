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

**La distancia es dirigida:** intercambiar `A` y `B` da otra respuesta.

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

Con entradas grandes conviene `--modo aprox`: la fuerza bruta es cuadrática.

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

## Los archivos

| Archivo | Qué hay |
|---|---|
| `khausdorff.py` | todo lo principal: lectura de archivos, el algoritmo, la fuerza bruta y el CLI |
| `buckets.py` | la variante de la Sección 5.2 (cola de buckets β), que **nadie importa** |
| `generar_datos.py` | el generador de datos de prueba |
| `tests/` | `python3 -m pytest tests/ -q` desde la raíz |

`buckets.py` está aparte a propósito. Es la variante que alcanza el tiempo de ejecución que
afirma el artículo, y da valores dentro de la misma garantía, pero **ni siquiera es reproducible
entre corridas**: el orden dentro de un bucket sale del hash de los objetos, o sea de las
direcciones de memoria. Eso es deliberado —no distinguir dos nodos cuyas cotas difieren en menos
de un factor β es justamente lo que compra el `O(1)`—, pero significa que no hay que compararla
por igualdad exacta, sino contra las cotas.

## Licencia

MIT, igual que `greedypermutation`. Ver [LICENSE](LICENSE).
