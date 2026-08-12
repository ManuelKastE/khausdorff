# khausdorff

Approximate **partial directed Hausdorff distances** — every `d_h^(k)(A, B)` at once, in a
single dual-tree traversal.

This implements Section 5 (`k-HAUSDORFF`) of

> O. A. Chubet, P. M. Parikh, D. R. Sheehy and S. S. Sheth,
> *Approximating the Directed Hausdorff Distance*,
> Computing in Geometry and Topology, 4(2):6:1–6:16, 2023.

Sections 3 and 4 of that paper — greedy permutations, greedy trees, the ordinary directed
Hausdorff distance and the generic dual-tree traversal — are already implemented in
[donsheehy/greedypermutation](https://github.com/donsheehy/greedypermutation). This package
builds on that one rather than duplicating it.

## What it computes

The **k-th partial directed Hausdorff distance** discards the `k` worst outliers before
measuring:

```
d_h^(k)(A, B) = min over S in A^(k) of d_h(S, B)
```

where `A^(k)` is the family of subsets of `A` with `k` points removed. `k = 0` is the ordinary
directed Hausdorff distance. It is the standard way to make the Hausdorff distance robust to
noise, since a single stray point of `A` otherwise dominates the answer.

The output is the whole sequence `(delta_0, ..., delta_{n-1})`, with the guarantee

```
delta_i  <=  d_h^(k=i)(A, B)  <=  (1 + epsilon) * delta_i
```

Computing one such distance naively costs `O(|A|·|B|)`. This computes *all n+1 of them* in
time near-linear in `|A| + |B|`, after a one-off preprocessing step per point set.

## Install

```bash
pip install git+https://github.com/ManuelKastE/khausdorff.git
```

That is all that is needed: `greedypermutation` comes from GitHub rather than from PyPI, but
`pip` resolves that nested git dependency on its own — see [Dependency note](#dependency-note)
for why it cannot come from PyPI. To run the tests and the demo, clone the repository instead;
see [Development](#development).

## Usage

```python
from metricspaces import MetricSpace
from greedypermutation.point import Point
from greedypermutation.balltree import greedy_tree
from khausdorff import all_k_hausdorff, k_hausdorff

A = MetricSpace([Point([x, y]) for x, y in [(0, 0), (1, 0), (0, 1), (5, 5)]])
B = MetricSpace([Point([x, y]) for x, y in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]])

# Preprocessing, once per point set.  The trees are reusable across queries.
G_A = greedy_tree(A)
G_B = greedy_tree(B)

deltas = all_k_hausdorff(G_A, G_B, epsilon=0.1)
print(deltas[0])   # the ordinary directed Hausdorff distance, dominated by (5,5)
print(deltas[1])   # the same distance after discarding the worst outlier

# Or ask for a single k:
print(k_hausdorff(G_A, G_B, k=1, epsilon=0.1))
```

Two caveats worth stating up front:

- The functions take **greedy trees**, not `MetricSpace` objects.
- The distance is **directed and asymmetric**: `all_k_hausdorff(G_A, G_B)` is not
  `all_k_hausdorff(G_B, G_A)`. The undirected version is the max of the two.
- A search **consumes** the trees it is given (it mutates the viability graph built from them).
  Build a fresh pair for each call.

### API

| Function | Returns |
|---|---|
| `all_k_hausdorff(G_A, G_B, epsilon=0, monotone=True)` | the full list `(delta_0, …, delta_{n-1})`, exact-heap variant |
| `k_hausdorff(G_A, G_B, k, epsilon=0)` | a single `delta_k` |
| `all_k_hausdorff_bucket(G_A, G_B, epsilon)` | same, β-bucket-queue variant (needs `epsilon > 0`) |
| `all_partial_hausdorff(A, B)` | the exact answer by brute force, `O(|A|·|B|)`; takes **points**, not trees |

`epsilon = 0` gives exact answers via `all_k_hausdorff`, at the cost of doing the full
traversal without any early finishing.

### Two variants

| | lower bound heap | per-update cost | notes |
|---|---|---|---|
| `KHausdorff` | exact max-heap | `O(log n)` | reference implementation, supports `epsilon = 0` |
| `KHausdorffBucket` | β-bucket queue, `β = 1 + ε/2` | `O(1)` amortised | Section 5.2; reaches the paper's `(2 + 1/ε)^O(d) n + O(log_β Δ)` bound |

Both satisfy the same guarantee. The bucket variant reports bucket endpoints rather than exact
lower bounds, so its answers are slightly coarser at the same `epsilon`.

## Deviations from the paper

Two places where a literal transcription of Section 5 does not hold up, both verified against
the brute-force reference on randomised inputs:

**1. Reported value.** The paper appends `ℓ(x)` to the output once for each point of `pts(x)`.
But `ℓ(x)` is a lower bound on `d(ctr(x), B)` only. A point of `pts(x)` lying closer to `B` than
the centre does then gets credited with too large a distance, and `delta_i <= d_h^(i)` fails —
observably, on about 1 in 20 random inputs at `epsilon = 0.5`. This package reports
`ℓ(x) − rad(x)`, which is a lower bound for *every* point of the node, since
`d(p, B) >= d(ctr(x), B) − rad(x) >= ℓ(x) − rad(x)`.

**2. Finishing constant.** Subtracting the radius costs accuracy, so the finishing condition has
to tighten to keep the upper bound. Writing `L = ℓ(x)` for the top of the heap, Lemma 4 gives
`d_h^(i) <= L + 2r`, and the reported value is `delta = L − rad(x) >= (1−c)L` under a finishing
condition `r <= c·L`. Then

```
d_h^(i)  <=  (1 + 2c) L  <=  (1 + 2c)/(1 − c) · delta
```

and requiring `(1 + 2c)/(1 − c) <= 1 + eps` gives **`c <= eps/(3 + eps)`**, where the paper uses
`c = eps/2`. The bucket variant's threshold level tightens the same way, from
`s = ceil(log_β(2rβ/(β−1)))` to `s = ceil(log_β(2r(3+eps)/eps))`.

Neither change affects the asymptotics: `c` is still `Θ(eps)`.

## Upstream bugs worked around

Found while building this, both in dependencies rather than in this package:

- **`ds2.priorityqueue.PriorityQueue.remove` corrupts the heap.** `_remove_at_index` fills the
  hole with the last entry and only sifts it *down*, never up. Fuzzing puts this at roughly 1%
  of removals. Nothing in `greedypermutation` calls `remove`, which is presumably why it went
  unnoticed; this algorithm calls it constantly. Fixed in
  [`khausdorff/lowerboundheap.py`](khausdorff/lowerboundheap.py), with a test pinning the
  upstream behaviour so the subclass can be dropped if `ds2` ever fixes it.
- **`greedypermutation.fvm.bucketqueue.BucketQueue` is unusable.** `insert` on an empty queue
  calls `max()` on an empty dict and raises `ValueError`, and its bucket index takes `log2` of
  the priority, which fails on the non-positive priorities that local lower bounds routinely
  take. This package ships its own
  [`BetaBucketQueue`](khausdorff/betabucketqueue.py) instead of depending on `fvm/`.

Also worth knowing: **`greedy_tree` cannot handle exactly duplicated points** (the greedy
permutation runs out of distinct centres and hands `None` to the metric). That is an upstream
precondition this package inherits.

## Development

```bash
git clone https://github.com/ManuelKastE/khausdorff.git
cd khausdorff
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m unittest discover -s tests -t .     # or: pytest tests/
python examples/demo.py --n 200 --epsilon 0.3
python benchmarks/bench_khausdorff.py --sizes 200,500,1000,2000 --variant both
```

The benchmark's `--compare` flag chooses whether to time the exact `O(n·m)` reference alongside
(`naive`), skip it (`none`), or decide by size (`auto`, the default). Measured on a laptop, 2-D
uniform clouds with `|B| = 0.6·|A|`, `epsilon = 0.5`:

| n | build (s) | k-Hausdorff (s) | naive (s) | speedup |
|---:|---:|---:|---:|---:|
| 500 | 0.32 | 0.23 | 0.23 | 1.0× |
| 1000 | 0.80 | 0.52 | 0.92 | 1.7× |
| 2000 | 1.88 | 1.25 | 3.63 | 2.9× |
| 4000 | 4.60 | 2.50 | — | — |
| 8000 | 10.75 | 5.73 | — | — |

Doubling `n` roughly doubles the traversal time, against the 4× the quadratic reference pays.

## Dependency note

`greedypermutation` is on PyPI, but **no installable build of it contains `dualtrees/`** — not
the PyPI release, and not `pip install git+https://github.com/donsheehy/greedypermutation.git`
either. The reason is a packaging bug rather than a stale release: upstream's
`greedypermutation/dualtrees/` and `greedypermutation/fvm/` have **no `__init__.py`**, so
`setuptools.find_packages()` in `setup.py` never sees them and every wheel comes out missing
the subpackages. (`hausdorff.py` is a top-level module, so it does get installed.)

That is why this package depends on a **fork**,
[ManuelKastE/greedypermutation](https://github.com/ManuelKastE/greedypermutation), branch
`packaging-fix`. It is upstream plus two commits, both one-liners in the packaging and none in
the algorithms:

1. **Add the missing `__init__.py`** to `greedypermutation/dualtrees/` and
   `greedypermutation/fvm/`, so `find_packages()` stops skipping them.
2. **Stop packaging the repository's own `tests/`.** `find_packages()` with no `exclude` also
   picks up the top-level `tests` package and installs it into `site-packages`, where it
   shadows the test suite of any project installed alongside it — a downstream `tests/` with no
   `__init__.py` is a namespace portion, which loses to that regular package regardless of
   `sys.path` order. This suite hit exactly that. The fork passes
   `exclude=["tests", "tests.*"]`.

The effect is visible directly in what `setup.py` discovers:

```console
$ python -c "import setuptools; print(sorted(setuptools.find_packages()))"
['greedypermutation', 'tests']                                                     # upstream
['greedypermutation', 'greedypermutation.dualtrees', 'greedypermutation.fvm']      # fork
```

`pyproject.toml` pins the fork's tip by SHA rather than by branch, so a later push to
`packaging-fix` cannot change what an existing install resolves to:

```toml
dependencies = [
  "greedypermutation @ git+https://github.com/ManuelKastE/greedypermutation.git@793a33f3ce3716989d9a9496429dc6a71fca8565",
]
```

This is worth a pull request upstream: the change is two empty files and it makes `dualtrees/`
installable for everyone.

One consequence of a direct-URL dependency: this package **cannot be uploaded to PyPI**.
Installing from GitHub works fine, and pip resolves the nested git dependency on its own.

## Credits

The algorithm is from Chubet, Parikh, Sheehy and Sheth (2023), cited above. The greedy
permutation, greedy tree and dual-tree machinery this builds on are Donald R. Sheehy's
[greedypermutation](https://github.com/donsheehy/greedypermutation), MIT licensed.

This package is MIT licensed; see [LICENSE](LICENSE).
