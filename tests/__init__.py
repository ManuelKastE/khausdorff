"""Test suite for `khausdorff`.

This file is not decorative.  The tests import each other as
`from tests.helpers import ...`, and without an `__init__.py` the directory is
only a PEP 420 namespace portion, which loses name resolution to any *regular*
top-level `tests` package installed in the environment.  `greedypermutation`
used to install exactly such a package; making this one regular means the suite
resolves to itself no matter what else is on the path.

`pyproject.toml` restricts packaging to `khausdorff*`, so this package is never
shipped -- adding it here does not repeat the mistake it guards against.
"""
