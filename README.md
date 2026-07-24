# kloosterman_twisted

Python package for computing **untwisted** and **modular-symbol twisted**
Kloosterman sums, as defined in Diamantis - Friedberg - Stromberg "SUMS OF KLOOSTERMAN SUMS FORMED WITH MODULAR SYMBOLS"

## Prerequisites

- Python 3.9+
- NumPy 1.23+

This package can be run using only Python with NumPy but if SageMath is present it will make use of Modular form coefficients computed by PARI as these are much more
optimized than the naive point countings in this module.


## Layout

```
kloosterman-twisted/
├── pyproject.toml
├── README.md
├── tests/
└── src/
    └── kloosterman_twisted/
        __init__.py       public API
        curves.py         a-invariants for prime-level newforms N <= 50
        coefficients.py   a_n by Legendre-symbol point counting + Hecke recursion
        eichler.py        G(j) = F((j+i)/c) via one FFT
        units.py          units and inverses mod c (vectorized for c > 3000)
        untwisted.py      ordinary Kloosterman sums S(m, n; c)
        twisted.py        S*(m, n; c), projected (even/odd), and eps-deformed
        zeta.py           Linnik-type partial sums zeta_kloosterman[_twisted](x)
        checkpoint.py     pluggable resumable-run backend (JSON file, in-memory, ...)
```

## Install

### With [uv](https://docs.astral.sh/uv/) (recommended)

`uv` creates an isolated virtualenv and installs the package (plus dev
extras) in one command:

```bash
# from the kloosterman-twisted/ directory:
uv venv                              # create .venv/ with a suitable Python
uv pip install -e '.[dev]'           # install package + pytest / ruff

# optional: use PARI (via cypari2) for the Fourier coefficients a_n --
# significantly faster than the built-in point-counting fallback.
# cypari2 is a small standalone package (no Sage needed):
uv pip install -e '.[dev,pari]'

# activate the venv for interactive use...
source .venv/bin/activate
pytest

# ...or run one-off commands without activating:
uv run pytest
uv run python -c "from kloosterman_twisted import CURVES; print(sorted(CURVES))"
```

Pin a specific interpreter with e.g. ``uv venv --python 3.12``.  ``uv sync``
is not needed here (no lockfile is committed); ``uv pip install`` is the
right entry point for a plain ``pyproject.toml`` project.

### With plain pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Quick start

The one-call convenience wrappers are the fastest way to a numerical answer:

```python
from kloosterman_twisted import (
    zeta_kloosterman_twisted_value, zeta_kloosterman_value,
)

# Zeta_kloosterman^*(x = 220; m=1, n=5, N=11)  -- coefficients a_n are
# computed automatically (PARI -> Sage -> point counting).
val_twisted   = zeta_kloosterman_twisted_value(N=11, m=1, n=5, x=220)
val_untwisted = zeta_kloosterman_value(N=11, m=1, n=5, x=220)

# ...or get the full running curve (x_arr, z_arr) instead of the scalar:
xs, zs = zeta_kloosterman_twisted_value(
    N=11, m=1, n=5, x=220, return_curve=True,
)
```

Lower-level building blocks are also exposed for callers that want to do
things by hand:

```python
from kloosterman_twisted import (
    CURVES, anlist_pointcount,
    kloosterman_sum,
    kloosterman_sum_twisted, kloosterman_sum_twisted_projected,
    zeta_kloosterman_run, zeta_kloosterman, zeta_kloosterman_twisted,
)

N = 11
an = anlist_pointcount(CURVES[N], 2000)

# Individual sums for one modulus c
S     = kloosterman_sum(m=1, n=5, c=22)
Sstar = kloosterman_sum_twisted(m=1, n=5, c=22, N=N, an=an)
proj  = kloosterman_sum_twisted_projected(m=1, n=5, c=22, N=N, an=an)
assert abs(proj["twisted_even"] + proj["twisted_odd"] - proj["twisted"]) < 1e-9

# Batch (many (m, n) pairs at once + checkpointing)
state = zeta_kloosterman_run(N=N, mn_pairs=[(1, 5)], cmax=220)
x, z_twisted = zeta_kloosterman_twisted(state["data"]["1,5"])
_, z_untwist = zeta_kloosterman(state["data"]["1,5"])
```

## Long batch runs and checkpointing

`zeta_kloosterman_run` accepts a pluggable **checkpoint** object so that
interrupted runs can resume without recomputation.  Three usage patterns:

```python
from kloosterman_twisted import (
    zeta_kloosterman_run, JSONFileCheckpoint, Checkpoint,
)

# 1. Path shorthand -- coerced into a JSONFileCheckpoint automatically:
zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=10_000,
                     checkpoint="run.json", checkpoint_every=200,
                     time_budget=3600.0, progress=True)

# 2. Fully constructed backend (any cadence, atomic writes):
ckpt = JSONFileCheckpoint("run.json", save_every=50)
zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=10_000, checkpoint=ckpt)

# 3. Custom backend -- implement three methods on the Checkpoint interface:
class InMemoryCheckpoint(Checkpoint):
    def __init__(self):        self.state = None; self._i = 0
    def load(self):            return self.state
    def tick(self):            self._i += 1; return True
    def save(self, state, *, force=False):  self.state = state
```

Re-invoking `zeta_kloosterman_run` with the same checkpoint resumes from the
last completed modulus.  Setting a `time_budget` (seconds) lets you cap
individual invocations without losing progress.

## Tests and doctests

`pytest` runs both the unit tests under `tests/` and the Sage-style
`EXAMPLES::` blocks in every module's docstrings:

```bash
pytest                          # unit tests + doctests
pytest tests/                   # unit tests only
pytest src/kloosterman_twisted  # doctests only
```

The docstrings use Sage's `sage:` / `....:` prompt style so they read the same
inside and outside a Sage session.  A small shim in `conftest.py` rewrites
those prompts to standard `>>>` / `...` on the fly before stdlib `doctest`
sees them, so the examples run under a plain Python + NumPy install.

Examples that genuinely depend on an optional backend are marked with a Sage
`# optional - X` tag:

```
sage: anlist_pari(CURVES[37], 7)              # optional - cypari2
sage: anlist_sage(CURVES[37], 7)              # optional - sage
```

The shim maps `# optional - cypari2` and `# optional - sage` to
`# doctest: +SKIP` when the corresponding module isn't importable, so those
examples are silently skipped rather than failing.  Install the matching
extras (`[pari]`, or a full Sage / `passagemath` install) to have them run.

## Notes

* Truncation of the q-expansion uses `TRUNCATION_FACTOR = 5.7` (error ~ 1e-16).
* Coefficient backend: `kloosterman_twisted.anlist` auto-selects in the order
  **PARI** (via `cypari2`; enable with the `[pari]` extra) → **Sage**
  (`sage.all.EllipticCurve`; used automatically if a Sage/`passagemath` install
  is present) → pure-NumPy **point counting** (default fallback, always
  available).  Force one explicitly with
  `anlist(ainvs, nmax, backend="pari"|"sage"|"pointcount")`.
* `cypari2` is a small standalone PyPI package — much lighter than
  `passagemath-schemes`, which would pull in a large slice of Sage just to
  reach the same `ellinit`/`ellan` calls.
* The `Checkpoint` abstraction lives in `checkpoint.py` and is fully generic:
  any batch loop of the form "iterate, update state, save periodically" can
  reuse it without depending on the rest of this package.
