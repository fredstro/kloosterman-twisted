r"""
kloosterman_twisted -- untwisted and modular-symbol twisted Kloosterman sums.

Public API:

Coefficients
    anlist(ainvs, nmax, ap_map=None, backend="auto") -> list[int]   # auto-select
    anlist_pointcount(ainvs, nmax, ap_map=None) -> list[int]        # pure NumPy
    anlist_pari(ainvs, nmax) -> list[int]                           # cypari2
    anlist_sage(ainvs, nmax) -> list[int]                           # Sage/PARI
    ap_table_chunked(ainvs, nmax, cache_path, time_budget=None) -> dict | None

Eichler integral
    eichler_values(c, an) -> np.ndarray

Units modulo c
    units_and_inverses(c) -> (d, a)

Untwisted Kloosterman sums
    kloosterman_sum(m, n, c) -> complex
    kloosterman_block(m, n, c, units=None) -> complex

Twisted (modular-symbol) Kloosterman sums
    kloosterman_sum_twisted(m, n, c, N, an, ...) -> complex
    kloosterman_sum_twisted_projected(m, n, c, N, an, ...) -> dict
    kloosterman_sum_twisted_deformed(m, n, c, N, an, eps_list, ...) -> dict

Partial sums (Linnik-type)
    zeta_kloosterman_twisted_value(N, m, n, x, ...) -> float   # one-call
    zeta_kloosterman_value(N, m, n, x, ...) -> float           # one-call
    zeta_kloosterman_run(N, mn_pairs, cmax, ...) -> dict       # advanced
    zeta_kloosterman(rows) -> (x, z)                           # advanced
    zeta_kloosterman_twisted(rows) -> (x, z)                   # advanced

Curve data
    CURVES -- a-invariants for prime level newforms N in {11,17,19,37,43,49}

The algorithms are drop-in reorganizations of the reference implementation in
``kloosterman_zstar.py`` in the parent directory.
"""

from .curves import CURVES
from .coefficients import (
    anlist,
    anlist_pari,
    anlist_pointcount,
    anlist_sage,
    ap_table_chunked,
)
from .eichler import TRUNCATION_FACTOR, eichler_values
from .units import units_and_inverses, units_and_inverses_fast
from .untwisted import kloosterman_sum, kloosterman_block
from .twisted import (
    kloosterman_sum_twisted,
    kloosterman_sum_twisted_block,
    kloosterman_sum_twisted_projected,
    kloosterman_sum_twisted_deformed,
)
from .zeta import (
    zeta_kloosterman,
    zeta_kloosterman_run,
    zeta_kloosterman_twisted,
    zeta_kloosterman_twisted_value,
    zeta_kloosterman_value,
)
from .checkpoint import Checkpoint, JSONFileCheckpoint, as_checkpoint

__all__ = [
    "CURVES",
    "TRUNCATION_FACTOR",
    "anlist",
    "anlist_pari",
    "anlist_pointcount",
    "anlist_sage",
    "ap_table_chunked",
    "eichler_values",
    "units_and_inverses",
    "units_and_inverses_fast",
    "kloosterman_sum",
    "kloosterman_block",
    "kloosterman_sum_twisted",
    "kloosterman_sum_twisted_block",
    "kloosterman_sum_twisted_projected",
    "kloosterman_sum_twisted_deformed",
    "zeta_kloosterman_run",
    "zeta_kloosterman",
    "zeta_kloosterman_twisted",
    "zeta_kloosterman_value",
    "zeta_kloosterman_twisted_value",
    "Checkpoint",
    "JSONFileCheckpoint",
    "as_checkpoint",
]
