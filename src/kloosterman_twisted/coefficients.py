r"""
Fourier coefficients `a_n` of the weight-2 newform attached to `E/\QQ`.

Three backends are available:

* ``"pari"``       -- calls PARI directly via ``cypari2`` (small standalone
  package, no Sage involved).  Requires the ``[pari]`` extra.
* ``"sage"``       -- calls ``sage.all.EllipticCurve(ainvs).anlist(nmax)``,
  used automatically when a Sage-flavoured install is already present
  (e.g. ``passagemath-schemes`` or full Sage).
* ``"pointcount"`` -- pure NumPy Legendre-symbol point counting + Hecke
  recursion + multiplicativity; no external dependency, always available.

:func:`anlist` picks the fastest available backend and falls back silently.
:func:`anlist_pari`, :func:`anlist_sage` and :func:`anlist_pointcount` are
exposed for callers that want to pin a specific backend.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


def _have_pari() -> bool:
    r"""Return ``True`` when ``cypari2`` is importable."""
    try:
        import cypari2  # noqa: F401
    except Exception:
        return False
    return True


def _have_sage() -> bool:
    r"""Return ``True`` when a Sage-flavoured ``EllipticCurve`` is importable."""
    try:
        from sage.all import EllipticCurve  # noqa: F401
    except Exception:
        return False
    return True


def anlist_pari(ainvs: Sequence[int], nmax: int) -> list:
    r"""
    Coefficients ``[a_0 = 0, a_1, ..., a_nmax]`` via PARI's ``ellan``.

    Uses ``cypari2`` directly -- no Sage required.  Enable via the ``[pari]``
    extra (``pip install kloosterman-twisted[pari]``).

    INPUT:

    - ``ainvs`` -- a-invariants ``[a1, a2, a3, a4, a6]`` of a minimal model
    - ``nmax`` -- last index `n`

    OUTPUT: list of integers of length ``nmax + 1``.

    EXAMPLES::

        sage: from kloosterman_twisted import CURVES
        sage: from kloosterman_twisted.coefficients import anlist_pari
        sage: anlist_pari(CURVES[37], 7)              # optional - cypari2
        [0, 1, -2, -3, 2, -2, 6, -1]
    """
    import cypari2  # local import: optional dep
    pari = cypari2.Pari()
    E = pari.ellinit(list(ainvs))
    # ellan returns a_1, ..., a_nmax (no a_0).  Prepend 0 for our convention.
    return [0] + [int(x) for x in pari.ellan(E, nmax)]


def anlist_sage(ainvs: Sequence[int], nmax: int) -> list:
    r"""
    Coefficients ``[a_0 = 0, a_1, ..., a_nmax]`` via ``sage.all.EllipticCurve``.

    Requires a Sage-flavoured install (full Sage or a ``passagemath`` distribution
    that ships ``sage.schemes.elliptic_curves``).  For a lighter-weight PARI-only
    path see :func:`anlist_pari`.

    INPUT:

    - ``ainvs`` -- a-invariants ``[a1, a2, a3, a4, a6]`` of a minimal model
    - ``nmax`` -- last index `n`

    OUTPUT: list of integers of length ``nmax + 1``.

    EXAMPLES::

        sage: from kloosterman_twisted import CURVES
        sage: from kloosterman_twisted.coefficients import anlist_sage
        sage: anlist_sage(CURVES[37], 7)              # optional - sage
        [0, 1, -2, -3, 2, -2, 6, -1]
    """
    from sage.all import EllipticCurve  # local import: optional dep
    E = EllipticCurve(list(ainvs))
    return list(E.anlist(nmax))


def anlist(ainvs: Sequence[int], nmax: int,
           ap_map: Optional[dict] = None,
           backend: str = "auto") -> list:
    r"""
    Coefficients ``[a_0 = 0, a_1, ..., a_nmax]`` via the fastest available backend.

    INPUT:

    - ``ainvs`` -- a-invariants ``[a1, a2, a3, a4, a6]`` of a minimal model
    - ``nmax`` -- last index `n`
    - ``ap_map`` -- (optional) precomputed ``{p: a_p}``; only consulted by the
      point-counting backend
    - ``backend`` -- one of ``"auto"`` (default; prefers ``"pari"``, then
      ``"sage"``, then ``"pointcount"``), ``"pari"``, ``"sage"``,
      or ``"pointcount"``

    OUTPUT: list of integers of length ``nmax + 1``.

    EXAMPLES::

        sage: from kloosterman_twisted import CURVES, anlist
        sage: anlist(CURVES[11], 10, backend="pointcount")
        [0, 1, -2, -1, 2, 1, 2, -2, 0, -2, -2]
    """
    if backend == "pari":
        return anlist_pari(ainvs, nmax)
    if backend == "sage":
        return anlist_sage(ainvs, nmax)
    if backend == "pointcount":
        return anlist_pointcount(ainvs, nmax, ap_map=ap_map)
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}")
    if _have_pari():
        return anlist_pari(ainvs, nmax)
    if _have_sage():
        return anlist_sage(ainvs, nmax)
    return anlist_pointcount(ainvs, nmax, ap_map=ap_map)


def _discriminant(ainvs: Sequence[int]) -> int:
    r"""Discriminant of the Weierstrass model with the given a-invariants."""
    a1, a2, a3, a4, a6 = ainvs
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    return -b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 * b6 + 9 * b2 * b4 * b6


def _pow_mod_small(x: np.ndarray, k: int, p: int) -> np.ndarray:
    r"""Elementwise x^k mod p for an int64 numpy array (k small)."""
    r = np.ones_like(x)
    for _ in range(k):
        r = (r * x) % p
    return r


def _ap_odd_prime(p: int, ainvs: Sequence[int]) -> int:
    r"""
    Trace of Frobenius a_p for an odd prime p, via vectorized Legendre symbols.

    Works for good and bad reduction: at p | Delta the count is over the non-singular
    locus, so a_p in {0, 1, -1}.
    """
    a1, a2, a3, a4, a6 = (ai % p for ai in ainvs)
    b2 = (a1 * a1 + 4 * a2) % p
    b4 = (2 * a4 + a1 * a3) % p
    b6 = (a3 * a3 + 4 * a6) % p
    x = np.arange(p, dtype=np.int64)
    f = (4 * x + b2) % p
    f *= x
    f += 2 * b4
    f %= p
    f *= x
    f += b6
    f %= p
    is_sq = np.zeros(p, dtype=bool)
    is_sq[(x * x) % p] = True
    s = int(np.count_nonzero(is_sq[f] & (f != 0))) - int(np.count_nonzero(~is_sq[f]))
    if _discriminant(ainvs) % p != 0:
        return -s
    fp = (12 * _pow_mod_small(x, 2, p) + 2 * b2 * x + 2 * b4) % p
    n_sing = int(np.count_nonzero((f == 0) & (fp == 0)))
    return n_sing - s - 1


def _ap_two(ainvs: Sequence[int]) -> int:
    r"""a_2 by brute force over F_2, counting the non-singular locus."""
    a1, a2, a3, a4, a6 = ainvs
    p = 2
    cnt = 0
    for xx in range(p):
        for yy in range(p):
            fval = (yy * yy + a1 * xx * yy + a3 * yy
                    - (xx ** 3 + a2 * xx * xx + a4 * xx + a6)) % p
            if fval != 0:
                continue
            fx = (a1 * yy - (3 * xx * xx + 2 * a2 * xx + a4)) % p
            fy = (2 * yy + a1 * xx + a3) % p
            if fx == 0 and fy == 0:
                continue
            cnt += 1
    if _discriminant(ainvs) % 2 != 0:
        return p + 1 - (cnt + 1)
    return p - (cnt + 1)


def _np_savez_atomic(path: str, **arrays) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def ap_table_chunked(ainvs: Sequence[int], nmax: int, cache_path: str,
                     time_budget: Optional[float] = None) -> Optional[dict]:
    r"""
    Resumable table `\{p : a_p\}` for all primes `p \leq \mathrm{nmax}`.

    Cached in an ``.npz`` file so that repeated invocations pick up where the
    previous run left off.

    INPUT:

    - ``ainvs`` -- a-invariants ``[a1, a2, a3, a4, a6]``
    - ``nmax`` -- upper bound on `p`
    - ``cache_path`` -- path of the ``.npz`` cache (created/updated in place)
    - ``time_budget`` -- (optional) wall-clock budget in seconds; if exceeded the
      partial cache is saved and ``None`` is returned so the caller can resume

    OUTPUT: dict ``{p: a_p}`` for all primes `p \leq \mathrm{nmax}`, or ``None``
    if the time budget expired before completion.

    EXAMPLES::

        sage: import os, tempfile
        sage: from kloosterman_twisted import CURVES, ap_table_chunked
        sage: with tempfile.TemporaryDirectory() as tmp:
        ....:     path = os.path.join(tmp, "ap.npz")
        ....:     table = ap_table_chunked(CURVES[11], 20, path)
        sage: table[2], table[3], table[5]
        (-2, -1, 1)
    """
    sieve = np.ones(nmax + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(nmax ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    primes = np.flatnonzero(sieve)
    aps = np.zeros(0, dtype=np.int64)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if list(cached["ainvs"]) == list(ainvs):
            aps = cached["aps"]
    t0 = time.time()
    while len(aps) < len(primes):
        p = int(primes[len(aps)])
        ap = _ap_two(ainvs) if p == 2 else _ap_odd_prime(p, ainvs)
        aps = np.append(aps, ap)
        if time_budget is not None and time.time() - t0 > time_budget:
            _np_savez_atomic(cache_path, ainvs=np.array(ainvs), aps=aps)
            log.info("ap_table_chunked: budget expired at p=%d (%d/%d)",
                     p, len(aps), len(primes))
            return None
    _np_savez_atomic(cache_path, ainvs=np.array(ainvs), aps=aps)
    return {int(p): int(a) for p, a in zip(primes, aps)}


def anlist_pointcount(ainvs: Sequence[int], nmax: int,
                      ap_map: Optional[dict] = None) -> list:
    r"""
    Coefficients ``[a_0 = 0, a_1, ..., a_nmax]`` of the newform attached to `E`.

    Uses Legendre-symbol point counting for `a_p`, the Hecke recursion at prime
    powers, and multiplicativity via a smallest-prime-factor sieve.

    INPUT:

    - ``ainvs`` -- a-invariants ``[a1, a2, a3, a4, a6]`` of a minimal model
    - ``nmax`` -- last index `n`
    - ``ap_map`` -- (optional) precomputed ``{p: a_p}`` dict (for example from
      :func:`ap_table_chunked`)

    OUTPUT: list of integers of length ``nmax + 1``; entry ``0`` is ``0`` and
    entry ``1`` is ``1``.

    EXAMPLES::

        sage: from kloosterman_twisted import CURVES, anlist_pointcount
        sage: anlist_pointcount(CURVES[11], 10)   # 11a1: q - 2q^2 - q^3 + ...
        [0, 1, -2, -1, 2, 1, 2, -2, 0, -2, -2]
        sage: anlist_pointcount(CURVES[37], 7)    # 37a1
        [0, 1, -2, -3, 2, -2, 6, -1]
    """
    spf = np.zeros(nmax + 1, dtype=np.int64)
    for i in range(2, nmax + 1):
        if spf[i] == 0:
            spf[i::i][spf[i::i] == 0] = i
    primes = [i for i in range(2, nmax + 1) if spf[i] == i]

    disc = _discriminant(ainvs)
    a = [0] * (nmax + 1)
    a[1] = 1
    for p in primes:
        if ap_map is not None and p in ap_map:
            ap = ap_map[p]
        else:
            ap = _ap_two(ainvs) if p == 2 else _ap_odd_prime(p, ainvs)
        good = disc % p != 0
        if not good and ap not in (-1, 0, 1):
            raise ArithmeticError(f"bad prime p={p}: a_p={ap} not in {{0, +-1}}")
        pk = p
        prev2, prev1 = None, ap
        a[pk] = ap
        while pk * p <= nmax:
            pk *= p
            if good:
                cur = ap * prev1 - p * (prev2 if prev2 is not None else 1)
            else:
                cur = ap * prev1
            a[pk] = cur
            prev2, prev1 = prev1, cur
    for m in range(2, nmax + 1):
        p = int(spf[m])
        pk, rest = p, m // p
        while rest % p == 0:
            pk *= p
            rest //= p
        if rest > 1:
            a[m] = a[pk] * a[rest]
    return a
