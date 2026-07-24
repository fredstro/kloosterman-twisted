r"""
Units and their inverses modulo `c`.

For small `c` a straight Python loop with ``pow(d, -1, c)`` is fastest; for
`c > 3000` a vectorized routine uses the Carmichael exponent `\lambda(c)` and a
full `d \cdot a \equiv 1 \pmod c` verification pass.
"""

from __future__ import annotations

import math

import numpy as np


def _prime_factors(c: int) -> list:
    r"""Distinct prime factors of ``c`` by trial division."""
    ps, p, n = [], 2, c
    while p * p <= n:
        if n % p == 0:
            ps.append(p)
            while n % p == 0:
                n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        ps.append(n)
    return ps


def _powmod_vec(base: np.ndarray, exp: int, mod: int) -> np.ndarray:
    r"""Elementwise ``base ** exp`` mod ``mod`` by binary exponentiation."""
    r = np.ones_like(base)
    b = base % mod
    while exp:
        if exp & 1:
            r = (r * b) % mod
        b = (b * b) % mod
        exp >>= 1
    return r


def units_and_inverses_fast(c: int) -> tuple:
    r"""
    Vectorized units and inverses modulo `c`.

    Uses `a = d^{\lambda(c) - 1} \bmod c` where `\lambda` is the Carmichael
    function, then verifies `d \cdot a \equiv 1 \pmod c` for every element so
    that failures cannot pass silently.

    INPUT:

    - ``c`` -- positive integer

    OUTPUT: pair of ``int64`` numpy arrays ``(d, a)`` of length `\varphi(c)`
    with ``d`` the units in `(\ZZ/c)^\times` in `[1, c)` and ``a = d^{-1} mod c``.

    EXAMPLES::

        sage: from kloosterman_twisted import units_and_inverses_fast
        sage: import numpy as np
        sage: d, a = units_and_inverses_fast(30)
        sage: bool(np.all((d * a) % 30 == 1))
        True
    """
    ps = _prime_factors(c)
    mask = np.ones(c, dtype=bool)
    mask[0] = False
    for p in ps:
        mask[::p] = False
    d = np.flatnonzero(mask).astype(np.int64)
    lam = 1
    for p in ps:
        e, n = 0, c
        while n % p == 0:
            n //= p
            e += 1
        if p > 2:
            lam_pe = (p - 1) * p ** (e - 1)
        else:
            lam_pe = 1 if e == 1 else 2 if e == 2 else 2 ** (e - 2)
        lam = lam * lam_pe // math.gcd(lam, lam_pe)
    a = _powmod_vec(d, lam - 1, c)
    if not np.all((d * a) % c == 1):
        raise ArithmeticError(f"inverse verification failed for c={c}")
    return d, a


def units_and_inverses(c: int) -> tuple:
    r"""
    Units and inverses modulo `c`, dispatching by size.

    Delegates to :func:`units_and_inverses_fast` for ``c > 3000`` (essential for
    `c \sim 10^6`, where a Python loop would dominate the total cost) and to a
    plain Python loop otherwise.

    INPUT:

    - ``c`` -- positive integer

    OUTPUT: pair of ``int64`` numpy arrays ``(d, a)`` of length `\varphi(c)`.

    EXAMPLES::

        sage: from kloosterman_twisted import units_and_inverses
        sage: d, a = units_and_inverses(12)
        sage: sorted(int(x) for x in d)
        [1, 5, 7, 11]
        sage: all((int(dd) * int(aa)) % 12 == 1 for dd, aa in zip(d, a))
        True
    """
    if c > 3000:
        return units_and_inverses_fast(c)
    ds, inv = [], []
    for d in range(1, c):
        if math.gcd(d, c) == 1:
            ds.append(d)
            inv.append(pow(d, -1, c))
    return np.array(ds, dtype=np.int64), np.array(inv, dtype=np.int64)
