r"""
Modular-symbol twisted Kloosterman sums.

For a weight-2 newform ``f`` at level `N` and `\gamma = (a, b; c, d) \in \Gamma_0(N)`,

.. MATH::

    \langle f, \gamma \rangle = -2 \pi i \int_{i \infty}^{\gamma i \infty} f(w)\, dw,

.. MATH::

    S^*(m, n; c) = \sum_{d \in (\ZZ/c)^\times} \langle f, \gamma_d \rangle
                   e\!\left(\frac{n a + m d}{c}\right),

with `a = d^{-1} \bmod c` and `\gamma_d` chosen so that `\gamma_d z_0 = (a + i)/c`
for `z_0 = (-d + i)/c`.  Using the Eichler integral
`F(z) = \sum_{n \geq 1} (a_n / n) q^n` this reduces to
`\langle f, \gamma_d \rangle = G(-d \bmod c) - G(a \bmod c)` with
`G(j) = F((j + i)/c)`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .eichler import eichler_values
from .units import units_and_inverses


def _phase_and_periods(m, n, c, N, an, gj, units):
    if c % N != 0:
        raise ValueError(f"{N} does not divide {c}")
    if gj is None:
        gj = eichler_values(c, an)
    d, a = units if units is not None else units_and_inverses(c)
    periods = gj[(-d) % c] - gj[a]
    phase = np.exp((2j * np.pi / c) * ((n * a + m * d) % c))
    return periods, phase


def kloosterman_sum_twisted_block(m: int, n: int, c: int, N: int, an: Sequence[float],
                                  gj: Optional[np.ndarray] = None,
                                  units: Optional[tuple] = None) -> dict:
    r"""
    Twisted `S^*(m, n; c)` and the ordinary `S(m, n; c)` sharing the same phase table.

    INPUT:

    - ``m``, ``n`` -- integers
    - ``c`` -- positive integer modulus, must satisfy ``N | c``
    - ``N`` -- level
    - ``an`` -- Fourier coefficients with ``an[k] = a_k`` for
      ``k <= TRUNCATION_FACTOR * c``
    - ``gj`` -- (optional) precomputed :func:`eichler_values(c, an)
      <kloosterman_twisted.eichler.eichler_values>`
    - ``units`` -- (optional) precomputed
      :func:`units_and_inverses(c) <kloosterman_twisted.units.units_and_inverses>`

    OUTPUT: dict with keys ``"twisted"`` (complex `S^*(m, n; c)`) and
    ``"untwisted"`` (complex `S(m, n; c)`).

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     CURVES, anlist_pointcount, kloosterman_sum_twisted_block)
        sage: an = anlist_pointcount(CURVES[11], 200)
        sage: r = kloosterman_sum_twisted_block(1, 5, 22, 11, an)
        sage: sorted(r)
        ['twisted', 'untwisted']
        sage: abs(r["twisted"].real - 3.407804064726443) < 1e-12
        True
        sage: abs(r["twisted"].imag) < 1e-9
        True
        sage: abs(r["untwisted"].real - (-5.716952715441702)) < 1e-12
        True
    """
    periods, phase = _phase_and_periods(m, n, c, N, an, gj, units)
    return {
        "twisted": complex(np.sum(periods * phase)),
        "untwisted": complex(np.sum(phase)),
    }


def kloosterman_sum_twisted(m: int, n: int, c: int, N: int, an: Sequence[float],
                            gj: Optional[np.ndarray] = None,
                            units: Optional[tuple] = None) -> complex:
    r"""
    Modular-symbol twisted Kloosterman sum `S^*(m, n; c)`.

    INPUT:

    - ``m``, ``n`` -- integers
    - ``c`` -- positive integer modulus, must satisfy ``N | c``
    - ``N`` -- level
    - ``an`` -- Fourier coefficients with ``an[k] = a_k`` for
      ``k <= TRUNCATION_FACTOR * c``
    - ``gj``, ``units`` -- as in :func:`kloosterman_sum_twisted_block`

    OUTPUT: a complex number.  For real ``an`` the imaginary part is zero up to
    floating-point roundoff.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     CURVES, anlist_pointcount, kloosterman_sum_twisted)
        sage: an = anlist_pointcount(CURVES[11], 200)
        sage: val = kloosterman_sum_twisted(1, 5, 22, 11, an)
        sage: abs(val.real - 3.407804064726443) < 1e-12
        True
        sage: abs(val.imag) < 1e-9
        True
    """
    return kloosterman_sum_twisted_block(m, n, c, N, an, gj=gj, units=units)["twisted"]


def kloosterman_sum_twisted_projected(m: int, n: int, c: int, N: int, an: Sequence[float],
                                      gj: Optional[np.ndarray] = None,
                                      units: Optional[tuple] = None) -> dict:
    r"""
    `S^*` split into its even (`\Omega^+`) and odd (`\Omega^-`) modular-symbol channels.

    Writing
    `T_2 = \sum_d \Re\langle f, \gamma_d \rangle e((na+md)/c)` and
    `T_1 = \sum_d \Im\langle f, \gamma_d \rangle e((na+md)/c)`,
    the identity `S^* = T_2 + i T_1` gives (since `S^*` is real)

    .. MATH::

        \mathrm{twisted\_even} = \Re(T_2), \qquad
        \mathrm{twisted\_odd}  = -\Im(T_1), \qquad
        \mathrm{twisted\_even} + \mathrm{twisted\_odd} = \mathrm{twisted}.

    INPUT:

    - ``m``, ``n``, ``c``, ``N``, ``an``, ``gj``, ``units`` -- as in
      :func:`kloosterman_sum_twisted_block`

    OUTPUT: dict with real keys ``"twisted"``, ``"twisted_even"``,
    ``"twisted_odd"`` and complex key ``"untwisted"``.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     CURVES, anlist_pointcount, kloosterman_sum_twisted_projected)
        sage: an = anlist_pointcount(CURVES[11], 200)
        sage: p = kloosterman_sum_twisted_projected(1, 5, 22, 11, an)
        sage: abs(p["twisted"] - 3.407804064726443) < 1e-12
        True
        sage: abs(p["twisted_even"] - 2.724868204914966) < 1e-12
        True
        sage: abs(p["twisted_odd"]  - 0.682935859811478) < 1e-12
        True
        sage: abs(p["twisted_even"] + p["twisted_odd"] - p["twisted"]) < 1e-12
        True
    """
    periods, phase = _phase_and_periods(m, n, c, N, an, gj, units)
    t2 = np.sum(periods.real * phase)
    t1 = np.sum(periods.imag * phase)
    return {
        "twisted": float((t2 + 1j * t1).real),
        "twisted_even": float(t2.real),
        "twisted_odd": float(-t1.imag),
        "untwisted": complex(np.sum(phase)),
    }


def kloosterman_sum_twisted_deformed(m: int, n: int, c: int, N: int, an: Sequence[float],
                                     eps_list: Sequence[float],
                                     gj: Optional[np.ndarray] = None,
                                     units: Optional[tuple] = None) -> dict:
    r"""
    `\epsilon`-deformed Kloosterman sums `S_{\chi^i_\epsilon}(m, n; c)`.

    For the unitary characters

    .. MATH::

        \chi^1_\epsilon(\gamma) = \exp( i \epsilon \Im\langle f, \gamma \rangle),
        \qquad
        \chi^2_\epsilon(\gamma) = \exp(-i \epsilon \Re\langle f, \gamma \rangle),

    the finite-difference identity

    .. MATH::

        S^*(m, n; c) = \left.\frac{d}{d\epsilon}\right|_{\epsilon = 0}
            \bigl(S_{\chi^1_\epsilon} + i S_{\chi^2_\epsilon}\bigr)

    provides a consistency check on the twisted sum.

    INPUT:

    - ``m``, ``n``, ``c``, ``N``, ``an``, ``gj``, ``units`` -- as in
      :func:`kloosterman_sum_twisted_block`
    - ``eps_list`` -- iterable of real deformation parameters

    OUTPUT: dict mapping each `\epsilon` to a pair
    `(S_{\chi^1_\epsilon},\ S_{\chi^2_\epsilon})` of complex numbers.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     CURVES, anlist_pointcount, kloosterman_sum_twisted_deformed,
        ....:     kloosterman_sum)
        sage: an = anlist_pointcount(CURVES[11], 200)
        sage: d = kloosterman_sum_twisted_deformed(1, 5, 22, 11, an, [0.0])
        sage: s1, s2 = d[0.0]
        sage: kl = kloosterman_sum(1, 5, 22)
        sage: abs(s1 - kl) < 1e-9 and abs(s2 - kl) < 1e-9
        True
        sage: abs(s1.real - (-5.716952715441702)) < 1e-12
        True
    """
    periods, phase = _phase_and_periods(m, n, c, N, an, gj, units)
    out = {}
    for eps in eps_list:
        s1 = complex(np.sum(np.exp(1j * eps * periods.imag) * phase))
        s2 = complex(np.sum(np.exp(-1j * eps * periods.real) * phase))
        out[eps] = (s1, s2)
    return out
