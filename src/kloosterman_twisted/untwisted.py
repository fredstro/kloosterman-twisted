r"""
Ordinary (untwisted) Kloosterman sums

.. MATH::

    S(m, n; c) = \sum_{d \in (\ZZ/c)^\times}
                 e\!\left(\frac{n a + m d}{c}\right),
    \qquad a = d^{-1} \bmod c.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .units import units_and_inverses


def kloosterman_block(m: int, n: int, c: int,
                      units: Optional[tuple] = None) -> complex:
    r"""
    Ordinary Kloosterman sum `S(m, n; c)` with optional precomputed units.

    INPUT:

    - ``m``, ``n`` -- integers
    - ``c`` -- positive integer modulus
    - ``units`` -- (optional) precomputed
      :func:`units_and_inverses(c) <kloosterman_twisted.units.units_and_inverses>`;
      passing this in lets many calls at the same ``c`` share work

    OUTPUT: complex `S(m, n; c)`.  For real inputs the imaginary part is zero
    up to floating-point roundoff.

    EXAMPLES::

        sage: from kloosterman_twisted import kloosterman_block
        sage: v = kloosterman_block(1, 1, 5)
        sage: abs(v.real - 0.3819660112501052) < 1e-12
        True
        sage: abs(v.imag) < 1e-9
        True
        sage: abs(v) < 2 * 5 ** 0.5            # Weil bound |S(1,1;5)| <= 2 sqrt 5
        True
    """
    d, a = units if units is not None else units_and_inverses(c)
    phase = np.exp((2j * np.pi / c) * ((n * a + m * d) % c))
    return complex(np.sum(phase))


def kloosterman_sum(m: int, n: int, c: int) -> complex:
    r"""
    Ordinary Kloosterman sum `S(m, n; c)`.

    Thin convenience wrapper around :func:`kloosterman_block`.

    INPUT:

    - ``m``, ``n`` -- integers
    - ``c`` -- positive integer modulus

    OUTPUT: complex number ``S(m, n; c)``.

    EXAMPLES::

        sage: from kloosterman_twisted import kloosterman_sum
        sage: v = kloosterman_sum(1, 5, 22)
        sage: abs(v.real - (-5.716952715441702)) < 1e-12
        True
        sage: abs(v.imag) < 1e-9
        True

    Cross-check against the naive definition::

        sage: import math
        sage: c = 22
        sage: ref = sum(complex(math.cos(2*math.pi*(pow(d,-1,c) + d)/c),
        ....:                   math.sin(2*math.pi*(pow(d,-1,c) + d)/c))
        ....:           for d in range(1, c) if math.gcd(d, c) == 1)
        sage: abs(kloosterman_sum(1, 1, c) - ref) < 1e-10
        True
    """
    return kloosterman_block(m, n, c)
