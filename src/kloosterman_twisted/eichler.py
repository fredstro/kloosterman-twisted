r"""
Eichler integral values `G(j) = F((j+i)/c)` for `j = 0, \ldots, c-1`, via one FFT.

With `b_n = (a_n / n) e^{-2 \pi n / c}` folded modulo `c`,

.. MATH::

    F\!\left(\frac{j + i}{c}\right)
        = \sum_r b^{(c)}_r\, e^{2 \pi i r j / c}
        = c \cdot \mathrm{ifft}(b^{(c)}),

where `b^{(c)}_r = \sum_{n \equiv r \bmod c} b_n`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Number of q-expansion terms is TRUNCATION_FACTOR * c; truncation error is
# roughly exp(-2 pi TRUNCATION_FACTOR) ~ 1e-16 for the default 5.7.
TRUNCATION_FACTOR = 5.7


def eichler_values(c: int, an: Sequence[float]) -> np.ndarray:
    r"""
    All Eichler-integral values `G(j) = F((j + i)/c)` for `j = 0, \ldots, c-1`.

    INPUT:

    - ``c`` -- positive integer modulus (evaluation height is `1/c`)
    - ``an`` -- Fourier coefficients with ``an[k] = a_k`` for
      ``k <= TRUNCATION_FACTOR * c``; ``an[0]`` is ignored

    OUTPUT: complex numpy array of length ``c``, with entry ``j`` equal to
    `F((j + i)/c)`.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     CURVES, anlist_pointcount, eichler_values)
        sage: an = anlist_pointcount(CURVES[11], 200)
        sage: g = eichler_values(22, an)
        sage: g.shape
        (22,)
        sage: bool(abs(g[0].real - 0.25383837352571625) < 1e-12)
        True
        sage: bool(abs(g[0].imag) < 1e-9)
        True
        sage: bool(abs(g[5].real - 0.8605982132602412) < 1e-12)
        True
        sage: bool(abs(g[5].imag - 0.7243777695605396) < 1e-12)
        True
    """
    m_trunc = min(int(TRUNCATION_FACTOR * c) + 1, len(an) - 1)
    n = np.arange(1, m_trunc + 1, dtype=np.float64)
    b = (np.asarray(an[1:m_trunc + 1], dtype=np.float64) / n) * np.exp(-2 * np.pi * n / c)
    folded = np.zeros(c, dtype=np.float64)
    np.add.at(folded, np.arange(1, m_trunc + 1) % c, b)
    return c * np.fft.ifft(folded)
