r"""
Linnik-type partial sums of Kloosterman sums.

Untwisted:

.. MATH::

    \zeta_{\mathrm{kloosterman}}(x)
        = \sum_{\substack{N \mid c \\ 0 < c \leq x}} \frac{S(m, n; c)}{c}.

Twisted (modular-symbol):

.. MATH::

    \zeta_{\mathrm{kloosterman}}^{*}(x)
        = \sum_{\substack{N \mid c \\ 0 < c \leq x}} \frac{S^*(m, n; c)}{c}.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, Union

import numpy as np

from .checkpoint import Checkpoint, as_checkpoint
from .coefficients import anlist
from .curves import CURVES
from .eichler import TRUNCATION_FACTOR, eichler_values
from .twisted import kloosterman_sum_twisted_block
from .units import units_and_inverses

log = logging.getLogger(__name__)


def _initial_state(N: int, cmax: int, mn_pairs: Sequence[tuple]) -> dict:
    return {
        "meta": {"N": N, "cmax": cmax,
                 "mn_pairs": [list(p) for p in mn_pairs],
                 "truncation_factor": TRUNCATION_FACTOR},
        "data": {f"{m},{n}": [] for (m, n) in mn_pairs},
    }


def _last_c_done(state: dict, mn_pairs: Sequence[tuple]) -> int:
    first_key = f"{mn_pairs[0][0]},{mn_pairs[0][1]}"
    rows = state["data"].get(first_key, [])
    return rows[-1][0] if rows else 0


def zeta_kloosterman_run(
    N: int,
    mn_pairs: Sequence[tuple],
    cmax: int,
    an: Optional[Sequence[float]] = None,
    ainvs: Optional[Sequence[int]] = None,
    checkpoint: Union[None, str, Checkpoint] = None,
    checkpoint_every: int = 200,
    progress: bool = False,
    time_budget: Optional[float] = None,
) -> dict:
    r"""
    Compute the rows needed for both partial sums
    :func:`zeta_kloosterman` and :func:`zeta_kloosterman_twisted` at all
    `N \mid c,\ 0 < c \leq c_{\max}` and every ``(m, n)`` pair.

    The Eichler-integral table `G(j)` and the unit/inverse arrays are computed
    once per ``c`` and shared across all ``(m, n)`` pairs, so extra pairs are
    nearly free.

    INPUT:

    - ``N`` -- level
    - ``mn_pairs`` -- list of ``(m, n)`` integer pairs
    - ``cmax`` -- largest modulus
    - ``an`` -- (optional) Fourier coefficients ``a_k`` for
      ``k <= TRUNCATION_FACTOR * cmax``; if ``None``, they are computed by
      point counting from ``ainvs`` (or ``CURVES[N]``)
    - ``ainvs`` -- (optional) a-invariants, used only when ``an`` is ``None``
    - ``checkpoint`` -- ``None``, a path (wrapped in
      :class:`~kloosterman_twisted.checkpoint.JSONFileCheckpoint`), or any
      :class:`~kloosterman_twisted.checkpoint.Checkpoint`-compatible object;
      supports resuming interrupted runs
    - ``checkpoint_every`` -- save cadence in moduli when ``checkpoint`` is a
      path; ignored when a fully constructed checkpoint object is passed
    - ``progress`` -- if ``True``, log progress lines
    - ``time_budget`` -- (optional) wall-clock budget in seconds; on expiry the
      checkpoint is written and the partial state returned with
      ``meta['complete'] = False``

    OUTPUT: dict with keys ``"meta"`` and ``"data"``.  ``"data"`` maps
    ``"m,n"`` to a list of rows ``[c, Re S*, Im S*, Re S, Im S]`` in
    increasing order of ``c``.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     zeta_kloosterman_run, zeta_kloosterman_twisted)
        sage: state = zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=44)
        sage: state["meta"]["complete"]
        True
        sage: x, z = zeta_kloosterman_twisted(state["data"]["1,5"])
        sage: [float(v) for v in x]
        [11.0, 22.0, 33.0, 44.0]
    """
    if an is None:
        if ainvs is None:
            ainvs = CURVES[N]
        nmax = int(TRUNCATION_FACTOR * cmax) + 2
        log.info("computing a_n up to %d", nmax)
        an = anlist(ainvs, nmax)

    ckpt = as_checkpoint(checkpoint, save_every=checkpoint_every)

    state = ckpt.load()
    if state is None:
        state = _initial_state(N, cmax, mn_pairs)
        c_done = 0
    else:
        c_done = _last_c_done(state, mn_pairs)
        log.info("resuming from checkpoint at c=%d", c_done)

    t0 = time.time()
    cs = [c for c in range(N, cmax + 1, N) if c > c_done]
    state["meta"]["cmax"] = cmax
    state["meta"]["complete"] = not cs

    for c in cs:
        gj = eichler_values(c, an)
        units = units_and_inverses(c)
        for (m, n) in mn_pairs:
            r = kloosterman_sum_twisted_block(m, n, c, N, an, gj=gj, units=units)
            state["data"][f"{m},{n}"].append(
                [c, r["twisted"].real, r["twisted"].imag,
                 r["untwisted"].real, r["untwisted"].imag])
        if ckpt.tick():
            ckpt.save(state)
            if progress:
                log.info("N=%d c=%d/%d  (%.1fs)", N, c, cmax, time.time() - t0)
        if time_budget is not None and time.time() - t0 > time_budget:
            if progress:
                log.info("N=%d: time budget reached at c=%d", N, c)
            break
    else:
        state["meta"]["complete"] = True

    ckpt.save(state, force=True)
    return state


def _partial(rows: Sequence[Sequence[float]], column: int) -> tuple:
    arr = np.asarray(rows, dtype=np.float64)
    x = arr[:, 0]
    z = np.cumsum(arr[:, column] / x)
    return x, z


def zeta_kloosterman_twisted(rows: Sequence[Sequence[float]]) -> tuple:
    r"""
    Partial sums of the twisted Kloosterman sums,

    .. MATH::

        \zeta_{\mathrm{kloosterman}}^{*}(x)
            = \sum_{c \leq x} \frac{\Re S^*(m, n; c)}{c},

    computed from rows ``[c, Re S*, Im S*, Re S, Im S]`` produced by
    :func:`zeta_kloosterman_run`.

    INPUT:

    - ``rows`` -- iterable of ``[c, Re S*, Im S*, Re S, Im S]`` rows

    OUTPUT: pair of numpy arrays ``(x, z)`` with ``x`` the moduli and ``z``
    the running partial sums.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     zeta_kloosterman_run, zeta_kloosterman_twisted)
        sage: state = zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=44)
        sage: x, z = zeta_kloosterman_twisted(state["data"]["1,5"])
        sage: x.shape == z.shape == (4,)
        True
        sage: ref = [0.4098962046748554, 0.5647963894351482,
        ....:        0.4514789417309632, 0.2491658375667683]
        sage: all(abs(float(v) - r) < 1e-12 for v, r in zip(z, ref))
        True
    """
    return _partial(rows, column=1)


def zeta_kloosterman(rows: Sequence[Sequence[float]]) -> tuple:
    r"""
    Partial sums of the ordinary Kloosterman sums,

    .. MATH::

        \zeta_{\mathrm{kloosterman}}(x)
            = \sum_{c \leq x} \frac{\Re S(m, n; c)}{c},

    computed from rows ``[c, Re S*, Im S*, Re S, Im S]`` produced by
    :func:`zeta_kloosterman_run`.

    INPUT:

    - ``rows`` -- iterable of ``[c, Re S*, Im S*, Re S, Im S]`` rows

    OUTPUT: pair of numpy arrays ``(x, z)``.

    EXAMPLES::

        sage: from kloosterman_twisted import (
        ....:     zeta_kloosterman_run, zeta_kloosterman)
        sage: state = zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=44)
        sage: x, z = zeta_kloosterman(state["data"]["1,5"])
        sage: len(x) == len(z) == 4
        True
        sage: ref = [-0.06887158542543956, -0.3287330724909714,
        ....:        -0.2848103948007431, -0.17763438285208366]
        sage: all(abs(float(v) - r) < 1e-12 for v, r in zip(z, ref))
        True
    """
    return _partial(rows, column=3)


# ---------------------------------------------------------------------------
# Top-level one-call convenience wrappers
# ---------------------------------------------------------------------------

def _zeta_value(
    N: int, m: int, n: int, x: float, column: int,
    ainvs: Optional[Sequence[int]] = None,
    an: Optional[Sequence[float]] = None,
    return_curve: bool = False,
):
    if x < N:
        raise ValueError(f"need x >= N (got x={x}, N={N})")
    cmax = int(x)
    state = zeta_kloosterman_run(
        N=N, mn_pairs=[(m, n)], cmax=cmax, an=an, ainvs=ainvs,
    )
    rows = state["data"][f"{m},{n}"]
    xs, zs = _partial(rows, column=column)
    if return_curve:
        return xs, zs
    return float(zs[-1])


def zeta_kloosterman_twisted_value(
    N: int, m: int, n: int, x: float,
    ainvs: Optional[Sequence[int]] = None,
    an: Optional[Sequence[float]] = None,
    return_curve: bool = False,
):
    r"""
    One-call evaluation of the twisted Kloosterman zeta function

    .. MATH::

        \zeta_{\mathrm{kloosterman}}^{*}(x; m, n, N)
            = \sum_{\substack{N \mid c \\ 0 < c \leq x}}
              \frac{\Re S^*(m, n; c)}{c}.

    Coefficients `a_n` for the newform attached to ``CURVES[N]`` are computed
    automatically (via PARI/Sage/point counting, whichever is available), so
    the caller does not need to prepare anything.

    INPUT:

    - ``N`` -- level (must be a key of :data:`~kloosterman_twisted.CURVES`
      unless ``ainvs`` or ``an`` is provided)
    - ``m``, ``n`` -- integers
    - ``x`` -- real cutoff; only moduli ``c`` with ``N | c, c <= x`` contribute
    - ``ainvs`` -- (optional) a-invariants to override ``CURVES[N]``
    - ``an`` -- (optional) precomputed Fourier coefficients (length
      ``>= TRUNCATION_FACTOR * int(x) + 2``); skips the internal ``anlist`` call
    - ``return_curve`` -- if ``True``, return the whole ``(x_arr, z_arr)`` pair
      of running partial sums instead of the scalar terminal value

    OUTPUT: ``float`` when ``return_curve=False`` (default), otherwise the pair
    of numpy arrays ``(x_arr, z_arr)``.

    EXAMPLES::

        sage: from kloosterman_twisted import zeta_kloosterman_twisted_value
        sage: v = zeta_kloosterman_twisted_value(N=11, m=1, n=5, x=220)
        sage: isinstance(v, float)
        True
        sage: abs(v - 0.6335327149272597) < 1e-12
        True
        sage: xs, zs = zeta_kloosterman_twisted_value(
        ....:     N=11, m=1, n=5, x=44, return_curve=True)
        sage: [float(c) for c in xs]
        [11.0, 22.0, 33.0, 44.0]
        sage: abs(float(zs[-1]) - 0.2491658375667683) < 1e-12
        True
    """
    return _zeta_value(N, m, n, x, column=1,
                       ainvs=ainvs, an=an, return_curve=return_curve)


def zeta_kloosterman_value(
    N: int, m: int, n: int, x: float,
    ainvs: Optional[Sequence[int]] = None,
    an: Optional[Sequence[float]] = None,
    return_curve: bool = False,
):
    r"""
    One-call evaluation of the ordinary Kloosterman zeta function

    .. MATH::

        \zeta_{\mathrm{kloosterman}}(x; m, n, N)
            = \sum_{\substack{N \mid c \\ 0 < c \leq x}}
              \frac{\Re S(m, n; c)}{c}.

    Same signature and behaviour as :func:`zeta_kloosterman_twisted_value`.
    The Fourier coefficients ``an`` are strictly speaking not required to
    evaluate the untwisted sum, but they are computed here anyway so that the
    twisted and untwisted values from a single run share a common state.

    INPUT: see :func:`zeta_kloosterman_twisted_value`.

    OUTPUT: ``float`` (default) or ``(x_arr, z_arr)`` when ``return_curve=True``.

    EXAMPLES::

        sage: from kloosterman_twisted import zeta_kloosterman_value
        sage: v = zeta_kloosterman_value(N=11, m=1, n=5, x=220)
        sage: isinstance(v, float)
        True
        sage: abs(v - (-0.20202121338278498)) < 1e-12
        True
    """
    return _zeta_value(N, m, n, x, column=3,
                       ainvs=ainvs, an=an, return_curve=return_curve)
