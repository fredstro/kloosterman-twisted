r"""Basic smoke tests for kloosterman_twisted."""

import math

import numpy as np
import pytest

from kloosterman_twisted import (
    CURVES,
    anlist,
    anlist_pointcount,
    eichler_values,
    kloosterman_sum,
    kloosterman_sum_twisted,
    kloosterman_sum_twisted_deformed,
    kloosterman_sum_twisted_projected,
    units_and_inverses,
    zeta_kloosterman,
    zeta_kloosterman_run,
    zeta_kloosterman_twisted,
)


@pytest.fixture(scope="module")
def an11():
    return anlist_pointcount(CURVES[11], 2000)


def test_anlist_11_known_values(an11):
    # 11a1 first coefficients
    assert an11[:10] == [0, 1, -2, -1, 2, 1, 2, -2, 0, -2]


def test_anlist_auto_backend_matches_pointcount():
    # auto picks Sage if available, pointcount otherwise; both must agree
    assert anlist(CURVES[11], 20, backend="pointcount") == anlist(CURVES[11], 20)


def test_anlist_unknown_backend_rejected():
    with pytest.raises(ValueError):
        anlist(CURVES[11], 5, backend="nope")


def test_kloosterman_matches_definition():
    m, n, c = 1, 5, 22
    got = kloosterman_sum(m, n, c)
    ref = sum(
        complex(math.cos(2 * math.pi * (n * pow(d, -1, c) + m * d) / c),
                math.sin(2 * math.pi * (n * pow(d, -1, c) + m * d) / c))
        for d in range(1, c) if math.gcd(d, c) == 1
    )
    assert abs(got - ref) < 1e-10


def test_twisted_is_real(an11):
    val = kloosterman_sum_twisted(1, 5, 22, 11, an11)
    assert abs(val.imag) < 1e-9


def test_projection_identity(an11):
    proj = kloosterman_sum_twisted_projected(1, 5, 22, 11, an11)
    assert abs(proj["twisted_even"] + proj["twisted_odd"] - proj["twisted"]) < 1e-9


def test_units_and_inverses_small():
    d, a = units_and_inverses(30)
    assert np.all((d * a) % 30 == 1)


def test_eichler_shape(an11):
    g = eichler_values(22, an11)
    assert g.shape == (22,)


def test_deformed_at_zero_matches_kloosterman(an11):
    d = kloosterman_sum_twisted_deformed(1, 5, 22, 11, an11, [0.0])
    s1, s2 = d[0.0]
    kl = kloosterman_sum(1, 5, 22)
    assert abs(s1 - kl) < 1e-9
    assert abs(s2 - kl) < 1e-9


def test_zeta_kloosterman_twisted_value_one_call():
    from kloosterman_twisted import (
        zeta_kloosterman_twisted_value, zeta_kloosterman_value,
    )
    v_t = zeta_kloosterman_twisted_value(N=11, m=1, n=5, x=44)
    v_u = zeta_kloosterman_value(N=11, m=1, n=5, x=44)
    assert isinstance(v_t, float) and isinstance(v_u, float)

    # curve mode: cross-check that the terminal value matches the scalar mode
    xs, zs = zeta_kloosterman_twisted_value(
        N=11, m=1, n=5, x=44, return_curve=True,
    )
    assert abs(float(zs[-1]) - v_t) < 1e-12
    assert list(int(c) for c in xs) == [11, 22, 33, 44]

    with pytest.raises(ValueError):
        zeta_kloosterman_twisted_value(N=11, m=1, n=5, x=5)   # x < N


def test_zeta_kloosterman_run_smoke():
    state = zeta_kloosterman_run(N=11, mn_pairs=[(1, 5)], cmax=44)
    assert state["meta"]["complete"]
    x_t, z_t = zeta_kloosterman_twisted(state["data"]["1,5"])
    x_u, z_u = zeta_kloosterman(state["data"]["1,5"])
    assert len(x_t) == len(z_t) == len(x_u) == len(z_u) > 0
    assert np.array_equal(x_t, x_u)
