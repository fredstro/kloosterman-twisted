r"""
Weierstrass a-invariants for the strong Weil curves `Na1` at prime levels `N`.

Only levels carrying an elliptic newform for `\Gamma_0(N)` with `N \leq 50`
are listed (Cremona labels 11a1, 17a1, 19a1, 37a1, 43a1, 49a1).

EXAMPLES::

    sage: from kloosterman_twisted import CURVES
    sage: sorted(CURVES)
    [11, 17, 19, 37, 43, 49]
    sage: CURVES[37]                      # 37a1: y^2 + y = x^3 - x
    [0, 0, 1, -1, 0]
"""

CURVES = {
    11: [0, -1, 1, -10, -20],
    17: [1, -1, 1, -1, -14],
    19: [0, 1, 1, -9, -15],
    37: [0, 0, 1, -1, 0],
    43: [0, 1, 1, 0, 0],
    49: [1, -1, 0, -2, -1],
}
