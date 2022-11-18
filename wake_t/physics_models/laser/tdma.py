"""
This module contais the tridiagonal matrix algorithm (TDMA) used by the laser
envelope solver.

Authors: Wilbert den Hertog, Ángel Ferran Pousa
"""

import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial(fastmath=True)
def TDMA(a, b, c, d, p):
    """TriDiagonal Matrix Algorithm: solve a linear system Ax=b,
    where A is a tridiagonal matrix. Source:
    https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-
    tdma-aka-thomas-algorithm-using-python-with-nump

    Parameters
    ----------
    a : array
        Lower diagonal of A. Dimension: nr-1.
    b : array
        Main diagonal of A. Dimension: nr.
    c : array
        Upper diagonal of A. Dimension: nr-1.
    d : array
        Solution vector. Dimension: nr.

    """
    n = len(d)
    w = np.empty(n - 1, dtype=np.complex128)
    g = np.empty(n, dtype=np.complex128)

    w[0] = c[0] / b[0]  # MAKE SURE THAT b[0]!=0
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        a_im1 = a[i - 1]
        inv_coef = 1. / (b[i] - a_im1 * w[i - 1])
        g[i] = (d[i] - a_im1 * g[i - 1]) * inv_coef
        w[i] = c[i] * inv_coef

    g[-1] = (d[-1] - a[-2] * g[-2]) / (b[-1] - a[-2] * w[-2])

    # Fill in output array.
    p[-1] = g[-1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
