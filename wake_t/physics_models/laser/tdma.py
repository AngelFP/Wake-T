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
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
