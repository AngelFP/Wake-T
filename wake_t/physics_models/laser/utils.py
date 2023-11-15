"""Utilities for the laser envelope solver."""
import numpy as np
from wake_t.utilities.numba import njit_serial


@njit_serial
def unwrap(p, discont=None, axis=-1, period=6.283185307179586):
    """Numba version of numpy.unwrap.

    The implementation is taken from
    https://github.com/numba/numba/blob/main/numba/np/arraymath.py,
    which currently is not yet included in the latest Numba release.
    """
    if axis != -1:
        msg = 'Value for argument "axis" is not supported'
        raise ValueError(msg)
    # Flatten to a 2D array, keeping axis -1
    p_init = np.asarray(p).astype(np.float64)
    init_shape = p_init.shape
    last_axis = init_shape[-1]
    p_new = p_init.reshape((p_init.size // last_axis, last_axis))
    # Manipulate discont and period
    if discont is None:
        discont = period / 2
    interval_high = period / 2
    boundary_ambiguous = True
    interval_low = -interval_high

    slice1 = (slice(1, None, None),)

    # Work on each row separately
    for i in range(p_init.size // last_axis):
        row = p_new[i]
        dd = np.diff(row)
        ddmod = np.mod(dd - interval_low, period) + interval_low
        if boundary_ambiguous:
            ddmod = np.where(
                (ddmod == interval_low) & (dd > 0),
                interval_high,
                ddmod
            )
        ph_correct = ddmod - dd

        ph_correct = np.where(
            np.array([abs(x) for x in dd]) < discont,
            0,
            ph_correct
        )
        ph_ravel = np.where(
            np.array([abs(x) for x in dd]) < discont,
            0,
            ph_correct
        )
        ph_correct = np.reshape(ph_ravel, ph_correct.shape)
        up = np.copy(row)
        up[slice1] = row[slice1] + ph_correct.cumsum()
        p_new[i] = up

    return p_new.reshape(init_shape)
