""" Methods for gathering fields """

from typing import List

import numpy as np

from .base import Field


def gather_fields(
    fields: List[Field],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> None:
    """Gather all fields at the specified locations and time.

    Parameters
    ----------
    fields : list
        List of `Field`s.
    x : ndarray
        1D array containing the x position where to gather the fields.
    y : ndarray
        1D array containing the x position where to gather the fields.
    z : ndarray
        1D array containing the x position where to gather the fields.
    t : float
        Time at which the field is being gathered.
    ex : ndarray
        1D array where the gathered Ex values will be stored.
    ey : ndarray
        1D array where the gathered Ey values will be stored
    ez : ndarray
        1D array where the gathered Ez values will be stored
    bx : ndarray
        1D array where the gathered Bx values will be stored
    by : ndarray
        1D array where the gathered By values will be stored
    bz : ndarray
        1D array where the gathered Bz values will be stored
    """
    # Initially, set all field values to zero.
    ex[:] = 0.
    ey[:] = 0.
    ez[:] = 0.
    bx[:] = 0.
    by[:] = 0.
    bz[:] = 0.

    # Gather contributions from all fields.
    for field in fields:
        field.gather(x, y, z, t, ex, ey, ez, bx, by, bz)
