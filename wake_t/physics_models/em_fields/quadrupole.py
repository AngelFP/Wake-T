""" Defines a magnetic field of a quadrupole """

from wake_t.fields.analytical_field import AnalyticalField
from wake_t.utilities.numba import prange


def b_x(x, y, z, t, bx, constants):
    """B_x component."""
    k = - constants[0]
    for i in prange(x.shape[0]):
        bx[i] += k * y[i]


def b_y(x, y, z, t, by, constants):
    """B_y component."""
    k = - constants[0]
    for i in prange(x.shape[0]):
        by[i] += k * x[i]


class QuadrupoleField(AnalyticalField):
    """Defines a field of a magnetic quadrupole of constant focusing gradient
    `k`.

    In Cartesian coordinates, the field is given by:
    ```
        b_x = - k * y
        b_y = - k * x
    ```

    When `k > 0`, it corresponds to focussing electrons in the `x` direction
    and defocussing in `y`. When `k < 0`, the result is the opposite.

    Parameters
    ----------
    foc_gradient : float
        Uniform focusing gradient in T/m.
    """

    def __init__(self, foc_gradient):
        super().__init__(b_x=b_x, b_y=b_y, constants=[foc_gradient])
