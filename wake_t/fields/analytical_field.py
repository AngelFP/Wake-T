"""Contains the class used to define analytic fields."""

from typing import Callable, Optional, List
import numpy as np

from .base import Field
from wake_t.utilities.numba import njit_parallel


# Define type alias.
FieldFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, List],
    np.ndarray
]


class AnalyticalField(Field):
    """Class used to define fields with analytical components.

    The given components (Ex, Ey, Ez, Bx, By, Bz) must be functions taking 5
    arguments (3 arrays containing the x, y, z positions where to calculate the
    field; 1 array with the same size where the field values will be stored;
    and a list of constants). The given functions must be written in a way
    which allows them to be compiled with ``numba``.

    Not all components need to be given. Those which are not specified will
    simply return a zero array when gathered.

    In addition to the field components, a list of constants can also be given.
    This list of constants is always passed to the field functions and can be
    used to compute the field.

    Parameters
    ----------
    e_x : callable, optional
        Function defining the Ex component.
    e_y : callable, optional
        Function defining the Ey component.
    e_z : callable, optional
        Function defining the Ez component.
    b_x : callable, optional
        Function defining the Bx component.
    b_y : callable, optional
        Function defining the By component.
    b_z : callable, optional
        Function defining the Bz component.
    constants : list, optional
        List of constants to be passed to each component.

    Examples
    --------
    >>> from numba import prange
    >>> def linear_ex(x, y, z, t, ex, constants):
    ...     ex_slope = constants[0]
    ...     for i in prange(x.shape[0]):
    ...         ex[i] = ex_slope * x[i]
    ...
    >>> ex = AnalyticField(e_x=linear_ex, constants=[1e6])

    """

    def __init__(
        self,
        e_x: Optional[FieldFunction] = None,
        e_y: Optional[FieldFunction] = None,
        e_z: Optional[FieldFunction] = None,
        b_x: Optional[FieldFunction] = None,
        b_y: Optional[FieldFunction] = None,
        b_z: Optional[FieldFunction] = None,
        constants: Optional[List] = None
    ) -> None:
        super().__init__()

        constants = [] if constants is None else constants

        def no_field(x, y, z, t, fld, k):
            """Default field component."""
            pass

        self.__e_x = njit_parallel(e_x) if e_x is not None else no_field
        self.__e_y = njit_parallel(e_y) if e_y is not None else no_field
        self.__e_z = njit_parallel(e_z) if e_z is not None else no_field
        self.__b_x = njit_parallel(b_x) if b_x is not None else no_field
        self.__b_y = njit_parallel(b_y) if b_y is not None else no_field
        self.__b_z = njit_parallel(b_z) if b_z is not None else no_field
        self.constants = np.array(constants)

    def _pre_gather(self, x, y, z, t):
        """Function that is automatically called just before gathering.

        This method can be overwritten by derived classes and used to,
        for example, pre-compute any useful quantities. This method is not
        compiled by numba.
        """
        pass

    def _gather(self, x, y, z, t, ex, ey, ez, bx, by, bz):
        self._pre_gather(x, y, z, t)
        self.__e_x(x, y, z, t, ex, self.constants)
        self.__e_y(x, y, z, t, ey, self.constants)
        self.__e_z(x, y, z, t, ez, self.constants)
        self.__b_x(x, y, z, t, bx, self.constants)
        self.__b_y(x, y, z, t, by, self.constants)
        self.__b_z(x, y, z, t, bz, self.constants)
