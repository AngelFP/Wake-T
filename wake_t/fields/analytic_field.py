"""Contains the class used to define analytic fields."""

import numpy as np
from numba import njit

from .base import Field


class AnalyticField(Field):
    """Class used to define fields with analytical components.

    The given components (Ex, Ey, Ez, Bx, By, Bz) must be functions taking 5
    arguments (3 arrays containing the x, y, z positions where to calculate the
    field; 1 array with the same size where the field values will be stored;
    and a list of constants). The given functions must be written in a way
    which allows them to be compiled with `numba`.

    Not all components need to be given. Those which are not specified will
    simply return a zero array when gathered.

    In addition to the field components, a list of constants can also be given.
    This list of constants is always passed to the field functions and can be
    used to compute the field.

    Example
    -------
    >>> def linear_ex(x, y, z, ex, constants):
    ...     ex_slope = constants[0]
    ...     for i in range(x.shape[0]):
    ...         ex[i] = ex_slope * x[i]
    ...
    >>> ex = AnalyticField(e_x=linear_ex, constants=[1e6])

    """

    def __init__(
            self, e_x=None, e_y=None, e_z=None, b_x=None, b_y=None, b_z=None,
            constants=[]
            ):
        """Initialize field.

        Parameters
        ----------
        e_x : function, optional
            Function defining the Ex component, by default None
        e_y : function, optional
            Function defining the Ey component, by default None
        e_z : function, optional
            Function defining the Ez component, by default None
        b_x : function, optional
            Function defining the Bx component, by default None
        b_y : function, optional
            Function defining the By component, by default None
        b_z : function, optional
            Function defining the Bz component, by default None
        constants : list, optional
            List of constants to be passed to each component, by default []
        """
        super().__init__()

        @njit()
        def zero_field(x, y, z, fld, k):
            """Default field component."""
            for i in range(len(fld)):
                fld[i] = 0.

        self.__e_x = njit()(e_x) if e_x is not None else zero_field
        self.__e_y = njit()(e_y) if e_y is not None else zero_field
        self.__e_z = njit()(e_z) if e_z is not None else zero_field
        self.__b_x = njit()(b_x) if b_x is not None else zero_field
        self.__b_y = njit()(b_y) if b_y is not None else zero_field
        self.__b_z = njit()(b_z) if b_z is not None else zero_field
        self.constants = np.array(constants)

    def _gather(self, x, y, z, ex, ey, ez, bx, by, bz):
        self.__e_x(x, y, z, ex, self.constants)
        self.__e_y(x, y, z, ey, self.constants)
        self.__e_z(x, y, z, ez, self.constants)
        self.__b_x(x, y, z, bx, self.constants)
        self.__b_y(x, y, z, by, self.constants)
        self.__b_z(x, y, z, bz, self.constants)
