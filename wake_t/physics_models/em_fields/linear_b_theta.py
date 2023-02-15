""" Defines a linearly-varying (in radius) azimuthal magnetic field """

from wake_t.fields.analytical_field import AnalyticalField


def b_x(x, y, z, t, bx, constants):
    """B_x component."""
    k = constants[0]
    for i in range(x.shape[0]):
        bx[i] -= k * y[i]


def b_y(x, y, z, t, by, constants):
    """B_y component."""
    k = constants[0]
    for i in range(x.shape[0]):
        by[i] += k * x[i]


class LinearBThetaField(AnalyticalField):
    """Defines a linear azimuthal magnetic field.

    Defining `k` as the constant focusing gradient, the field can be expressed
    as:

        b_theta = k * r

    where r is the radius.

    In cartesian coordinates, this is equivalent to:

        b_x = - k * y
        b_y = k * x

    Parameters
    ----------
    foc_gradient : float
        Uniform focusing gradient in T/m.
    """

    def __init__(self, foc_gradient):
        super().__init__(b_x=b_x, b_y=b_y, constants=[foc_gradient])
