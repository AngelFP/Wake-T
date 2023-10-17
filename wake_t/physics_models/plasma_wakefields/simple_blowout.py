import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.fields.analytical_field import AnalyticalField
from wake_t.utilities.numba import prange


class SimpleBlowoutWakefield(AnalyticalField):
    def __init__(self, n_p, laser, field_offset=0):
        """
        [n_p] = m^-3
        """
        self.density = n_p
        self.laser = laser
        self.field_offset = field_offset

        def e_x(x, y, xi, t, ex, constants):
            k = constants[0]
            for i in prange(x.shape[0]):
                ex[i] = ct.c * k * x[i]

        def e_y(x, y, xi, t, ey, constants):
            k = constants[0]
            for i in prange(x.shape[0]):
                ey[i] = ct.c * k * y[i]

        def e_z(x, y, xi, t, ez, constants):
            e_z_p = constants[1]
            l_p = constants[2]
            l_c = constants[3]
            field_off = constants[4]
            b_w = constants[5]

            # Precalculate offset.
            xi_off = l_p/2 - l_c - field_off + (1-b_w)*ct.c*t
            for i in prange(x.shape[0]):
                ez[i] = e_z_p * (xi[i] + xi_off)

        super().__init__(e_x=e_x, e_y=e_y, e_z=e_z)

    def _pre_gather(self, x, y, xi, t):
        n_p = self.density(t*ct.c)
        w_p = ge.plasma_frequency(n_p*1e-6)
        l_p = 2*np.pi*ct.c / w_p
        g_x = w_p**2/2 * ct.m_e / (ct.e * ct.c)
        e_z_p = w_p**2/2 * ct.m_e / ct.e
        l_c = self.laser.xi_c
        b_w = self.laser.get_group_velocity(n_p)
        self.constants = np.array(
            [g_x, e_z_p, l_p, l_c, self.field_offset, b_w])
