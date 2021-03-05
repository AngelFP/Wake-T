import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class SimpleBlowoutWakefield(Wakefield):
    def __init__(self, n_p, laser, field_offset=0):
        """
        [n_p] = m^-3
        """
        super().__init__()
        self.n_p = n_p
        self.field_off = field_offset
        self.laser = laser
        self._calculate_base_quantities()

    def _calculate_base_quantities(self):
        w_p = ge.plasma_frequency(self.n_p*1e-6)
        self.l_p = 2*np.pi*ct.c / w_p
        self.g_x = w_p**2/2 * ct.m_e / (ct.e * ct.c)
        self.E_z_p = w_p**2/2 * ct.m_e / ct.e
        self.l_c = self.laser.xi_c
        self.b_w = self.laser.get_group_velocity(self.n_p)

    def Wx(self, x, y, xi, px, py, pz, q, t):
        return ct.c * self.g_x*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        return ct.c * self.g_x*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        return self.E_z_p * (self.l_p/2 + xi - self.l_c - self.field_off +
                             (1-self.b_w)*ct.c*t)
