import numpy as np
import scipy.constants as ct

from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class CustomBlowoutWakefield(Wakefield):
    def __init__(self, n_p, laser, lon_field=None, lon_field_slope=None,
                 foc_strength=None, xi_fields=None):
        """
        [n_p] = m^-3
        """
        super().__init__()
        self.n_p = n_p
        self.xi_fields = xi_fields
        self.laser = laser
        self._calculate_base_quantities(lon_field, lon_field_slope,
                                        foc_strength)

    def _calculate_base_quantities(self, lon_field, lon_field_slope,
                                   foc_strength):
        self.g_x = foc_strength
        self.E_z_0 = lon_field
        self.E_z_p = lon_field_slope
        self.l_c = self.laser.xi_c
        self.b_w = self.laser.get_group_velocity(self.n_p)

    def Wx(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.g_x*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.g_x*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        if self.xi_fields is None:
            self.xi_fields = np.average(xi, weights=q)
        return self.E_z_0 + self.E_z_p*(xi - self.xi_fields
                                        + (1-self.b_w)*ct.c*t)
