""" This module contains the possible plasma wakefields """

import numpy as np
import scipy.constants as ct

class Wakefield():

    """ Base class for all wakefields """

    def __init__(self):
        pass

    def Wx(self, x, y, xi, t):
        raise NotImplementedError

    def Wy(self, x, y, xi, t):
        raise NotImplementedError

    def Ez(self, x, y, xi, t):
        raise NotImplementedError

    def Kx(self, x, y, xi, t):
        raise NotImplementedError


class CustomBlowoutWakefield(Wakefield):
    def __init__(self, n_p, driver, beam_center, lon_field=None,
                 lon_field_slope=None, foc_strength=None):
        """
        [n_p] = cm^-3
        """
        self.n_p = n_p
        self.xi_c = beam_center	
        self.driver = driver
        self._calculate_base_quantities(lon_field, lon_field_slope,
                                        foc_strength)

    def _calculate_base_quantities(self, lon_field, lon_field_slope,
                                   foc_strength):
        self.g_x = foc_strength
        self.E_z_0 = lon_field
        self.E_z_p = lon_field_slope
        self.l_c = self.driver.xi_c
        self.b_w = self.driver.get_group_velocity(self.n_p*1e-6)

    def Wx(self, x, y, xi, t):
        return ct.c*self.g_x*x

    def Wy(self, x, y, xi, t):
        return ct.c*self.g_x*y

    def Ez(self, x, y, xi, t):
        return self.E_z_0 + self.E_z_p*(xi-self.xi_c + (1-self.b_w)*ct.c*t)

    def Kx(self, x, y, xi, t):
        return self.g_x*np.ones_like(x)
