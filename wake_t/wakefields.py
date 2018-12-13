""" This module contains the possible plasma wakefields """

import numpy as np
import scipy.constants as ct

class Wakefield():

    """ Base class for all wakefields """

    def __init__(self):
        pass

    def Wx(self, x, y, xi, px, py, pz, gamma, t):
        raise NotImplementedError

    def Wy(self, x, y, xi, px, py, pz, gamma, t):
        raise NotImplementedError

    def Wz(self, x, y, xi, px, py, pz, gamma, t):
        raise NotImplementedError

    def Kx(self, x, y, xi, px, py, pz, gamma, t):
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

    def Wx(self, x, y, xi, px, py, pz, gamma, t):
        return ct.c*self.g_x*x

    def Wy(self, x, y, xi, px, py, pz, gamma, t):
        return ct.c*self.g_x*y

    def Wz(self, x, y, xi, px, py, pz, gamma, t):
        return self.E_z_0 + self.E_z_p*(xi-self.xi_c + (1-self.b_w)*ct.c*t)

    def Kx(self, x, y, xi, px, py, pz, gamma, t):
        return self.g_x*np.ones_like(x)


class PlasmaRampBlowoutField(Wakefield):
    def __init__(self, density_function):
        self.density_function = density_function

    def Wx(self, x, y, xi, px, py, pz, gamma, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*x

    def Wy(self, x, y, xi, px, py, pz, gamma, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*y

    def Wz(self, x, y, xi, px, py, pz, gamma, t):
        return np.zeros(len(xi))

    def Kx(self, x, y, xi, px, py, pz, gamma, t):
        kx = self.calculate_focusing(xi, t)
        return np.ones(len(xi))*kx

    def calculate_focusing(self, xi, t):
        z = t*ct.c + xi # z postion of each particle at time t
        n_p = self.density_function(z)
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        return (ct.m_e/(2*ct.e*ct.c))*w_p**2
