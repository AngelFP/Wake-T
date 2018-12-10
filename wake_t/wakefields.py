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


class PlasmaUprampBlowoutField(Wakefield):
    def __init__(self, ramp_length, initial_dens, final_dens,
                 initial_dens_pos=0, profile='exponential'):
        self.ramp_duration = ramp_length/ct.c
        self.initial_dens = initial_dens
        self.initial_dens_pos = initial_dens_pos/ct.c
        self.final_dens = final_dens
        self.current_time = None
        self.current_kx = None
        self.profile = profile

    def Wx(self, x, y, xi, px, py, pz, gamma, t):
        self.calculate_current_focusing(xi, t)
        return ct.c*self.current_kx*x

    def Wy(self, x, y, xi, px, py, pz, gamma, t):
        self.calculate_current_focusing(xi, t)
        return ct.c*self.current_kx*y

    def Wz(self, x, y, xi, px, py, pz, gamma, t):
        return np.zeros(len(x))

    def Kx(self, x, y, xi, px, py, pz, gamma, t):
        self.calculate_current_focusing(xi, t)
        return np.ones(len(x))*self.current_kx

    def calculate_current_focusing(self, xi, t):
        t = t + xi/ct.c # particles with different xi have a different t
        n_p = self.calculate_denstity(t)
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        self.current_kx = (ct.m_e/(2*ct.e*ct.c))*w_p**2

    def calculate_denstity(self, t):
        if self.profile == 'exponential':
            b = (np.log(self.final_dens / self.initial_dens)
                 /(self.ramp_duration - self.initial_dens_pos))
            a = self.initial_dens*np.exp(-self.initial_dens_pos*b)
            n_p = a*np.exp(b*t)
        elif self.profile == 'linear':
            b = ((self.final_dens - self.initial_dens)
                 /(self.ramp_duration - self.initial_dens_pos))
            a = self.initial_dens - b*self.initial_dens_pos
            n_p = a + b*t
            # make negative densities 0
            n_p[n_p<0] = 0
        elif self.profile == 'inverse square':
            a = np.sqrt(self.final_dens/self.initial_dens) - 1
            b = (self.ramp_duration - self.initial_dens_pos)/a
            n_p = self.final_dens/np.square(1+(self.ramp_duration-t)/b)
        return n_p

