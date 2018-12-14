"""
This module contains the classes for laser pulses and particle bunches to be
used as driver and witness

"""

import numpy as np
import scipy.constants as ct


class LaserPulse():

    """ Stores the laser pulse parameters. """

    def __init__(self, xi_c, l_0, w_0, a_0=None, s_z=None, prop_distance=0):
        """
        Initialize laser pulse parameters.

        Parameters:
        -----------
        xi_c : float
            Central position of the pulse along xi in units of m.
        l_0 : float
            Laser wavelength in units of m.
        w_0 : float
            Spot size (w_0) of the laser pulse in units of m.
        a_0 : float
            Peak normalized vector potential.
        s_z : float
            Longitudinal pulse length (standard deviation) in units of fs.
        prop_distance : float
            Propagation distance of the bunch along the beamline.

        """
        self.xi_c = xi_c
        self.l_0 = l_0
        #self.x_c = x_c
        #self.y_c = y_c
        self.a_0 = a_0
        self.s_z = s_z
        self.w_0 = w_0
        self.prop_distance = prop_distance

    def increase_prop_distance(self, dist):
        """Increase the propagation distance of the laser pulse"""
        self.prop_distance += dist

    def get_group_velocity(self, n_p):
        """
        Get group velocity of the laser pulse for a given plasma density.
        
        Parameters:
        -----------
        n_p : float
            Plasma density in units of cm^{-3}.

        Returns:
        --------
        A float containing the group velocity.
            
        """
        n_p_SI = n_p*1e6
        w_p = np.sqrt(n_p_SI*ct.e**2/(ct.m_e*ct.epsilon_0))
        k = 2*np.pi/self.l_0
        w = np.sqrt(w_p**2+k**2*ct.c**2)
        v_g = k*ct.c**2/np.sqrt(w_p**2+k**2*ct.c**2)/ct.c
        return v_g


class ParticleBunch():

    """ Defines a particle bunch. """

    def __init__(self, q, x, y, xi, px, py, pz, tags=None, prop_distance=0,
                 t_flight=0):
        """
        Initialize particle bunch.

        Parameters:
        -----------
        q : array
            Charge of each particle in units of C.
        x : array
            Position of each particle in the x-plane in units of m.
        y : array
            Position of each particle in the y-plane in units of m.
        xi : array
            Position of each particle in the xi-plane in units of m.
        px : array
            Momentum of each particle in the x-plane in non-dimensional units
            (beta*gamma).
        py : array
            Momentum of each particle in the y-plane in non-dimensional units
            (beta*gamma).
        pz : array
            Momentum of each particle in the z-plane in non-dimensional units
            (beta*gamma).
        tags : array
            Individual tags assigned to each particle.
        prop_distance : float
            Propagation distance of the bunch along the beamline.
        t_flight : float
            Time of flight of the bunch along the beamline.

        """
        self.q = q
        self.x = x
        self.y = y
        self.xi = xi
        self.px = px
        self.py = py
        self.pz = pz
        #self.mu = 0
        self.tags = tags
        self.prop_distance = prop_distance
        self.t_flight = t_flight

    def set_phase_space(self, x, y, xi, px, py, pz):
        """Sets the 6D phase space coordinates"""
        self.x = x
        self.y = y
        self.xi = xi
        self.px = px
        self.py = py
        self.pz = pz

    def get_bunch_matrix(self):
        """Returns a matrix with the 6D phase space and charge of the bunch"""
        return np.array([self.x, self.y, self.xi, self.px, self.py, self.pz,
                         self.q])

    def get_6D_matrix(self):
        """Returns the 6D phase space matrix of the bunch"""
        return np.array([self.x, self.px, self.y, self.py, self.xi, self.pz])

    def increase_prop_distance(self, dist):
        """Increases the propagation distance"""
        self.prop_distance += dist

    def reposition_xi(self, xi_c):
        """Recenter bunch along xi arounf the specified xi_c"""
        current_xi_c = np.average(self.xi, weights=self.q)
        dxi = xi_c - current_xi_c
        self.xi += dxi
