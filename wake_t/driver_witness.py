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

    def __init__(self, q, x=None, y=None, xi=None, px=None, py=None, pz=None,
                 bunch_matrix=None, matrix_type='standard', gamma_ref=None,
                 tags=None, prop_distance=0, t_flight=0):
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
        bunch_matrix : array
            6 x N matrix, where N is the number of partciles, containing the
            phase-space information of the bunch. If provided, the argumets x
            to pz are not considered. The matrix contains (x, px, y, py, z, pz)
            if matrix_type='standard' or (x, x', y, y', xi, dp) if
            matrix_type='alternative'.
        matrix_type : string
            Indicates the type of bunch_matrix. Possible values are 'standard'
            or 'alternative' (see above).
        gamma_ref : float
            Reference energy with respect to which the particle momentum dp is
            calculated. Only needed if bunch_matrix is used and
            matrix_type='alternative'.
        tags : array
            Individual tags assigned to each particle.
        prop_distance : float
            Propagation distance of the bunch along the beamline.
        t_flight : float
            Time of flight of the bunch along the beamline.

        """
        if bunch_matrix is not None:
            if matrix_type == 'standard':
                self.set_phase_space_from_matrix(bunch_matrix)
            elif matrix_type == 'alternative':
                self.set_phase_space_from_alternative_matrix(bunch_matrix,
                                                             gamma_ref)
        else:
            self.x = x
            self.y = y
            self.xi = xi
            self.px = px
            self.py = py
            self.pz = pz
        self.q = q
        #self.mu = 0
        self.tags = tags
        self.prop_distance = prop_distance
        self.t_flight = t_flight

    def set_phase_space(self, x, y, xi, px, py, pz):
        """Sets the phase space coordinates"""
        self.x = x
        self.y = y
        self.xi = xi
        self.px = px
        self.py = py
        self.pz = pz

    def set_phase_space_from_matrix(self, beam_matrix):
        """
        Sets the phase space coordinates from a matrix with the values of
        (x, px, y, py, xi, pz).
        
        """
        self.x = beam_matrix[0]
        self.y = beam_matrix[2]
        self.xi = beam_matrix[4]
        self.px = beam_matrix[1]
        self.py = beam_matrix[3]
        self.pz = beam_matrix[5]

    def set_phase_space_from_alternative_matrix(self, beam_matrix, gamma_ref):
        """
        Sets the phase space coordinates from a matrix with the values of
        (x, x', y, y', xi, dp).

        Parameters:
        -----------
        bunch_matrix : array
            6 x N matrix, where N is the number of partciles, containing the
            phase-space information of the bunch as (x, x', y, y', xi, dp) in
            units of (m, rad, m, rad, m, -). dp is defined as
            dp = (g-g_ref)/g_ref, while x' = px/p_kin and y' = py/p_kin, where
            p_kin is the kinetic momentum of each particle.

        gamma_ref : float
            Reference energy with respect to which the particle momentum dp is
            calculated. 

        """
        dp = beam_matrix[5]
        gamma = (dp + 1)*gamma_ref
        p_kin = np.sqrt(gamma**2 - 1)
        self.x = beam_matrix[0]
        self.px = beam_matrix[1] * p_kin
        self.y = beam_matrix[2]
        self.py = beam_matrix[3] * p_kin
        self.xi = beam_matrix[4]
        self.pz = np.sqrt(gamma**2 - self.px**2 - self.py**2 - 1)

    def get_bunch_matrix(self):
        """Returns a matrix with the 6D phase space and charge of the bunch"""
        return np.array([self.x, self.y, self.xi, self.px, self.py, self.pz,
                         self.q])

    def get_6D_matrix(self):
        """
        Returns the 6D phase space matrix of the bunch containing
        (x, px, y, py, xi, pz)
        """
        return np.array([self.x, self.px, self.y, self.py, self.xi, self.pz])

    def get_alternative_6D_matrix(self):
        """
        Returns the 6D matrix of the bunch containing
        (x, x', y, y', xi, dp)
        """
        g = np.sqrt(1 + self.px**2 + self.py**2 + self.pz**2)
        g_avg = np.average(g, weights=self.q)
        b_avg = np.sqrt(1 - g_avg**(-2))
        dp = (g-g_avg)/(g_avg*b_avg)
        p_kin = np.sqrt(g**2 - 1)
        return np.array([self.x, self.px/p_kin, self.y, self.py/p_kin,
                         self.xi, dp]), g_avg

    def increase_prop_distance(self, dist):
        """Increases the propagation distance"""
        self.prop_distance += dist

    def reposition_xi(self, xi_c):
        """Recenter bunch along xi arounf the specified xi_c"""
        current_xi_c = np.average(self.xi, weights=self.q)
        dxi = xi_c - current_xi_c
        self.xi += dxi
