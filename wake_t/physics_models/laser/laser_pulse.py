"""
This module contains the class defining a laser pulse.
"""
import numpy as np
import scipy.constants as ct


class LaserPulse():

    """ Stores the laser pulse parameters. """

    def __init__(self, xi_c, l_0, w_0, a_0=None, tau=None,
                 polarization='linear', prop_distance=0):
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
        tau : float
            Longitudinal pulse length (FWHM in intensity) in units of s.
        prop_distance : float
            Propagation distance of the bunch along the beamline.

        """
        self.xi_c = xi_c
        self.l_0 = l_0
        # self.x_c = x_c
        # self.y_c = y_c
        self.a_0 = a_0
        self.tau = tau
        self.w_0 = w_0
        self.z_r = np.pi * w_0**2 / l_0
        self.polarization = polarization
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
            Plasma density in units of m^{-3}.

        Returns:
        --------
        A float containing the group velocity.

        """
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        k = 2*np.pi/self.l_0
        v_g = k*ct.c**2/np.sqrt(w_p**2+k**2*ct.c**2)/ct.c
        return v_g

    def get_a0_profile(self, r, xi, dz_foc=0):
        """
        Return the normalized vector potential profile of the laser averaged
        envelope.

        Parameters:
        -----------
        r : array
            Radial position at which to calculate normalized potential.
        xi : array
            Longitudinal position at which to calculate normalized potential.

        dz_foc : float
            Distance to focal point (beam waist).

        Returns:
        --------
        An array containing the values of the normalized vector potential at
        the specified positions.

        """
        w_fac = np.sqrt(1 + (dz_foc/self.z_r)**2)
        s_r = self.w_0 * w_fac / np.sqrt(2)
        s_z = self.tau * ct.c / (2*np.sqrt(2*np.log(2))) * np.sqrt(2)
        avg_amplitude = self.a_0
        if self.polarization == 'linear':
            avg_amplitude /= np.sqrt(2)
        return avg_amplitude/w_fac * (np.exp(-(r)**2/(2*s_r**2)) *
                                      np.exp(-(xi-self.xi_c)**2/(2*s_z**2)))
