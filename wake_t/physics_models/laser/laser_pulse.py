"""
This module contains the class definitions for different laser pulses.

"""
import numpy as np
import scipy.constants as ct

from .envelope_solver import evolve_envelope


class LaserPulse():

    """ Base class for all Laser pulses. """

    def __init__(self, l_0):
        """
        Initialize pulse.

        Parameters:
        -----------
        l_0 : float
            Laser wavelength in units of m.

        """
        self.l_0 = l_0
        self.a_env_old = None
        self.a_env = None
        self.solver_params = None

    def set_envelope_solver_params(self, xi_min, xi_max, r_max, nz, nr, dt,
                                   nt=1):
        """
        Set the parameters for the laser envelope solver.

        Parameters:
        -----------
        xi_min, xi_max : float
            Position of the left and right boundaries of the the grid in SI
            units.
        r_max : float
            Radial extension of the grid in SI units.
        nz, nr : int
            Number of grid points along the longitudinal and radial directions.
        dt : float
            Time step, in SI units, that the laser pulse should advance every
            time `evolve` is called.
        nt : int
            Number of sub-time-steps that should be computed every time
            `evolve` is called. The internal time step used by the solver is
            therefore dt/nt, so that the laser effectively advances by `dt`
            every time `evolve` is called. All these time steps are therefore
            computed using the same `chi`.

        """
        solver_params = {
            'zmin': xi_min,
            'zmax': xi_max,
            'rmax': r_max,
            'nz': nz,
            'nr': nr,
            'nt': nt,
            'dt': dt / nt
        }
        if self.a_env is not None and solver_params != self.solver_params:
            raise ValueError(
                'Solver parameters cannot be changed once envelope has been '
                'initialized.'
            )
        else:
            self.solver_params = solver_params

    def initialize_envelope(self):
        """ Initialize laser envelope arrays. """
        if self.solver_params is None:
            raise ValueError(
                'Envelope solver parameters not yet set.'
                'Cannot initialize envelope.')
        if self.a_env is None:
            z_min = self.solver_params['zmin']
            z_max = self.solver_params['zmax']
            r_max = self.solver_params['rmax']
            nz = self.solver_params['nz']
            nr = self.solver_params['nr']
            dt = self.solver_params['dt']
            dr = r_max / nr
            z = np.linspace(z_min, z_max, nz)
            r = np.linspace(dr/2, r_max-dr/2, nr)
            ZZ, RR = np.meshgrid(z, r, indexing='ij')
            self.a_env = self.envelope_function(ZZ, RR, 0.)
            self.a_env_old = self.envelope_function(ZZ, RR, -dt*ct.c)

    def get_envelope(self):
        """ Get the current laser envelope array. """
        return self.a_env

    def evolve(self, chi, n_p):
        """
        Evolve laser envelope to next time step.

        Parameters:
        -----------
        chi : ndarray
            A (nz x nr) array containing the plasma susceptibility.
        n_p : float
            Plasma density in SI units.
        
        """
        k_0 = 2*np.pi / self.l_0
        k_p = np.sqrt(ct.e**2 * n_p / (ct.m_e*ct.epsilon_0)) / ct.c
        a_env_old, a_env = evolve_envelope(
            self.a_env, self.a_env_old, chi, k_0, k_p, **self.solver_params)
        self.a_env_old[:] = a_env_old[0: -2]
        self.a_env = a_env[0: -2]

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

    def envelope_function(self, xi, r, z_pos):
        """
        Complex envelope of a Gaussian beam in the paraxial approximation.
        
        """
        return np.zeros_like(r)


class GaussianPulse(LaserPulse):

    """ Class defining a Gaussian laser pulse. """
    
    def __init__(self, xi_c, l_0, w_0, a_0=None, tau=None, z_foc=None,
                 polarization='linear'):
        """
        Initialize Gaussian pulse.

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
        z_foc : float
            Focal position of the pulse.
        polarization : str
            Polarization of the laser pulse. Accepted values are 'linear'
            (default) or 'circular'.

        """
        super().__init__(l_0)
        self.xi_c = xi_c
        self.a_0 = a_0
        self.tau = tau
        self.w_0 = w_0
        self.z_foc = z_foc
        self.z_r = np.pi * w_0**2 / l_0
        self.polarization = polarization

    def envelope_function(self, xi, r, z_pos):
        """
        Complex envelope of a Gaussian beam in the paraxial approximation.
        
        """
        z = xi + z_pos - self.z_foc
        diff_factor = 1. + 1j * (z - self.z_foc) / self.z_r
        s_z = self.tau * ct.c / (2*np.sqrt(2*np.log(2))) * np.sqrt(2)
        # Phases
        exp_r = r**2 / (self.w_0**2 * diff_factor)
        exp_z = (xi-self.xi_c)**2 / (2*s_z**2)
        # Profile
        gaussian_profile = np.exp(- exp_r - exp_z)
        # Amplitude
        avg_amplitude = self.a_0
        if self.polarization == 'linear':
            avg_amplitude /= np.sqrt(2)
        return avg_amplitude / diff_factor * gaussian_profile
