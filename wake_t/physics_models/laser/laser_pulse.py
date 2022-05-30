"""
This module contains the class definitions for different laser pulses.

The implementation of the Laguerre-Gauss and Flattened Gaussian
profiles, as well as the SummedPulse approach is based on FBPIC
(https://github.com/fbpic/fbpic).

Authors: Angel Ferran Pousa, Remi Lehe, Manuel Kirchen, Pierre Pelletier.

"""
import numpy as np
import scipy.constants as ct
from scipy.special import genlaguerre, binom

from .envelope_solver import evolve_envelope


class LaserPulse():
    """Base class for all Laser pulses."""

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
        self.init_outside_plasma = False
        self.n_steps = 0

    def __add__(self, pulse_2):
        """Overload the add operator to allow summing of laser pulses."""
        return SummedPulse(self, pulse_2)

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
        """Initialize laser envelope arrays."""
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
            self.init_outside_plasma = True

    def get_envelope(self):
        """Get the current laser envelope array."""
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

        # Determine if laser starts evolution outside plasma.
        start_outside_plasma = (self.n_steps == 0 and self.init_outside_plasma)

        # Compute evolution.
        a_env_old, a_env = evolve_envelope(
            self.a_env, self.a_env_old, chi, k_0, k_p, **self.solver_params,
            start_outside_plasma=start_outside_plasma)

        # Update arrays and step count.
        self.a_env_old[:] = a_env_old[0: -2]
        self.a_env = a_env[0: -2]
        self.n_steps += 1

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
        """Return the complex envelope of the laser pulse."""
        return np.zeros_like(r)


class SummedPulse(LaserPulse):
    """Class defining a laser pulse made up of the addition of two pulses."""

    def __init__(self, pulse_1, pulse_2):
        """
        Initialize summed pulse.

        Parameters:
        -----------
        pulse_1 : LaserPulse
            A LaserPulse instance.
        pulse_2 : LaserPulse
            Another LaserPulse instance to be summed to `pulse_1`.
        """
        if pulse_1.l_0 != pulse_2.l_0:
            raise ValueError(
                'Only laser pulses with the same central wavelength'
                ' can currently be summed.')
        super().__init__(pulse_1.l_0)
        self.pulse_1 = pulse_1
        self.pulse_2 = pulse_2

    def envelope_function(self, xi, r, z_pos):
        """Return the summed envelope of the two pulses."""
        a_env_1 = self.pulse_1.envelope_function(xi, r, z_pos)
        a_env_2 = self.pulse_2.envelope_function(xi, r, z_pos)
        return a_env_1 + a_env_2


class GaussianPulse(LaserPulse):
    """Class defining a Gaussian laser pulse."""

    def __init__(self, xi_c, a_0, w_0, tau, z_foc=None, l_0=0.8e-6,
                 cep_phase=0., polarization='linear'):
        """
        Initialize Gaussian pulse.

        Parameters:
        -----------
        xi_c : float
            Initial central position of the pulse along xi in units of m.
        a_0 : float
            The peak normalized vector potential at the focal plane.
        w_0 : float
            Spot size of the laser pulse, in units of m, at the focal plane.
        tau : float
            Longitudinal pulse length (FWHM in intensity) in units of s.
        z_foc : float
            Focal position of the pulse.
        l_0 : float
            Laser wavelength in units of m. By default, a Ti:Sa laser with
            `l_0=0.8e-6` is assumed.
        cep_phase: float
            The Carrier Envelope Phase (CEP) in radian. This is the phase of
            the laser oscillation at the position where the envelope is
            maximum.
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
        self.cep_phase = cep_phase
        self.polarization = polarization

    def envelope_function(self, xi, r, z_pos):
        """
        Complex envelope of a Gaussian beam in the paraxial approximation.
        """
        z = xi - self.xi_c + z_pos
        diff_factor = 1. + 1j * (z - self.z_foc) / self.z_r
        s_z = self.tau * ct.c / (2*np.sqrt(2*np.log(2))) * np.sqrt(2)
        # Phases
        exp_cep = -1j * self.cep_phase
        exp_r = -r**2 / (self.w_0**2 * diff_factor)
        exp_z = -(xi-self.xi_c)**2 / (2*s_z**2)
        # Profile
        gaussian_profile = np.exp(exp_cep + exp_r + exp_z)
        # Amplitude
        avg_amplitude = self.a_0
        if self.polarization == 'linear':
            avg_amplitude /= np.sqrt(2)
        return avg_amplitude / diff_factor * gaussian_profile


class LaguerreGaussPulse(LaserPulse):
    """Class defining a Laguerre-Gauss pulse."""

    def __init__(self, xi_c, p, a_0, w_0, tau, z_foc=None,
                 l_0=0.8e-6, cep_phase=0., polarization='linear'):
        """
        Initialize a Laguerre-Gauss laser profile.

        Due to the cylindrical geometry of Wake-T, only the `0` azimuthal mode
        is supported.

        Parameters
        ----------
        xi_c : float
            Initial central position of the pulse along xi in units of m.
        p : int (positive)
            The order of the Laguerre polynomial. Increasing ``p`` increases
            the number of "rings" in the radial intensity profile of the laser.
        a_0 : float
            The peak normalized vector potential at the focal plane, defined
            so that the total energy of the pulse is the same as that of a
            Gaussian pulse with the same `a_0`, `w_0` `tau`.
            (i.e. The energy of the pulse is independent of `p`.)
        w_0 : float
            Spot size of the laser pulse, in units of m, at the focal plane.
        tau : float
            Longitudinal pulse length (FWHM in intensity) in units of s.
        z_foc : float
            Focal position of the pulse.
        l_0 : float
            Laser wavelength in units of m. By default, a Ti:Sa laser with
            `l_0=0.8e-6` is assumed.
        cep_phase: float
            The Carrier Envelope Phase (CEP) in radian. This is the phase of
            the laser oscillation at the position where the envelope is
            maximum.
        polarization : str
            Polarization of the laser pulse. Accepted values are 'linear'
            (default) or 'circular'.
        """
        # Initialize parent class
        super().__init__(l_0)

        # If no focal plane position is given, use xi_c
        if z_foc is None:
            z_foc = xi_c

        # Store the parameters
        self.p = p
        self.laguerre_pm = genlaguerre(self.p, 0)  # Laguerre polynomial
        self.z_r = np.pi * w_0**2 / l_0
        self.z_foc = z_foc
        self.xi_c = xi_c
        self.a0 = a_0
        self.w0 = w_0
        self.cep_phase = cep_phase
        self.tau = tau
        self.polarization = polarization

    def envelope_function(self, xi, r, z_pos):
        """Complex envelope of a Laguerre-Gauss beam."""
        z = xi - self.xi_c + z_pos
        # Diffraction factor, waist and Gouy phase
        diffract_factor = 1. + 1j * (z - self.z_foc) / self.z_r
        s_z = self.tau * ct.c / (2*np.sqrt(2*np.log(2))) * np.sqrt(2)
        w = self.w0 * abs(diffract_factor)
        psi = np.angle(diffract_factor)
        # Calculate the scaled radius
        scaled_radius_squared = 2 * r**2 / w**2
        # Calculate the argument of the complex exponential
        exp_argument = - 1j*self.cep_phase \
            - r**2 / (self.w0**2 * diffract_factor) \
            - (xi-self.xi_c)**2 / (2*s_z**2) \
            - 1j*(2*self.p)*psi  # *Additional* Gouy phase
        # Get the transverse profile
        profile = (np.exp(exp_argument) / diffract_factor
                   * self.laguerre_pm(scaled_radius_squared))

        a = self.a0 * profile
        if self.polarization == 'linear':
            a /= np.sqrt(2)
        return a


class FlattenedGaussianPulse(LaserPulse):
    """Class defining a flattened Gaussian pulse."""

    def __init__(self, xi_c, a_0, w_0, tau, N=6, z_foc=None, l_0=0.8e-6,
                 cep_phase=0., polarization='linear'):
        """
        Initialize flattened Gaussian pulse.

        The laser pulse is defined such that the transverse intensity
        profile is a flattened Gaussian far from focus, and a distribution
        with rings in the focal plane. (See `Santarsiero et al., J.
        Modern Optics, 1997 <http://doi.org/10.1080/09500349708232927>`_)
        Increasing the parameter ``N`` increases the
        flatness of the transverse profile far from focus,
        and increases the number of rings in the focal plane.

        Parameters
        ----------
        xi_c : float
            Initial central position of the pulse along xi in units of m.
        a_0: float
            The peak normalized vector potential at the focal plane.
        w_0 : float
            Spot size of the laser pulse, in units of m, at the focal plane.
        tau : float
            Longitudinal pulse length (FWHM in intensity) in units of s.
        N : int
            Determines the "flatness" of the transverse profile, far from
            focus.
            Default: ``N=6`` ; somewhat close to an 8th order supergaussian.
        z_foc : float
            Focal position of the pulse.
        l_0 : float
            Laser wavelength in units of m. By default, a Ti:Sa laser with
            `l_0=0.8e-6` is assumed.
        cep_phase: float
            The Carrier Envelope Phase (CEP) in radian. This is the phase of
            the laser oscillation at the position where the envelope is
            maximum.
        polarization : str
            Polarization of the laser pulse. Accepted values are 'linear'
            (default) or 'circular'.
        """
        # Initialize parent class.
        super().__init__(l_0)

        # Store parameters.
        self.xi_c = xi_c

        # Ensure that N is an integer.
        N = int(round(N))
        # Calculate effective waist of the Laguerre-Gauss modes, at focus.
        w_foc = w_0*(N+1)**.5

        # Sum the Laguerre-Gauss modes that constitute this pulse.
        # See equation (2) and (3) in Santarsiero et al.
        for n in range(N+1):
            cep_phase_n = cep_phase + 2*n*np.pi/2
            m_values = np.arange(n, N+1)
            cn = (-1)**n * np.sum(0.5**m_values * binom(m_values, n)) / (N+1)
            pulse = LaguerreGaussPulse(
                xi_c=xi_c, p=n, a_0=cn*a_0, cep_phase=cep_phase_n, w_0=w_foc,
                tau=tau, z_foc=z_foc, l_0=l_0, polarization=polarization)
            if n == 0:
                summed_pulse = pulse
            else:
                summed_pulse += pulse

        # Register the summed_pulse.
        self.summed_pulse = summed_pulse

    def envelope_function(self, xi, r, z_pos):
        """Complex envelope of the flattened Gaussian beam."""
        return self.summed_pulse.envelope_function(xi, r, z_pos)
