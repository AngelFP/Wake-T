"""
This module contains the class definitions for different laser pulses.

The implementation of the Laguerre-Gauss and Flattened Gaussian
profiles, as well as the SummedPulse approach is based on FBPIC
(https://github.com/fbpic/fbpic).

Authors: Angel Ferran Pousa, Remi Lehe, Manuel Kirchen, Pierre Pelletier.

"""
from typing import Optional, Union, Iterable

import numpy as np
import scipy.constants as ct
from scipy.special import genlaguerre, binom
from scipy.ndimage import gaussian_filter
try:
    from lasy.profiles.from_openpmd_profile import FromOpenPMDProfile
    from lasy.laser import Laser
    from lasy.utils.laser_utils import field_to_vector_potential
    lasy_installed = True
except ImportError:
    lasy_installed = False


from .envelope_solver import evolve_envelope
from .envelope_solver_non_centered import evolve_envelope_non_centered
from wake_t.fields.interpolation import interpolate_rz_field


class LaserPulse():
    """Base class for all Laser pulses.

    Parameters
    ----------
    l_0 : float
        Laser wavelength in units of m.
    polarization : str, optional
        Polarization of the laser pulse. Accepted values are 'linear'
        (default) or 'circular'.

    """

    def __init__(
        self,
        l_0: float,
        polarization: Optional[str] = 'linear'
    ) -> None:
        self.l_0 = l_0
        self.polarization = polarization
        self.a_env = None
        self.solver_params = None
        self.n_steps = 0

    def __add__(self, pulse_2):
        """Overload the add operator to allow summing of laser pulses."""
        return SummedPulse(self, pulse_2)

    def set_envelope_solver_params(
        self,
        xi_min: float,
        xi_max: float,
        r_max: float,
        nz: int,
        nr: int,
        dt: float,
        nt: Optional[int] = 1,
        subgrid_nz: Optional[int] = None,
        subgrid_nr: Optional[int] = None,
        use_phase: Optional[bool] = True
    ) -> None:
        """
        Set the parameters for the laser envelope solver.

        Parameters
        ----------
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
        subgrid_nz, subgrid_nr : int, optional
            If specified, run laser envelope solver in a subgrid of resolution
            (subgrid_nz, subgrid_nr), which is different than the size of the
            plasma grid (nz, nr). Linear interpolation is used to transform
            the plasma susceptibility into the laser envelope grid, and to
            transform the laser envelope into the plasma grid.
        use_phase : bool
            Determines whether to take into account the terms related to the
            longitudinal derivative of the complex phase in the envelope
            solver.

        """
        if nt < 1:
            raise ValueError(
                'Number of laser envelope substeps cannot be smaller than 1.')

        # Determine whether to run laser envelope in a subgrid.
        self.use_subgrid = subgrid_nz is not None or subgrid_nr is not None
        if self.use_subgrid:
            subgrid_nz = nz if subgrid_nz is None else subgrid_nz
            subgrid_nr = nr if subgrid_nr is None else subgrid_nr
            self._create_laser_subgrid(
                nz, nr, subgrid_nz, subgrid_nr, xi_max, xi_min, r_max)

        solver_params = {
            'zmin': xi_min,
            'zmax': xi_max,
            'rmax': r_max,
            'nz': subgrid_nz if self.use_subgrid else nz,
            'nr': subgrid_nr if self.use_subgrid else nr,
            'nt': nt,
            'dt': dt / nt,
            'use_phase': use_phase
        }
        # Check if laser pulse has already been initialized.
        if self.n_steps > 0:
            # Check that grid parameters have not changed.
            if (
                solver_params['zmin'] != self.solver_params['zmin'] and
                solver_params['zmax'] != self.solver_params['zmax'] and
                solver_params['rmax'] != self.solver_params['rmax'] and
                solver_params['nz'] != self.solver_params['nz'] and
                solver_params['nr'] != self.solver_params['nr']
            ):
                raise ValueError(
                    'Laser envelope grid parameters cannot be changed once '
                    'envelope has been initialized.'
                )
            # If time step has changed, set `n_steps` to 0 so that the
            # non-centered envelope solver is used for the next step.
            if solver_params['dt'] != self.solver_params['dt']:
                self.n_steps = 0
        self.solver_params = solver_params

    def initialize_envelope(self) -> None:
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
            dr = r_max / nr
            z = np.linspace(z_min, z_max, nz)
            r = np.linspace(dr/2, r_max-dr/2, nr)
            ZZ, RR = np.meshgrid(z, r, indexing='ij')
            self._a_env_old = np.zeros((nz + 2, nr), dtype=np.complex128)
            self._a_env = np.zeros((nz + 2, nr), dtype=np.complex128)
            self._a_env[0:-2] = self.envelope_function(ZZ, RR, 0.)
            self._update_output_envelope()

    def get_envelope(self) -> np.ndarray:
        """Get the current laser envelope array."""
        return self.a_env

    def evolve(
        self,
        chi: np.ndarray,
        n_p: float
    ) -> None:
        """
        Evolve laser envelope to next time step.

        Parameters
        ----------
        chi : ndarray
            A (nz x nr) array containing the plasma susceptibility.
        n_p : float
            Plasma density in SI units.
        """
        k_0 = 2*np.pi / self.l_0
        k_p = np.sqrt(ct.e**2 * n_p / (ct.m_e*ct.epsilon_0)) / ct.c

        # If needed, interpolate chi to subgrid.
        if self.use_subgrid:
            chi = self._interpolate_chi_to_subgrid(chi)

        # Compute evolution.
        if self.n_steps == 0:
            evolve_envelope_non_centered(
                self._a_env, self._a_env_old, chi, k_0, k_p,
                **self.solver_params)
        else:
            evolve_envelope(
                self._a_env, self._a_env_old, chi, k_0, k_p,
                **self.solver_params)

        # Update arrays and step count.
        self._update_output_envelope()
        self.n_steps += 1

    def get_group_velocity(
        self,
        n_p: float
    ) -> float:
        """
        Get group velocity of the laser pulse for a given plasma density.

        Parameters
        ----------
        n_p : float
            Plasma density in units of m^{-3}.

        Returns
        -------
        A float containing the group velocity.
        """
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        k = 2*np.pi/self.l_0
        v_g = k*ct.c**2/np.sqrt(w_p**2+k**2*ct.c**2)/ct.c
        return v_g

    def envelope_function(
        self,
        xi: np.ndarray,
        r: np.ndarray,
        z_pos: float
    ) -> np.ndarray:
        """Return the complex envelope of the laser pulse."""
        return self._envelope_function(xi, r, z_pos)

    def _envelope_function(self, xi, r, z_pos):
        return np.zeros_like(r)

    def _create_laser_subgrid(self, nz, nr, subgrid_nz, subgrid_nr, xi_max,
                              xi_min, r_max):
        """
        Create the parameters needed to run the laser envelope in a subgrid.
        """
        # Grid spacing and minimum radius of the main grid.
        dr = r_max / nr
        dz = (xi_max - xi_min) / (nz - 1)
        grid_r_min = dr / 2
        grid_r_max = r_max - dr / 2

        # Grid spacing and minimum radius of the subgrid.
        subgrid_dr = r_max / subgrid_nr
        subgrid_dz = (xi_max - xi_min) / (subgrid_nz - 1)
        subgrid_r_min = subgrid_dr / 2
        subgrid_r_max = r_max - subgrid_dr / 2

        # Store parameters in dictionary.
        self.subgrid_params = {
            'grid': {
                'nz': nz,
                'nr': nr,
                'z_min': xi_min,
                'r_min': grid_r_min,
                'dr': dr,
                'dz': dz,
                'z': np.linspace(xi_min, xi_max, nz),
                'r': np.linspace(grid_r_min, grid_r_max, nr),
            },
            'subgrid': {
                'nz': subgrid_nz,
                'nr': subgrid_nr,
                'z_min': xi_min,
                'r_min': subgrid_r_min,
                'dr': subgrid_dr,
                'dz': subgrid_dz,
                'z': np.linspace(xi_min, xi_max, subgrid_nz),
                'r': np.linspace(subgrid_r_min, subgrid_r_max, subgrid_nr),
                'chi': np.zeros((subgrid_nz, subgrid_nr))
            }
        }

    def _update_output_envelope(self):
        """ Update the publicly-accessible laser envelope array. """
        # If running on a subgrid, interpolate envelope array to main grid.
        if self.use_subgrid:
            if self.a_env is None:
                nz = self.subgrid_params['grid']['nz']
                nr = self.subgrid_params['grid']['nr']
                self.a_env = np.zeros((nz, nr), dtype=np.complex128)
            z_min = self.subgrid_params['subgrid']['z_min']
            r_min = self.subgrid_params['subgrid']['r_min']
            dz = self.subgrid_params['subgrid']['dz']
            dr = self.subgrid_params['subgrid']['dr']
            z_f = self.subgrid_params['grid']['z']
            r_f = self.subgrid_params['grid']['r']
            interpolate_rz_field(
                self._a_env[0:-2], z_min, r_min, dz, dr, z_f, r_f, self.a_env)
        # Otherwise, simply remove guard cells.
        else:
            if self.a_env is None:
                nz = self.solver_params['nz']
                nr = self.solver_params['nr']
                self.a_env = np.zeros((nz, nr), dtype=np.complex128)
            self.a_env[:] = self._a_env[0: -2]

    def _interpolate_chi_to_subgrid(self, chi):
        """ Interpolate the plasma susceptibility to the envelope subgrid. """
        z_min = self.subgrid_params['grid']['z_min']
        r_min = self.subgrid_params['grid']['r_min']
        dz = self.subgrid_params['grid']['dz']
        dr = self.subgrid_params['grid']['dr']
        subgrid_chi = self.subgrid_params['subgrid']['chi']
        z_f = self.subgrid_params['subgrid']['z']
        r_f = self.subgrid_params['subgrid']['r']
        interpolate_rz_field(
            chi, z_min, r_min, dz, dr, z_f, r_f, subgrid_chi)
        return subgrid_chi


class SummedPulse(LaserPulse):
    """Class defining a laser pulse made up of the addition of two pulses.

    Parameters
    ----------
    pulse_1 : LaserPulse
        A LaserPulse instance.
    pulse_2 : LaserPulse
        Another LaserPulse instance to be summed to `pulse_1`.
    """

    def __init__(
        self,
        pulse_1: LaserPulse,
        pulse_2: LaserPulse
    ) -> None:
        if pulse_1.l_0 != pulse_2.l_0:
            raise ValueError(
                'Only laser pulses with the same central wavelength'
                ' can currently be summed.')
        super().__init__(pulse_1.l_0)
        self.pulse_1 = pulse_1
        self.pulse_2 = pulse_2

    def _envelope_function(self, xi, r, z_pos):
        """Return the summed envelope of the two pulses."""
        a_env_1 = self.pulse_1.envelope_function(xi, r, z_pos)
        a_env_2 = self.pulse_2.envelope_function(xi, r, z_pos)
        return a_env_1 + a_env_2


class GaussianPulse(LaserPulse):
    """Class defining a Gaussian laser pulse.

    Parameters
    ----------
    xi_c : float
        Initial central position of the pulse along xi in units of m.
    a_0 : float
        The peak normalized vector potential at the focal plane.
    w_0 : float
        Spot size of the laser pulse, in units of m, at the focal plane.
    tau : float
        Longitudinal pulse length (FWHM in intensity) in units of s.
    z_foc : float, optional
        Focal position of the pulse.
    l_0 : float, optional
        Laser wavelength in units of m. By default, a Ti:Sa laser with
        `l_0=0.8e-6` is assumed.
    cep_phase : float, optional
        The Carrier Envelope Phase (CEP) in radian. This is the phase of
        the laser oscillation at the position where the envelope is
        maximum.
    polarization : str, optional
        Polarization of the laser pulse. Accepted values are 'linear'
        (default) or 'circular'.
    """

    def __init__(
        self,
        xi_c: float,
        a_0: float,
        w_0: float,
        tau: float,
        z_foc: Optional[float] = 0.,
        l_0: Optional[float] = 0.8e-6,
        cep_phase: Optional[float] = 0.,
        polarization: Optional[str] = 'linear'
    ) -> None:
        super().__init__(l_0=l_0, polarization=polarization)
        self.xi_c = xi_c
        self.a_0 = a_0
        self.tau = tau
        self.w_0 = w_0
        self.z_foc = z_foc
        self.z_r = np.pi * w_0**2 / l_0
        self.cep_phase = cep_phase

    def _envelope_function(self, xi, r, z_pos):
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
        return avg_amplitude / diff_factor * gaussian_profile


class LaguerreGaussPulse(LaserPulse):
    """Class defining a Laguerre-Gauss pulse.

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
    z_foc : float, optional
        Focal position of the pulse.
    l_0 : float, optional
        Laser wavelength in units of m. By default, a Ti:Sa laser with
        `l_0=0.8e-6` is assumed.
    cep_phase : float, optional
        The Carrier Envelope Phase (CEP) in radian. This is the phase of
        the laser oscillation at the position where the envelope is
        maximum.
    polarization : str, optional
        Polarization of the laser pulse. Accepted values are 'linear'
        (default) or 'circular'.

    """

    def __init__(
        self,
        xi_c: float,
        p: int,
        a_0: float,
        w_0: float,
        tau: float,
        z_foc: Optional[float] = 0.,
        l_0: Optional[float] = 0.8e-6,
        cep_phase: Optional[float] = 0.,
        polarization: Optional[str] = 'linear'
    ) -> None:
        # Initialize parent class
        super().__init__(l_0=l_0, polarization=polarization)

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

    def _envelope_function(self, xi, r, z_pos):
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
        return a


class FlattenedGaussianPulse(LaserPulse):
    """Class defining a flattened Gaussian pulse.

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
    N : int, optional
        Determines the "flatness" of the transverse profile, far from
        focus.
        Default: ``N=6`` ; somewhat close to an 8th order supergaussian.
    z_foc : float, optional
        Focal position of the pulse.
    l_0 : float, optional
        Laser wavelength in units of m. By default, a Ti:Sa laser with
        `l_0=0.8e-6` is assumed.
    cep_phase : float, optional
        The Carrier Envelope Phase (CEP) in radian. This is the phase of
        the laser oscillation at the position where the envelope is
        maximum.
    polarization : str, optional
        Polarization of the laser pulse. Accepted values are 'linear'
        (default) or 'circular'.

    """

    def __init__(
        self,
        xi_c: float,
        a_0: float,
        w_0: float,
        tau: float,
        N: Optional[int] = 6,
        z_foc: Optional[float] = 0.,
        l_0: Optional[float] = 0.8e-6,
        cep_phase: Optional[float] = 0.,
        polarization: Optional[str] = 'linear'
    ) -> None:
        # Initialize parent class.
        super().__init__(l_0=l_0, polarization=polarization)

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

    def _envelope_function(self, xi, r, z_pos):
        """Complex envelope of the flattened Gaussian beam."""
        return self.summed_pulse.envelope_function(xi, r, z_pos)


class OpenPMDPulse(LaserPulse):
    """Read a laser pulse from an openPMD file.

    This class requires `lasy <https://lasydoc.readthedocs.io>`_ to be
    installed.

    Parameters
    ----------
    path : str
        Path to the openPMD file or folder containing the laser data.
    iteration : int
        Iteration at which to read the laser pulse.
    field : str, optional
        Name of the field containing the laser pulse. By default `'E'`.
    coord : string, optional
        Coordinate of the field containing the laser pulse.. By default `'x'`.
    prefix : string, optional
        Prefix of the openPMD file from which the envelope is read.
        Only used when envelope=True.
        The provided iteration is read from <path>/<prefix>_%T.h5.
    theta : float or None, optional
        Only used if the openPMD input is in thetaMode geometry.
        The angle of the plane of observation, with respect to the x axis.
        By default `0`.
    smooth_edges : bool, optional
        Whether to smooth the edges of the laser profile along `r` using a
        super-Gaussian function of power 8. This is useful when the laser
        profile in the openPMD file has a sharp edge (e.g., due to the finite
        width of the domain in `r`). Smoothing this edge can help reduce noise
        in the simulation. By default `False`.
    apply_gaussian_filter : bool, optional
        Whether to apply a Gaussian filter to the laser profile. This is
        useful, for example, when the openpmd laser pulse comes from a
        noisy simulation. In this case, applying the filter can improve the
        stability of the simulation. By default `False`.
    gaussian_filter_sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes. By default `(5, 0)`, which only smooths along the radial
        direction.

    Notes
    -----
    When the grid of the openPMD laser pulse and the grid of the Wake-T
    simulation have a different extent or resolution, the original
    pulse will be linearly interpolated into the Wake-T grid. This is can
    sometimes lead to numerical issues that manifest as a radial oscillation
    of the laser pulse (and, as a result, of the plasma wake). To mitigate
    this, it is important to avoid interpolating along the radial direction.
    This can be achieved if the ratios between the original and the Wake-T
    resolution and extent are an integer.
    """
    def __init__(
        self,
        path: str,
        iteration: int,
        field: Optional[str] = 'E',
        coord: Optional[str] = 'x',
        prefix: Optional[str] = None,
        theta: Optional[float] = 0.,
        smooth_edges: Optional[bool] = False,
        apply_gaussian_filter: Optional[bool] = False,
        gaussian_filter_sigma: Optional[Union[int, float, Iterable]] = (5, 0)
    ) -> None:
        assert lasy_installed, (
            "Using an `OpenPMDPulse` requires `lasy` to be installed. "
            "You can do so with `pip install lasy`."
        )
        self.lasy_profile = FromOpenPMDProfile(
            path=path,
            iteration=iteration,
            pol=(1, 0),  # dummy value, currently not needed
            field=field,
            coord=coord,
            prefix=prefix,
            theta=theta
        )
        super().__init__(self.lasy_profile.lambda0, 'linear')
        self._smooth_edges = smooth_edges
        self._apply_gaussian_filter = apply_gaussian_filter
        self._gaussian_filter_sigma = gaussian_filter_sigma

    def _envelope_function(self, xi, r, z_pos):
        # Create laser
        t = -xi / ct.c
        t_min = np.min(t)
        t_max = np.max(t)
        t_max -= t_min
        t_min = 0
        r_min = np.min(r)
        r_max = np.max(r)
        laser = Laser(
            dim='rt',
            lo=(r_min, t_min),
            hi=(r_max, t_max),
            npoints=(xi.shape[1], xi.shape[0]),
            profile=self.lasy_profile,
            n_azimuthal_modes=1
        )
        a_env = field_to_vector_potential(laser.grid, laser.profile.omega0)

        # Get 2D slice and change to Wake-T ordering.
        a_env = a_env[0].T[::-1]

        # Apply Gaussian filter.
        if self._apply_gaussian_filter:
            a_env = gaussian_filter(a_env, self._gaussian_filter_sigma)

        # Smooth radial edges of profile.
        if self._smooth_edges:
            r_smooth = min(np.max(self.lasy_profile.axes['r']), np.max(r))
            a_env *= np.exp(- 2 * (r / (r_smooth * 0.85)) ** 8)

        return a_env
