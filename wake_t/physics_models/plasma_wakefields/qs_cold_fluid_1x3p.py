from typing import Optional, Callable

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.particles.deposition import deposit_3d_distribution
from wake_t.particles.interpolation import (
    gather_field_cyl_linear)
from wake_t.fields.rz_wakefield import RZWakefield
from wake_t.physics_models.laser.laser_pulse import LaserPulse


class NonLinearColdFluidWakefield(RZWakefield):
    """
    This class computes the plasma wakefields using a nonlinear cold fluid
    theory in one spatial dimension with a three-component fluid momentum, as
    described in [1]_. This implies that only longitudinal plasma waves are
    modeled, i.e., it assumes infinitely broad laser pulses and particle beams.

    This 1D model is in Wake-T extended to 2D in r-z geometry by computing the
    1D plasma response at each radial slice.

    Given the assumptions of the model, it is only accurate for broad drivers
    where the radial plasma waves can be neglected. For a laser driver, it is
    typically accurate up to :math:`a_0 \\lesssim 1`. For electron beams it has
    not been yet fully tested, but given the typically narrow width of the
    beams, only very low charges can be accurately modeled.

    For a much more general model, see :class:`Quasistatic2DWakefield`.

    Parameters
    ----------
    density_function : callable
        Function that returns the density value at the given position z.
        This parameter is given by the `PlasmaStage` and does not need
        to be specified by the user.
    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated. Required only if mode='cold_fluid_1d'.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    n_r : int
        Number of grid elements along r to calculate the wakefields.
    n_xi : int
        Number of grid elements along xi to calculate the wakefields.
    dz_fields : float, optional
        Determines how often the plasma wakefields should be updated. If
        dz_fields=0 (default value), the wakefields are calculated at every
        step of the Runge-Kutta solver for the beam particle evolution
        (most expensive option). If specified, the wakefields are only
        updated in steps determined by dz_fields. For example, if
        dz_fields=10e-6, the plasma wakefields are only updated every time
        the simulation window advances by 10 micron. By default, if not
        specified, the value of `dz_fields` will be `xi_max-xi_min`, i.e.,
        the length the simulation box.
    beam_wakefields : bool, optional
        Whether to take into account beam-driven wakefields (False by
        default). This should be set to True for any beam-driven case or
        in order to take into account the beam-loading of the witness in
        a laser-driven case.
    p_shape : str, optional
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'.
    laser : LaserPulse, optional
        Laser driver of the plasma stage.
    laser_evolution : bool, optional
        If True (default), the laser pulse is evolved
        using a laser envelope model. If False, the pulse envelope stays
        unchanged throughout the computation.
    laser_envelope_substeps : int, optional
        Number of substeps of the laser envelope solver per `dz_fields`.
        The time step of the envelope solver is therefore
        `dz_fields / c / laser_envelope_substeps`.
    laser_envelope_nxi, laser_envelope_nr : int, optional
        If given, the laser envelope will run in a grid of size
        (`laser_envelope_nxi`, `laser_envelope_nr`) instead
        of (`n_xi`, `n_r`). This allows the laser to run in a finer (or
        coarser) grid than the plasma wake. It is not necessary to specify
        both parameters. If one of them is not given, the resolution of
        the plasma grid with be used for that direction.
    laser_envelope_use_phase : bool, optional
        Determines whether to take into account the terms related to the
        longitudinal derivative of the complex phase in the envelope
        solver.

    See Also
    --------
    Quasistatic2DWakefield

    References
    ----------
    .. [1] T. Mehrling, "Theoretical and numerical studies on the transport of
       transverse beam quality in plasma-based accelerators," PhD thesis
       (2014), http://dx.doi.org/10.3204/DESY-THESIS-2014-040

    """
    def __init__(
        self,
        density_function: Callable[[float], float],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        dz_fields: Optional[float] = None,
        beam_wakefields: Optional[bool] = False,
        p_shape: Optional[str] = 'linear',
        laser: Optional[LaserPulse] = None,
        laser_evolution: Optional[bool] = True,
        laser_envelope_substeps: Optional[int] = 1,
        laser_envelope_nxi: Optional[int] = None,
        laser_envelope_nr: Optional[int] = None,
        laser_envelope_use_phase: Optional[bool] = True,
    ) -> None:
        self.beam_wakefields = beam_wakefields
        self.p_shape = p_shape
        super().__init__(
            density_function=density_function,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            laser=laser,
            laser_evolution=laser_evolution,
            laser_envelope_substeps=laser_envelope_substeps,
            laser_envelope_nxi=laser_envelope_nxi,
            laser_envelope_nr=laser_envelope_nr,
            laser_envelope_use_phase=laser_envelope_use_phase,
            model_name='cold_fluid_1d'
        )

    def __wakefield_ode_system(self, u_1, u_2, laser_a0, n_beam):
        if self.beam_wakefields:
            return np.array(
                [u_2, (1+laser_a0**2)/(2*(1+u_1)**2) - n_beam - 1/2])
        else:
            return np.array([u_2, (1+laser_a0**2)/(2*(1+u_1)**2) - 1/2])

    def _calculate_wakefield(self, bunches):
        # Get laser envelope
        if self.laser is not None:
            a_env = np.abs(self.laser.get_envelope())
            # If linearly polarized, divide by sqrt(2) so that the
            # ponderomotive force on the plasma particles is correct.
            if self.laser.polarization == 'linear':
                a_env /= np.sqrt(2)
        else:
            a_env = np.zeros((self.n_xi, self.n_r))

        # Calculate and allocate laser quantities, including guard cells.
        a_rz = np.zeros((self.n_xi+4, self.n_r+4))
        a_rz[2:-2, 2:-2] = a_env

        s_d = ge.plasma_skin_depth(self.n_p*1e-6)
        dz = self.dxi / s_d
        dr = self.dr / s_d
        r_fld = self.r_fld / s_d

        # Get charge distribution and remove guard cells.
        beam_hist = np.zeros((self.n_xi+4, self.n_r+4))
        for bunch in bunches:
            x = bunch.x
            y = bunch.y
            xi = bunch.xi
            w = bunch.w * (bunch.q_species / ct.e)
            deposit_3d_distribution(
                xi/s_d, x/s_d, y/s_d, w, self.xi_min/s_d, r_fld[0],
                self.n_xi, self.n_r, dz, dr, beam_hist, p_shape=self.p_shape,
                use_ruyten=True)
        beam_hist = beam_hist[2:-2, 2:-2]

        n = np.arange(self.n_r)
        disc_area = np.pi * dr ** 2 * (1 + 2 * n)
        beam_hist *= 1 / (disc_area * dz * self.n_p) / s_d ** 3
        n_iter = self.n_xi - 1
        u_1 = np.zeros((n_iter + 1, len(r_fld)))
        u_2 = np.zeros((n_iter + 1, len(r_fld)))
        z_fld = self.xi_fld / s_d

        for i in np.arange(n_iter):
            z_i = z_fld[-1 - i]
            # get laser a0 at z, z+dz/2 and z+dz
            if self.laser is not None:
                x = r_fld
                y = np.zeros_like(r_fld)
                z = np.full_like(r_fld, z_i)
                a0_0 = gather_field_cyl_linear(
                    a_rz, self.xi_min/s_d, self.xi_max/s_d, r_fld[0],
                    r_fld[-1], dz, dr, x, y, z)
                a0_1 = gather_field_cyl_linear(
                    a_rz, self.xi_min/s_d, self.xi_max/s_d, r_fld[0],
                    r_fld[-1], dz, dr, x, y, z - dz/2)
                a0_2 = gather_field_cyl_linear(
                    a_rz, self.xi_min/s_d, self.xi_max/s_d, r_fld[0],
                    r_fld[-1], dz, dr, x, y, z - dz)
            else:
                a0_0 = np.zeros(r_fld.shape[0])
                a0_1 = np.zeros(r_fld.shape[0])
                a0_2 = np.zeros(r_fld.shape[0])
            # perform runge-kutta
            A = dz*self.__wakefield_ode_system(
                u_1[-1-i], u_2[-1-i], a0_0, beam_hist[-i-1])
            B = dz*self.__wakefield_ode_system(
                u_1[-1-i] + A[0]/2, u_2[-1-i] + A[1]/2, a0_1, beam_hist[-i-1])
            C = dz*self.__wakefield_ode_system(
                u_1[-1-i] + B[0]/2, u_2[-1-i] + B[1]/2, a0_1, beam_hist[-i-1])
            D = dz*self.__wakefield_ode_system(
                u_1[-1-i] + C[0], u_2[-1-i] + C[1], a0_2, beam_hist[-i-1])
            u_1[-2-i] = u_1[-1-i] + 1/6*(A[0] + 2*B[0] + 2*C[0] + D[0])
            u_2[-2-i] = u_2[-1-i] + 1/6*(A[1] + 2*B[1] + 2*C[1] + D[1])
        E_z = -np.gradient(u_1, dz, axis=0, edge_order=2)
        W_r = -np.gradient(u_1, dr, axis=1, edge_order=2)
        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(
            self.n_p*1e-6)

        # Calculate rho and chi.
        gamma_fl = (1 + a_env**2 + (1 + u_1)**2) / (2 * (1 + u_1))
        rho_fl = gamma_fl / (1 + u_1)
        self.rho[2:-2, 2:-2] = rho_fl
        self.chi[2:-2, 2:-2] = rho_fl / gamma_fl

        # Calculate B_theta and E_r.
        u_z = (1 + a_env**2 - (1 + u_1)**2) / (2 * (1 + u_1))
        dE_z = np.gradient(E_z, dz, axis=0, edge_order=2)
        v_z = u_z / gamma_fl
        nv_z = rho_fl * v_z
        integrand = (dE_z - nv_z + beam_hist) * r_fld
        subs = integrand / 2
        B_theta = (np.cumsum(integrand, axis=1) - subs) * dr / np.abs(r_fld)
        E_r = W_r + B_theta

        # Store fields.
        self.b_t[2:-2, 2:-2] = B_theta * E_0 / ct.c
        self.e_r[2:-2, 2:-2] = E_r * E_0
        self.e_z[2:-2, 2:-2] = E_z * E_0
