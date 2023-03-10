from typing import Optional, Callable

import numpy as np
import scipy.constants as ct

from .solver import calculate_wakefields
from wake_t.fields.rz_wakefield import RZWakefield
from wake_t.physics_models.laser.laser_pulse import LaserPulse


class Quasistatic2DWakefield(RZWakefield):
    """
    This class calculates the plasma wakefields using the gridless
    quasi-static model in r-z geometry originally developed by P. Baxevanis
    and G. Stupakov [1]_.

    The model implemented here includes additional features with respect to the
    original version in [1]_. Among them is the support for laser drivers,
    particle beams (instead of an analytic charge distribution), non-uniform
    and finite plasma density profiles, as well as an Adams-Bashforth pusher
    for the plasma particles (in addition the original Runke-Kutta pusher).

    As a kinetic quasi-static model in r-z geometry, it computes the plasma
    response by evolving a 1D radial plasma column from the front to the back
    of the simulation domain. A special feature of this model is that it does
    not need a grid in order to compute this evolution, and it allows the
    fields ``E`` and ``B`` to be calculated at any radial position in the
    plasma column.

    In the Wake-T implementation, a grid is used only in order to be able
    to easily interpolate the fields to the beam particles. After evolving the
    plasma column, ``E`` and ``B`` are calculated at the locations of the grid
    nodes. Similarly, the charge density ``rho`` and susceptibility ``chi``
    of the plasma are computed by depositing the charge of the plasma
    particles on the same grid. This useful for diagnostics and for evolving
    a laser pulse.

    Parameters
    ----------
    density_function : callable
        Function that returns the density value at the given position z.
        This parameter is given by the ``PlasmaStage`` and does not need
        to be specified by the user.
    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    n_r : int
        Number of grid elements along `r` to calculate the wakefields.
    n_xi : int
        Number of grid elements along `xi` to calculate the wakefields.
    ppc : int, optional
        Number of plasma particles per radial cell. By default ``ppc=2``.
    dz_fields : float, optional
        Determines how often the plasma wakefields should be updated.
        For example, if ``dz_fields=10e-6``, the plasma wakefields are
        only updated every time the simulation window advances by
        10 micron. By default ``dz_fields=xi_max-xi_min``, i.e., the
        length the simulation box.
    r_max_plasma : float, optional
        Maximum radial extension of the plasma column. If ``None``, the
        plasma extends up to the ``r_max`` boundary of the simulation box.
    parabolic_coefficient : float or callable, optional
        The coefficient for the transverse parabolic density profile. The
        radial density distribution is calculated as
        ``n_r = n_p * (1 + parabolic_coefficient * r**2)``, where n_p is
        the local on-axis plasma density. If a ``float`` is provided, the
        same value will be used throwout the stage. Alternatively, a
        function which returns the value of the coefficient at the given
        position ``z`` (e.g. ``def func(z)``) might also be provided.
    p_shape : str, optional
        Particle shape to be used for the beam charge deposition. Possible
        values are ``'linear'`` or ``'cubic'`` (default).
    max_gamma : float, optional
        Plasma particles whose ``gamma`` exceeds ``max_gamma`` are
        considered to violate the quasistatic condition and are put at
        rest (i.e., ``gamma=1.``, ``pr=pz=0.``). By default
        ``max_gamma=10``.
    plasma_pusher : str, optional
        The pusher used to evolve the plasma particles. Possible values
        are ``'rk4'`` (Runge-Kutta 4th order) or ``'ab5'`` (Adams-Bashforth
        5th order).
    laser : LaserPulse, optional
        Laser driver of the plasma stage.
    laser_evolution : bool, optional
        If True (default), the laser pulse is evolved
        using a laser envelope model. If ``False``, the pulse envelope
        stays unchanged throughout the computation.
    laser_envelope_substeps : int, optional
        Number of substeps of the laser envelope solver per ``dz_fields``.
        The time step of the envelope solver is therefore
        ``dz_fields / c / laser_envelope_substeps``.
    laser_envelope_nxi, laser_envelope_nr : int, optional
        If given, the laser envelope will run in a grid of size
        (``laser_envelope_nxi``, ``laser_envelope_nr``) instead
        of (``n_xi``, ``n_r``). This allows the laser to run in a finer (or
        coarser) grid than the plasma wake. It is not necessary to specify
        both parameters. If one of them is not given, the resolution of
        the plasma grid with be used for that direction.
    laser_envelope_use_phase : bool, optional
        Determines whether to take into account the terms related to the
        longitudinal derivative of the complex phase in the envelope
        solver.

    References
    ----------
    .. [1] P. Baxevanis and G. Stupakov, "Novel fast simulation technique
        for axisymmetric plasma wakefield acceleration configurations in
        the blowout regime," Phys. Rev. Accel. Beams 21, 071301 (2018),
        https://link.aps.org/doi/10.1103/PhysRevAccelBeams.21.071301

    """

    def __init__(
        self,
        density_function: Callable[[float], float],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        ppc: Optional[int] = 2,
        dz_fields: Optional[float] = None,
        r_max_plasma: Optional[float] = None,
        parabolic_coefficient: Optional[float] = 0.,
        p_shape: Optional[str] = 'cubic',
        max_gamma: Optional[float] = 10,
        plasma_pusher: Optional[str] = 'rk4',
        laser: Optional[LaserPulse] = None,
        laser_evolution: Optional[bool] = True,
        laser_envelope_substeps: Optional[int] = 1,
        laser_envelope_nxi: Optional[int] = None,
        laser_envelope_nr: Optional[int] = None,
        laser_envelope_use_phase: Optional[bool] = True,
    ) -> None:
        self.ppc = ppc
        self.r_max_plasma = r_max_plasma
        self.parabolic_coefficient = self._get_parabolic_coefficient_fn(
            parabolic_coefficient)
        self.p_shape = p_shape
        self.max_gamma = max_gamma
        self.plasma_pusher = plasma_pusher
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
            model_name='quasistatic_2d'
        )

    def _calculate_wakefield(self, bunches):
        parabolic_coefficient = self.parabolic_coefficient(self.t*ct.c)

        # Get square of laser envelope
        if self.laser is not None:
            a_env_2 = np.abs(self.laser.get_envelope()) ** 2
            # If linearly polarized, divide by 2 so that the ponderomotive
            # force on the plasma particles is correct.
            if self.laser.polarization == 'linear':
                a_env_2 /= 2
        else:
            a_env_2 = np.zeros((self.n_xi, self.n_r))

        # Calculate plasma wakefields
        calculate_wakefields(
            a_env_2, bunches, self.r_max, self.xi_min, self.xi_max,
            self.n_r, self.n_xi, self.ppc, self.n_p,
            r_max_plasma=self.r_max_plasma,
            parabolic_coefficient=parabolic_coefficient,
            p_shape=self.p_shape, max_gamma=self.max_gamma,
            plasma_pusher=self.plasma_pusher,
            fld_arrays=[self.rho, self.chi, self.e_r, self.e_z, self.b_t,
                        self.xi_fld, self.r_fld])

    def _get_parabolic_coefficient_fn(self, parabolic_coefficient):
        """ Get parabolic_coefficient profile function """
        if isinstance(parabolic_coefficient, float):
            def uniform_parabolic_coefficient(z):
                return np.ones_like(z) * parabolic_coefficient
            return uniform_parabolic_coefficient
        elif callable(parabolic_coefficient):
            return parabolic_coefficient
        else:
            raise ValueError(
                'Type {} not supported for parabolic_coefficient.'.format(
                    type(parabolic_coefficient)))
