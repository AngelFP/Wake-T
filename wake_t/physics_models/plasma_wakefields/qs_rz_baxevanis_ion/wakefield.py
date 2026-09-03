from typing import Optional, Callable, List, Union, Dict
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .solver import calculate_wakefields, calculate_wakefields_salame
from .b_theta_bunch import calculate_bunch_source, deposit_bunch_charge
from .adaptive_grid import AdaptiveGrid
from .utils import calculate_laser_a2
from wake_t.fields.rz_wakefield import RZWakefield
from wake_t.physics_models.laser.laser_pulse import LaserPulse
from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.particles.interpolation import gather_main_fields_cyl_linear
from wake_t.particles.gather_z0r1 import gather_z0r1
from wake_t.utilities.other import ProfStart, ProfStop


class Quasistatic2DWakefieldIon(RZWakefield):
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
        Function that returns the density value at the given position (z, r).
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
    ppc : array_like, optional
        Number of plasma particles per radial cell. It can be a single number
        (e.g., ``ppc=2``) if the plasma should have the same number of
        particles per cell everywhere. Alternatively, a different number
        of particles per cell at different radial locations can also be
        specified. This can be useful, for example, when using adaptive grids
        with very narrow beams that might require more plasma particles close
        to the axis. To achieve this, an array-like structure should be given
        where each item contains two values: the number of particles per cell
        and the radius up to which this number should be used. For example
        to have 8 ppc up to a radius of 100µm and 2 ppc for higher radii up to
        500µm ``ppc=[[100e-6, 8], [500e-6, 2]]``. When using this step option
        for ``ppc`` the ``r_max_plasma`` argument is ignored. By default
        ``ppc=2``.
    dz_fields : float, optional
        Determines how often the plasma wakefields should be updated.
        For example, if ``dz_fields=10e-6``, the plasma wakefields are
        only updated every time the simulation window advances by
        10 micron. By default ``dz_fields=xi_max-xi_min``, i.e., the
        length the simulation box.
    r_max_plasma : float, optional
        Maximum radial extension of the plasma column. If ``None``, the
        plasma extends up to the ``r_max`` boundary of the simulation box.
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
        are ``'ab2'`` (Adams-Bashforth 2nd order).
    ion_motion : bool, optional
        Whether to allow the plasma ions to move. By default, False.
    ion_mass : float, optional
        Mass of the plasma ions. By default, the mass of a proton.
    free_electrons_per_ion : int, optional
        Number of free electrons per ion. The ion charge is adjusted
        accordingly to maintain a quasi-neutral plasma (i.e.,
        ion charge = e * free_electrons_per_ion). By default, 1.
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
    field_diags : list, optional
        List of fields to save to openpmd diagnostics. By default ['rho', 'E',
        'B', 'a_mod', 'a_phase']. Can also be 'all'.
        Each entry can also be a dict to give more precise control of the
        outputted data. The dict can have the following keys:
        {
            "field" : list of field names to output
            "r_min" , "r_max" , "xi_min" , "xi_max" :
                Cut the diagnostic box in r and xi.
            "r_stride" , "xi_stride" :
                Add a stride in units of dr and dxi to r and xi.
            "do_transpose" :
                Wheater to transpose data for the output (default True).
            "diag_name" :
                Name to append to the field name if there are multiple
                diagnostics with the same field.
        }
    particle_diags : list, optional
        List of particle quantities to save to openpmd diagnostics. By default
        [].
    use_adaptive_grids : bool, optional
        Whether to use adaptive grids for each particle bunch, instead of the
        general (n_xi x n_r) grid.
    adaptive_grid_nr : int or list of int, optional
        Radial resolution of the adaptive grids. If only one value is given,
        the same resolution will be used for the adaptive grids of all bunches.
        Otherwise, a list of values can be given (one per bunch and in the same
        order as the list of bunches given to the `track` method). If the
        value is `None`, no adaptive grid will be used for the corresponding
        bunch, which will instead use the base grid.
    adaptive_grid_r_max : float or list of float, optional
        Specify a fixed radial extent for the adaptive grids. If not given,
        the radial extent of the grids is continuously adapted with the
        transverse size of the bunches. If only one value is given,
        the same extent will be used for all adaptive grids.
        Otherwise, a list of values can be given (one per bunch and in the same
        order as the list of bunches given to the `track` method). The
        individual values can be `float` or `None` (in which case, no fixed
        radial extent is used for the corresponding grid).
    adaptive_grid_r_lim : float or list of float, optional
        Specify a limit to the radial extent of the adaptive grids. If not
        given, the radial extent of the grids is continuously adapted to fit
        the whole transverse size of the bunches. If only one value is given,
        the same limit will be used for all adaptive grids.
        Otherwise, a list of values can be given (one per bunch and in the same
        order as the list of bunches given to the `track` method). The
        individual values can be `float` or `None` (in which case, no radial
        limit is used for the corresponding grid). Bunch particles that escape
        the grid transversely with deposit to and gather from the base grid
        (if they haven't escaped from it too).
    adaptive_grid_diags : list, optional
        List of fields from the adaptive grids to save to openpmd diagnostics.
        By default ['E', 'B'].

    References
    ----------
    .. [1] P. Baxevanis and G. Stupakov, "Novel fast simulation technique
        for axisymmetric plasma wakefield acceleration configurations in
        the blowout regime," Phys. Rev. Accel. Beams 21, 071301 (2018),
        https://link.aps.org/doi/10.1103/PhysRevAccelBeams.21.071301

    """

    def __init__(
        self,
        density_function: Callable[[float, float], float],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        ppc: Optional[ArrayLike] = 2,
        dz_fields: Optional[float] = None,
        r_max_plasma: Optional[float] = None,
        parabolic_coefficient: Optional[float] = None,
        p_shape: Optional[str] = "cubic",
        max_gamma: Optional[float] = 10,
        plasma_pusher: Optional[str] = "ab2",
        ion_motion: Optional[bool] = False,
        ion_mass: Optional[float] = ct.m_p,
        free_electrons_per_ion: Optional[int] = 1,
        laser: Optional[LaserPulse] = None,
        laser_evolution: Optional[bool] = True,
        laser_envelope_substeps: Optional[int] = 1,
        laser_envelope_nxi: Optional[int] = None,
        laser_envelope_nr: Optional[int] = None,
        laser_envelope_use_phase: Optional[bool] = True,
        field_diags: Optional[List[str]] = None,
        particle_diags: Optional[List[str]] = [],
        use_adaptive_grids: Optional[bool] = False,
        adaptive_grid_nr: Optional[Union[int, List[int]]] = 16,
        adaptive_grid_r_max: Optional[Union[float, List[float]]] = None,
        adaptive_grid_r_lim: Optional[Union[float, List[float]]] = None,
        adaptive_grid_diags: Optional[List[str]] = ["E", "B"],
    ) -> None:
        # Checks for backward compatibility.
        if plasma_pusher not in ["ab2"]:
            if plasma_pusher in ["rk4", "ab5"]:
                plasma_pusher = "ab2"
                warn(
                    "Plasma pusher {plasma_pusher} has been deprecated. "
                    "Using 'ab2' pusher instead.",
                    DeprecationWarning,
                )
            else:
                raise ValueError(
                    "Plasma pusher {plasma_pusher} not recognized. "
                    "Possible values are ['ab2']."
                )
        if parabolic_coefficient is not None:
            warn(
                "The 'parabolic_coefficient' parameter has been deprecated. "
                "It will still work in order to maintain backward "
                "compatibility, but the "
                "new recommended approach is to provide a density function "
                "that takes `z` and `r` as input parameters.",
                DeprecationWarning,
            )
            parabolic_coefficient = self._get_parabolic_coefficient_fn(
                parabolic_coefficient
            )
        if r_max_plasma is None:
            r_max_plasma = r_max - r_max / n_r / 2
        self.r_max_plasma = r_max_plasma
        self.ppc = np.array(ppc)
        self.p_shape = p_shape
        self.max_gamma = max_gamma
        self.plasma_pusher = plasma_pusher
        self.ion_motion = ion_motion
        self.ion_mass = ion_mass
        self.free_electrons_per_ion = free_electrons_per_ion
        self.use_adaptive_grids = use_adaptive_grids
        self.adaptive_grid_nr = adaptive_grid_nr
        self.adaptive_grid_r_max = adaptive_grid_r_max
        self.adaptive_grid_r_lim = adaptive_grid_r_lim
        self.adaptive_grid_diags = adaptive_grid_diags
        self.bunch_grids: Dict[str, AdaptiveGrid] = {}
        self._t_reset_bunch_arrays = -1.0
        if len(self.ppc.shape) in [0, 1]:
            self.ppc = np.array([[self.r_max_plasma, self.ppc.flatten()[0]]])
        self.parabolic_coefficient = parabolic_coefficient
        if field_diags is None:
            field_diags = ["rho", "E", "B"]
            if laser is not None:
                field_diags.append("a")

        super().__init__(
            density_function=density_function,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            species_rho_diags=True,
            laser=laser,
            laser_evolution=laser_evolution,
            laser_envelope_substeps=laser_envelope_substeps,
            laser_envelope_nxi=laser_envelope_nxi,
            laser_envelope_nr=laser_envelope_nr,
            laser_envelope_use_phase=laser_envelope_use_phase,
            field_diags=field_diags,
            particle_diags=particle_diags,
            model_name="quasistatic_2d_ion",
        )

        # --- one-time init & one-time application flags ---
        # apply beam-loading only on first solve
        self._initial_condition_done = False

    def _initialize_properties(self, bunches):
        super()._initialize_properties(bunches)
        # Add bunch source array (needed if not using adaptive grids).
        self.b_t_bunch = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.q_bunch = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.q_var = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.laser_a2 = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.fld_arrays = [
            self.rho,
            self.rho_e,
            self.rho_i,
            self.chi,
            self.e_r,
            self.e_z,
            self.b_t,
            self.xi_fld,
            self.r_fld,
        ]

    def _select_witness_bunch(self, bunches: List[ParticleBunch]) -> ParticleBunch:
        # Option A: by do_salame attribute (highest priority)
        for b in bunches:
            if getattr(b, "do_salame", False):
                return b
        # Option B: assume last bunch is witness in beam-driven case
        return bunches[-1]

    def _calculate_wakefield(self, bunches: List[ParticleBunch]):
        radial_density = self._get_radial_density(self.t * ct.c)

        # Get square of laser envelope
        if self.laser is not None:
            calculate_laser_a2(self.laser.get_envelope(), self.laser_a2)
            # If linearly polarized, divide by 2 so that the ponderomotive
            # force on the plasma particles is correct.
            if self.laser.polarization == "linear":
                self.laser_a2 /= 2.0
            laser_a2 = self.laser_a2
        else:
            laser_a2 = None

        # Store plasma history if required by the diagnostics.
        store_plasma_history = len(self.particle_diags) > 0

        witness = self._select_witness_bunch(bunches) if bunches else None
        use_salame = getattr(witness, "do_salame", False)
        if use_salame and self.use_adaptive_grids:
            raise ValueError(
                "SALAME is not supported with adaptive grids. "
                "Set use_adaptive_grids=False or do_salame=False."
            )
        is_ic = use_salame and not self._initial_condition_done

        # Initialize empty lists with correct type so that numba can use
        # them even if there are no bunch sources.
        bunch_source_arrays = []
        bunch_source_xi_indices = []
        bunch_source_metadata = []

        # Calculate bunch sources and create adaptive grids if needed.
        s_d = ge.plasma_skin_depth(self.n_p * 1e-6)
        deposit_outliers_on_base_grid = False
        if self.use_adaptive_grids:
            ProfStart("adaptive_grids.deposit_bunch")

            store_plasma_history = True
            # Get radial grid resolution.
            if isinstance(self.adaptive_grid_nr, list):
                assert len(self.adaptive_grid_nr) == len(bunches), (
                    "Several resolutions for the adaptive grids have been "
                    "given, but they do not match the number of tracked "
                    "bunches"
                )
                nr_grids = self.adaptive_grid_nr
            else:
                nr_grids = [self.adaptive_grid_nr] * len(bunches)
            # Get radial extent.
            if isinstance(self.adaptive_grid_r_max, list):
                assert len(self.adaptive_grid_r_max) == len(bunches), (
                    "Several `r_max` for the adaptive grids have been "
                    "given, but they do not match the number of tracked "
                    "bunches"
                )
                r_max_grids = self.adaptive_grid_r_max
            else:
                r_max_grids = [self.adaptive_grid_r_max] * len(bunches)
            # Get radial extent limit.
            if isinstance(self.adaptive_grid_r_lim, list):
                assert len(self.adaptive_grid_r_lim) == len(bunches), (
                    "Several `r_lim` for the adaptive grids have been "
                    "given, but they do not match the number of tracked "
                    "bunches"
                )
                r_lim_grids = self.adaptive_grid_r_lim
            else:
                r_lim_grids = [self.adaptive_grid_r_lim] * len(bunches)
            # Check that the given extents are not larger than the limits.
            for r_lim, r_max in zip(r_lim_grids, r_max_grids):
                if r_lim is not None and r_max is not None:
                    if r_max > r_lim:
                        raise ValueError("`r_max` cannot be larger than `r_lim`")
            # Create adaptive grids for each bunch.
            bunches_with_grid: List[ParticleBunch] = []
            bunches_without_grid: List[ParticleBunch] = []
            for i, bunch in enumerate(bunches):
                if nr_grids[i] is not None:
                    bunches_with_grid.append(bunch)
                    if bunch.name not in self.bunch_grids:
                        self.bunch_grids[bunch.name] = AdaptiveGrid(
                            bunch.x,
                            bunch.y,
                            bunch.xi,
                            bunch.name,
                            nr_grids[i],
                            self.n_xi,
                            self.xi_fld,
                            r_max_grids[i],
                            r_lim_grids[i],
                        )
                else:
                    bunches_without_grid.append(bunch)
            # Calculate bunch sources at each grid.
            for bunch in bunches_with_grid:
                grid = self.bunch_grids[bunch.name]
                all_deposited = grid.calculate_bunch_source(
                    bunch, self.n_p, self.p_shape
                )

                bunch_source_arrays.append(grid.b_t_bunch)
                bunch_source_xi_indices.append(grid.i_grid)
                bunch_source_metadata.append(
                    np.array([grid.r_min_cell, grid.r_max_cell_guard, grid.dr]) / s_d
                )
                if not all_deposited:
                    self._reset_bunch_arrays()
                    deposit_bunch_charge(
                        bunch.x,
                        bunch.y,
                        bunch.xi,
                        bunch.q,
                        self.n_p,
                        self.n_r,
                        self.n_xi,
                        self.r_fld,
                        self.xi_fld,
                        self.dr,
                        self.dxi,
                        self.p_shape,
                        self.q_bunch,
                        r_min_deposit=grid.r_max,
                    )
                    deposit_outliers_on_base_grid = True
            ProfStop("adaptive_grids.deposit_bunch")
        else:
            bunches_without_grid = bunches
        # If not using adaptive grids, add all sources to the same array.
        if bunches_without_grid or deposit_outliers_on_base_grid:
            self._reset_bunch_arrays()
            if is_ic:
                # Non-witness bunches: deposit with self.p_shape into q_fixed.
                q_fixed = np.zeros_like(self.q_bunch)
                for bunch in bunches_without_grid:
                    if bunch is witness:
                        continue
                    deposit_bunch_charge(
                        bunch.x,
                        bunch.y,
                        bunch.xi,
                        bunch.q,
                        self.n_p,
                        self.n_r,
                        self.n_xi,
                        self.r_fld,
                        self.xi_fld,
                        self.dr,
                        self.dxi,
                        self.p_shape,
                        q_fixed,
                    )
                # Witness: deposit with z0r1 into self.q_var.
                self.q_var[:] = 0.0
                deposit_bunch_charge(
                    witness.x,
                    witness.y,
                    witness.xi,
                    witness.q,
                    self.n_p,
                    self.n_r,
                    self.n_xi,
                    self.r_fld,
                    self.xi_fld,
                    self.dr,
                    self.dxi,
                    "z0r1",
                    self.q_var,
                )
                self.q_bunch[:] = q_fixed + self.q_var
                # calculate_wakefields_salame_inline builds its own source
                # internally from q_fixed and q_var — no append needed here.
            else:
                for bunch in bunches_without_grid:
                    deposit_bunch_charge(
                        bunch.x,
                        bunch.y,
                        bunch.xi,
                        bunch.q,
                        self.n_p,
                        self.n_r,
                        self.n_xi,
                        self.r_fld,
                        self.xi_fld,
                        self.dr,
                        self.dxi,
                        self.p_shape,
                        self.q_bunch,
                    )
                calculate_bunch_source(
                    self.q_bunch, self.n_r, self.n_xi, self.b_t_bunch
                )
                bunch_source_arrays.append(self.b_t_bunch)
                bunch_source_xi_indices.append(np.arange(self.n_xi))
                bunch_source_metadata.append(
                    np.array(
                        [
                            self.r_fld[0],
                            self.r_fld[-1] + 2 * self.dr,  # r of last guard cell.
                            self.dr,
                        ]
                    )
                    / s_d
                )

        # Calculate rho only if requested in the diagnostics.
        calculate_rho = any(
            (
                any(
                    "rho" in f or f == "all"
                    for f in (
                        diag["field"]
                        if isinstance(diag["field"], list)
                        else [diag["field"]]
                    )
                )
                if isinstance(diag, dict) and "field" in diag
                else "rho" in diag or diag == "all"
            )
            for diag in (
                self.field_diags
                if isinstance(self.field_diags, list)
                else [self.field_diags]
            )
        )

        # Calculate plasma wakefields.
        if is_ic:
            ProfStart("salame.initial_condition")
            # Gather from q_var before SALAME to get per-particle baseline.
            w_gathered_old, _ = gather_z0r1(
                witness.xi,
                witness.x,
                witness.y,
                self.xi_fld[0],
                self.r_fld[0],
                self.n_xi,
                self.n_r,
                self.dxi,
                self.dr,
                self.q_var,
                use_ruyten=True,
            )

            self.pp = calculate_wakefields_salame(
                laser_a2,
                self.r_max,
                self.xi_min,
                self.xi_max,
                self.n_r,
                self.n_xi,
                self.ppc,
                self.n_p,
                self.q_bunch,
                q_fixed,
                self.q_var,
                self.b_t_bunch,
                salame_max_iter=getattr(witness, "salame_n_iter", 10),
                salame_tol=getattr(witness, "salame_relative_tolerance", 1e-4),
                use_avg_psi=getattr(witness, "use_avg_psi", True),
                r_max_plasma=self.r_max_plasma,
                radial_density=radial_density,
                p_shape=self.p_shape,
                max_gamma=self.max_gamma,
                plasma_pusher=self.plasma_pusher,
                ion_motion=self.ion_motion,
                ion_mass=self.ion_mass,
                free_electrons_per_ion=self.free_electrons_per_ion,
                store_plasma_history=store_plasma_history,
                calculate_rho=calculate_rho,
                particle_diags=self.particle_diags,
                fld_arrays=self.fld_arrays,
            )

            # Gather from shaped q_var and apply the ratio as a per-particle
            # scale factor. Using the ratio cancels the N-particle accumulation
            # that a direct inversion would produce.
            w_gathered_new, _ = gather_z0r1(
                witness.xi,
                witness.x,
                witness.y,
                self.xi_fld[0],
                self.r_fld[0],
                self.n_xi,
                self.n_r,
                self.dxi,
                self.dr,
                self.q_var,
                use_ruyten=True,
            )
            safe_old = np.where(np.abs(w_gathered_old) > 0, w_gathered_old, 1.0)
            witness.w[:] = witness.w * (w_gathered_new / safe_old)
            ProfStop("salame.initial_condition")
        else:
            self.pp = calculate_wakefields(
                laser_a2,
                self.r_max,
                self.xi_min,
                self.xi_max,
                self.n_r,
                self.n_xi,
                self.ppc,
                self.n_p,
                r_max_plasma=self.r_max_plasma,
                radial_density=radial_density,
                p_shape=self.p_shape,
                max_gamma=self.max_gamma,
                plasma_pusher=self.plasma_pusher,
                ion_motion=self.ion_motion,
                ion_mass=self.ion_mass,
                free_electrons_per_ion=self.free_electrons_per_ion,
                fld_arrays=self.fld_arrays,
                bunch_source_arrays=bunch_source_arrays,
                bunch_source_xi_indices=bunch_source_xi_indices,
                bunch_source_metadata=bunch_source_metadata,
                store_plasma_history=store_plasma_history,
                calculate_rho=calculate_rho,
                particle_diags=self.particle_diags,
            )

        # Add bunch density to total density.
        if calculate_rho:
            rho_bunch = -self.q_bunch[2:-2, 2:-2] / (self.r_fld / s_d)
            self.rho[2:-2, 2:-2] += rho_bunch

        # Calculate fields on adaptive grids.
        if self.use_adaptive_grids:
            for _, grid in self.bunch_grids.items():
                grid.calculate_fields(self.n_p, self.pp)

        # After first wake solve, disable one-time beam-loading effect
        self._initial_condition_done = True

    def _reset_bunch_arrays(self):
        """Reset to zero the bunch arrays of the base grid."""
        if self.t > self._t_reset_bunch_arrays:
            self.b_t_bunch[:] = 0.0
            self.q_bunch[:] = 0.0
            self._t_reset_bunch_arrays = self.t

    def _get_radial_density(self, z_current):
        """Get radial density profile function"""

        def radial_density(r):
            n_p = self.density_function(z_current, r)
            if self.parabolic_coefficient is not None:
                pc = self.parabolic_coefficient(z_current)
                n_p = n_p + n_p * pc * r**2
            return n_p

        return radial_density

    def _get_parabolic_coefficient_fn(self, parabolic_coefficient):
        """Get parabolic_coefficient profile function"""
        if isinstance(parabolic_coefficient, float):

            def uniform_parabolic_coefficient(z):
                return np.ones_like(z) * parabolic_coefficient

            return uniform_parabolic_coefficient
        elif callable(parabolic_coefficient):
            return parabolic_coefficient
        else:
            raise ValueError(
                "Type {} not supported for parabolic_coefficient.".format(
                    type(parabolic_coefficient)
                )
            )

    def _gather(self, x, y, z, t, ex, ey, ez, bx, by, bz, bunch_name):
        # If using adaptive grids, gather fields from them.
        if bunch_name in self.bunch_grids:
            grid = self.bunch_grids[bunch_name]
            grid.update_if_needed(x, y, z, self.n_p, self.pp)
            all_gathered = grid.gather_fields(x, y, z, ex, ey, ez, bx, by, bz)
            # If not all particles managed to gather from adaptive grid (for
            # example, because they escaped from it), try to gather from the
            # base grid.
            if not all_gathered:
                gather_main_fields_cyl_linear(
                    self.e_r,
                    self.e_z,
                    self.b_t,
                    self.xi_fld[0],
                    self.xi_fld[-1],
                    self.r_fld[0],
                    self.r_fld[-1],
                    self.dxi,
                    self.dr,
                    x,
                    y,
                    z,
                    ex,
                    ey,
                    ez,
                    bx,
                    by,
                    bz,
                    r_min_gather=grid.r_max_cell,
                )

        # Otherwise, use base implementation.
        else:
            super()._gather(x, y, z, t, ex, ey, ez, bx, by, bz, bunch_name)

    def _get_openpmd_diagnostics_data(self, global_time):
        diag_data = super()._get_openpmd_diagnostics_data(global_time)

        # Add fields from adaptive grids to openpmd diagnostics.
        if self.use_adaptive_grids:
            for _, grid in self.bunch_grids.items():
                grid_data = grid.get_openpmd_data(global_time, self.adaptive_grid_diags)
                diag_data["fields"] += grid_data["fields"]
                for field in grid_data["fields"]:
                    diag_data[field] = grid_data[field]
        # Add plasma particles to openpmd diagnostics.
        particle_diags = self._get_plasma_particle_diagnostics(global_time)
        diag_data = {**diag_data, **particle_diags}
        diag_data["species"] = list(particle_diags.keys())
        return diag_data

    def _get_plasma_particle_diagnostics(self, global_time):
        """Return dict with plasma particle diagnostics."""
        diag_dict = {}
        elec_name = "plasma_electrons"
        ions_name = "plasma_ions"
        if len(self.particle_diags) > 0:
            for name, hist in zip((elec_name, ions_name), self.pp):
                s_d = ge.plasma_skin_depth(self.n_p * 1e-6)

                if name == elec_name:
                    diag_dict[name] = {
                        "q": -ct.e,
                        "m": ct.m_e,
                        "geometry": "rz",
                    }
                else:
                    diag_dict[name] = {
                        "q": ct.e * self.free_electrons_per_ion,
                        "m": self.ion_mass,
                        "geometry": "rz",
                    }
                diag_dict[name]["name"] = name
                if "r" in self.particle_diags:
                    diag_dict[name]["r"] = hist["r_hist"] * s_d
                if "z" in self.particle_diags:
                    diag_dict[name]["z"] = hist["xi_hist"] * s_d + self.xi_max
                    diag_dict[name]["z_off"] = global_time * ct.c
                if "pr" in self.particle_diags:
                    diag_dict[name]["pr"] = (
                        hist["pr_hist"] * diag_dict[name]["mass"] * ct.c
                    )
                if "pz" in self.particle_diags:
                    diag_dict[name]["pz"] = (
                        hist["pz_hist"] * diag_dict[name]["mass"] * ct.c
                    )
                if "w" in self.particle_diags:
                    diag_dict[name]["w"] = hist["w_hist"] * (
                        self.n_p
                        if name == elec_name
                        else self.n_p / self.free_electrons_per_ion
                    )
                if "r_to_x" in self.particle_diags:
                    diag_dict[name]["r_to_x"] = hist["r_to_x_hist"]
                if "id" in self.particle_diags:
                    diag_dict[name]["id"] = hist["id_hist"]

        return diag_dict
