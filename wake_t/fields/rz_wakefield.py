"""This module contains the base class for plasma wakefields in r-z geometry"""
import numpy as np
import scipy.constants as ct

from wake_t.particles.interpolation import gather_main_fields_cyl_linear
from wake_t.utilities.other import generate_field_diag_dictionary
from .numerical_field import NumericalField


class RZWakefield(NumericalField):
    """Base class for plasma wakefields in r-z geometry"""

    def __init__(self, density_function, laser=None, laser_evolution=True,
                 laser_envelope_substeps=1, laser_envelope_subgrid_nz=None,
                 laser_envelope_subgrid_nr=None,
                 r_max=None, xi_min=None, xi_max=None, n_r=100, n_xi=100,
                 dz_fields=None, model_name=''):
        """Initialize wakefield.

        Parameters
        ----------
        density_function : callable
            Function of that returns the relative value of the plasma density
            at each `z` position.
        laser : LaserPulse
            Laser driver of the plasma stage.
        laser_evolution : bool
            If True (default), the laser pulse is evolved
            using a laser envelope model. If False, the pulse envelope stays
            unchanged throughout the computation.
        laser_envelope_substeps : int
            Number of substeps of the laser envelope solver per `dz_fields`.
            The time step of the envelope solver is therefore
            `dz_fields / c / laser_envelope_substeps`.
        laser_envelope_subgrid : int
            Number of substeps of the laser envelope solver per `dz`.
            The number of grid points in `z` of the envelope solver is
            `n_xi * laser_envelope_subgrid`
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
            Number of grid elements along r to calculate the wakefields.
        n_xi : int
            Number of grid elements along xi to calculate the wakefields.
        dz_fields : float (optional)
            Determines how often the plasma wakefields should be updated. If
            dz_fields=0 (default value), the wakefields are calculated at every
            step of the Runge-Kutta solver for the beam particle evolution
            (most expensive option). If specified, the wakefields are only
            updated in steps determined by dz_fields. For example, if
            dz_fields=10e-6, the plasma wakefields are only updated every time
            the simulation window advances by 10 micron. By default, if not
            specified, the value of `dz_fields` will be `xi_max-xi_min`, i.e.,
            the length the simulation box.
        model_name : str, optional
            Name of the wakefield model. This will be stored in the openPMD
            diagnostics.
        """
        dz_fields = xi_max - xi_min if dz_fields is None else dz_fields
        self.density_function = density_function
        self.laser = laser
        self.laser_evolution = laser_evolution
        self.laser_envelope_substeps = laser_envelope_substeps
        self.laser_envelope_subgrid_nz = laser_envelope_subgrid_nz
        self.laser_envelope_subgrid_nr = laser_envelope_subgrid_nr
        self.r_max = r_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.n_r = n_r
        self.n_xi = n_xi
        self.dr = r_max / n_r
        self.dxi = (xi_max - xi_min) / (n_xi - 1)
        self.model_name = model_name
        # If a laser is included, make sure it is evolved for the whole
        # duration of the plasma stage. See `force_even_updates` parameter.
        super().__init__(
            dt_update=dz_fields/ct.c,
            openpmd_diag_supported=True,
            force_even_updates=laser is not None
        )

    def _initialize_properties(self, bunches):
        # Initialize laser.
        if self.laser is not None:
            self.laser.set_envelope_solver_params(
                self.xi_min, self.xi_max, self.r_max, self.n_xi, self.n_r,
                self.dt_update, self.laser_envelope_substeps,
                self.laser_envelope_subgrid_nz,
                self.laser_envelope_subgrid_nr)
            self.laser.initialize_envelope()

        # Initialize field arrays
        self.rho = np.zeros((self.n_xi+4, self.n_r+4))
        self.chi = np.zeros((self.n_xi+4, self.n_r+4))
        self.e_z = np.zeros((self.n_xi+4, self.n_r+4))
        self.e_r = np.zeros((self.n_xi+4, self.n_r+4))
        self.e_t = np.zeros((self.n_xi+4, self.n_r+4))
        self.b_z = np.zeros((self.n_xi+4, self.n_r+4))
        self.b_r = np.zeros((self.n_xi+4, self.n_r+4))
        self.b_t = np.zeros((self.n_xi+4, self.n_r+4))
        self.r_fld = np.linspace(self.dr/2, self.r_max - self.dr/2, self.n_r)
        self.xi_fld = np.linspace(self.xi_min, self.xi_max, self.n_xi)

    def _evolve_properties(self, bunches):
        if self.laser is not None:
            # Evolve laser envelope
            if self.laser_evolution:
                self.laser.evolve(self.chi[2:-2, 2:-2], self.n_p)

    def _calculate_field(self, bunches):
        self.n_p = self.density_function(self.t*ct.c)
        self.rho[:] = 0.
        self.chi[:] = 0.
        self.e_z[:] = 0.
        self.e_r[:] = 0.
        self.b_t[:] = 0.
        self._calculate_wakefield(bunches)

    def _calculate_wakefield(bunches):
        """To be implemented by the subclasses."""
        raise NotImplementedError

    def _gather(self, x, y, z, t, ex, ey, ez, bx, by, bz):
        dr = self.r_fld[1] - self.r_fld[0]
        dxi = self.xi_fld[1] - self.xi_fld[0]
        gather_main_fields_cyl_linear(
            self.e_r, self.e_z, self.b_t, self.xi_fld[0], self.xi_fld[-1],
            self.r_fld[0], self.r_fld[-1], dxi, dr, x, y, z,
            ex, ey, ez, bx, by, bz)

    def _get_openpmd_diagnostics_data(self, global_time):
        # Prepare necessary data.
        fld_solver = 'other'
        fld_solver_params = self.model_name
        fld_boundary = ['other'] * 4
        part_boundary = ['other'] * 4
        fld_boundary_params = ['none'] * 4
        part_boundary_params = ['none'] * 4
        current_smoothing = 'none'
        charge_correction = 'none'
        dr = np.abs(self.r_fld[1] - self.r_fld[0])
        dz = np.abs(self.xi_fld[1] - self.xi_fld[0])
        grid_spacing = [dr, dz]
        grid_labels = ['r', 'z']
        grid_global_offset = [0., global_time*ct.c+self.xi_min]
        # Cell-centered in 'r' and node centered in 'z'.
        fld_position = [0.5, 0.]
        fld_names = ['E', 'B', 'rho']
        fld_comps = [['r', 't', 'z'], ['r', 't', 'z'], None]
        fld_arrays = [
            [np.ascontiguousarray(self.e_r.T[2:-2, 2:-2]),
             np.ascontiguousarray(self.e_t.T[2:-2, 2:-2]),
             np.ascontiguousarray(self.e_z.T[2:-2, 2:-2])],
            [np.ascontiguousarray(self.b_r.T[2:-2, 2:-2]),
             np.ascontiguousarray(self.b_t.T[2:-2, 2:-2]),
             np.ascontiguousarray(self.b_z.T[2:-2, 2:-2])],
            [np.ascontiguousarray(self.rho.T[2:-2, 2:-2]) * self.n_p * (-ct.e)]
        ]
        if self.laser is not None:
            fld_names += ['a_mod', 'a_phase']
            fld_comps += [None, None]
            fld_arrays += [
                [np.ascontiguousarray(np.abs(self.laser.get_envelope().T))],
                [np.ascontiguousarray(np.angle(self.laser.get_envelope().T))]
            ]
        fld_comp_pos = [fld_position] * len(fld_names)

        # Generate dictionary for openPMD diagnostics.
        diag_data = generate_field_diag_dictionary(
            fld_names, fld_comps, fld_arrays, fld_comp_pos, grid_labels,
            grid_spacing, grid_global_offset, fld_solver, fld_solver_params,
            fld_boundary, fld_boundary_params, part_boundary,
            part_boundary_params, current_smoothing, charge_correction)

        return diag_data
