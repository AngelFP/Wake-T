import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .solver import calculate_wakefields
from wake_t.particles.interpolation import gather_main_fields_cyl_linear
from wake_t.utilities.other import generate_field_diag_dictionary
from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class Quasistatic2DWakefield(Wakefield):

    def __init__(self, density_function, laser=None, laser_evolution=True,
                 r_max=None, xi_min=None, xi_max=None, n_r=100,
                 n_xi=100, ppc=2, dz_fields=None, r_max_plasma=None,
                 parabolic_coefficient=0., p_shape='cubic', max_gamma=10):
        super().__init__()
        self.openpmd_diag_supported = True
        self.density_function = density_function
        self.laser = laser
        self.laser_evolution = laser_evolution
        self.r_max = r_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.n_r = n_r
        self.n_xi = n_xi
        self.ppc = ppc
        self.dz_fields = xi_max - xi_min if dz_fields is None else dz_fields
        self.r_max_plasma = r_max_plasma
        self.parabolic_coefficient = self._get_parabolic_coefficient_fn(
            parabolic_coefficient)
        self.p_shape = p_shape
        self.max_gamma = max_gamma
        # Last time at which the fields where requested.
        self.current_t = None
        # Last step at which the fields where calculated.
        self.current_wf_step = None
        # Last time at which the fields where interpolated to the particles.
        self.current_t_interp = None

    def Wx(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        self.__interpolate_fields_to_particles(x, y, xi, t)
        return self.wx_part

    def Wy(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        self.__interpolate_fields_to_particles(x, y, xi, t)
        return self.wy_part

    def Wz(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        self.__interpolate_fields_to_particles(x, y, xi, t)
        return self.ez_part

    def __calculate_wakefields(self, x, y, xi, px, py, pz, q, t):
        self.current_t = t
        required_wf_step = int(np.round(t * ct.c / self.dz_fields))
        if self.current_wf_step is None:
            self.current_wf_step = required_wf_step
            if self.laser is not None:
                # Initialize laser.
                self.laser.set_envelope_solver_params(
                    self.xi_min, self.xi_max, self.r_max, self.n_xi, self.n_r,
                    self.dz_fields/ct.c)
                self.laser.initialize_envelope()
            d_step = required_wf_step
        elif required_wf_step > self.current_wf_step:
            d_step = required_wf_step - self.current_wf_step
            self.current_wf_step = required_wf_step
        else:
            return
        n_p = self.density_function(t*ct.c)
        parabolic_coefficient = self.parabolic_coefficient(t*ct.c)

        if self.laser is not None:
            # Evolve laser envelope
            if self.laser_evolution:
                for s in range(d_step):
                    # Evolve laser in the current chi (removing guard cells).
                    self.laser.evolve(self.chi[2:-2, 2:-2], n_p)

            # Laser envelope
            a_env = np.abs(self.laser.get_envelope()) ** 2

        else:
            a_env = np.zeros((self.n_xi, self.n_r))

        # Calculate plasma wakefields
        rho, chi, W_r, E_z, xi_arr, r_arr = calculate_wakefields(
            a_env, [x, y, xi, q], self.r_max, self.xi_min, self.xi_max,
            self.n_r, self.n_xi, self.ppc, n_p, r_max_plasma=self.r_max_plasma,
            parabolic_coefficient=parabolic_coefficient,
            p_shape=self.p_shape, max_gamma=self.max_gamma)

        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p*1e-6)
        s_d = ge.plasma_skin_depth(n_p*1e-6)

        self.rho = rho
        self.chi = chi
        self.E_z = E_z*E_0
        self.W_x = W_r*E_0
        self.xi_fld = xi_arr*s_d
        self.r_fld = r_arr*s_d
        self.n_p = n_p

    def __interpolate_fields_to_particles(self, x, y, xi, t):
        if (self.current_t_interp is None) or (self.current_t_interp != t):
            self.current_t_interp = t
            # Gather fields
            dr = self.r_fld[1] - self.r_fld[0]
            dxi = self.xi_fld[1] - self.xi_fld[0]
            interp_flds = gather_main_fields_cyl_linear(
                self.W_x, self.E_z, self.xi_fld[0], self.xi_fld[-1],
                self.r_fld[0], self.r_fld[-1], dxi, dr, x, y, xi)
            self.wx_part, self.wy_part, self.ez_part = interp_flds

    def _get_openpmd_diagnostics_data(self):
        # Prepare necessary data.
        fld_solver = 'other'
        fld_solver_params = 'quasistatic_2d'
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
        grid_global_offset = [0., self.current_t*ct.c+self.xi_min]
        # Cell-centered in 'r' and node centered in 'z'.
        fld_position = [0.5, 0.]
        fld_names = ['E', 'W', 'rho']
        fld_comps = [['z'], ['r'], None]
        fld_arrays = [
            [np.ascontiguousarray(self.E_z.T[2:-2, 2:-2])],
            [np.ascontiguousarray(self.W_x.T[2:-2, 2:-2])],
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
