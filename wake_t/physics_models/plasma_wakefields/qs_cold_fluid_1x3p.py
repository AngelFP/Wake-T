import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.particles.deposition import deposit_3d_distribution
from wake_t.particles.interpolation import (
    gather_field_cyl_linear, gather_main_fields_cyl_linear)
from wake_t.utilities.other import generate_field_diag_dictionary
from wake_t.fields.numerical_field import NumericalField


class NonLinearColdFluidWakefield(NumericalField):
    def __init__(self, density_function, laser=None, laser_evolution=True,
                 r_max=None, xi_min=None, xi_max=None, n_r=100,
                 n_xi=100, dz_fields=None, beam_wakefields=False,
                 p_shape='linear'):
        self.density_function = density_function
        self.laser = laser
        self.laser_evolution = laser_evolution
        self.r_max = r_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.n_r = n_r
        self.n_xi = n_xi
        self.dz_fields = xi_max - xi_min if dz_fields is None else dz_fields
        self.beam_wakefields = beam_wakefields
        self.p_shape = p_shape
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
                self.dt_update)
            self.laser.initialize_envelope()

    def _evolve_properties(self, bunches):
        if self.laser is not None and self.laser_evolution:
            # Evolve laser envelope
            self.laser.evolve(self.chi_fl, self.current_n_p)

    def __wakefield_ode_system(self, u_1, u_2, laser_a0, n_beam):
        if self.beam_wakefields:
            return np.array(
                [u_2, (1+laser_a0**2)/(2*(1+u_1)**2) + n_beam - 1/2])
        else:
            return np.array([u_2, (1+laser_a0**2)/(2*(1+u_1)**2) - 1/2])

    def _calculate_field(self, bunches):
        n_p = self.density_function(self.t*ct.c)

        # Get laser envelope
        if self.laser is not None:
            a_env = np.abs(self.laser.get_envelope())
        else:
            a_env = np.zeros((self.n_xi, self.n_r))

        # Calculate and allocate laser quantities, including guard cells.
        a_rz = np.zeros((self.n_xi+4, self.n_r+4))
        a_rz[2:-2, 2:-2] = a_env

        s_d = ge.plasma_skin_depth(n_p*1e-6)
        dz = (self.xi_max - self.xi_min) / (self.n_xi - 1) / s_d
        dr = self.r_max / self.n_r / s_d
        r = np.linspace(dr/2, self.r_max/s_d-dr/2, self.n_r)

        # Currently, only one bunch supported
        bunch = bunches[0]
        x = bunch.x
        y = bunch.y
        xi = bunch.xi
        q = bunch.q

        # Get charge distribution and remove guard cells.
        beam_hist = np.zeros((self.n_xi+4, self.n_r+4))
        deposit_3d_distribution(
            xi/s_d, x/s_d, y/s_d, q/ct.e, self.xi_min/s_d, r[0],
            self.n_xi, self.n_r, dz, dr, beam_hist, p_shape=self.p_shape,
            use_ruyten=True)
        beam_hist = beam_hist[2:-2, 2:-2]

        n = np.arange(self.n_r)
        disc_area = np.pi * dr**2*(1+2*n)
        beam_hist *= 1/(disc_area*dz*n_p)/s_d**3
        n_iter = self.n_xi - 1
        u_1 = np.zeros((n_iter+1, len(r)))
        u_2 = np.zeros((n_iter+1, len(r)))
        z_arr = np.linspace(self.xi_min, self.xi_max, self.n_xi) / s_d

        for i in np.arange(n_iter):
            z_i = z_arr[-1 - i]
            # get laser a0 at z, z+dz/2 and z+dz
            if self.laser is not None:
                x = r
                y = np.zeros_like(r)
                z = np.full_like(r, z_i)
                a0_0 = gather_field_cyl_linear(
                    a_rz, self.xi_min/s_d, self.xi_max/s_d, r[0], r[-1],
                    dz, dr, x, y, z)
                a0_1 = gather_field_cyl_linear(
                    a_rz, self.xi_min/s_d, self.xi_max/s_d, r[0], r[-1],
                    dz, dr, x, y, z - dz/2)
                a0_2 = gather_field_cyl_linear(
                    a_rz, self.xi_min/s_d, self.xi_max/s_d, r[0], r[-1],
                    dz, dr, x, y, z - dz)
            else:
                a0_0 = np.zeros(r.shape[0])
                a0_1 = np.zeros(r.shape[0])
                a0_2 = np.zeros(r.shape[0])
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
        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p*1e-6)

        self.current_n_p = n_p
        self.gamma_fl = (1 + a_env**2 + (1 + u_1)**2) / (2 * (1 + u_1))
        self.n_fl = self.gamma_fl / (1 + u_1)
        self.chi_fl = self.n_fl / self.gamma_fl

        # Calculate B_theta and E_r.
        u_z = (1 + a_env**2 - (1 + u_1)**2) / (2 * (1 + u_1))
        dE_z = np.gradient(E_z, dz, axis=0, edge_order=2)
        v_z = u_z / self.gamma_fl
        nv_z = self.n_fl * v_z
        integrand = (dE_z - nv_z - beam_hist) * r
        subs = integrand / 2
        B_theta = (np.cumsum(integrand, axis=1) - subs) * dr / np.abs(r)
        E_r = W_r + B_theta

        # Store fields.
        self.B_t = np.zeros((self.n_xi+4, self.n_r+4))
        self.E_r = np.zeros((self.n_xi+4, self.n_r+4))
        self.E_z = np.zeros((self.n_xi+4, self.n_r+4))
        self.B_t[2:-2, 2:-2] = B_theta*E_0/ct.c
        self.E_r[2:-2, 2:-2] = E_r*E_0
        self.E_z[2:-2, 2:-2] = E_z*E_0
        self.xi_fld = z_arr * s_d
        self.r_fld = r * s_d

    def _gather(self, x, y, z, t, ex, ey, ez, bx, by, bz):
        dr = self.r_fld[1] - self.r_fld[0]
        dxi = self.xi_fld[1] - self.xi_fld[0]
        gather_main_fields_cyl_linear(
            self.E_r, self.E_z, self.B_t, self.xi_fld[0], self.xi_fld[-1],
            self.r_fld[0], self.r_fld[-1], dxi, dr, x, y, z,
            ex, ey, ez, bx, by, bz)

    def _get_openpmd_diagnostics_data(self, global_time):
        # Prepare necessary data.
        fld_solver = 'other'
        fld_solver_params = 'cold_fluid_1d'
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
        fld_comps = [['r', 'z'], ['t'], None]
        # Need to make sure it is a contiguous array to prevent incorrect
        # openPMD output.
        fld_arrays = [
            [np.ascontiguousarray(self.E_r.T)],
            [np.ascontiguousarray(self.E_z.T)],
            [np.ascontiguousarray(self.B_t.T)],
            [np.ascontiguousarray(self.n_fl.T) * self.current_n_p * (-ct.e)]
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
