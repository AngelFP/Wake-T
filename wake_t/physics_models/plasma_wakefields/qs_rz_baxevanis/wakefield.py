import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .solver import calculate_wakefields
from wake_t.particles.interpolation import (
    gather_field_cyl_linear, gather_main_fields_cyl_linear)
from wake_t.utilities.other import generate_field_diag_dictionary
from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class Quasistatic2DWakefield(Wakefield):

    def __init__(self, density_function, laser=None, laser_evolution=False,
                 laser_z_foc=0, r_max=None, xi_min=None, xi_max=None, n_r=100,
                 n_xi=100, ppc=2, dz_fields=0, p_shape='linear'):
        super().__init__()
        self.openpmd_diag_supported = True
        self.density_function = density_function
        self.laser = laser
        self.laser_evolution = laser_evolution
        self.laser_z_foc = laser_z_foc
        self.r_max = r_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.n_r = n_r
        self.n_xi = n_xi
        self.ppc = ppc
        self.dz_fields = np.inf if dz_fields is None else dz_fields
        self.p_shape = p_shape
        # Last time at which the fields where requested.
        self.current_t = None
        # Last time at which the fields where calculated.
        self.current_t_wf = None
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

    def Kx(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        return gather_field_cyl_linear(
            self.K_x, self.xi_fld, self.r_fld, x, y, xi)

    def Ez_p(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        return gather_field_cyl_linear(
            self.E_z_p, self.xi_fld, self.r_fld, x, y, xi)

    def __calculate_wakefields(self, x, y, xi, px, py, pz, q, t):
        self.current_t = t
        if self.current_t_wf is None:
            self.current_t_wf = t
        elif (self.current_t_wf != t and
              t >= self.current_t_wf + self.dz_fields/ct.c):
            self.current_t_wf = t
        else:
            return
        z_beam = t*ct.c + np.average(xi)  # z postion of beam center
        n_p = self.density_function(z_beam)

        # calculate distance to laser focus
        if self.laser_evolution:
            dz_foc = self.laser_z_foc - ct.c*t
        else:
            dz_foc = 0

        flds = calculate_wakefields(
            self.laser, [x, y, xi, q], self.r_max, self.xi_min, self.xi_max,
            self.n_r, self.n_xi, self.ppc, n_p, dz_foc, p_shape=self.p_shape)
        n_p_mesh, W_r, E_z, E_z_p, K_r, psi_mesh, xi_arr, r_arr = flds

        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p*1e-6)
        s_d = ge.plasma_skin_depth(n_p*1e-6)

        self.rho = n_p_mesh.T
        self.E_z = E_z.T*E_0
        self.W_x = W_r.T*E_0
        self.K_x = K_r.T*E_0/s_d/ct.c
        self.E_z_p = E_z_p.T*E_0/s_d
        self.xi_fld = xi_arr*s_d
        self.r_fld = r_arr*s_d

    def __interpolate_fields_to_particles(self, x, y, xi, t):
        if (self.current_t_interp is None) or (self.current_t_interp != t):
            self.current_t_interp = t
            interp_flds = gather_main_fields_cyl_linear(
                self.W_x, self.E_z, self.xi_fld, self.r_fld, x, y, xi)
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
        grid_local_offset = [0., self.current_t*ct.c+self.xi_min]
        # Cell-centered in 'r' anf 'z'. TODO: check correctness.
        fld_position = [0.5, 0.5]
        fld_names = ['E', 'W', 'rho']
        fld_comps = [['z'], ['r'], None]
        fld_arrays = [[self.E_z.T], [self.W_x.T], [self.rho.T]]
        fld_comp_pos = [fld_position] * len(fld_names)

        # Generate dictionary for openPMD diagnostics.
        diag_data = generate_field_diag_dictionary(
            fld_names, fld_comps, fld_arrays, fld_comp_pos, grid_labels,
            grid_spacing, grid_local_offset, fld_solver, fld_solver_params,
            fld_boundary, fld_boundary_params, part_boundary,
            part_boundary_params, current_smoothing, charge_correction)

        return diag_data


class PlasmaRampBlowoutField(Wakefield):
    def __init__(self, density_function):
        super().__init__()
        self.density_function = density_function

    def Wx(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        return np.zeros(len(xi))

    def Kx(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return np.ones(len(xi))*kx

    def calculate_focusing(self, xi, t):
        z = t*ct.c + xi  # z postion of each particle at time t
        n_p = self.density_function(z)
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        return (ct.m_e/(2*ct.e*ct.c))*w_p**2
