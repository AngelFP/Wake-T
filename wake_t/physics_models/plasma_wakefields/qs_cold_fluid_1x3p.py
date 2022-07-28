import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.particles.deposition import deposit_3d_distribution
from wake_t.particles.interpolation import (
    gather_field_cyl_linear)
from wake_t.fields.rz_wakefield import RZWakefield


class NonLinearColdFluidWakefield(RZWakefield):
    def __init__(self, density_function, laser=None, laser_evolution=True,
                 laser_envelope_substeps=1, r_max=None, xi_min=None,
                 xi_max=None, n_r=100, n_xi=100, dz_fields=None,
                 beam_wakefields=False, p_shape='linear'):
        self.beam_wakefields = beam_wakefields
        self.p_shape = p_shape
        super().__init__(
            density_function=density_function,
            laser=laser,
            laser_evolution=laser_evolution,
            laser_envelope_substeps=laser_envelope_substeps,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            model_name='cold_fluid_1d'
        )

    def __wakefield_ode_system(self, u_1, u_2, laser_a0, n_beam):
        if self.beam_wakefields:
            return np.array(
                [u_2, (1+laser_a0**2)/(2*(1+u_1)**2) + n_beam - 1/2])
        else:
            return np.array([u_2, (1+laser_a0**2)/(2*(1+u_1)**2) - 1/2])

    def _calculate_wakefield(self, bunches):
        # Get laser envelope
        if self.laser is not None:
            a_env = np.abs(self.laser.get_envelope())
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
            q = bunch.q
            deposit_3d_distribution(
                xi/s_d, x/s_d, y/s_d, q/ct.e, self.xi_min/s_d, r_fld[0],
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
        integrand = (dE_z - nv_z - beam_hist) * r_fld
        subs = integrand / 2
        B_theta = (np.cumsum(integrand, axis=1) - subs) * dr / np.abs(r_fld)
        E_r = W_r + B_theta

        # Store fields.
        self.b_t[2:-2, 2:-2] = B_theta * E_0 / ct.c
        self.e_r[2:-2, 2:-2] = E_r * E_0
        self.e_z[2:-2, 2:-2] = E_z * E_0
