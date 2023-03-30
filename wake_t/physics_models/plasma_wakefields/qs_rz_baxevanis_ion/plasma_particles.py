"""Contains the definition of the `PlasmaParticles` class."""

import numpy as np
import scipy.constants as ct
import matplotlib.pyplot as plt

from wake_t.utilities.numba import njit_serial
from .psi_and_derivatives import (
    calculate_psi,
    calculate_cumulative_sums, calculate_cumulative_sum_1, calculate_cumulative_sum_2, calculate_cumulative_sum_3,
    calculate_psi_dr_psi_at_particles_bg,
    determine_neighboring_points, calculate_psi_and_dr_psi, calculate_dxi_psi,
    calculate_dxi_psi_at_particles_bg)
from .deposition import deposit_plasma_particles
from .gather import gather_sources, gather_psi_bg, gather_dr_psi_bg
from .b_theta import (calculate_ai_bi_from_axis,
                      calculate_b_theta_at_particles, calculate_b_theta,
                      calculate_b_theta_at_ions)
from .plasma_push.ab5 import evolve_plasma_ab5


class PlasmaParticles():
    """
    Class containing the 1D slice of plasma particles used in the quasi-static
    Baxevanis wakefield model.

    Parameters
    ----------
    r_max : float
        Maximum radial extension of the simulation box in normalized units.
    r_max_plasma : float
        Maximum radial extension of the plasma column in normalized units.
    parabolic_coefficient : float
        The coefficient for the transverse parabolic density profile.
    dr : float
        Radial step size of the discretized simulation box.
    ppc : int
        Number of particles per cell.
    pusher : str
        Particle pusher used to evolve the plasma particles. Possible
        values are `'rk4'` and `'ab5'`.

    """

    def __init__(self, r_max, r_max_plasma, parabolic_coefficient, dr, ppc,
                 r_grid, nr, ion_motion=True, pusher='ab5', shape='linear'):
        # Calculate total number of plasma particles.
        n_elec = int(np.round(r_max_plasma / dr * ppc))
        n_part = n_elec * 2

        # Readjust plasma extent to match number of particles.
        dr_p = dr / ppc
        r_max_plasma = n_elec * dr_p

        # Store parameters.
        self.r_max = r_max
        self.r_max_plasma = r_max_plasma
        self.parabolic_coefficient = parabolic_coefficient
        self.dr = dr
        self.ppc = ppc
        self.dr_p = dr / ppc
        self.pusher = pusher
        self.n_elec = n_elec
        self.n_part = n_part
        self.shape = shape
        self.r_grid = r_grid
        self.nr = nr
        self.ion_motion = ion_motion

    def initialize(self):
        """Initialize column of plasma particles."""

        # Initialize particle arrays.
        r = np.linspace(
            self.dr_p / 2, self.r_max_plasma - self.dr_p / 2, self.n_elec)
        pr = np.zeros(self.n_elec)
        pz = np.zeros(self.n_elec)
        gamma = np.ones(self.n_elec)
        q = self.dr_p * r + self.dr_p * self.parabolic_coefficient * r**3
        m_e = np.ones(self.n_elec)
        m_i = np.ones(self.n_elec) * ct.m_p / ct.m_e
        q_species_e = np.ones(self.n_elec)
        q_species_i = np.ones(self.n_elec) * -1

        self.r = np.concatenate((r, r))
        self.pr = np.concatenate((pr, pr))
        self.pz = np.concatenate((pz, pz))
        self.gamma = np.concatenate((gamma, gamma))
        self.q = np.concatenate((q, -q))
        self.q_species = np.concatenate((q_species_e, q_species_i))
        self.m = np.concatenate((m_e, m_i))

        self.r_elec = self.r[:self.n_elec]
        self.pr_elec = self.pr[:self.n_elec]
        self.pz_elec = self.pz[:self.n_elec]
        self.gamma_elec = self.gamma[:self.n_elec]
        self.q_elec = self.q[:self.n_elec]
        self.q_species_elec = self.q_species[:self.n_elec]
        self.m_elec = self.m[:self.n_elec]

        self.r_ion = self.r[self.n_elec:]
        self.pr_ion = self.pr[self.n_elec:]
        self.pz_ion = self.pz[self.n_elec:]
        self.gamma_ion = self.gamma[self.n_elec:]
        self.q_ion = self.q[self.n_elec:]
        self.q_species_ion = self.q_species[self.n_elec:]
        self.m_ion = self.m[self.n_elec:]

        self.ions_computed = False

        # Allocate arrays that will contain the fields experienced by the
        # particles.
        self._allocate_field_arrays()

        # Allocate arrays needed for the particle pusher.
        if self.pusher == 'ab5':
            self._allocate_ab5_arrays()
        elif self.pusher == 'rk4':
            self._allocate_rk4_arrays()
            self._allocate_rk4_field_arrays()

    def sort(self):
        self.i_sort_e = np.argsort(self.r_elec, kind='stable')
        if self.ion_motion or not self.ions_computed:
            self.i_sort_i = np.argsort(self.r_ion, kind='stable')

    def _allocate_field_arrays(self):
        """Allocate arrays for the fields experienced by the particles.

        In order to evolve the particles to the next longitudinal position,
        it is necessary to know the fields that they are experiencing. These
        arrays are used for storing the value of these fields at the location
        of each particle.
        """
        self._a2 = np.zeros(self.n_part)
        self._nabla_a2 = np.zeros(self.n_part)
        self._b_t_0 = np.zeros(self.n_part)
        self._b_t = np.zeros(self.n_part)
        self._psi = np.zeros(self.n_part)
        self._dr_psi = np.zeros(self.n_part)
        self._dxi_psi = np.zeros(self.n_part)
        self._psi_e = self._psi[:self.n_elec]
        self._dr_psi_e = self._dr_psi[:self.n_elec]
        self._dxi_psi_e = self._dxi_psi[:self.n_elec]
        self._psi_i = self._psi[self.n_elec:]
        self._dr_psi_i = self._dr_psi[self.n_elec:]
        self._dxi_psi_i = self._dxi_psi[self.n_elec:]
        self._b_t_e = self._b_t[:self.n_elec]
        self._b_t_i = self._b_t[self.n_elec:]
        self._b_t_0_e = self._b_t_0[:self.n_elec]
        self._b_t_0_i = self._b_t_0[self.n_elec:]
        self._nabla_a2_e = self._nabla_a2[:self.n_elec]
        self._nabla_a2_i = self._nabla_a2[self.n_elec:]
        self._a2_e = self._a2[:self.n_elec]
        self._a2_i = self._a2[self.n_elec:]
        self._sum_1 = np.zeros(self.n_part)
        self._sum_2 = np.zeros(self.n_part)
        self._sum_3 = np.zeros(self.n_part)
        self._sum_1_e = np.zeros(self.n_elec)
        self._sum_2_e = np.zeros(self.n_elec)
        self._sum_3_e = np.zeros(self.n_elec)
        self._sum_1_i = np.zeros(self.n_elec)
        self._sum_2_i = np.zeros(self.n_elec)
        self._sum_3_i = np.zeros(self.n_elec)
        # self._sum_1_e = self._sum_1[:self.n_elec]
        # self._sum_2_e = self._sum_2[:self.n_elec]
        # self._sum_1_i = self._sum_1[self.n_elec:]
        # self._sum_2_i = self._sum_2[self.n_elec:]
        self._rho = np.zeros(self.n_part)
        self._psi_bg_grid_e = np.zeros(self.nr + 4)
        self._dr_psi_bg_grid_e = np.zeros(self.nr + 4)
        self._psi_bg_grid_i = np.zeros(self.nr + 4)
        self._dr_psi_bg_grid_i = np.zeros(self.nr + 4)        
        self._psi_bg_e = np.zeros(self.n_elec+1)
        self._dr_psi_bg_e = np.zeros(self.n_elec+1)
        self._dxi_psi_bg_e = np.zeros(self.n_elec+1)
        self._psi_bg_i = np.zeros(self.n_elec+1)
        self._dr_psi_bg_i = np.zeros(self.n_elec+1)
        self._dxi_psi_bg_i = np.zeros(self.n_elec+1)
        self._chi = np.zeros(self.n_part)
        self._a_0 = np.zeros(1)
        self._a_i = np.zeros(self.n_part)
        self._b_i = np.zeros(self.n_part)
        self._a_i_e = self._a_i[:self.n_elec]
        self._b_i_e = self._b_i[:self.n_elec]
        self._a_i_i = self._a_i[self.n_elec:]
        self._b_i_i = self._b_i[self.n_elec:]
        self._i_left = np.zeros(self.n_part, dtype=np.int)
        self._i_right = np.zeros(self.n_part, dtype=np.int)
        self._r_neighbor_e = np.zeros(self.n_elec+1)
        self._r_neighbor_i = np.zeros(self.n_elec+1)

        self._r_max = np.zeros(1)
        self._psi_max = np.zeros(1)
        self._dxi_psi_max = np.zeros(1)

        self._field_arrays = [
            self._a2, self._nabla_a2, self._b_t_0, self._b_t,
            self._psi, self._dr_psi, self._dxi_psi
        ]

    def _allocate_ab5_arrays(self):
        """Allocate the arrays needed for the 5th order Adams-Bashforth pusher.

        The AB5 pusher needs the derivatives of r and pr for each particle
        at the last 5 plasma slices. This method allocates the arrays that will
        store these derivatives.
        """
        if self.ion_motion:
            size = self.n_part
        else:
            size = self.n_elec
        self._dr_1 = np.zeros(size)
        self._dr_2 = np.zeros(size)
        self._dr_3 = np.zeros(size)
        self._dr_4 = np.zeros(size)
        self._dr_5 = np.zeros(size)
        self._dpr_1 = np.zeros(size)
        self._dpr_2 = np.zeros(size)
        self._dpr_3 = np.zeros(size)
        self._dpr_4 = np.zeros(size)
        self._dpr_5 = np.zeros(size)
        self._dr_arrays = [
            self._dr_1, self._dr_2, self._dr_3, self._dr_4, self._dr_5]
        self._dpr_arrays = [
            self._dpr_1, self._dpr_2, self._dpr_3, self._dpr_4,
            self._dpr_5]

    def _allocate_rk4_arrays(self):
        """Allocate the arrays needed for the 4th order Runge-Kutta pusher.

        The RK4 pusher needs the derivatives of r and pr for each particle at
        the current slice and at 3 intermediate substeps. This method allocates
        the arrays that will store these derivatives.
        """
        self._dr_1 = np.zeros(self.n_part)
        self._dr_2 = np.zeros(self.n_part)
        self._dr_3 = np.zeros(self.n_part)
        self._dr_4 = np.zeros(self.n_part)
        self._dpr_1 = np.zeros(self.n_part)
        self._dpr_2 = np.zeros(self.n_part)
        self._dpr_3 = np.zeros(self.n_part)
        self._dpr_4 = np.zeros(self.n_part)
        self._dr_arrays = [self._dr_1, self._dr_2, self._dr_3, self._dr_4]
        self._dpr_arrays = [
            self._dpr_1, self._dpr_2, self._dpr_3, self._dpr_4]

    def _allocate_rk4_field_arrays(self):
        """Allocate field arrays needed by the 4th order Runge-Kutta pusher.

        In order to compute the derivatives of r and pr at the 3 subteps
        of the RK4 pusher, the field values at the location of the particles
        in these substeps are needed. This method allocates the arrays
        that will store these field values.
        """
        self._a2_2 = np.zeros(self.n_part)
        self._nabla_a2_2 = np.zeros(self.n_part)
        self._b_t_0_2 = np.zeros(self.n_part)
        self._b_t_2 = np.zeros(self.n_part)
        self._psi_2 = np.zeros(self.n_part)
        self._dr_psi_2 = np.zeros(self.n_part)
        self._dxi_psi_2 = np.zeros(self.n_part)
        self._a2_3 = np.zeros(self.n_part)
        self._nabla_a2_3 = np.zeros(self.n_part)
        self._b_t_0_3 = np.zeros(self.n_part)
        self._b_t_3 = np.zeros(self.n_part)
        self._psi_3 = np.zeros(self.n_part)
        self._dr_psi_3 = np.zeros(self.n_part)
        self._dxi_psi_3 = np.zeros(self.n_part)
        self._a2_4 = np.zeros(self.n_part)
        self._nabla_a2_4 = np.zeros(self.n_part)
        self._b_t_0_4 = np.zeros(self.n_part)
        self._b_t_4 = np.zeros(self.n_part)
        self._psi_4 = np.zeros(self.n_part)
        self._dr_psi_4 = np.zeros(self.n_part)
        self._dxi_psi_4 = np.zeros(self.n_part)
        self._rk4_flds = [
            [self._a2, self._nabla_a2, self._b_t_0, self._b_t,
             self._psi, self._dr_psi, self._dxi_psi],
            [self._a2_2, self._nabla_a2_2, self._b_t_0_2, self._b_t_2,
             self._psi_2, self._dr_psi_2, self._dxi_psi_2],
            [self._a2_3, self._nabla_a2_3, self._b_t_0_3, self._b_t_3,
             self._psi_3, self._dr_psi_3, self._dxi_psi_3],
            [self._a2_4, self._nabla_a2_4, self._b_t_0_4, self._b_t_4,
             self._psi_4, self._dr_psi_4, self._dxi_psi_4]
        ]
    
    def deposit_rho(self, rho, r_fld, nr, dr):
        w_rho = self.q / (1 - self.pz/self.gamma)
        deposit_plasma_particles(self.r, w_rho, r_fld[0], nr, dr, rho, self.shape)
        rho[2: -2] /= r_fld * dr

    def deposit_rho_e(self, rho, r_fld, nr, dr):
        w_rho = self.q_elec / (1 - self.pz_elec/self.gamma_elec)
        deposit_plasma_particles(self.r_elec, w_rho, r_fld[0], nr, dr, rho, self.shape)
        rho[2: -2] /= r_fld * dr

    def deposit_rho_i(self, rho, r_fld, nr, dr):
        w_rho = self.q_ion / ((1 - self.pz_ion/self.gamma_ion))
        deposit_plasma_particles(self.r_ion, w_rho, r_fld[0], nr, dr, rho, self.shape)
        rho[2: -2] /= r_fld * dr

    def gather_particle_background(self):
        calculate_psi_and_dr_psi(
            self._r_neighbor_e, self._log_r_neighbor_e, self.r_ion, self.dr_p, self.i_sort_i,
            self._sum_1_i, self._sum_2_i, self._psi_bg_i, self._dr_psi_bg_i)
        if self.ion_motion:
            calculate_psi_and_dr_psi(
                self._r_neighbor_i, self._log_r_neighbor_i, self.r_elec, self.dr_p, self.i_sort_e,
                self._sum_1_e, self._sum_2_e, self._psi_bg_e,
                self._dr_psi_bg_e)

    def gather_particle_background_dxi_psi(self):
        calculate_dxi_psi(
            self._r_neighbor_e, self.r_ion, self.i_sort_i, self._sum_3_i,
            self._dxi_psi_bg_i)
        if self.ion_motion:
            calculate_dxi_psi(
                self._r_neighbor_i, self.r_elec, self.i_sort_e, self._sum_3_e,
                self._dxi_psi_bg_e)

    def deposit_chi(self, chi, r_fld, nr, dr):
        w_chi = self.q_elec / ((1 - self.pz_elec/self.gamma_elec)) / self.gamma_elec
        # w_chi = self.q / (self.dr * self.r * (1 - self.pz/self.gamma)) / (self.gamma * self.m)
        # w_chi = w_chi[:self.n_elec]
        # r_elec = self.r[:self.n_elec]
        deposit_plasma_particles(self.r_elec, w_chi, r_fld[0], nr, dr, chi, self.shape)
        chi[2: -2] /= r_fld * dr

    def gather_sources(self, a2, nabla_a2, b_theta, r_min, r_max, dr):
        if self.ion_motion:
            gather_sources(a2, nabla_a2, b_theta, r_min, r_max, dr, self.r,
                           self._a2, self._nabla_a2, self._b_t_0)
        else:
            gather_sources(a2, nabla_a2, b_theta, r_min, r_max, dr, self.r_elec,
                           self._a2_e, self._nabla_a2_e, self._b_t_0_e)


    def calculate_cumulative_sums(self):
        # calculate_cumulative_sums(self.r_elec, self.q_elec, self.i_sort_e,
        #                           self._sum_1_e, self._sum_2_e)
        calculate_cumulative_sum_1(self.q_elec, self.i_sort_e, self._sum_1_e)
        calculate_cumulative_sum_2(self.r_elec, self.q_elec, self.i_sort_e, self._sum_2_e)
        if self.ion_motion or not self.ions_computed:
            # calculate_cumulative_sums(self.r_ion, self.q_ion, self.i_sort_i,
            #                         self._sum_1_i, self._sum_2_i)            
            calculate_cumulative_sum_1(self.q_ion, self.i_sort_i, self._sum_1_i)
            calculate_cumulative_sum_2(self.r_ion, self.q_ion, self.i_sort_i, self._sum_2_i)
        
    def calculate_cumulative_sum_3(self):
        calculate_cumulative_sum_3(
            self.r_elec, self.pr_elec, self.q_elec, self._psi_e, self.i_sort_e,
            self._sum_3_e)
        if self.ion_motion or not self.ions_computed:
            calculate_cumulative_sum_3(
            self.r_ion, self.pr_ion, self.q_ion, self._psi_i, self.i_sort_i,
            self._sum_3_i)

    def calculate_ai_bi(self):
        calculate_ai_bi_from_axis(
            self.r_elec, self.pr_elec, self.q_elec, self.gamma_elec,
            self._psi_e, self._dr_psi_e, self._dxi_psi_e, self._b_t_0_e,
            self._nabla_a2_e, self.i_sort_e, self._a_0, self._a_i_e,
            self._b_i_e)
        
    def calculate_psi_dr_psi(self):
        calculate_psi_dr_psi_at_particles_bg(
            self.r_elec, self._sum_1_e, self._sum_2_e,
            self._psi_bg_i, self._r_neighbor_e, self._log_r_neighbor_e,
            self.i_sort_e, self._psi_e, self._dr_psi_e)
        if self.ion_motion:
            calculate_psi_dr_psi_at_particles_bg(
                self.r_ion, self._sum_1_i, self._sum_2_i,
                self._psi_bg_e, self._r_neighbor_i, self._log_r_neighbor_e,
                self.i_sort_i, self._psi_i, self._dr_psi_i)

        # self._i_max = np.argmax(self.r)
        # self._psi_max = self._psi[self._i_max]
        # self._psi -= self._psi_max

        

        r_max_e = self.r_elec[self.i_sort_e[-1]]
        r_max_i = self.r_ion[self.i_sort_i[-1]]
        self._r_max[:] = max(r_max_e, r_max_i) + self.dr_p/2
        log_r_max = np.log(self._r_max)

        self._psi_max[:] = 0.

        calculate_psi(self._r_max, log_r_max, self.r_elec, self._sum_1_e, self._sum_2_e, self.i_sort_e, self._psi_max)
        calculate_psi(self._r_max, log_r_max, self.r_ion, self._sum_1_i, self._sum_2_i, self.i_sort_i, self._psi_max)

        self._psi_e -= self._psi_max
        if self.ion_motion:
            self._psi_i -= self._psi_max
        
        self._psi[self._psi < -0.9] = -0.9
        if np.max(self._dr_psi_e) > 1:
            print(np.abs(np.max(self._dr_psi)))     

    def calculate_dxi_psi(self):            
        calculate_dxi_psi_at_particles_bg(
            self.r_elec, self._sum_3_e, self._dxi_psi_bg_i, self._r_neighbor_e,
            self.i_sort_e, self._dxi_psi_e)
        if self.ion_motion:
            calculate_dxi_psi_at_particles_bg(
                self.r_ion, self._sum_3_i, self._dxi_psi_bg_e, self._r_neighbor_i,
                self.i_sort_i, self._dxi_psi_i)

        # Apply boundary condition (dxi_psi = 0 after last particle).
        self._dxi_psi += (self._sum_3_e[self.i_sort_e[-1]] +
                          self._sum_3_i[self.i_sort_i[-1]])

        self._dxi_psi[self._dxi_psi < -3.] = -3.
        self._dxi_psi[self._dxi_psi > 3.] = 3.

    def calculate_b_theta(self):
        calculate_b_theta_at_particles(
            self.r_elec, self._a_0[0], self._a_i_e, self._b_i_e,
            self._r_neighbor_e, self.i_sort_e, self._b_t_e)
        if self.ion_motion:
            # calculate_b_theta_at_particles(
            #     self.r_ion, self._a_0, self._a_i_e, self._b_i_e,
            #     self._r_neighbor_i, self.i_sort_i, self._b_t_i)
            calculate_b_theta_at_ions(
                self.r_ion, self.r_elec, self._a_0[0], self._a_i_e, self._b_i_e,
                self.i_sort_i, self.i_sort_e, self._b_t_i)

    def calculate_psi_grid(self, r_eval, log_r_eval, psi):
        calculate_psi(r_eval, log_r_eval, self.r_elec, self._sum_1_e, self._sum_2_e, self.i_sort_e, psi)
        calculate_psi(r_eval, log_r_eval, self.r_ion, self._sum_1_i, self._sum_2_i, self.i_sort_i, psi)
        psi -= self._psi_max

    def calculate_b_theta_grid(self, r_eval, b_theta):
        calculate_b_theta(r_eval, self._a_0[0], self._a_i_e, self._b_i_e,
                          self.r_elec, self.i_sort_e, b_theta)

    def evolve(self, dxi):
        if self.ion_motion:
            evolve_plasma_ab5(dxi, self.r, self.pr, self.gamma, self.m, self.q_species, self._nabla_a2,
                            self._b_t_0, self._b_t, self._psi, self._dr_psi, 
                            self._dr_arrays, self._dpr_arrays)
        else:
            evolve_plasma_ab5(dxi, self.r_elec, self.pr_elec, self.gamma_elec, self.m_elec, self.q_species_elec, self._nabla_a2_e,
                            self._b_t_0_e, self._b_t_e, self._psi_e, self._dr_psi_e, 
                            self._dr_arrays, self._dpr_arrays)

        
    def update_gamma_pz(self):
        if self.ion_motion:
            update_gamma_and_pz(
                self.gamma, self.pz, self.pr,
                self._a2, self._psi, self.q_species, self.m)
        else:
            update_gamma_and_pz(
                self.gamma_elec, self.pz_elec, self.pr_elec,
                self._a2_e, self._psi_e, self.q_species_elec, self.m_elec)
            if np.max(self.pz_elec/self.gamma_elec) > 0.999:
                print('p'+str(np.max(self.pz_elec/self.gamma_elec)))
        
    def determine_neighboring_points(self):
        determine_neighboring_points(
            self.r_elec, self.dr_p, self.i_sort_e, self._r_neighbor_e)
        self._log_r_neighbor_e = np.log(self._r_neighbor_e)
        if self.ion_motion:
            determine_neighboring_points(
                self.r_ion, self.dr_p, self.i_sort_i, self._r_neighbor_i)
            self._log_r_neighbor_i = np.log(self._r_neighbor_i)


def radial_integral(f_r):
    subs = f_r / 2
    subs += f_r[0]/4
    return (np.cumsum(f_r) - subs)


@njit_serial()
def update_gamma_and_pz(gamma, pz, pr, a2, psi, q, m):
    """
    Update the gamma factor and longitudinal momentum of the plasma particles.

    Parameters
    ----------
    gamma, pz : ndarray
        Arrays containing the current gamma factor and longitudinal momentum
        of the plasma particles (will be modified here).
    pr, a2, psi : ndarray
        Arrays containing the radial momentum of the particles and the
        value of a2 and psi at the position of the particles.

    """
    for i in range(pr.shape[0]):
        q_over_m = q[i] / m[i]
        psi_i = psi[i] * q_over_m
        pz_i = (1 + pr[i]**2 + q_over_m**2 * a2[i] - (1+psi_i)**2) / (2 * (1+psi_i))
        pz[i] = pz_i
        gamma[i] = 1. + pz_i + psi_i
