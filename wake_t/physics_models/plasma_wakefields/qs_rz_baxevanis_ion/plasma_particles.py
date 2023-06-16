"""Contains the definition of the `PlasmaParticles` class."""

import numpy as np
import scipy.constants as ct
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba.core.types import float64, int64, string, boolean

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
                      calculate_b_theta_at_ions,
                      calculate_ABC, calculate_KU)
from .plasma_push.ab5 import evolve_plasma_ab5


spec = [
    ('dr_p', float64),
    ('r', float64[::1]),
    ('pr', float64[::1]),
    ('pz', float64[::1]),
    ('gamma', float64[::1]),
    ('q', float64[::1]),
    ('q_species', float64[::1]),
    ('m', float64[::1]),
    ('r_elec', float64[::1]),
    ('pr_elec', float64[::1]),
    ('q_elec', float64[::1]),
    ('gamma_elec', float64[::1]),
    ('pz_elec', float64[::1]),
    ('q_species_elec', float64[::1]),
    ('m_elec', float64[::1]),
    ('i_sort_e', int64[::1]),
    ('r_ion', float64[::1]),
    ('pr_ion', float64[::1]),
    ('q_ion', float64[::1]),
    ('gamma_ion', float64[::1]),
    ('pz_ion', float64[::1]),
    ('q_species_ion', float64[::1]),
    ('m_ion', float64[::1]),
    ('i_sort_i', int64[::1]),
    ('_a2', float64[::1]),
    ('_nabla_a2', float64[::1]),
    ('_b_t_0', float64[::1]),
    ('_b_t', float64[::1]),
    ('_a2_e', float64[::1]),
    ('_b_t_0_e', float64[::1]),
    ('_nabla_a2_e', float64[::1]),
    ('_r_neighbor_e', float64[::1]),
    ('_r_neighbor_i', float64[::1]),
    ('_log_r_neighbor_e', float64[::1]),
    ('_log_r_neighbor_i', float64[::1]),
    ('_sum_1_e', float64[::1]),
    ('_sum_2_e', float64[::1]),
    ('_sum_3_e', float64[::1]),
    ('_psi_bg_e', float64[::1]),
    ('_dr_psi_bg_e', float64[::1]),
    ('_dxi_psi_bg_e', float64[::1]),
    ('_psi_e', float64[::1]),
    ('_dr_psi_e', float64[::1]),
    ('_dxi_psi_e', float64[::1]),
    ('_b_t_e', float64[::1]),
    ('_A', float64[::1]),
    ('_B', float64[::1]),
    ('_C', float64[::1]),
    ('_K', float64[::1]),
    ('_U', float64[::1]),
    ('_a_i', float64[::1]),
    ('_b_i', float64[::1]),
    ('_a_0', float64[::1]),
    ('_sum_1_i', float64[::1]),
    ('_sum_2_i', float64[::1]),
    ('_sum_3_i', float64[::1]),
    ('_psi_bg_i', float64[::1]),
    ('_dr_psi_bg_i', float64[::1]),
    ('_dxi_psi_bg_i', float64[::1]),
    ('_psi_i', float64[::1]),
    ('_dr_psi_i', float64[::1]),
    ('_dxi_psi_i', float64[::1]),
    ('_b_t_i', float64[::1]),
    ('_psi', float64[::1]),
    ('_dr_psi', float64[::1]),
    ('_dxi_psi', float64[::1]),
    ('ion_motion', boolean),
    ('ions_computed', boolean),
    ('_r_max', float64[::1]),
    ('_psi_max', float64[::1]),
    ('nr', int64),
    ('dr', float64),
    ('shape', string),
    ('pusher', string),
    ('r_max', float64),
    ('r_max_plasma', float64),
    ('parabolic_coefficient', float64),
    ('ppc', int64),
    ('n_elec', int64),
    ('n_part', int64),
    ('_dr', float64[:, ::1]),
    ('_dpr', float64[:, ::1]),
]

@jitclass(spec)
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
                 nr, ion_motion=True, pusher='ab5', shape='linear'):
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
        # self.r_grid = r_grid
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
        # elif self.pusher == 'rk4':
        #     self._allocate_rk4_arrays()
        #     self._allocate_rk4_field_arrays()

    def sort(self):
        self.i_sort_e = np.argsort(self.r_elec)
        if self.ion_motion or not self.ions_computed:
            self.i_sort_i = np.argsort(self.r_ion)

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
        # self._b_t_0_i = self._b_t_0[self.n_elec:]
        self._nabla_a2_e = self._nabla_a2[:self.n_elec]
        # self._nabla_a2_i = self._nabla_a2[self.n_elec:]
        self._a2_e = self._a2[:self.n_elec]
        # self._a2_i = self._a2[self.n_elec:]
        # self._sum_1 = np.zeros(self.n_part)
        # self._sum_2 = np.zeros(self.n_part)
        # self._sum_3 = np.zeros(self.n_part)
        self._sum_1_e = np.zeros(self.n_elec)
        self._sum_2_e = np.zeros(self.n_elec)
        self._sum_3_e = np.zeros(self.n_elec)
        self._sum_1_i = np.zeros(self.n_elec)
        self._sum_2_i = np.zeros(self.n_elec)
        self._sum_3_i = np.zeros(self.n_elec)
        # self._rho = np.zeros(self.n_part)
        # self._psi_bg_grid_e = np.zeros(self.nr + 4)
        # self._dr_psi_bg_grid_e = np.zeros(self.nr + 4)
        # self._psi_bg_grid_i = np.zeros(self.nr + 4)
        # self._dr_psi_bg_grid_i = np.zeros(self.nr + 4)        
        self._psi_bg_e = np.zeros(self.n_elec+1)
        self._dr_psi_bg_e = np.zeros(self.n_elec+1)
        self._dxi_psi_bg_e = np.zeros(self.n_elec+1)
        self._psi_bg_i = np.zeros(self.n_elec+1)
        self._dr_psi_bg_i = np.zeros(self.n_elec+1)
        self._dxi_psi_bg_i = np.zeros(self.n_elec+1)
        # self._chi = np.zeros(self.n_part)
        self._a_0 = np.zeros(1)
        self._a_i = np.zeros(self.n_elec)
        self._b_i = np.zeros(self.n_elec)
        self._A = np.zeros(self.n_elec)
        self._B = np.zeros(self.n_elec)
        self._C = np.zeros(self.n_elec)
        self._K = np.zeros(self.n_elec)
        self._U = np.zeros(self.n_elec)
        # self._i_left = np.zeros(self.n_part, dtype=np.int)
        # self._i_right = np.zeros(self.n_part, dtype=np.int)
        self._r_neighbor_e = np.zeros(self.n_elec+1)
        self._r_neighbor_i = np.zeros(self.n_elec+1)

        self._r_max = np.zeros(1)
        self._psi_max = np.zeros(1)
        # self._dxi_psi_max = np.zeros(1)

        # self._field_arrays = [
        #     self._a2, self._nabla_a2, self._b_t_0, self._b_t,
        #     self._psi, self._dr_psi, self._dxi_psi
        # ]

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

        self._dr = np.zeros((5, size))
        self._dpr = np.zeros((5, size))

    # def _allocate_rk4_arrays(self):
    #     """Allocate the arrays needed for the 4th order Runge-Kutta pusher.

    #     The RK4 pusher needs the derivatives of r and pr for each particle at
    #     the current slice and at 3 intermediate substeps. This method allocates
    #     the arrays that will store these derivatives.
    #     """
    #     self._dr_1 = np.zeros(self.n_part)
    #     self._dr_2 = np.zeros(self.n_part)
    #     self._dr_3 = np.zeros(self.n_part)
    #     self._dr_4 = np.zeros(self.n_part)
    #     self._dpr_1 = np.zeros(self.n_part)
    #     self._dpr_2 = np.zeros(self.n_part)
    #     self._dpr_3 = np.zeros(self.n_part)
    #     self._dpr_4 = np.zeros(self.n_part)
    #     self._dr_arrays = [self._dr_1, self._dr_2, self._dr_3, self._dr_4]
    #     self._dpr_arrays = [
    #         self._dpr_1, self._dpr_2, self._dpr_3, self._dpr_4]

    # def _allocate_rk4_field_arrays(self):
    #     """Allocate field arrays needed by the 4th order Runge-Kutta pusher.

    #     In order to compute the derivatives of r and pr at the 3 subteps
    #     of the RK4 pusher, the field values at the location of the particles
    #     in these substeps are needed. This method allocates the arrays
    #     that will store these field values.
    #     """
    #     self._a2_2 = np.zeros(self.n_part)
    #     self._nabla_a2_2 = np.zeros(self.n_part)
    #     self._b_t_0_2 = np.zeros(self.n_part)
    #     self._b_t_2 = np.zeros(self.n_part)
    #     self._psi_2 = np.zeros(self.n_part)
    #     self._dr_psi_2 = np.zeros(self.n_part)
    #     self._dxi_psi_2 = np.zeros(self.n_part)
    #     self._a2_3 = np.zeros(self.n_part)
    #     self._nabla_a2_3 = np.zeros(self.n_part)
    #     self._b_t_0_3 = np.zeros(self.n_part)
    #     self._b_t_3 = np.zeros(self.n_part)
    #     self._psi_3 = np.zeros(self.n_part)
    #     self._dr_psi_3 = np.zeros(self.n_part)
    #     self._dxi_psi_3 = np.zeros(self.n_part)
    #     self._a2_4 = np.zeros(self.n_part)
    #     self._nabla_a2_4 = np.zeros(self.n_part)
    #     self._b_t_0_4 = np.zeros(self.n_part)
    #     self._b_t_4 = np.zeros(self.n_part)
    #     self._psi_4 = np.zeros(self.n_part)
    #     self._dr_psi_4 = np.zeros(self.n_part)
    #     self._dxi_psi_4 = np.zeros(self.n_part)
    #     self._rk4_flds = [
    #         [self._a2, self._nabla_a2, self._b_t_0, self._b_t,
    #          self._psi, self._dr_psi, self._dxi_psi],
    #         [self._a2_2, self._nabla_a2_2, self._b_t_0_2, self._b_t_2,
    #          self._psi_2, self._dr_psi_2, self._dxi_psi_2],
    #         [self._a2_3, self._nabla_a2_3, self._b_t_0_3, self._b_t_3,
    #          self._psi_3, self._dr_psi_3, self._dxi_psi_3],
    #         [self._a2_4, self._nabla_a2_4, self._b_t_0_4, self._b_t_4,
    #          self._psi_4, self._dr_psi_4, self._dxi_psi_4]
    #     ]

# @njit_serial()
# def deposit_rho_pp(pp, rho, r_fld, nr, dr):
#     w_rho = pp.q / (1 - pp.pz/pp.gamma)
#     deposit_plasma_particles(pp.r, w_rho, r_fld[0], nr, dr, rho, pp.shape)
#     rho[2: -2] /= r_fld * dr

@njit_serial()
def deposit_rho_e_pp(pp, rho, r_fld, nr, dr):
    w_rho = pp.q_elec / (1 - pp.pz_elec/pp.gamma_elec)
    deposit_plasma_particles(pp.r_elec, w_rho, r_fld[0], nr, dr, rho, pp.shape)
    rho[2: -2] /= r_fld * dr

@njit_serial()
def deposit_rho_i_pp(pp, rho, r_fld, nr, dr):
    w_rho = pp.q_ion / ((1 - pp.pz_ion/pp.gamma_ion))
    deposit_plasma_particles(pp.r_ion, w_rho, r_fld[0], nr, dr, rho, pp.shape)
    rho[2: -2] /= r_fld * dr

@njit_serial()
def gather_particle_background_pp(pp):
    calculate_psi_and_dr_psi(
        pp._r_neighbor_e, pp._log_r_neighbor_e, pp.r_ion, pp.dr_p, pp.i_sort_i,
        pp._sum_1_i, pp._sum_2_i, pp._psi_bg_i, pp._dr_psi_bg_i)
    if pp.ion_motion:
        calculate_psi_and_dr_psi(
            pp._r_neighbor_i, pp._log_r_neighbor_i, pp.r_elec, pp.dr_p, pp.i_sort_e,
            pp._sum_1_e, pp._sum_2_e, pp._psi_bg_e,
            pp._dr_psi_bg_e)

@njit_serial()
def gather_particle_background_dxi_psi_pp(pp):
    calculate_dxi_psi(
        pp._r_neighbor_e, pp.r_ion, pp.i_sort_i, pp._sum_3_i,
        pp._dxi_psi_bg_i)
    if pp.ion_motion:
        calculate_dxi_psi(
            pp._r_neighbor_i, pp.r_elec, pp.i_sort_e, pp._sum_3_e,
            pp._dxi_psi_bg_e)

@njit_serial()
def deposit_chi_pp(pp, chi, r_fld, nr, dr):
    w_chi = pp.q_elec / ((1 - pp.pz_elec/pp.gamma_elec)) / pp.gamma_elec
    # w_chi = pp.q / (pp.dr * pp.r * (1 - pp.pz/pp.gamma)) / (pp.gamma * pp.m)
    # w_chi = w_chi[:pp.n_elec]
    # r_elec = pp.r[:pp.n_elec]
    deposit_plasma_particles(pp.r_elec, w_chi, r_fld[0], nr, dr, chi, pp.shape)
    chi[2: -2] /= r_fld * dr

@njit_serial()
def gather_sources_pp(pp, a2, nabla_a2, b_theta, r_min, r_max, dr):
    if pp.ion_motion:
        gather_sources(a2, nabla_a2, b_theta, r_min, r_max, dr, pp.r,
                        pp._a2, pp._nabla_a2, pp._b_t_0)
    else:
        gather_sources(a2, nabla_a2, b_theta, r_min, r_max, dr, pp.r_elec,
                        pp._a2_e, pp._nabla_a2_e, pp._b_t_0_e)


@njit_serial()
def calculate_cumulative_sums_pp(pp):
    # calculate_cumulative_sums(pp.r_elec, pp.q_elec, pp.i_sort_e,
    #                           pp._sum_1_e, pp._sum_2_e)
    calculate_cumulative_sum_1(pp.q_elec, pp.i_sort_e, pp._sum_1_e)
    calculate_cumulative_sum_2(pp.r_elec, pp.q_elec, pp.i_sort_e, pp._sum_2_e)
    if pp.ion_motion or not pp.ions_computed:
        # calculate_cumulative_sums(pp.r_ion, pp.q_ion, pp.i_sort_i,
        #                         pp._sum_1_i, pp._sum_2_i)            
        calculate_cumulative_sum_1(pp.q_ion, pp.i_sort_i, pp._sum_1_i)
        calculate_cumulative_sum_2(pp.r_ion, pp.q_ion, pp.i_sort_i, pp._sum_2_i)

@njit_serial()
def calculate_cumulative_sum_3_pp(pp):
    calculate_cumulative_sum_3(
        pp.r_elec, pp.pr_elec, pp.q_elec, pp._psi_e, pp.i_sort_e,
        pp._sum_3_e)
    if pp.ion_motion or not pp.ions_computed:
        calculate_cumulative_sum_3(
        pp.r_ion, pp.pr_ion, pp.q_ion, pp._psi_i, pp.i_sort_i,
        pp._sum_3_i)

@njit_serial()
def calculate_ai_bi_pp(pp):
    calculate_ABC(
        pp.r_elec, pp.pr_elec, pp.q_elec, pp.gamma_elec,
        pp._psi_e, pp._dr_psi_e, pp._dxi_psi_e, pp._b_t_0_e,
        pp._nabla_a2_e, pp.i_sort_e, pp._A, pp._B, pp._C)
    calculate_KU(pp.r_elec, pp._A, pp.i_sort_e, pp._K, pp._U)
    calculate_ai_bi_from_axis(
        pp.r_elec, pp._A, pp._B, pp._C, pp._K, pp._U, pp.i_sort_e, pp._a_0, pp._a_i,
        pp._b_i)
    
@njit_serial()
def calculate_psi_dr_psi_pp(pp):
    calculate_psi_dr_psi_at_particles_bg(
        pp.r_elec, pp._sum_1_e, pp._sum_2_e,
        pp._psi_bg_i, pp._r_neighbor_e, pp._log_r_neighbor_e,
        pp.i_sort_e, pp._psi_e, pp._dr_psi_e)
    if pp.ion_motion:
        calculate_psi_dr_psi_at_particles_bg(
            pp.r_ion, pp._sum_1_i, pp._sum_2_i,
            pp._psi_bg_e, pp._r_neighbor_i, pp._log_r_neighbor_e,
            pp.i_sort_i, pp._psi_i, pp._dr_psi_i)

    # pp._i_max = np.argmax(pp.r)
    # pp._psi_max = pp._psi[pp._i_max]
    # pp._psi -= pp._psi_max

    

    r_max_e = pp.r_elec[pp.i_sort_e[-1]]
    r_max_i = pp.r_ion[pp.i_sort_i[-1]]
    pp._r_max[:] = max(r_max_e, r_max_i) + pp.dr_p/2
    log_r_max = np.log(pp._r_max)

    pp._psi_max[:] = 0.

    calculate_psi(pp._r_max, log_r_max, pp.r_elec, pp._sum_1_e, pp._sum_2_e, pp.i_sort_e, pp._psi_max)
    calculate_psi(pp._r_max, log_r_max, pp.r_ion, pp._sum_1_i, pp._sum_2_i, pp.i_sort_i, pp._psi_max)

    pp._psi_e -= pp._psi_max
    if pp.ion_motion:
        pp._psi_i -= pp._psi_max
    
    pp._psi[pp._psi < -0.9] = -0.9
    # if np.max(pp._dr_psi_e) > 1:
    #     print(np.abs(np.max(pp._dr_psi)))     

@njit_serial()
def calculate_dxi_psi_pp(pp):            
    calculate_dxi_psi_at_particles_bg(
        pp.r_elec, pp._sum_3_e, pp._dxi_psi_bg_i, pp._r_neighbor_e,
        pp.i_sort_e, pp._dxi_psi_e)
    if pp.ion_motion:
        calculate_dxi_psi_at_particles_bg(
            pp.r_ion, pp._sum_3_i, pp._dxi_psi_bg_e, pp._r_neighbor_i,
            pp.i_sort_i, pp._dxi_psi_i)

    # Apply boundary condition (dxi_psi = 0 after last particle).
    pp._dxi_psi += (pp._sum_3_e[pp.i_sort_e[-1]] +
                        pp._sum_3_i[pp.i_sort_i[-1]])

    pp._dxi_psi[pp._dxi_psi < -3.] = -3.
    pp._dxi_psi[pp._dxi_psi > 3.] = 3.

@njit_serial()
def calculate_b_theta_pp(pp):
    calculate_b_theta_at_particles(
        pp.r_elec, pp._a_0[0], pp._a_i, pp._b_i,
        pp._r_neighbor_e, pp.i_sort_e, pp._b_t_e)
    if pp.ion_motion:
        # calculate_b_theta_at_particles(
        #     pp.r_ion, pp._a_0, pp._a_i, pp._b_i,
        #     pp._r_neighbor_i, pp.i_sort_i, pp._b_t_i)
        calculate_b_theta_at_ions(
            pp.r_ion, pp.r_elec, pp._a_0[0], pp._a_i, pp._b_i,
            pp.i_sort_i, pp.i_sort_e, pp._b_t_i)

@njit_serial()
def calculate_psi_grid_pp(pp, r_eval, log_r_eval, psi):
    calculate_psi(r_eval, log_r_eval, pp.r_elec, pp._sum_1_e, pp._sum_2_e, pp.i_sort_e, psi)
    calculate_psi(r_eval, log_r_eval, pp.r_ion, pp._sum_1_i, pp._sum_2_i, pp.i_sort_i, psi)
    psi -= pp._psi_max

@njit_serial()
def calculate_b_theta_grid_pp(pp, r_eval, b_theta):
    calculate_b_theta(r_eval, pp._a_0[0], pp._a_i, pp._b_i,
                        pp.r_elec, pp.i_sort_e, b_theta)

@njit_serial()
def evolve(pp, dxi):
    if pp.ion_motion:
        evolve_plasma_ab5(dxi, pp.r, pp.pr, pp.gamma, pp.m, pp.q_species, pp._nabla_a2,
                        pp._b_t_0, pp._b_t, pp._psi, pp._dr_psi, 
                        pp._dr, pp._dpr
                        )
    else:
        evolve_plasma_ab5(dxi, pp.r_elec, pp.pr_elec, pp.gamma_elec, pp.m_elec, pp.q_species_elec, pp._nabla_a2_e,
                        pp._b_t_0_e, pp._b_t_e, pp._psi_e, pp._dr_psi_e, 
                        pp._dr, pp._dpr
                        )

@njit_serial()
def update_gamma_pz_pp(pp):
    if pp.ion_motion:
        update_gamma_and_pz(
            pp.gamma, pp.pz, pp.pr,
            pp._a2, pp._psi, pp.q_species, pp.m)
    else:
        update_gamma_and_pz(
            pp.gamma_elec, pp.pz_elec, pp.pr_elec,
            pp._a2_e, pp._psi_e, pp.q_species_elec, pp.m_elec)
        # if np.max(pp.pz_elec/pp.gamma_elec) > 0.999:
        #     print('p'+str(np.max(pp.pz_elec/pp.gamma_elec)))
        idx_keep = np.where(pp.gamma_elec >= 25)
        if idx_keep[0].size > 0:
            pp.pz_elec[idx_keep] = 0.
            pp.gamma_elec[idx_keep] = 1.
            pp.pr_elec[idx_keep] = 0.

@njit_serial()
def determine_neighboring_points_pp(pp):
    determine_neighboring_points(
        pp.r_elec, pp.dr_p, pp.i_sort_e, pp._r_neighbor_e)
    pp._log_r_neighbor_e = np.log(pp._r_neighbor_e)
    if pp.ion_motion:
        determine_neighboring_points(
            pp.r_ion, pp.dr_p, pp.i_sort_i, pp._r_neighbor_i)
        pp._log_r_neighbor_i = np.log(pp._r_neighbor_i)


# def radial_integral(f_r):
#     subs = f_r / 2
#     subs += f_r[0]/4
#     return (np.cumsum(f_r) - subs)


@njit_serial()
def all_work(
        pp,
        r_fld, log_r_fld, psi_grid, b_theta_grid,
        rho_e, rho_i, rho, chi
    ):
    # pp.determine_neighboring_points()
    determine_neighboring_points(
        pp.r_elec, pp.dr_p, pp.i_sort_e, pp._r_neighbor_e)
    _log_r_neighbor_e = np.log(pp._r_neighbor_e)
    if pp.ion_motion:
        determine_neighboring_points(
            pp.r_ion, pp.dr_p, pp.i_sort_i, pp._r_neighbor_i)
        _log_r_neighbor_i = np.log(pp._r_neighbor_i)

    # pp.calculate_cumulative_sums()
    calculate_cumulative_sum_1(pp.q_elec, pp.i_sort_e, pp._sum_1_e)
    calculate_cumulative_sum_2(pp.r_elec, pp.q_elec, pp.i_sort_e, pp._sum_2_e)
    if pp.ion_motion or not pp.ions_computed:           
        calculate_cumulative_sum_1(pp.q_ion, pp.i_sort_i, pp._sum_1_i)
        calculate_cumulative_sum_2(pp.r_ion, pp.q_ion, pp.i_sort_i, pp._sum_2_i)
      
    # pp.gather_particle_background()
    calculate_psi_and_dr_psi(
        pp._r_neighbor_e, _log_r_neighbor_e, pp.r_ion, pp.dr_p, pp.i_sort_i,
        pp._sum_1_i, pp._sum_2_i, pp._psi_bg_i, pp._dr_psi_bg_i)
    if pp.ion_motion:
        calculate_psi_and_dr_psi(
            pp._r_neighbor_i, _log_r_neighbor_i, pp.r_elec, pp.dr_p, pp.i_sort_e,
            pp._sum_1_e, pp._sum_2_e, pp._psi_bg_e,
            pp._dr_psi_bg_e)

    # pp.calculate_psi_dr_psi()
    calculate_psi_dr_psi_at_particles_bg(
        pp.r_elec, pp._sum_1_e, pp._sum_2_e,
        pp._psi_bg_i, pp._r_neighbor_e, _log_r_neighbor_e,
        pp.i_sort_e, pp._psi_e, pp._dr_psi_e)
    if pp.ion_motion:
        calculate_psi_dr_psi_at_particles_bg(
            pp.r_ion, pp._sum_1_i, pp._sum_2_i,
            pp._psi_bg_e, pp._r_neighbor_i, _log_r_neighbor_e,
            pp.i_sort_i, pp._psi_i, pp._dr_psi_i)
    r_max_e = pp.r_elec[pp.i_sort_e[-1]]
    r_max_i = pp.r_ion[pp.i_sort_i[-1]]
    pp._r_max[:] = max(r_max_e, r_max_i) + pp.dr_p/2
    log_r_max = np.log(pp._r_max)
    pp._psi_max[:] = 0.
    calculate_psi(pp._r_max, log_r_max, pp.r_elec, pp._sum_1_e, pp._sum_2_e, pp.i_sort_e, pp._psi_max)
    calculate_psi(pp._r_max, log_r_max, pp.r_ion, pp._sum_1_i, pp._sum_2_i, pp.i_sort_i, pp._psi_max)
    pp._psi_e -= pp._psi_max
    if pp.ion_motion:
        pp._psi_i -= pp._psi_max
    pp._psi[pp._psi < -0.9] = -0.9

    # pp.calculate_cumulative_sum_3()
    calculate_cumulative_sum_3(
        pp.r_elec, pp.pr_elec, pp.q_elec, pp._psi_e, pp.i_sort_e,
        pp._sum_3_e)
    if pp.ion_motion or not pp.ions_computed:
        calculate_cumulative_sum_3(
        pp.r_ion, pp.pr_ion, pp.q_ion, pp._psi_i, pp.i_sort_i,
        pp._sum_3_i)

    # pp.gather_particle_background_dxi_psi()
    calculate_dxi_psi(
        pp._r_neighbor_e, pp.r_ion, pp.i_sort_i, pp._sum_3_i,
        pp._dxi_psi_bg_i)
    if pp.ion_motion:
        calculate_dxi_psi(
            pp._r_neighbor_i, pp.r_elec, pp.i_sort_e, pp._sum_3_e,
            pp._dxi_psi_bg_e)
        
    # pp.calculate_dxi_psi()
    calculate_dxi_psi_at_particles_bg(
        pp.r_elec, pp._sum_3_e, pp._dxi_psi_bg_i, pp._r_neighbor_e,
        pp.i_sort_e, pp._dxi_psi_e)
    if pp.ion_motion:
        calculate_dxi_psi_at_particles_bg(
            pp.r_ion, pp._sum_3_i, pp._dxi_psi_bg_e, pp._r_neighbor_i,
            pp.i_sort_i, pp._dxi_psi_i)

    # Apply boundary condition (dxi_psi = 0 after last particle).
    pp._dxi_psi += (pp._sum_3_e[pp.i_sort_e[-1]] +
                        pp._sum_3_i[pp.i_sort_i[-1]])

    pp._dxi_psi[pp._dxi_psi < -3.] = -3.
    pp._dxi_psi[pp._dxi_psi > 3.] = 3.

    # pp.update_gamma_pz()
    if pp.ion_motion:
        update_gamma_and_pz(
            pp.gamma, pp.pz, pp.pr,
            pp._a2, pp._psi, pp.q_species, pp.m)
    else:
        update_gamma_and_pz(
            pp.gamma_elec, pp.pz_elec, pp.pr_elec,
            pp._a2_e, pp._psi_e, pp.q_species_elec, pp.m_elec)
        idx_keep = np.where(pp.gamma_elec >= 25)
        if idx_keep[0].size > 0:
            pp.pz_elec[idx_keep] = 0.
            pp.gamma_elec[idx_keep] = 1.
            pp.pr_elec[idx_keep] = 0.

    # pp.calculate_ai_bi()
    calculate_ABC(
        pp.r_elec, pp.pr_elec, pp.q_elec, pp.gamma_elec,
        pp._psi_e, pp._dr_psi_e, pp._dxi_psi_e, pp._b_t_0_e,
        pp._nabla_a2_e, pp.i_sort_e, pp._A, pp._B, pp._C)
    calculate_KU(pp.r_elec, pp._A, pp.i_sort_e, pp._K, pp._U)
    calculate_ai_bi_from_axis(
        pp.r_elec, pp._A, pp._B, pp._C, pp._K, pp._U, pp.i_sort_e, pp._a_0,pp. _a_i,
        pp._b_i)
    
    # pp.calculate_b_theta()
    calculate_b_theta_at_particles(
        pp.r_elec, pp._a_0[0], pp._a_i, pp._b_i,
        pp._r_neighbor_e, pp.i_sort_e, pp._b_t_e)
    if pp.ion_motion:
        calculate_b_theta_at_ions(
            pp.r_ion, pp.r_elec, pp._a_0[0], pp._a_i, pp._b_i,
            pp.i_sort_i, pp.i_sort_e, pp._b_t_i)
        
    # pp.calculate_psi_grid(r_fld, log_r_fld, psi[slice_i+2, 2:-2])
    calculate_psi(r_fld, log_r_fld, pp.r_elec, pp._sum_1_e, pp._sum_2_e, pp.i_sort_e, psi_grid)
    calculate_psi(r_fld, log_r_fld, pp.r_ion, pp._sum_1_i, pp._sum_2_i, pp.i_sort_i, psi_grid)
    psi_grid -= pp._psi_max
    
    # pp.calculate_b_theta_grid(r_fld, b_t_bar[slice_i+2, 2:-2])
    calculate_b_theta(r_fld, pp._a_0[0], pp._a_i, pp._b_i, pp.r_elec, pp.i_sort_e, b_theta_grid)

    if pp.ion_motion:
    #     pp.deposit_rho_e(rho_e[slice_i+2], r_fld, n_r, dr)
        w_rho = pp.q_elec / (1 - pp.pz_elec/pp.gamma_elec)
        deposit_plasma_particles(pp.r_elec, w_rho, r_fld[0], pp.nr, pp.dr, rho_e, pp.shape)
        rho_e[2: -2] /= r_fld * pp.dr

    #     pp.deposit_rho_i(rho_i[slice_i+2], r_fld, n_r, dr)
        w_rho = pp.q_ion / ((1 - pp.pz_ion/pp.gamma_ion))
        deposit_plasma_particles(pp.r_ion, w_rho, r_fld[0], pp.nr, pp.dr, rho_i, pp.shape)
        rho_i[2: -2] /= r_fld * pp.dr
        rho += rho_e + rho_i
    else:
    #     pp.deposit_rho_e(rho[slice_i+2], r_fld, n_r, dr)
        w_rho = pp.q_elec / (1 - pp.pz_elec/pp.gamma_elec)
        deposit_plasma_particles(pp.r_elec, w_rho, r_fld[0], pp.nr, pp.dr, rho, pp.shape)
        rho[2: -2] /= r_fld * pp.dr

    # pp.deposit_chi(chi[slice_i+2], r_fld, n_r, dr)
    w_chi = pp.q_elec / ((1 - pp.pz_elec/pp.gamma_elec)) / pp.gamma_elec
    deposit_plasma_particles(pp.r_elec, w_chi, r_fld[0], pp.nr, pp.dr, chi, pp.shape)
    chi[2: -2] /= r_fld * pp.dr


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
