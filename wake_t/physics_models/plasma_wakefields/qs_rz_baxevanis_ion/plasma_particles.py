"""Contains the definition of the `PlasmaParticles` class."""

import numpy as np
import scipy.constants as ct
from numba.experimental import jitclass
from numba.core.types import float64, int64, string, boolean

from wake_t.utilities.numba import njit_serial
from .psi_and_derivatives import (
    calculate_psi, calculate_cumulative_sum_1, calculate_cumulative_sum_2,
    calculate_cumulative_sum_3, calculate_psi_dr_psi_at_particles_bg,
    determine_neighboring_points, calculate_psi_and_dr_psi, calculate_dxi_psi,
    calculate_dxi_psi_at_particles_bg
)
from .deposition import deposit_plasma_particles
from .gather import gather_sources
from .b_theta import (
    calculate_ai_bi_from_axis, calculate_b_theta_at_particles,
    calculate_b_theta, calculate_b_theta_at_ions, calculate_ABC, calculate_KU
)
from .plasma_push.ab5 import evolve_plasma_ab5


spec = [
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
    ('dr_p', float64),
    ('max_gamma', float64),
    ('ion_motion', boolean),
    ('ions_computed', boolean),

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

    ('_psi', float64[::1]),
    ('_dr_psi', float64[::1]),
    ('_dxi_psi', float64[::1]),
    ('_a2', float64[::1]),
    ('_nabla_a2', float64[::1]),
    ('_b_t_0', float64[::1]),
    ('_b_t', float64[::1]),
    ('_r_max', float64[::1]),
    ('_psi_max', float64[::1]),
    ('_dr', float64[:, ::1]),
    ('_dpr', float64[:, ::1]),
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
                 nr, max_gamma=10., ion_motion=True, pusher='ab5', shape='linear'):
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
        self.max_gamma = max_gamma
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

    def sort(self):
        self.i_sort_e = np.argsort(self.r_elec)
        if self.ion_motion or not self.ions_computed:
            self.i_sort_i = np.argsort(self.r_ion)

    def determine_neighboring_points(self):
        determine_neighboring_points(
            self.r_elec, self.dr_p, self.i_sort_e, self._r_neighbor_e
        )
        self._log_r_neighbor_e = np.log(self._r_neighbor_e)
        if self.ion_motion:
            determine_neighboring_points(
                self.r_ion, self.dr_p, self.i_sort_i, self._r_neighbor_i
            )
            self._log_r_neighbor_i = np.log(self._r_neighbor_i)

    def gather_sources(self, a2, nabla_a2, b_theta, r_min, r_max, dr):
        if self.ion_motion:
            gather_sources(
                a2, nabla_a2, b_theta, r_min, r_max, dr,
                self.r, self._a2, self._nabla_a2, self._b_t_0
            )
        else:
            gather_sources(
                a2, nabla_a2, b_theta, r_min, r_max, dr,
                self.r_elec, self._a2_e, self._nabla_a2_e, self._b_t_0_e
            )
            
    def calculate_fields(self):
        self._calculate_cumulative_sums_psi_dr_psi()
        self._gather_particle_background_psi_dr_psi()
        self._calculate_psi_dr_psi()
        self._calculate_cumulative_sum_dxi_psi()
        self._gather_particle_background_dxi_psi()
        self._calculate_dxi_psi()
        self._update_gamma_pz()
        self._calculate_ai_bi()
        self._calculate_b_theta()

    def calculate_psi_at_grid(self, r_eval, log_r_eval, psi):
        calculate_psi(
            r_eval, log_r_eval, self.r_elec, self._sum_1_e, self._sum_2_e,
            self.i_sort_e, psi
        )
        calculate_psi(
            r_eval, log_r_eval, self.r_ion, self._sum_1_i, self._sum_2_i,
            self.i_sort_i, psi
        )
        psi -= self._psi_max

    def calculate_b_theta_at_grid(self, r_eval, b_theta):
        calculate_b_theta(
            r_eval, self._a_0[0], self._a_i, self._b_i, self.r_elec,
            self.i_sort_e, b_theta
        )

    def evolve(self, dxi):
        if self.ion_motion:
            evolve_plasma_ab5(
                dxi, self.r, self.pr, self.gamma, self.m, self.q_species,
                self._nabla_a2, self._b_t_0, self._b_t, self._psi,
                self._dr_psi, self._dr, self._dpr
            )
        else:
            evolve_plasma_ab5(
                dxi, self.r_elec, self.pr_elec, self.gamma_elec, self.m_elec,
                self.q_species_elec, self._nabla_a2_e, self._b_t_0_e,
                self._b_t_e, self._psi_e, self._dr_psi_e, self._dr, self._dpr
            )

    def deposit_rho(self, rho, rho_e, rho_i, r_fld, nr, dr):
        # Deposit electrons
        w_rho_e = self.q_elec / (1 - self.pz_elec/self.gamma_elec)
        deposit_plasma_particles(
            self.r_elec, w_rho_e, r_fld[0], nr, dr, rho, self.shape
        )
        rho[2: -2] /= r_fld * dr

        if self.ion_motion:
            # Deposit ions
            w_rho_i = self.q_ion / (1 - self.pz_ion/self.gamma_ion)
            deposit_plasma_particles(
                self.r_ion, w_rho_i, r_fld[0], nr, dr, rho_i, self.shape
            )
            rho_i[2: -2] /= r_fld * dr
            rho_e[:] = rho
            rho[:] += rho_i

    def deposit_chi(self, chi, r_fld, nr, dr):
        w_chi = (
            self.q_elec / (1 - self.pz_elec / self.gamma_elec) /
            self.gamma_elec
        )
        # w_chi = self.q / (self.dr * self.r * (1 - self.pz/self.gamma)) / (self.gamma * self.m)
        # w_chi = w_chi[:self.n_elec]
        # r_elec = self.r[:self.n_elec]
        deposit_plasma_particles(
            self.r_elec, w_chi, r_fld[0], nr, dr, chi, self.shape
        )
        chi[2: -2] /= r_fld * dr

    def _calculate_cumulative_sums_psi_dr_psi(self):
        calculate_cumulative_sum_1(self.q_elec, self.i_sort_e, self._sum_1_e)
        calculate_cumulative_sum_2(self.r_elec, self.q_elec, self.i_sort_e,
                                   self._sum_2_e)
        if self.ion_motion or not self.ions_computed:           
            calculate_cumulative_sum_1(self.q_ion, self.i_sort_i,
                                       self._sum_1_i)
            calculate_cumulative_sum_2(self.r_ion, self.q_ion, self.i_sort_i,
                                       self._sum_2_i)
            
    def _calculate_cumulative_sum_dxi_psi(self):
        calculate_cumulative_sum_3(
            self.r_elec, self.pr_elec, self.q_elec, self._psi_e, self.i_sort_e,
            self._sum_3_e)
        if self.ion_motion or not self.ions_computed:
            calculate_cumulative_sum_3(
            self.r_ion, self.pr_ion, self.q_ion, self._psi_i, self.i_sort_i,
            self._sum_3_i)

    def _gather_particle_background_psi_dr_psi(self):
        calculate_psi_and_dr_psi(
            self._r_neighbor_e, self._log_r_neighbor_e, self.r_ion, self.dr_p,
            self.i_sort_i, self._sum_1_i, self._sum_2_i, self._psi_bg_i,
            self._dr_psi_bg_i
        )
        if self.ion_motion:
            calculate_psi_and_dr_psi(
                self._r_neighbor_i, self._log_r_neighbor_i, self.r_elec,
                self.dr_p, self.i_sort_e, self._sum_1_e, self._sum_2_e,
                self._psi_bg_e, self._dr_psi_bg_e)
            
    def _gather_particle_background_dxi_psi(self):
        calculate_dxi_psi(
            self._r_neighbor_e, self.r_ion, self.i_sort_i, self._sum_3_i,
            self._dxi_psi_bg_i
        )
        if self.ion_motion:
            calculate_dxi_psi(
                self._r_neighbor_i, self.r_elec, self.i_sort_e, self._sum_3_e,
                self._dxi_psi_bg_e
            )
            
    def _calculate_psi_dr_psi(self):
        calculate_psi_dr_psi_at_particles_bg(
            self.r_elec, self._sum_1_e, self._sum_2_e, self._psi_bg_i,
            self._r_neighbor_e, self._log_r_neighbor_e, self.i_sort_e,
            self._psi_e, self._dr_psi_e
        )
        if self.ion_motion:
            calculate_psi_dr_psi_at_particles_bg(
                self.r_ion, self._sum_1_i, self._sum_2_i, self._psi_bg_e,
                self._r_neighbor_i, self._log_r_neighbor_i, self.i_sort_i,
                self._psi_i, self._dr_psi_i
            )        

        r_max_e = self.r_elec[self.i_sort_e[-1]]
        r_max_i = self.r_ion[self.i_sort_i[-1]]
        self._r_max[:] = max(r_max_e, r_max_i) + self.dr_p/2
        log_r_max = np.log(self._r_max)

        self._psi_max[:] = 0.

        calculate_psi(
            self._r_max, log_r_max, self.r_elec, self._sum_1_e, self._sum_2_e,
            self.i_sort_e, self._psi_max
        )
        calculate_psi(
            self._r_max, log_r_max, self.r_ion, self._sum_1_i, self._sum_2_i,
            self.i_sort_i, self._psi_max
        )

        self._psi_e -= self._psi_max
        if self.ion_motion:
            self._psi_i -= self._psi_max
        
        self._psi[self._psi < -0.9] = -0.9

    def _calculate_dxi_psi(self):            
        calculate_dxi_psi_at_particles_bg(
            self.r_elec, self._sum_3_e, self._dxi_psi_bg_i, self._r_neighbor_e,
            self.i_sort_e, self._dxi_psi_e
        )
        if self.ion_motion:
            calculate_dxi_psi_at_particles_bg(
                self.r_ion, self._sum_3_i, self._dxi_psi_bg_e,
                self._r_neighbor_i, self.i_sort_i, self._dxi_psi_i
            )

        # Apply boundary condition (dxi_psi = 0 after last particle).
        self._dxi_psi += (self._sum_3_e[self.i_sort_e[-1]] +
                            self._sum_3_i[self.i_sort_i[-1]])

        self._dxi_psi[self._dxi_psi < -3.] = -3.
        self._dxi_psi[self._dxi_psi > 3.] = 3.

    def _update_gamma_pz(self):
        if self.ion_motion:
            update_gamma_and_pz(
                self.gamma, self.pz, self.pr,
                self._a2, self._psi, self.q_species, self.m
            )
        else:
            update_gamma_and_pz(
                self.gamma_elec, self.pz_elec, self.pr_elec,
                self._a2_e, self._psi_e, self.q_species_elec, self.m_elec
            )
            # if np.max(self.pz_elec/self.gamma_elec) > 0.999:
            #     print('p'+str(np.max(self.pz_elec/self.gamma_elec)))
        idx_keep = np.where(self.gamma_elec >= self.max_gamma)
        if idx_keep[0].size > 0:
            self.pz_elec[idx_keep] = 0.
            self.gamma_elec[idx_keep] = 1.
            self.pr_elec[idx_keep] = 0.
        
    def _calculate_ai_bi(self):
        calculate_ABC(
            self.r_elec, self.pr_elec, self.q_elec, self.gamma_elec,
            self._psi_e, self._dr_psi_e, self._dxi_psi_e, self._b_t_0_e,
            self._nabla_a2_e, self.i_sort_e, self._A, self._B, self._C
        )
        calculate_KU(self.r_elec, self._A, self.i_sort_e, self._K, self._U)
        calculate_ai_bi_from_axis(
            self.r_elec, self._A, self._B, self._C, self._K, self._U,
            self.i_sort_e, self._a_0, self._a_i, self._b_i
        )

    def _calculate_b_theta(self):
        calculate_b_theta_at_particles(
            self.r_elec, self._a_0[0], self._a_i, self._b_i,
            self._r_neighbor_e, self.i_sort_e, self._b_t_e
        )
        if self.ion_motion:
            # calculate_b_theta_at_particles(
            #     self.r_ion, self._a_0, self._a_i, self._b_i,
            #     self._r_neighbor_i, self.i_sort_i, self._b_t_i)
            calculate_b_theta_at_ions(
                self.r_ion, self.r_elec, self._a_0[0], self._a_i, self._b_i,
                self.i_sort_i, self.i_sort_e, self._b_t_i
            )
    
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
        self._nabla_a2_e = self._nabla_a2[:self.n_elec]
        self._a2_e = self._a2[:self.n_elec]
        self._sum_1_e = np.zeros(self.n_elec)
        self._sum_2_e = np.zeros(self.n_elec)
        self._sum_3_e = np.zeros(self.n_elec)
        self._sum_1_i = np.zeros(self.n_elec)
        self._sum_2_i = np.zeros(self.n_elec)
        self._sum_3_i = np.zeros(self.n_elec)
        self._psi_bg_e = np.zeros(self.n_elec+1)
        self._dr_psi_bg_e = np.zeros(self.n_elec+1)
        self._dxi_psi_bg_e = np.zeros(self.n_elec+1)
        self._psi_bg_i = np.zeros(self.n_elec+1)
        self._dr_psi_bg_i = np.zeros(self.n_elec+1)
        self._dxi_psi_bg_i = np.zeros(self.n_elec+1)
        self._a_0 = np.zeros(1)
        self._a_i = np.zeros(self.n_elec)
        self._b_i = np.zeros(self.n_elec)
        self._A = np.zeros(self.n_elec)
        self._B = np.zeros(self.n_elec)
        self._C = np.zeros(self.n_elec)
        self._K = np.zeros(self.n_elec)
        self._U = np.zeros(self.n_elec)
        self._r_neighbor_e = np.zeros(self.n_elec+1)
        self._r_neighbor_i = np.zeros(self.n_elec+1)

        self._r_max = np.zeros(1)
        self._psi_max = np.zeros(1)

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
        pz_i = (
            (1 + pr[i] ** 2 + q_over_m ** 2 * a2[i] - (1 + psi_i) ** 2) /
            (2 * (1 + psi_i))
        )
        pz[i] = pz_i
        gamma[i] = 1. + pz_i + psi_i
