"""Contains the definition of the `PlasmaParticles` class."""
from typing import Optional, List, Callable

import numpy as np
import scipy.constants as ct

from wake_t.utilities.numba import njit_serial
from .deposition import deposit_plasma_particles
from .gather import gather_bunch_sources, gather_laser_sources
from .plasma_push.ab2 import evolve_plasma_ab2
from .utils import (log, calculate_chi, calculate_rho,
                    determine_neighboring_points)


class PlasmaSpecies():
    """
    Class containing the 1D slice of plasma particles used in the quasi-static
    Baxevanis wakefield model.

    Parameters
    ----------
    r_max : float
        Maximum radial extension of the simulation box in normalized units.
    r_max_plasma : float
        Maximum radial extension of the plasma column in normalized units.
    dr : float
        Radial step size of the discretized simulation box.
    ppc : float
        Number of particles per cell.
    nr, nz : int
        Number of grid elements along `r` and `z`.
    radial_density : callable
        Function defining the radial density profile.
    max_gamma : float, optional
        Plasma particles whose ``gamma`` exceeds ``max_gamma`` are
        considered to violate the quasistatic condition and are put at
        rest (i.e., ``gamma=1.``, ``pr=pz=0.``). By default 10.
    ion_motion : bool, optional
        Whether to allow the plasma ions to move. By default, False.
    ion_mass : float, optional
        Mass of the plasma ions. By default, the mass of a proton.
    free_electrons_per_ion : int, optional
        Number of free electrons per ion. The ion charge is adjusted
        accordingly to maintain a quasi-neutral plasma (i.e.,
        ion charge = e * free_electrons_per_ion). By default, 1.
    pusher : str, optional
        The pusher used to evolve the plasma particles. Possible values
        are ``'ab2'`` (Adams-Bashforth 2nd order).
    shape : str
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'. By default 'linear'.
    store_history : bool, optional
        Whether to store the plasma particle evolution. This might be needed
        for diagnostics or the use of adaptive grids. By default, False.
    diags : list, optional
        List of particle quantities to save to diagnostics.
    """

    def __init__(
        self,
        r_max: float,
        r_max_plasma: float,
        dr: float,
        ppc: float,
        nr: int,
        nz: int,
        radial_density: Callable[[float], float],
        max_gamma: Optional[float] = 10.,
        can_move: Optional[bool] = True,
        mass: Optional[float] = ct.m_p,
        charge: Optional[float] = ct.e,
        pusher: Optional[str] = 'ab2',
        shape: Optional[str] = 'linear',
        store_history: Optional[bool] = False,
        diags: Optional[List[str]] = []
    ):

        # Store parameters.
        self.r_max = r_max
        self.r_max_plasma = r_max_plasma
        self.radial_density = radial_density
        self.dr = dr
        self.ppc = ppc
        self.pusher = pusher
        self.shape = shape
        self.max_gamma = max_gamma
        self.nr = nr
        self.nz = nz
        self.can_move = can_move
        self.mass = mass
        self.charge = charge
        self.store_history = store_history
        self.diags = diags
        self.rho_species = None

    def initialize(self):
        """Initialize column of plasma particles."""

        # Create radial distribution of plasma particles.
        rmin = 0.
        for i in range(self.ppc.shape[0]):
            rmax = self.ppc[i, 0]
            ppc = self.ppc[i, 1]

            n_part = int(np.round((rmax - rmin) / self.dr * ppc))
            dr_p_i = self.dr / ppc
            rmax = rmin + n_part * dr_p_i

            r_i = np.linspace(rmin + dr_p_i / 2, rmax - dr_p_i / 2, n_part)
            dr_p_i = np.ones(n_part) * dr_p_i
            if i == 0:
                r = r_i
                dr_p = dr_p_i
            else:
                r = np.concatenate((r, r_i))
                dr_p = np.concatenate((dr_p, dr_p_i))

            rmin = rmax

        # Determine number of particles.
        self.n_part = r.shape[0]

        # Initialize particle arrays.
        self.r = r
        self.dr_p = dr_p
        self.pr = np.zeros(self.n_part)
        self.pz = np.zeros(self.n_part)
        self.gamma = np.ones(self.n_part)
        self.w = dr_p * r * self.radial_density(r)
        self.w *= - self.charge / ct.e
        self.m = np.ones(self.n_part) * self.mass / ct.m_e
        self.q = - np.ones(self.n_part) * self.charge / ct.e

        # Create history arrays.
        if self.store_history:
            self.r_hist = np.zeros((self.nz, self.n_part))
            self.xi_hist = np.zeros((self.nz, self.n_part))
            self.pr_hist = np.zeros((self.nz, self.n_part))
            self.pz_hist = np.zeros((self.nz, self.n_part))
            self.w_hist = np.zeros((self.nz, self.n_part))
            self.sum_1_hist = np.zeros((self.nz, self.n_part))
            self.sum_2_hist = np.zeros((self.nz, self.n_part))
            self.i_sort_hist = np.zeros((self.nz, self.n_part), dtype=np.int64)
            self.psi_max_hist = np.zeros(self.nz)
            self.a_i_hist = np.zeros((self.nz, self.n_part))
            self.b_i_hist = np.zeros((self.nz, self.n_part))
            self.a_0_hist = np.zeros(self.nz)
            self.i_push = 0
            self.xi_current = 0.

        self.first_iteration_computed = False

        # Allocate arrays that will contain the fields experienced by the
        # particles.
        self._allocate_field_arrays()

        # Allocate arrays needed for the particle pusher.
        if self.can_move and self.pusher == 'ab2':
            self._allocate_ab2_arrays()

    def sort(self):
        """Sort plasma particles radially (only by index)."""
        if self.can_move or not self.first_iteration_computed:
            self.i_sort = np.argsort(self.r, kind='stable')

    def determine_neighboring_points(self):        
        """Determine the neighboring points of each plasma particle."""
        if self.can_move:
            determine_neighboring_points(
                self.r, self.dr_p, self.i_sort, self._r_neighbor
            )
            log(self._r_neighbor, self._log_r_neighbor)

    def gather_laser_sources(self, a2, nabla_a2, r_min, r_max, dr):
        """Gather the source terms (a^2 and nabla(a)^2) from the laser."""
        if self.can_move:
            gather_laser_sources(
                a2, nabla_a2, r_min, r_max, dr,
                self.r, self._a2, self._nabla_a2
            )

    def gather_bunch_sources(self, source_arrays, source_xi_indices,
                             source_metadata, slice_i):
        """Gather the source terms (b_theta) from the particle bunches."""
        self._b_t_0[:] = 0.
        for i in range(len(source_arrays)):
            array = source_arrays[i]
            idx = source_xi_indices[i]
            md = source_metadata[i]
            r_min = md[0]
            r_max = md[1]
            dr = md[2]
            if slice_i in idx:
                xi_index = slice_i + 2 - idx[0]
                if self.can_move:
                    gather_bunch_sources(array[xi_index], r_min, r_max, dr,
                                         self.r, self._b_t_0)

    def update_gamma_and_pz(self):
        if self.can_move:
            update_gamma_and_pz(
                self.gamma, self.pz, self.pr,
                self._a2, self._psi, self.q, self.m
            )
            check_gamma(self.gamma, self.pz, self.pr, self.max_gamma)

    def evolve(self, dxi):
        """Evolve plasma particles to next longitudinal slice."""
        if self.can_move:
            evolve_plasma_ab2(
                dxi, self.r, self.pr, self.gamma, self.m, self.q,
                self._nabla_a2, self._b_t_0, self._b_t, self._psi,
                self._dr_psi, self._dr, self._dpr
            )

        if self.store_history:
            self.i_push += 1
            self.xi_current -= dxi
            self._move_auxiliary_arrays_to_next_slice()

    def calculate_weights(self):
        """Calculate the plasma density weights of each particle."""
        calculate_rho(self.w, self.pz, self.gamma, self._rho)

    def deposit_rho(self, rho, slice_i, r_fld, nr, dr):
        """Deposit plasma density on a grid slice."""
        self.calculate_weights()
        # Deposit electrons
        deposit_plasma_particles(
            self.r, self._rho, r_fld[0], nr, dr, self.rho_species[slice_i], self.shape
        )
        rho += self.rho_species[slice_i]

    def deposit_chi(self, chi, r_fld, nr, dr):
        """Deposit plasma susceptibility on a grid slice."""
        calculate_chi(self.w, self.pz, self.gamma, self._chi)
        deposit_plasma_particles(
            self.r, self._chi, r_fld[0], nr, dr, chi, self.shape
        )

    def get_history(self):
        """Get the history of the evolution of the plasma particles.

        Returns
        -------
        dict
            A dictionary containing the particle history arrays.
        """
        if self.store_history:
            history = {
                'r_hist': self.r_hist,
                'xi_hist': self.xi_hist,
                'pr_hist': self.pr_hist,
                'pz_hist': self.pz_hist,
                'w_hist': self.w_hist,
                'sum_1_hist': self.sum_1_hist,
                'sum_2_hist': self.sum_2_hist,
                'a_i_hist': self.a_i_hist,
                'b_i_hist': self.b_i_hist,
                'a_0_hist': self.a_0_hist,
                'psi_max_hist': self.psi_max_hist,
                'i_sort_hist': self.i_sort_hist,
            }
            return history

    def store_current_step(self):
        """Store current particle properties in the history arrays."""
        if 'r' in self.diags or self.store_history:
            self.r_hist[-1 - self.i_push] = self.r
        if 'z' in self.diags:
            self.xi_hist[-1 - self.i_push] = self.xi_current
        if 'pr' in self.diags:
            self.pr_hist[-1 - self.i_push] = self.pr
        if 'pz' in self.diags:
            self.pz_hist[-1 - self.i_push] = self.pz
        if 'w' in self.diags:
            self.w_hist[-1 - self.i_push] = self._rho
        if self.store_history:
            self.i_sort_hist[-1 - self.i_push] = self.i_sort
            self.psi_max_hist[-1 - self.i_push] = self._psi_max[0]
            self.a_0_hist[-1 - self.i_push] = self._a_0[0]

    def _allocate_field_arrays(self):
        """Allocate arrays for the fields experienced by the particles.

        In order to evolve the particles to the next longitudinal position,
        it is necessary to know the fields that they are experiencing. These
        arrays are used for storing the value of these fields at the location
        of each particle.
        """
        # When storing the particle history, define the following auxiliary
        # arrays as views of a 1D slice of the history arrays.
        if self.store_history:
            self._a_i = self.a_i_hist[-1]
            self._b_i = self.b_i_hist[-1]
            self._sum_1 = self.sum_1_hist[-1]
            self._sum_2 = self.sum_2_hist[-1]
            self._rho = self.w_hist[-1]
        else:
            self._a_i = np.zeros(self.n_part)
            self._b_i = np.zeros(self.n_part)
            self._sum_1 = np.zeros(self.n_part)
            self._sum_2 = np.zeros(self.n_part)
            self._rho = np.zeros(self.n_part)

        self._a2 = np.zeros(self.n_part)
        self._nabla_a2 = np.zeros(self.n_part)
        self._b_t_0 = np.zeros(self.n_part)
        self._b_t = np.zeros(self.n_part)
        self._psi = np.zeros(self.n_part)
        self._dr_psi = np.zeros(self.n_part)
        self._dxi_psi = np.zeros(self.n_part)
        self._chi = np.zeros(self.n_part)
        self._sum_3 = np.zeros(self.n_part)
        self._psi_bg = np.zeros(self.n_part+1)
        self._dr_psi_bg = np.zeros(self.n_part+1)
        self._dxi_psi_bg = np.zeros(self.n_part+1)
        self._b_t_bg = np.zeros(self.n_part+1)
        self._a_0 = np.zeros(1)
        self._A = np.zeros(self.n_part)
        self._B = np.zeros(self.n_part)
        self._C = np.zeros(self.n_part)
        self._K = np.zeros(self.n_part)
        self._U = np.zeros(self.n_part)
        self._r_neighbor = np.zeros(self.n_part+1)
        self._log_r_neighbor = np.zeros(self.n_part+1)

        self._psi_max = np.zeros(1)

    def _allocate_ab2_arrays(self):
        """Allocate the arrays needed for the 5th order Adams-Bashforth pusher.

        The AB2 pusher needs the derivatives of r and pr for each particle
        at the last 2 plasma slices. This method allocates the arrays that will
        store these derivatives.
        """
        self._dr = np.zeros((2, self.n_part))
        self._dpr = np.zeros((2, self.n_part))

    def _move_auxiliary_arrays_to_next_slice(self):
        """Point auxiliary 1D arrays to next slice of the 2D history arrays.

        When storing the particle history, some auxiliary arrays (e.g., those
        storing the cumulative sums, the a_i, b_i coefficients, ...) have to be
    	stored at every longitudinal step. In principle, this used to be done
        by writing the 1D auxiliary arrays into the corresponding slice of the
        2D history arrays. However, this is time consuming as it leads to
        copying data at every step. In order to avoid this, the auxiliary
        arrays are defined simply as views of a 1D slice of the history arrays
        so that the data is written directly to the history without it being a
        copy. In order to make this work, the slice to which the auxiliary
        arrays point to needs to be moved at each step. This is what this
        method does.
        """
        self._a_i = self.a_i_hist[-1 - self.i_push]
        self._b_i = self.b_i_hist[-1 - self.i_push]
        self._sum_1 = self.sum_1_hist[-1 - self.i_push]
        self._sum_2 = self.sum_2_hist[-1 - self.i_push]
        self._rho = self.w_hist[-1 - self.i_push]

        if not self.can_move:
            self._sum_1[:] = self.sum_1_hist[-self.i_push]
            self._sum_2[:] = self.sum_2_hist[-self.i_push]


@njit_serial(error_model='numpy')
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


@njit_serial()
def check_gamma(gamma, pz, pr, max_gamma):
    """Check that the gamma of particles does not exceed `max_gamma`"""
    for i in range(gamma.shape[0]):
        if gamma[i] > max_gamma:
            gamma[i] = 1.
            pz[i] = 0.
            pr[i] = 0.
