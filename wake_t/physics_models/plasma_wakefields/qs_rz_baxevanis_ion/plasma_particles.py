"""Contains the definition of the `PlasmaParticles` class."""
from typing import Optional, List, Callable

import numpy as np
import scipy.constants as ct

from .psi_and_derivatives import (calculate_psi_with_interpolation,
                                  calculate_psi_and_derivatives_at_particles)
from .deposition import deposit_plasma_particles
from .gather import gather_bunch_sources, gather_laser_sources
from .b_theta import (calculate_b_theta_at_particles,
                      calculate_b_theta_with_interpolation)
from .plasma_push.ab2 import evolve_plasma_ab2
from .utils import (
    calculate_chi, calculate_rho, update_gamma_and_pz, sort_particle_arrays,
    check_gamma, log)


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
        values are `'ab2'`.


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
        ion_motion: Optional[bool] = True,
        ion_mass: Optional[float] = ct.m_p,
        free_electrons_per_ion: Optional[int] = 1,
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
        self.ion_motion = ion_motion
        self.ion_mass = ion_mass
        self.free_electrons_per_ion = free_electrons_per_ion
        self.store_history = store_history
        self.diags = diags

    def initialize(self):
        """Initialize column of plasma particles."""

        # Create radial distribution of plasma particles.
        rmin = 0.
        for i in range(self.ppc.shape[0]):
            rmax = self.ppc[i, 0]
            ppc = self.ppc[i, 1]

            n_elec = int(np.round((rmax - rmin) / self.dr * ppc))
            dr_p_i = self.dr / ppc
            rmax = rmin + n_elec * dr_p_i

            r_i = np.linspace(rmin + dr_p_i / 2, rmax - dr_p_i / 2, n_elec)
            dr_p_i = np.ones(n_elec) * dr_p_i
            if i == 0:
                r = r_i
                dr_p = dr_p_i
            else:
                r = np.concatenate((r, r_i))
                dr_p = np.concatenate((dr_p, dr_p_i))

            rmin = rmax

        # Determine number of particles.
        self.n_elec = r.shape[0]
        self.n_part = self.n_elec * 2

        # Initialize particle arrays.
        # `q_center` represents the charge until the particle center. That is,
        # the charge of the first half of the particle.
        pr = np.zeros(self.n_elec)
        pz = np.zeros(self.n_elec)
        gamma = np.ones(self.n_elec)
        q = dr_p * r * self.radial_density(r)
        q_center = q / 2 - dr_p ** 2 / 8
        q *= self.free_electrons_per_ion
        q_center *= self.free_electrons_per_ion
        m_e = np.ones(self.n_elec)
        m_i = np.ones(self.n_elec) * self.ion_mass / ct.m_e
        q_species_e = np.ones(self.n_elec)
        q_species_i = - np.ones(self.n_elec) * self.free_electrons_per_ion
        tag = np.arange(self.n_elec, dtype=np.int32)

        # Combine arrays of both species.
        self.r = np.concatenate((r, r))
        self.dr_p = np.concatenate((dr_p, dr_p))
        self.pr = np.concatenate((pr, pr))
        self.pz = np.concatenate((pz, pz))
        self.gamma = np.concatenate((gamma, gamma))
        self.q = np.concatenate((q, -q))
        self.q_center = np.concatenate((q_center, -q_center))
        self.q_species = np.concatenate((q_species_e, q_species_i))
        self.m = np.concatenate((m_e, m_i))
        self.r_to_x = np.ones(self.n_part, dtype=np.int32)
        self.tag = np.concatenate((tag, tag))

        # Create history arrays.
        if self.store_history:
            self.r_hist = np.zeros((self.nz, self.n_part))
            self.log_r_hist = np.zeros((self.nz, self.n_part))
            self.xi_hist = np.zeros((self.nz, self.n_part))
            self.pr_hist = np.zeros((self.nz, self.n_part))
            self.pz_hist = np.zeros((self.nz, self.n_part))
            self.w_hist = np.zeros((self.nz, self.n_part))
            self.r_to_x_hist = np.zeros((self.nz, self.n_part), dtype=np.int32)
            self.tag_hist = np.zeros((self.nz, self.n_part), dtype=np.int32)
            self.sum_1_hist = np.zeros((self.nz, self.n_part + 2))
            self.sum_2_hist = np.zeros((self.nz, self.n_part + 2))
            self.a_i_hist = np.zeros((self.nz, self.n_elec))
            self.b_i_hist = np.zeros((self.nz, self.n_elec))
            self.a_0_hist = np.zeros(self.nz)
            self.i_push = 0
            self.xi_current = 0.

        self.ions_computed = False

        # Allocate arrays that will contain the fields experienced by the
        # particles.
        self._allocate_field_arrays()
        self._make_species_views()

        # Allocate arrays needed for the particle pusher.
        if self.pusher == 'ab2':
            self._allocate_ab2_arrays()

    def sort(self):
        """Sort plasma particles radially (only by index).
        
        The `q_species` and `m` arrays do not need to be sorted because all
        particles have the same value.
        """
        i_sort_e = np.argsort(self.r_elec, kind='stable')
        sort_particle_arrays(
            self.r_elec,
            self.dr_p_elec,
            self.pr_elec,
            self.pz_elec,
            self.gamma_elec,
            self.q_elec,
            self.q_center_elec,
            self.r_to_x_elec,
            self.tag_elec,
            self._dr_e,
            self._dpr_e,
            i_sort_e,
        )
        if self.ion_motion:
            i_sort_i = np.argsort(self.r_ion, kind='stable')
            sort_particle_arrays(
                self.r_ion,
                self.dr_p_ion,
                self.pr_ion,
                self.pz_ion,
                self.gamma_ion,
                self.q_ion,
                self.q_center_ion,
                self.r_to_x_ion,
                self.tag_ion,
                self._dr_i,
                self._dpr_i,
                i_sort_i,
            )

    def gather_laser_sources(self, a2, nabla_a2, r_min, r_max, dr):
        """Gather the source terms (a^2 and nabla(a)^2) from the laser."""
        if self.ion_motion:
            gather_laser_sources(
                a2, nabla_a2, r_min, r_max, dr,
                self.r, self._a2, self._nabla_a2
            )
        else:
            gather_laser_sources(
                a2, nabla_a2, r_min, r_max, dr,
                self.r_elec, self._a2_e, self._nabla_a2_e
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
                if self.ion_motion:
                    gather_bunch_sources(array[xi_index], r_min, r_max, dr,
                                         self.r, self._b_t_0)
                else:
                    gather_bunch_sources(array[xi_index], r_min, r_max, dr,
                                         self.r_elec, self._b_t_0_e)

    def calculate_fields(self):
        """Calculate the fields at the plasma particles."""
        # Precalculate logarithms (expensive) to avoid doing so several times.
        log(self.r_elec, self.log_r_elec)
        if self.ion_motion or not self.ions_computed:
            log(self.r_ion, self.log_r_ion)

        calculate_psi_and_derivatives_at_particles(
            self.r_elec, self.log_r_elec, self.pr_elec, self.q_elec,
            self.q_center_elec,
            self.r_ion, self.log_r_ion, self.pr_ion, self.q_ion,
            self.q_center_ion,
            self.ion_motion, self.ions_computed,
            self._sum_1_e, self._sum_2_e, self._sum_3_e,
            self._sum_1_i, self._sum_2_i, self._sum_3_i,
            self._psi_e, self._dr_psi_e, self._dxi_psi_e,
            self._psi_i, self._dr_psi_i, self._dxi_psi_i,
            self._psi, self._dr_psi, self._dxi_psi
        )
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
        check_gamma(self.gamma_elec, self.pz_elec, self.pr_elec,
                    self.max_gamma)
        calculate_b_theta_at_particles(
            self.r_elec, self.pr_elec, self.q_elec, self.q_center_elec,
            self.gamma_elec,
            self.r_ion,
            self.ion_motion,
            self._psi_e, self._dr_psi_e, self._dxi_psi_e,
            self._b_t_0_e, self._nabla_a2_e,
            self._A, self._B, self._C,
            self._K, self._U,
            self._a_0, self._a_i, self._b_i,
            self._b_t_e, self._b_t_i
        )

    def calculate_psi_at_grid(self, r_eval, psi):
        """Calculate psi on the current grid slice."""
        calculate_psi_with_interpolation(
            r_eval, self.r_elec, self.log_r_elec, self._sum_1_e, self._sum_2_e,
            psi
        )
        calculate_psi_with_interpolation(
            r_eval, self.r_ion, self.log_r_ion, self._sum_1_i, self._sum_2_i,
            psi, add=True
        )

    def calculate_b_theta_at_grid(self, r_eval, b_theta):
        """Calculate b_theta on the current grid slice."""
        calculate_b_theta_with_interpolation(
            r_eval, self._a_0[0], self._a_i, self._b_i, self.r_elec,
            b_theta
        )

    def evolve(self, dxi):
        """Evolve plasma particles to next longitudinal slice."""
        if self.ion_motion:
            evolve_plasma_ab2(
                dxi, self.r, self.pr, self.gamma, self.m, self.q_species,
                self.r_to_x, self._nabla_a2, self._b_t_0, self._b_t,
                self._psi, self._dr_psi, self._dr, self._dpr
            )
        else:
            evolve_plasma_ab2(
                dxi, self.r_elec, self.pr_elec, self.gamma_elec, self.m_elec,
                self.r_to_x_elec, self.q_species_elec,
                self._nabla_a2_e, self._b_t_0_e,
                self._b_t_e, self._psi_e, self._dr_psi_e, self._dr, self._dpr
            )

        if self.store_history:
            self.i_push += 1
            self.xi_current -= dxi
            self._move_auxiliary_arrays_to_next_slice()

    def calculate_weights(self):
        """Calculate the plasma density weights of each particle."""
        calculate_rho(self.q, self.pz, self.gamma, self._rho)


    def deposit_rho(self, rho, rho_e, rho_i, r_fld, nr, dr):
        """Deposit plasma density on a grid slice."""
        self.calculate_weights()
        # Deposit electrons
        deposit_plasma_particles(
            self.r_elec, self._rho_e, r_fld[0], nr, dr, rho_e, self.shape
        )

        # Deposit ions
        deposit_plasma_particles(
            self.r_ion, self._rho_i, r_fld[0], nr, dr, rho_i, self.shape
        )
        rho[:] = rho_e
        rho += rho_i

    def deposit_chi(self, chi, r_fld, nr, dr):
        """Deposit plasma susceptibility on a grid slice."""
        calculate_chi(self.q_elec, self.pz_elec, self.gamma_elec, self._chi_e)
        deposit_plasma_particles(
            self.r_elec, self._chi_e, r_fld[0], nr, dr, chi, self.shape
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
                'log_r_hist': self.log_r_hist,
                'xi_hist': self.xi_hist,
                'pr_hist': self.pr_hist,
                'pz_hist': self.pz_hist,
                'w_hist': self.w_hist,
                'r_to_x_hist': self.r_to_x_hist,
                'tag_hist': self.tag_hist,
                'sum_1_hist': self.sum_1_hist,
                'sum_2_hist': self.sum_2_hist,
                'a_i_hist': self.a_i_hist,
                'b_i_hist': self.b_i_hist,
                'a_0_hist': self.a_0_hist,
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
        if 'r_to_x' in self.diags:
            self.r_to_x_hist[-1 - self.i_push] = self.r_to_x
        if 'tag' in self.diags:
            self.tag_hist[-1 - self.i_push] = self.tag
        if self.store_history:
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
            self._log_r = self.log_r_hist[-1]
        else:
            self._a_i = np.zeros(self.n_elec)
            self._b_i = np.zeros(self.n_elec)
            self._sum_1 = np.zeros(self.n_part + 2)
            self._sum_2 = np.zeros(self.n_part + 2)
            self._rho = np.zeros(self.n_part)
            self._log_r = np.zeros(self.n_part)

        self._a2 = np.zeros(self.n_part)
        self._nabla_a2 = np.zeros(self.n_part)
        self._b_t_0 = np.zeros(self.n_part)
        self._b_t = np.zeros(self.n_part)
        self._psi = np.zeros(self.n_part)
        self._dr_psi = np.zeros(self.n_part)
        self._dxi_psi = np.zeros(self.n_part)
        self._chi = np.zeros(self.n_part)
        self._sum_3_e = np.zeros(self.n_elec + 1)
        self._sum_3_i = np.zeros(self.n_elec + 1)
        self._a_0 = np.zeros(1)
        self._A = np.zeros(self.n_elec)
        self._B = np.zeros(self.n_elec)
        self._C = np.zeros(self.n_elec)
        self._K = np.zeros(self.n_elec)
        self._U = np.zeros(self.n_elec)

    def _make_species_views(self):
        """Make species arrays as partial views of the particle arrays."""
        self.r_elec = self.r[:self.n_elec]
        self.log_r_elec = self._log_r[:self.n_elec]
        self.dr_p_elec = self.dr_p[:self.n_elec]
        self.pr_elec = self.pr[:self.n_elec]
        self.pz_elec = self.pz[:self.n_elec]
        self.gamma_elec = self.gamma[:self.n_elec]
        self.q_elec = self.q[:self.n_elec]
        self.q_center_elec = self.q_center[:self.n_elec]
        self.q_species_elec = self.q_species[:self.n_elec]
        self.m_elec = self.m[:self.n_elec]
        self.r_to_x_elec = self.r_to_x[:self.n_elec]
        self.tag_elec = self.tag[:self.n_elec]

        self.r_ion = self.r[self.n_elec:]
        self.log_r_ion = self._log_r[self.n_elec:]
        self.dr_p_ion = self.dr_p[self.n_elec:]
        self.pr_ion = self.pr[self.n_elec:]
        self.pz_ion = self.pz[self.n_elec:]
        self.gamma_ion = self.gamma[self.n_elec:]
        self.q_ion = self.q[self.n_elec:]
        self.q_center_ion = self.q_center[self.n_elec:]
        self.q_species_ion = self.q_species[self.n_elec:]
        self.m_ion = self.m[self.n_elec:]
        self.r_to_x_ion = self.r_to_x[self.n_elec:]
        self.tag_ion = self.tag[self.n_elec:]

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
        self._sum_1_e = self._sum_1[:self.n_elec + 1]
        self._sum_2_e = self._sum_2[:self.n_elec + 1]
        self._sum_1_i = self._sum_1[self.n_elec + 1:]
        self._sum_2_i = self._sum_2[self.n_elec + 1:]
        self._rho_e = self._rho[:self.n_elec]
        self._rho_i = self._rho[self.n_elec:]
        self._chi_e = self._chi[:self.n_elec]
        self._chi_i = self._chi[self.n_elec:]

    def _allocate_ab2_arrays(self):
        """Allocate the arrays needed for the 5th order Adams-Bashforth pusher.

        The AB2 pusher needs the derivatives of r and pr for each particle
        at the last 2 plasma slices. This method allocates the arrays that will
        store these derivatives.
        """
        if self.ion_motion:
            size = self.n_part
        else:
            size = self.n_elec
        self._dr = np.zeros((2, size))
        self._dpr = np.zeros((2, size))
        self._dr_e = self._dr[:, :self.n_elec]
        self._dpr_e = self._dpr[:, :self.n_elec]
        self._dr_i = self._dr[:, self.n_elec:]
        self._dpr_i = self._dpr[:, self.n_elec:]

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
        self._log_r = self.log_r_hist[-1 - self.i_push]

        self._sum_1_e = self._sum_1[:self.n_elec + 1]
        self._sum_2_e = self._sum_2[:self.n_elec + 1]
        self._sum_1_i = self._sum_1[self.n_elec + 1:]
        self._sum_2_i = self._sum_2[self.n_elec + 1:]
        self._rho_e = self._rho[:self.n_elec]
        self._rho_i = self._rho[self.n_elec:]
        self.log_r_elec = self._log_r[:self.n_elec]
        self.log_r_ion = self._log_r[self.n_elec:]

        if not self.ion_motion:
            self._sum_1_i[:] = self.sum_1_hist[-self.i_push, self.n_elec + 1:]
            self._sum_2_i[:] = self.sum_2_hist[-self.i_push, self.n_elec + 1:]
