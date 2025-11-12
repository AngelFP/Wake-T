"""Contains the definition of the `PlasmaParticles` class."""

from typing import Optional, List, Callable

import numpy as np
import scipy.constants as ct
from numba.experimental import jitclass
from .utils import sort_particle_arrays


# @jitclass
class PlasmaParticles:
    """
    Class containing a 1D slice of plasma particles.

    In the current implementation, this class stores both the plasma electrons
    and ions. It would be useful to change this in the future so that it
    stores only a single species. This would allow us to more easily
    extend the wakefield model to cases with more than 2 species, which would
    be great to model ionization, for example.

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
    mass : float, optional
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
        for diagnostics or because of the use of adaptive grids. By default,
        ``False``.
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
        max_gamma: Optional[float] = 10.0,
        ion_motion: Optional[bool] = True,
        mass: Optional[float] = ct.m_p,
        free_electrons_per_ion: Optional[int] = 1,
        pusher: Optional[str] = "ab2",
        shape: Optional[str] = "linear",
        store_history: Optional[bool] = False,
        diags: Optional[List[str]] = [],
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
        self.mass = mass
        self.free_electrons_per_ion = free_electrons_per_ion
        self.store_history = store_history
        self.diags = diags
        self.rho_species = np.zeros((nz + 4, nr + 4))
        self.chi_species = np.zeros((nz + 4, nr + 4))

    def initialize(self):
        """Initialize column of plasma particles."""

        # Create radial distribution of plasma particles.
        rmin = 0.0
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
        # `q_center` represents the charge until the particle center. That is,
        # the charge of the first half of the particle.
        pr = np.zeros(self.n_part)
        pz = np.zeros(self.n_part)
        gamma = np.ones(self.n_part)
        id = np.arange(self.n_part, dtype=np.int32)
        w = dr_p * r * self.radial_density(r)
        w_center = w / 2 - dr_p**2 / 8

        # Charge and mass of the macroparticles of each species.
        self.m = self.mass / ct.m_e
        self.q = -self.free_electrons_per_ion

        # Combine arrays of both species.
        self.r = r
        self.dr_p = dr_p
        self.pr = pr
        self.pz = pz
        self.gamma = gamma
        self.w = w
        self.w_center = w_center
        self.r_to_x = np.ones(self.n_part, dtype=np.int32)
        self.id = id

        # Create history arrays.
        if self.store_history:
            self.r_hist = np.zeros((self.nz, self.n_part))
            self.log_r_hist = np.zeros((self.nz, self.n_part))
            self.xi_hist = np.zeros((self.nz, self.n_part))
            self.pr_hist = np.zeros((self.nz, self.n_part))
            self.pz_hist = np.zeros((self.nz, self.n_part))
            self.w_hist = np.zeros((self.nz, self.n_part))
            self.r_to_x_hist = np.zeros((self.nz, self.n_part), dtype=np.int32)
            self.id_hist = np.zeros((self.nz, self.n_part), dtype=np.int32)
            self.sum_1_hist = np.zeros((self.nz, self.n_part + 2))
            self.sum_2_hist = np.zeros((self.nz, self.n_part + 2))
            self.a_i_hist = np.zeros((self.nz, self.n_part))
            self.b_i_hist = np.zeros((self.nz, self.n_part))
            self.a_0_hist = np.zeros(self.nz)
            self.i_push = 0
            self.xi_current = 0.0
            self.i_sort_hist = np.zeros((self.nz, self.n_part), dtype=np.int64)
            self.psi_max_hist = np.zeros(self.nz)

        self.ions_computed = False

        # Allocate arrays that will contain the fields experienced by the
        # particles.
        self._allocate_field_arrays()

        # Allocate arrays needed for the particle pusher.
        if self.pusher == "ab2":
            self._allocate_ab2_arrays()

    @property
    def is_empty(self):
        return self.r.size == 0

    def sort(self):
        """Sort plasma particles radially.

        The `q_species` and `m` arrays do not need to be sorted because all
        particles have the same value.
        """
        if self.ion_motion or not self.ions_computed:
            self.i_sort = np.argsort(self.r, kind="stable")
            sort_particle_arrays(
                self.r,
                self.dr_p,
                self.pr,
                self.pz,
                self.gamma,
                self.w,
                self.w_center,
                self.r_to_x,
                self.id,
                self._dr,
                self._dpr,
                self.i_sort,
            )

    def store_current_step(self):
        """Store current particle properties in the history arrays."""
        if "r" in self.diags or self.store_history:
            self.r_hist[-1 - self.i_push] = self.r
        if "z" in self.diags:
            self.xi_hist[-1 - self.i_push] = self.xi_current
        if "pr" in self.diags:
            self.pr_hist[-1 - self.i_push] = self.pr
        if "pz" in self.diags:
            self.pz_hist[-1 - self.i_push] = self.pz
        if "w" in self.diags:
            self.w_hist[-1 - self.i_push] = self._rho
        if "r_to_x" in self.diags:
            self.r_to_x_hist[-1 - self.i_push] = self.r_to_x
        if "id" in self.diags:
            self.id_hist[-1 - self.i_push] = self.id
        if self.store_history:
            self.i_sort_hist[-1 - self.i_push] = self.i_sort
            self.psi_max_hist[-1 - self.i_push] = self._psi_max[0]
            self.a_0_hist[-1 - self.i_push] = self._a_0[0]

    def get_history(self):
        """Get the history of the evolution of the plasma particles.

        Returns
        -------
        dict
            A dictionary containing the particle history arrays.
        """
        if self.store_history:
            history = {
                "r_hist": self.r_hist,
                "log_r_hist": self.log_r_hist,
                "xi_hist": self.xi_hist,
                "pr_hist": self.pr_hist,
                "pz_hist": self.pz_hist,
                "w_hist": self.w_hist,
                "r_to_x_hist": self.r_to_x_hist,
                "id_hist": self.id_hist,
                "sum_1_hist": self.sum_1_hist,
                "sum_2_hist": self.sum_2_hist,
                "a_i_hist": self.a_i_hist,
                "b_i_hist": self.b_i_hist,
                "a_0_hist": self.a_0_hist,
                'psi_max_hist': self.psi_max_hist,
                "i_sort_hist": self.i_sort_hist,
            }
            return history

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
            self._a_i = np.zeros(self.n_part)
            self._b_i = np.zeros(self.n_part)
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
        self._sum_3 = np.zeros(self.n_part + 1)
        self._a_0 = np.zeros(1)
        self._A = np.zeros(self.n_part)
        self._B = np.zeros(self.n_part)
        self._C = np.zeros(self.n_part)
        self._K = np.zeros(self.n_part)
        self._U = np.zeros(self.n_part)
        self._psi_max = np.zeros(1)

    def _allocate_ab2_arrays(self):
        """Allocate the arrays needed for the 5th order Adams-Bashforth pusher.

        The AB2 pusher needs the derivatives of r and pr for each particle
        at the last 2 plasma slices. This method allocates the arrays that will
        store these derivatives.
        """

        size = self.n_part
        self._dr = np.zeros((2, size))
        self._dpr = np.zeros((2, size))

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

        if not self.ion_motion:
            self._sum_1[:] = self.sum_1_hist[-self.i_push]
            self._sum_2[:] = self.sum_2_hist[-self.i_push]
