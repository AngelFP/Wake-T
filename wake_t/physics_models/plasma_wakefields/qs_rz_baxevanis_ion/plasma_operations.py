from .deposition import deposit_plasma_particles
from .gather import gather_bunch_sources, gather_laser_sources

from .plasma_push.ab2 import evolve_plasma_ab2
from .utils import (
    calculate_chi,
    calculate_rho,
    update_gamma_and_pz,
    check_gamma,
)
from .utils import sort_particle_arrays
import numpy as np


def gather_laser_sources_b(self, a2, nabla_a2, r_min, r_max, dr):
    """Gather the source terms (a^2 and nabla(a)^2) from the laser."""
    if self.ion_motion:
        gather_laser_sources(
            a2,
            nabla_a2,
            r_min,
            r_max,
            dr,
            self.r,
            self._a2,
            self._nabla_a2,
        )


def gather_bunch_sources_b(
    self, source_arrays, source_xi_indices, source_metadata, slice_i
):
    """Gather the source terms (b_theta) from the particle bunches."""
    self._b_t_0[:] = 0.0
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
                gather_bunch_sources(
                    array[xi_index], r_min, r_max, dr, self.r, self._b_t_0
                )


def update_gamma_and_pz_b(self):
    if self.ion_motion:
        update_gamma_and_pz(
            self.gamma,
            self.pz,
            self.pr,
            self._a2,
            self._psi,
            self.q,
            self.m,
        )
    check_gamma(self.gamma, self.pz, self.pr, self.max_gamma)


def evolve(self, dxi):
    """Evolve plasma particles to next longitudinal slice."""
    if self.ion_motion:
        evolve_plasma_ab2(
            dxi,
            self.r,
            self.pr,
            self.gamma,
            self.m,
            self.q,
            self.r_to_x,
            self._nabla_a2,
            self._b_t_0,
            self._b_t,
            self._psi,
            self._dr_psi,
            self._dr,
            self._dpr,
        )

    if self.store_history:
        self.i_push += 1
        self.xi_current -= dxi
        self._move_auxiliary_arrays_to_next_slice()


def calculate_weights(self):
    """Calculate the plasma density weights of each particle."""
    if self.ion_motion or not self.ions_computed:
        calculate_rho(
            self.q,
            self.w,
            self.pz,
            self.gamma,
            self._rho,
        )


def deposit_rho(self, rho, slice_i, r_fld, nr, dr):
    """Deposit plasma density on a grid slice."""
    calculate_weights(self)
    # Deposit species
    deposit_plasma_particles(
        self.r, self._rho, r_fld[0], nr, dr, self.rho_species[slice_i], self.shape
    )
    rho += self.rho_species[slice_i]


def deposit_chi(self, chi, slice_i, r_fld, nr, dr):
    """Deposit plasma susceptibility on a grid slice."""
    calculate_chi(
        self.q,
        self.w,
        self.pz,
        self.gamma,
        self._chi,
    )
    deposit_plasma_particles(self.r, self._chi, r_fld[0], nr, dr, chi, self.shape)
    chi += self.chi_species[slice_i]
