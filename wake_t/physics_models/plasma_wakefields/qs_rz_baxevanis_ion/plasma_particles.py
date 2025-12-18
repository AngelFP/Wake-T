"""Contains the definition of the `PlasmaParticles` class."""

import numpy as np
import numba

from .psi_and_derivatives import (
    calculate_psi_with_interpolation,
    calculate_psi_and_derivatives_at_particles,
)
from .deposition import deposit_plasma_particles, deposit_plasma_finish
from .gather import gather_bunch_sources, gather_laser_sources
from .b_theta import (
    calculate_b_theta_at_particles,
    calculate_b_theta_with_interpolation,
)
from .plasma_push.ab2 import evolve_plasma_ab2
from .utils import (
    calculate_chi,
    calculate_rho,
    update_gamma_and_pz,
    sort_particle_arrays,
    check_gamma,
    log,
)

from wake_t.utilities.numba import njit_serial

from .plasma_particle_container import PlasmaParticleContainerPy


def pp_initialize(
    init_list,
    nz,
    ppc,
    dr,
    radial_density,
    ion_motion,
    store_history,
    pusher,
):
    """
    Initialize column of plasma particles.

    Parameters
    ----------
    init_list : List[Dict]
        Dict containing charge, mass and is_ion in normalized units
        for all plasma species.
    nz : int
        Number of grid elements along `z`.
    ppc: array_like
        see Quasistatic2DWakefieldIons.
    dr : float
        Radial step size of the discretized simulation box.
    radial_density : callable
        Function defining the radial density profile.
    ion_motion : bool
        Whether to allow the plasma ions to move.
    store_history : bool
        Whether to store the plasma particle evolution.
    pusher : str
        The pusher used to evolve the plasma particles.
    """
    # Create radial distribution of plasma particles.
    rmin = 0.0
    for i in range(ppc.shape[0]):
        rmax = ppc[i, 0]
        local_ppc = ppc[i, 1]

        n_elec = int(np.round((rmax - rmin) / dr * local_ppc))
        dr_p_i = dr / local_ppc
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
    num_per_species = r.shape[0]

    # Initialize particle arrays.
    # `w_center` represents the weight until the particle center. That is,
    # the weight of the first half of the particle.
    pr = np.zeros(num_per_species)
    pz = np.zeros(num_per_species)
    gamma = np.ones(num_per_species)
    idd = np.arange(num_per_species, dtype=np.int32)
    w = dr_p * r * radial_density(r)
    w_center = w / 2 - dr_p**2 / 8

    species_list = [PlasmaParticleContainerPy() for _ in init_list]

    for s, init in zip(species_list, init_list):
        # Charge and mass of the macroparticles of each species.
        s.is_ion = bool(init["is_ion"])
        s.charge = float(init["charge"])
        s.mass = float(init["mass"])

        s.num_particles = num_per_species
        s.do_push = not s.is_ion or ion_motion
        s.store_history = store_history

        # Make copy to avoid multiple species sharing the same array
        s.r = np.copy(r)
        s.pr = np.copy(pr)
        s.pz = np.copy(pz)
        s.gamma = np.copy(gamma)
        s.w = np.copy(w)
        s.w_center = np.copy(w_center)
        s.r_to_x = np.ones(s.num_particles, dtype=np.int32)
        s.id = np.copy(idd)

        # Create history arrays.
        if s.store_history:
            s.r_hist = np.zeros((nz, s.num_particles))
            s.log_r_hist = np.zeros((nz, s.num_particles))
            s.xi_hist = np.zeros((nz, s.num_particles))
            s.pr_hist = np.zeros((nz, s.num_particles))
            s.pz_hist = np.zeros((nz, s.num_particles))
            s.w_hist = np.zeros((nz, s.num_particles))
            s.r_to_x_hist = np.zeros((nz, s.num_particles), dtype=np.int32)
            s.id_hist = np.zeros((nz, s.num_particles), dtype=np.int32)
            s.sum_1_hist = np.zeros((nz, s.num_particles + 1))
            s.sum_2_hist = np.zeros((nz, s.num_particles + 1))
            if not s.is_ion:
                s.a_i_hist = np.zeros((nz, s.num_particles))
                s.b_i_hist = np.zeros((nz, s.num_particles))
                s.a_0_hist = np.zeros(nz)
            else:
                s.a_i_hist = np.zeros((nz, 0))
                s.b_i_hist = np.zeros((nz, 0))
                s.a_0_hist = np.zeros(0)
            s.i_push = 0
            s.xi_current = 0.0
        else:
            s.r_hist = np.zeros((nz, 0))
            s.log_r_hist = np.zeros((nz, 0))
            s.xi_hist = np.zeros((nz, 0))
            s.pr_hist = np.zeros((nz, 0))
            s.pz_hist = np.zeros((nz, 0))
            s.w_hist = np.zeros((nz, 0))
            s.r_to_x_hist = np.zeros((nz, 0), dtype=np.int32)
            s.id_hist = np.zeros((nz, 0), dtype=np.int32)
            s.sum_1_hist = np.zeros((nz, 0))
            s.sum_2_hist = np.zeros((nz, 0))
            s.a_i_hist = np.zeros((nz, 0))
            s.b_i_hist = np.zeros((nz, 0))
            s.a_0_hist = np.zeros((0))
            s.i_push = 0
            s.xi_current = 0.0

    # Allocate arrays that will contain the fields experienced by the
    # particles.
    _pp_allocate_field_arrays(species_list)

    # Allocate arrays needed for the particle pusher.
    if pusher == "ab2":
        _pp_allocate_ab2_arrays(species_list)

    return species_list


@njit_serial
def pp_sort(species_list):
    """Sort plasma particles radially.
    Only some of the arrays need to be sorted
    """
    for s in species_list:
        if s.do_push:
            # When a NumPy function is called within a Numba JIT-compiled function, it does not
            # actually call NumPy. Instead, a reimplementation from Numba is used. Unfortunately,
            # the Numba version of np.argsort (both quicksort and mergesort) is significantly
            # slower than the actual NumPy argsort (here using timsort), especially for the usually
            # partially sorted arrays used here. Timsort is also available from Numba, however,
            # it is also slow and does not have a user interface. So we are forced to take the
            # performance penalty from going into object mode to call the actual NumPy sorting
            # function. Make sure that we do not pass s into object mode.
            plasma_radius = s.r
            with numba.objmode(indices="int64[::1]"):
                indices = np.argsort(plasma_radius, kind="stable")

            sort_particle_arrays(s, indices)


@njit_serial
def pp_gather_laser_sources(species_list, a2, nabla_a2, r_min, r_max, dr):
    """Gather the source terms (a^2 and nabla(a)^2) from the laser."""
    for s in species_list:
        if s.do_push:
            gather_laser_sources(a2, nabla_a2, r_min, r_max, dr, s.r, s.a2, s.nabla_a2)


@njit_serial
def pp_gather_bunch_sources(
    species_list, source_arrays, source_xi_indices, source_metadata, slice_i
):
    """Gather the source terms (b_theta) from the particle bunches."""
    for s in species_list:
        s.b_t_0[:] = 0.0
    for i in range(len(source_arrays)):
        array = source_arrays[i]
        idx = source_xi_indices[i]
        md = source_metadata[i]
        r_min = md[0]
        r_max = md[1]
        dr = md[2]
        if slice_i in idx:
            xi_index = slice_i + 2 - idx[0]
            for s in species_list:
                if s.do_push:
                    gather_bunch_sources(
                        array[xi_index], r_min, r_max, dr, s.r, s.b_t_0
                    )


@njit_serial
def pp_calculate_fields(species_list, ions_computed, max_gamma):
    """Calculate the fields at the plasma particles."""
    # Precalculate logarithms (expensive) to avoid doing so several times.
    for s in species_list:
        if s.do_push or not ions_computed:
            log(s.r, s.log_r)

    calculate_psi_and_derivatives_at_particles(species_list, ions_computed)

    for s in species_list:
        if s.do_push:
            update_gamma_and_pz(
                s.gamma,
                s.pz,
                s.pr,
                s.a2,
                s.psi,
                s.charge,
                s.mass,
            )
        if not s.is_ion:
            check_gamma(s.gamma, s.pz, s.pr, max_gamma)
    calculate_b_theta_at_particles(species_list)


@njit_serial
def pp_calculate_psi_at_grid(species_list, r_eval, psi):
    """Calculate psi on the current grid slice."""
    add = False
    for s in species_list:
        calculate_psi_with_interpolation(
            r_eval, s.r, s.log_r, s.sum_1, s.sum_2, psi, add
        )
        add = True


@njit_serial
def pp_calculate_b_theta_at_grid(species_list, r_eval, b_theta):
    """Calculate b_theta on the current grid slice."""
    for s in species_list:
        if not s.is_ion:
            calculate_b_theta_with_interpolation(
                r_eval, s.a_0[0], s.a_i, s.b_i, s.r, b_theta
            )
            return


@njit_serial
def pp_calculate_weights(species_list, ions_computed):
    """Calculate the plasma density weights of each particle."""
    for s in species_list:
        if s.do_push or not ions_computed:
            calculate_rho(
                s.charge,
                s.w,
                s.pz,
                s.gamma,
                s.rho,
            )


@njit_serial
def pp_deposit_rho(
    species_list, ions_computed, shape, rho, rho_e, rho_i, r_fld, nr, dr
):
    """Deposit plasma density on a grid slice."""
    pp_calculate_weights(species_list, ions_computed)
    # Deposit electrons
    for s in species_list:
        deposit_plasma_particles(
            s.r, s.rho, r_fld[0], nr, dr, rho_i if s.is_ion else rho_e, shape
        )
    deposit_plasma_finish(r_fld[0], nr, dr, rho_e)
    deposit_plasma_finish(r_fld[0], nr, dr, rho_i)
    rho[:] = rho_e
    rho += rho_i


@njit_serial
def pp_deposit_chi(species_list, shape, chi, r_fld, nr, dr):
    """Deposit plasma susceptibility on a grid slice."""
    for s in species_list:
        if not s.is_ion:
            calculate_chi(
                s.charge,
                s.w,
                s.pz,
                s.gamma,
                s.chi,
            )
            deposit_plasma_particles(s.r, s.chi, r_fld[0], nr, dr, chi, shape)
    deposit_plasma_finish(r_fld[0], nr, dr, chi)


@njit_serial
def pp_store_current_step(species_list, diags):
    """Store current particle properties in the history arrays."""
    for s in species_list:
        if "r" in diags or s.store_history:
            s.r_hist[-1 - s.i_push] = s.r
        if "z" in diags:
            s.xi_hist[-1 - s.i_push] = s.xi_current
        if "pr" in diags:
            s.pr_hist[-1 - s.i_push] = s.pr
        if "pz" in diags:
            s.pz_hist[-1 - s.i_push] = s.pz
        if "w" in diags:
            s.w_hist[-1 - s.i_push] = s.rho
        if "r_to_x" in diags:
            s.r_to_x_hist[-1 - s.i_push] = s.r_to_x
        if "id" in diags:
            s.id_hist[-1 - s.i_push] = s.id
        if s.store_history and not s.is_ion:
            s.a_0_hist[-1 - s.i_push] = s.a_0[0]


@njit_serial
def pp_evolve(species_list, dxi):
    """Evolve plasma particles to next longitudinal slice."""
    for s in species_list:
        if s.do_push:
            evolve_plasma_ab2(
                dxi,
                s.r,
                s.pr,
                s.gamma,
                s.mass,
                s.charge,
                s.r_to_x,
                s.nabla_a2,
                s.b_t_0,
                s.b_t,
                s.psi,
                s.dr_psi,
                s.dr,
                s.dpr,
            )
        if s.store_history:
            s.i_push += 1
            s.xi_current -= dxi

            if not s.is_ion:
                s.a_i = s.a_i_hist[-1 - s.i_push]
                s.b_i = s.b_i_hist[-1 - s.i_push]
            s.sum_1 = s.sum_1_hist[-1 - s.i_push]
            s.sum_2 = s.sum_2_hist[-1 - s.i_push]
            s.rho = s.w_hist[-1 - s.i_push]
            s.log_r = s.log_r_hist[-1 - s.i_push]

            if not s.do_push:
                s.sum_1[:] = s.sum_1_hist[-s.i_push, :]
                s.sum_2[:] = s.sum_2_hist[-s.i_push, :]
                s.log_r[:] = s.log_r_hist[-s.i_push, :]


def pp_get_history(species_list, store_history):
    """Get the history of the evolution of the plasma particles.

    Returns
    -------
    List[Dict]
        A dictionary containing the particle history arrays for each plasma species.
    """
    if store_history:
        history = list()
        for s in species_list:
            history.append(
                {
                    "r_hist": s.r_hist,
                    "log_r_hist": s.log_r_hist,
                    "xi_hist": s.xi_hist,
                    "pr_hist": s.pr_hist,
                    "pz_hist": s.pz_hist,
                    "w_hist": s.w_hist,
                    "r_to_x_hist": s.r_to_x_hist,
                    "id_hist": s.id_hist,
                    "sum_1_hist": s.sum_1_hist,
                    "sum_2_hist": s.sum_2_hist,
                    "a_i_hist": s.a_i_hist,
                    "b_i_hist": s.b_i_hist,
                    "a_0_hist": s.a_0_hist,
                }
            )
        return history


def _pp_allocate_field_arrays(species_list):
    """Allocate arrays for the fields experienced by the particles.

    In order to evolve the particles to the next longitudinal position,
    it is necessary to know the fields that they are experiencing. These
    arrays are used for storing the value of these fields at the location
    of each particle.
    """
    for s in species_list:
        if s.store_history:
            if not s.is_ion:
                s.a_i = s.a_i_hist[-1]
                s.b_i = s.b_i_hist[-1]
            else:
                s.a_i = np.zeros((0))
                s.b_i = np.zeros((0))
            s.sum_1 = s.sum_1_hist[-1]
            s.sum_2 = s.sum_2_hist[-1]
            s.rho = s.w_hist[-1]
            s.log_r = s.log_r_hist[-1]
        else:
            if not s.is_ion:
                s.a_i = np.zeros(s.num_particles)
                s.b_i = np.zeros(s.num_particles)
            else:
                s.a_i = np.zeros((0))
                s.b_i = np.zeros((0))
            s.sum_1 = np.zeros(s.num_particles + 1)
            s.sum_2 = np.zeros(s.num_particles + 1)
            s.rho = np.zeros(s.num_particles)
            s.log_r = np.zeros(s.num_particles)

        s.a2 = np.zeros(s.num_particles)
        s.nabla_a2 = np.zeros(s.num_particles)
        s.b_t_0 = np.zeros(s.num_particles)
        s.b_t = np.zeros(s.num_particles)
        s.psi = np.zeros(s.num_particles)
        s.dr_psi = np.zeros(s.num_particles)
        s.dxi_psi = np.zeros(s.num_particles)
        s.chi = np.zeros(s.num_particles)
        s.sum_3 = np.zeros(s.num_particles + 1)

        if not s.is_ion:
            s.a_0 = np.zeros(1)
            s.A = np.zeros(s.num_particles)
            s.B = np.zeros(s.num_particles)
            s.C = np.zeros(s.num_particles)
            s.K = np.zeros(s.num_particles)
            s.U = np.zeros(s.num_particles)
        else:
            s.a_0 = np.zeros(0)
            s.A = np.zeros(0)
            s.B = np.zeros(0)
            s.C = np.zeros(0)
            s.K = np.zeros(0)
            s.U = np.zeros(0)


def _pp_allocate_ab2_arrays(species_list):
    """Allocate the arrays needed for the 2nd order Adams-Bashforth pusher.

    The AB2 pusher needs the derivatives of r and pr for each particle
    at the last 2 plasma slices. This method allocates the arrays that will
    store these derivatives.
    """
    for s in species_list:
        if s.do_push:
            s.dr = np.zeros((2, s.num_particles))
            s.dpr = np.zeros((2, s.num_particles))
        else:
            s.dr = np.zeros((0, 0))
            s.dpr = np.zeros((0, 0))
