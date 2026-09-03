"""
This module implements the methods for calculating the plasma wakefields
using the 2D r-z reduced model from P. Baxevanis and G. Stupakov.

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this model.
"""

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge
from copy import deepcopy


from .plasma_particles import (
    pp_initialize,
    pp_sort,
    pp_gather_laser_sources,
    pp_gather_bunch_sources,
    pp_calculate_fields,
    pp_calculate_psi_at_grid,
    pp_calculate_b_theta_at_grid,
    pp_calculate_weights,
    pp_deposit_rho,
    pp_deposit_chi,
    pp_store_current_step,
    pp_evolve,
    pp_get_history,
)
from .b_theta_bunch import calculate_bunch_source, calculate_bunch_source_slice
from .utils import longitudinal_gradient, radial_gradient

from wake_t.utilities.numba import njit_serial
from wake_t.utilities.other import ProfStart, ProfStop

from .plasma_particle_container import PlasmaParticleContainer


def _normalize_grid(r_max, xi_min, xi_max, n_r, n_xi, n_p, r_fld):
    """Return (s_d, dr, dxi, r_fld_n) in normalized units."""
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    dr = (r_max / s_d) / n_r
    dxi = ((xi_max - xi_min) / s_d) / (n_xi - 1)
    return s_d, dr, dxi, r_fld / s_d


def _setup_laser(laser_a2, dr, n_xi, n_r):
    """Build nabla_a2 and JIT-safe dummy arrays if no laser.
    Returns (laser_a2, nabla_a2, has_laser_source)."""
    nabla_a2 = np.zeros((n_xi + 4, n_r + 4))
    has_laser_source = laser_a2 is not None
    if has_laser_source:
        radial_gradient(laser_a2[2:-2, 2:-2], dr, nabla_a2[2:-2, 2:-2])
    else:
        laser_a2 = np.zeros((0, 0))
        nabla_a2 = np.zeros((0, 0))
    return laser_a2, nabla_a2, has_laser_source


@njit_serial
def evolve_one_step(
    pp_serialized_list,
    n_xi,
    n_r,
    dxi,
    dr,
    r_fld,
    has_laser_source,
    laser_a2,
    nabla_a2,
    has_beam_source,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    max_gamma,
    psi,
    B_t,
    shape,
    calculate_rho,
    rho,
    rho_e,
    rho_i,
    chi,
    store_plasma_history,
    particle_diags,
    start_slice_i,
    stop_slice_i,
):
    """
    Evolve the plasma from slice start_slice_i down to stop_slice_i (inclusive).

    For performance reasons, this is done in a single JIT-compiled
    function to minimize the number of Python-to-Numba function calls.

    pp_serialized_list is passed in as a Tuple[Tuple[np.ndarray]]
    instead of a class so that Numba can cache the JIT compiled function.

    See calculate_wakefields() for parameters.
    """
    ions_computed = False
    pp_species_list = [
        PlasmaParticleContainer(species) for species in pp_serialized_list
    ]

    # Evolve plasma from right to left over the requested slice window.
    for slice_i in range(start_slice_i, stop_slice_i - 1, -1):
        pp_sort(pp_species_list)

        if has_laser_source:
            pp_gather_laser_sources(
                pp_species_list,
                laser_a2[slice_i + 2],
                nabla_a2[slice_i + 2],
                r_fld[0],
                r_fld[-1],
                dr,
            )

        if has_beam_source:
            pp_gather_bunch_sources(
                pp_species_list,
                bunch_source_arrays,
                bunch_source_xi_indices,
                bunch_source_metadata,
                slice_i,
            )

        pp_calculate_fields(pp_species_list, ions_computed, max_gamma)

        pp_calculate_psi_at_grid(pp_species_list, r_fld, psi[slice_i + 2, 2:-2])
        pp_calculate_b_theta_at_grid(pp_species_list, r_fld, B_t[slice_i + 2, 2:-2])

        if calculate_rho:
            pp_deposit_rho(
                pp_species_list,
                ions_computed,
                shape,
                rho[slice_i + 2],
                rho_e[slice_i + 2],
                rho_i[slice_i + 2],
                r_fld,
                n_r,
                dr,
            )
        elif "w" in particle_diags:
            pp_calculate_weights(pp_species_list, ions_computed)

        if has_laser_source:
            pp_deposit_chi(pp_species_list, shape, chi[slice_i + 2], r_fld, n_r, dr)

        ions_computed = True

        if store_plasma_history:
            pp_store_current_step(pp_species_list, particle_diags)

        if slice_i > 0:
            pp_evolve(pp_species_list, dxi)


def calculate_wakefields(
    laser_a2,
    r_max,
    xi_min,
    xi_max,
    n_r,
    n_xi,
    ppc,
    n_p,
    r_max_plasma=None,
    radial_density=None,
    p_shape="cubic",
    max_gamma=10.0,
    plasma_pusher="ab2",
    ion_motion=False,
    ion_mass=ct.m_p,
    free_electrons_per_ion=1,
    bunch_source_arrays=[],
    bunch_source_xi_indices=[],
    bunch_source_metadata=[],
    store_plasma_history=False,
    calculate_rho=True,
    particle_diags=[],
    fld_arrays=[],
    profiler=None,
):
    """
    Calculate the plasma wakefields generated by the given laser pulse and
    electron beam in the specified grid points.

    Parameters
    ----------
    laser_a2 : ndarray
        A (nz x nr) array containing the square of the laser envelope.
    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    n_r : int
        Number of grid elements along r in which to calculate the wakefields.
    n_xi : int
        Number of grid elements along xi in which to calculate the wakefields.
    ppc : array_like
        see Quasistatic2DWakefieldIons.
    n_p : float
        On-axis plasma density in units of m^{-3}.
    r_max_plasma : float
        Maximum radial extension of the plasma column. If `None`, the plasma
        extends up to the `r_max` boundary of the simulation box.
    radial_density : callable
        Function defining the radial density profile.
    p_shape : str
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'.
    max_gamma : float
        Plasma particles whose `gamma` exceeds `max_gamma` are considered to
        violate the quasistatic condition and are put at rest (i.e.,
        `gamma=1.`, `pr=pz=0.`).
    plasma_pusher : str
        Numerical pusher for the plasma particles. Possible values are `'ab2'`.
    ion_motion : bool, optional
        Whether to allow the plasma ions to move. By default, False.
    ion_mass : float, optional
        Mass of the plasma ions. By default, the mass of a proton.
    free_electrons_per_ion : int, optional
        Number of free electrons per ion. The ion charge is adjusted
        accordingly to maintain a quasi-neutral plasma (i.e.,
        ion charge = e * free_electrons_per_ion). By default, 1.
    bunch_source_arrays : list, optional
        List containing the array from which the bunch source terms (the
        azimuthal magnetic field) will be gathered. It can be a single
        array for the whole domain, or one array per bunch when using
        adaptive grids.
    bunch_source_xi_indices : list, optional
        List containing 1d arrays that with the indices of the longitudinal
        plasma slices that can gather from them. This is needed because the
        adaptive grids might not extend the whole longitudinal domain of the
        plasma, so the plasma slices should only try to gather the source terms
        if they are available at the current slice.
    bunch_source_metadata : list, optional
        Metadata of each bunch source array.
    store_plasma_history : bool, optional
        Whether to store the plasma particle evolution. This might be needed
        for diagnostics or the use of adaptive grids. By default, False.
    calculate_rho : bool, optional
        Whether to deposit the plasma density. This might be needed for
        diagnostics. By default, False.
    particle_diags : list, optional
        List of particle quantities to save to diagnostics.
    fld_arrays : list, optional
        List of all the fields.
    """
    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    # Convert to normalized units.
    s_d, dr, dxi, r_fld = _normalize_grid(r_max, xi_min, xi_max, n_r, n_xi, n_p, r_fld)
    ppc = ppc.copy()
    ppc[:, 0] /= s_d
    r_max_plasma = r_max_plasma / s_d
    xi_fld = xi_fld / s_d

    def radial_density_normalized(r):
        return radial_density(r * s_d) / n_p

    # Initialize field arrays, including guard cells.
    psi = np.zeros((n_xi + 4, n_r + 4))

    # Laser source.
    laser_a2, nabla_a2, has_laser_source = _setup_laser(laser_a2, dr, n_xi, n_r)

    has_beam_source = len(bunch_source_arrays) > 0
    if not has_beam_source:
        # need to set the dtype for JIT
        bunch_source_arrays.append(np.zeros((0, 0)))
        bunch_source_xi_indices.append(np.zeros(0, dtype=np.int64))
        bunch_source_metadata.append(np.zeros(0))

    if len(particle_diags) == 0:
        # need to set the type for JIT
        particle_diags = ["none"]

    # Calculate plasma response (including density, susceptibility, potential
    # and magnetic field)

    # Initialize plasma particles.
    # Set parameters for electron and ion species in normalized units
    init_list = [
        {
            "charge": free_electrons_per_ion,
            "mass": free_electrons_per_ion,
            "is_ion": False,
        },
        {
            "charge": -free_electrons_per_ion,
            "mass": ion_mass / ct.m_e,
            "is_ion": True,
        },
    ]

    ProfStart("wakefield.initialize")

    species_list = pp_initialize(
        init_list,
        n_xi,
        ppc,
        dr,
        radial_density_normalized,
        ion_motion,
        store_plasma_history,
        plasma_pusher,
    )

    ProfStop("wakefield.initialize")

    ProfStart("wakefield.evolve")

    evolve_one_step(
        tuple(s.serialize() for s in species_list),
        n_xi,
        n_r,
        dxi,
        dr,
        r_fld,
        has_laser_source,
        laser_a2,
        nabla_a2,
        has_beam_source,
        tuple(bunch_source_arrays),
        tuple(bunch_source_xi_indices),
        tuple(bunch_source_metadata),
        max_gamma,
        psi,
        B_t,
        p_shape,
        calculate_rho,
        rho,
        rho_e,
        rho_i,
        chi,
        store_plasma_history,
        tuple(particle_diags),
        n_xi - 1,
        0,
    )

    ProfStop("wakefield.evolve")

    # Calculate derived fields (E_z, W_r, and E_r).
    E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)
    longitudinal_gradient(psi[2:-2, 2:-2], dxi, E_z[2:-2, 2:-2])
    radial_gradient(psi[2:-2, 2:-2], dr, E_r[2:-2, 2:-2])
    E_r -= B_t
    E_z *= -E_0
    E_r *= -E_0
    # B_t[:] = (b_t_bar + b_t_beam) * E_0 / ct.c
    B_t *= E_0 / ct.c
    return pp_get_history(species_list, store_plasma_history)


def calculate_wakefields_salame(
    laser_a2,
    r_max,
    xi_min,
    xi_max,
    n_r,
    n_xi,
    ppc,
    n_p,
    q_bunch,
    q_fixed,
    q_var,
    b_t_bunch,
    salame_max_iter=10,
    salame_tol=1e-4,
    use_avg_psi=False,
    r_max_plasma=None,
    radial_density=None,
    p_shape="cubic",
    max_gamma=10.0,
    plasma_pusher="ab2",
    ion_motion=False,
    ion_mass=ct.m_p,
    free_electrons_per_ion=1,
    store_plasma_history=False,
    calculate_rho=True,
    particle_diags=None,
    fld_arrays=None,
):
    """
    Full wakefield solve with inline SALAME bisection.

    Compared to the two-pass approach (beamloading_initial_condition +
    calculate_wakefields), this function performs a single plasma initialization
    and evolves the plasma column once:
      Phase 1 (JIT): slices n_xi-1 ... k_head+2  (pre-witness, fixed source)
      Phase 2 (Python): SALAME bisection for each witness slice (k_head ... k_tail)
      Phase 3 (JIT): slices k_tail-1 ... 0  (post-witness, shaped source)

    q_var is updated in-place with the shaped witness deposit.
    q_bunch is updated in-place to q_fixed + q_var (shaped).
    b_t_bunch is updated in-place.
    """

    if particle_diags is None:
        particle_diags = []
    if fld_arrays is None:
        fld_arrays = []

    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    s_d, dr, dxi, r_fld_n = _normalize_grid(
        r_max, xi_min, xi_max, n_r, n_xi, n_p, r_fld
    )
    ppc_n = ppc.copy()
    ppc_n[:, 0] /= s_d

    def radial_density_normalized(r):
        return radial_density(r * s_d) / n_p

    psi = np.zeros((n_xi + 4, n_r + 4))
    laser_a2, nabla_a2, has_laser_source = _setup_laser(laser_a2, dr, n_xi, n_r)

    if len(particle_diags) == 0:
        particle_diags = ["none"]

    # --- Bunch source: Phase 1 uses fixed-only source ---
    calculate_bunch_source(q_fixed, n_r, n_xi, b_t_bunch)
    dr_phys = r_max / n_r
    bsmd_base = np.array([r_fld[0], r_fld[-1] + 2 * dr_phys, dr_phys]) / s_d
    bsa = [b_t_bunch]
    bsxi = [np.arange(n_xi, dtype=np.int64)]
    bsmd = [bsmd_base]

    # --- Find witness longitudinal support ---
    g_line_var = np.sum(q_var[2:-2, 2:-2], axis=1)  # (n_xi,)
    support = np.where(np.abs(g_line_var) > 0.0)[0]
    k_head = int(support[-1])
    k_tail = int(support[0])

    # --- Initialize plasma particles ---
    init_list = [
        {
            "charge": free_electrons_per_ion,
            "mass": free_electrons_per_ion,
            "is_ion": False,
        },
        {"charge": -free_electrons_per_ion, "mass": ion_mass / ct.m_e, "is_ion": True},
    ]
    species_list = pp_initialize(
        init_list,
        n_xi,
        ppc_n,
        dr,
        radial_density_normalized,
        ion_motion,
        store_plasma_history,
        plasma_pusher,
    )
    pp_state = tuple(s.serialize() for s in species_list)

    # Scratch arrays reused across bisection trials (calculate_rho=False in trials)
    psi_sc = np.zeros((n_xi + 4, n_r + 4))
    B_t_sc = np.zeros((n_xi + 4, n_r + 4))
    _rho_sc = np.zeros((n_xi + 4, n_r + 4))
    _chi_sc = np.zeros((n_xi + 4, n_r + 4))

    E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)

    def _eval_ez_weighted_km1(qv_trial, pp_state_in, k, _use_avg_psi=None):
        """
        Deepcopy pp_state_in, update b_t_bunch at slice k with trial witness,
        evolve k..k-2, return weighted Ez at k-1.
        pp_state_in must be "ready for slice k" (pp_evolve done for k+1).
        """
        q_bunch[:] = q_fixed + qv_trial
        calculate_bunch_source_slice(q_bunch, n_r, k, b_t_bunch)

        pp_trial = deepcopy(pp_state_in)
        evolve_one_step(
            pp_trial,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            True,
            tuple(bsa),
            tuple(bsxi),
            tuple(bsmd),
            max_gamma,
            psi_sc,
            B_t_sc,
            p_shape,
            False,
            _rho_sc,
            _rho_sc,
            _rho_sc,
            _chi_sc,
            False,
            ("none",),
            k,
            k - 2,
        )

        psi_k = psi_sc[2 + k, :]
        psi_km2 = psi_sc[2 + k - 2, :]
        psi_km1 = psi_sc[2 + k - 1, :]

        if _use_avg_psi is None:
            _use_avg_psi = use_avg_psi

        if not _use_avg_psi:
            Ez_r = -(psi_k - psi_km2) / (2.0 * dxi) * E_0
        else:
            Ez_r = -((psi_k + psi_km1) / 2.0 - psi_km2) / (1.5 * dxi) * E_0

        # Weight by witness charge at k-1
        qb_slice = qv_trial[2 + k - 1, 2:-2]
        w = np.abs(qb_slice)
        s = w.sum()
        if s == 0.0:
            w = np.ones(len(w))
            s = w.sum()
        return float((Ez_r[2:-2] * w).sum() / s)

    def _set_var_slice(qv2d, k_, g_new):
        """Return a copy of qv2d with slice k_ rescaled to line charge g_new."""
        qv_new = qv2d.copy()
        g_old = float(np.sum(qv_new[2 + k_, 2:-2]))
        if g_old == 0.0:
            return qv_new
        qv_new[2 + k_, 2:-2] *= g_new / g_old
        return qv_new

    # -------------------------------------------------------------------
    # Phase 1: evolve nxi-1 → k_head+2 (pp_state becomes ready for k_head+1)
    # -------------------------------------------------------------------
    if n_xi - 1 >= k_head + 2:
        evolve_one_step(
            pp_state,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            True,
            tuple(bsa),
            tuple(bsxi),
            tuple(bsmd),
            max_gamma,
            psi,
            B_t,
            p_shape,
            calculate_rho,
            rho,
            rho_e,
            rho_i,
            chi,
            store_plasma_history,
            tuple(particle_diags),
            n_xi - 1,
            k_head + 2,
        )

    # -------------------------------------------------------------------
    # Compute Ez_target: evolve trial k_head+1..k_head-1 from pp_state
    # (pp_state is now ready for k_head+1)
    # -------------------------------------------------------------------
    Ez_target = _eval_ez_weighted_km1(q_var, pp_state, k_head + 1, _use_avg_psi=True)

    # Commit k_head+1 → pp_state ready for k_head
    evolve_one_step(
        pp_state,
        n_xi,
        n_r,
        dxi,
        dr,
        r_fld_n,
        has_laser_source,
        laser_a2,
        nabla_a2,
        True,
        tuple(bsa),
        tuple(bsxi),
        tuple(bsmd),
        max_gamma,
        psi,
        B_t,
        p_shape,
        calculate_rho,
        rho,
        rho_e,
        rho_i,
        chi,
        store_plasma_history,
        tuple(particle_diags),
        k_head + 1,
        k_head + 1,
    )

    # -------------------------------------------------------------------
    # Phase 2: SALAME bisection for each witness slice k_head .. k_tail+1
    # -------------------------------------------------------------------
    g_line_var0 = np.sum(q_var[2:-2, 2:-2], axis=1).copy()
    qv_current = q_var.copy()

    for k in range(k_head, k_tail, -1):
        if np.abs(g_line_var0[k]) == 0.0:
            # No witness charge at this slice — just advance pp_state
            q_bunch[:] = q_fixed + qv_current
            calculate_bunch_source_slice(q_bunch, n_r, k, b_t_bunch)
            evolve_one_step(
                pp_state,
                n_xi,
                n_r,
                dxi,
                dr,
                r_fld_n,
                has_laser_source,
                laser_a2,
                nabla_a2,
                True,
                tuple(bsa),
                tuple(bsxi),
                tuple(bsmd),
                max_gamma,
                psi,
                B_t,
                p_shape,
                calculate_rho,
                rho,
                rho_e,
                rho_i,
                chi,
                store_plasma_history,
                tuple(particle_diags),
                k,
                k,
            )
            continue

        g_min = -1e-100
        g_max = 5.0 * g_line_var0[k]

        qv_min = _set_var_slice(qv_current, k, g_min)
        qv_max = _set_var_slice(qv_current, k, g_max)
        Ez_min = _eval_ez_weighted_km1(qv_min, pp_state, k)
        Ez_max = _eval_ez_weighted_km1(qv_max, pp_state, k)

        while np.abs(Ez_max) > np.abs(Ez_target):
            g_max *= 5.0
            qv_max = _set_var_slice(qv_current, k, g_max)
            Ez_max = _eval_ez_weighted_km1(qv_max, pp_state, k)
            if g_max == 0.0 or not np.isfinite(g_max):
                break

        qv_new = qv_current
        for _ in range(salame_max_iter):
            den = np.abs(Ez_max - Ez_min)
            if den == 0.0:
                break

            if Ez_target < Ez_min:
                g_new = g_min
                qv_new = _set_var_slice(qv_current, k, g_new)
                print(
                    f"SALAME needs positive charge for slice {k} at xi= {xi_fld[k]}. The charge at this slice is set as 0."
                )
                break

            wg = np.abs(Ez_target - Ez_min) / den
            g_new = wg * g_max + (1.0 - wg) * g_min
            qv_try = _set_var_slice(qv_current, k, g_new)
            Ez_try = _eval_ez_weighted_km1(qv_try, pp_state, k)

            if np.abs(Ez_try) > np.abs(Ez_target):
                g_min, Ez_min = g_new, Ez_try
            else:
                g_max, Ez_max = g_new, Ez_try

            rel = np.abs(Ez_try - Ez_target) / (np.abs(Ez_target) + 1e-300)
            qv_new = qv_try
            if rel < salame_tol:
                break

        qv_current = qv_new

        # Commit slice k with final shaped witness
        q_bunch[:] = q_fixed + qv_current
        calculate_bunch_source_slice(q_bunch, n_r, k, b_t_bunch)
        evolve_one_step(
            pp_state,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            True,
            tuple(bsa),
            tuple(bsxi),
            tuple(bsmd),
            max_gamma,
            psi,
            B_t,
            p_shape,
            calculate_rho,
            rho,
            rho_e,
            rho_i,
            chi,
            store_plasma_history,
            tuple(particle_diags),
            k,
            k,
        )

    # -------------------------------------------------------------------
    # Phase 3: evolve k_tail..0 with fully shaped source
    # -------------------------------------------------------------------
    # Rebuild b_t_bunch for all slices using final shaped q_bunch
    q_bunch[:] = q_fixed + qv_current
    calculate_bunch_source(q_bunch, n_r, n_xi, b_t_bunch)

    if k_tail >= 0:
        evolve_one_step(
            pp_state,
            n_xi,
            n_r,
            dxi,
            dr,
            r_fld_n,
            has_laser_source,
            laser_a2,
            nabla_a2,
            True,
            tuple(bsa),
            tuple(bsxi),
            tuple(bsmd),
            max_gamma,
            psi,
            B_t,
            p_shape,
            calculate_rho,
            rho,
            rho_e,
            rho_i,
            chi,
            store_plasma_history,
            tuple(particle_diags),
            k_tail,
            0,
        )

    # Update caller's q_var in-place with shaped witness
    q_var[:] = qv_current

    # --- Derived fields (same as calculate_wakefields) ---
    longitudinal_gradient(psi[2:-2, 2:-2], dxi, E_z[2:-2, 2:-2])
    radial_gradient(psi[2:-2, 2:-2], dr, E_r[2:-2, 2:-2])
    E_r -= B_t
    E_z *= -E_0
    E_r *= -E_0
    B_t *= E_0 / ct.c

    return pp_get_history(species_list, store_plasma_history)
