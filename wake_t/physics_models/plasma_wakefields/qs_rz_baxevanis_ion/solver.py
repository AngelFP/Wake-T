"""
This module implements the methods for calculating the plasma wakefields
using the 2D r-z reduced model from P. Baxevanis and G. Stupakov.

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this model.
"""

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from .plasma_particles import PlasmaParticles
from .utils import longitudinal_gradient, radial_gradient


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
):
    """
    Calculate the plasma wakefields generated by the given laser pulse and
    electron beam in the specified grid points.

    Parameters
    ----------
    laser_a2 : ndarray
        A (nz x nr) array containing the square of the laser envelope.
    beam_part : list
        List of numpy arrays containing the spatial coordinates and charge of
        all beam particles, i.e [x, y, xi, q].
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
    ppc : int (optional)
        Number of plasma particles per 1d cell along the radial direction.
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
    """
    rho, rho_e, rho_i, chi, E_r, E_z, B_t, xi_fld, r_fld = fld_arrays

    # Convert to normalized units.
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max = r_max / s_d
    xi_min = xi_min / s_d
    xi_max = xi_max / s_d
    dr = r_max / n_r
    dxi = (xi_max - xi_min) / (n_xi - 1)
    ppc = ppc.copy()
    ppc[:, 0] /= s_d
    r_max_plasma = r_max_plasma / s_d

    def radial_density_normalized(r):
        return radial_density(r * s_d) / n_p

    # Field node coordinates.
    r_fld = r_fld / s_d
    xi_fld = xi_fld / s_d

    # Initialize field arrays, including guard cells.
    nabla_a2 = np.zeros((n_xi + 4, n_r + 4))
    psi = np.zeros((n_xi + 4, n_r + 4))

    # Laser source.
    laser_source = laser_a2 is not None
    if laser_source:
        radial_gradient(laser_a2[2:-2, 2:-2], dr, nabla_a2[2:-2, 2:-2])

    # Calculate plasma response (including density, susceptibility, potential
    # and magnetic field)
    pp_hist = calculate_plasma_response(
        r_max,
        r_max_plasma,
        radial_density_normalized,
        dr,
        ppc,
        n_r,
        plasma_pusher,
        p_shape,
        max_gamma,
        ion_motion,
        ion_mass,
        free_electrons_per_ion,
        n_xi,
        laser_a2,
        nabla_a2,
        laser_source,
        bunch_source_arrays,
        bunch_source_xi_indices,
        bunch_source_metadata,
        r_fld,
        psi,
        B_t,
        rho,
        rho_e,
        rho_i,
        chi,
        dxi,
        store_plasma_history=store_plasma_history,
        calculate_rho=calculate_rho,
        particle_diags=particle_diags,
    )

    # Calculate derived fields (E_z, W_r, and E_r).
    E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)
    longitudinal_gradient(psi[2:-2, 2:-2], dxi, E_z[2:-2, 2:-2])
    radial_gradient(psi[2:-2, 2:-2], dr, E_r[2:-2, 2:-2])
    E_r -= B_t
    E_z *= -E_0
    E_r *= -E_0
    # B_t[:] = (b_t_bar + b_t_beam) * E_0 / ct.c
    B_t *= E_0 / ct.c
    return pp_hist


def calculate_plasma_response(
    r_max,
    r_max_plasma,
    radial_density_normalized,
    dr,
    ppc,
    n_r,
    plasma_pusher,
    p_shape,
    max_gamma,
    ion_motion,
    ion_mass,
    free_electrons_per_ion,
    n_xi,
    a2,
    nabla_a2,
    laser_source,
    bunch_source_arrays,
    bunch_source_xi_indices,
    bunch_source_metadata,
    r_fld,
    psi,
    b_t_bar,
    rho,
    rho_e,
    rho_i,
    chi,
    dxi,
    store_plasma_history,
    calculate_rho,
    particle_diags,
):
    # Initialize plasma particles.
    pp = PlasmaParticles(
        r_max,
        r_max_plasma,
        dr,
        ppc,
        n_r,
        n_xi,
        radial_density_normalized,
        max_gamma,
        ion_motion,
        ion_mass,
        free_electrons_per_ion,
        plasma_pusher,
        p_shape,
        store_plasma_history,
        particle_diags,
    )
    pp.initialize()

    # Evolve plasma from right to left and calculate psi, b_t_bar, rho and
    # chi on a grid.
    for step in range(n_xi):
        slice_i = n_xi - step - 1

        pp.sort()

        if laser_source:
            pp.gather_laser_sources(
                a2[slice_i + 2], nabla_a2[slice_i + 2], r_fld[0], r_fld[-1], dr
            )
        pp.gather_bunch_sources(
            bunch_source_arrays,
            bunch_source_xi_indices,
            bunch_source_metadata,
            slice_i,
        )

        pp.calculate_fields()

        pp.calculate_psi_at_grid(r_fld, psi[slice_i + 2, 2:-2])
        pp.calculate_b_theta_at_grid(r_fld, b_t_bar[slice_i + 2, 2:-2])

        if calculate_rho:
            pp.deposit_rho(
                rho[slice_i + 2],
                rho_e[slice_i + 2],
                rho_i[slice_i + 2],
                r_fld,
                n_r,
                dr,
            )
        elif "w" in particle_diags:
            pp.calculate_weights()
        if laser_source:
            pp.deposit_chi(chi[slice_i + 2], r_fld, n_r, dr)

        pp.ions_computed = True

        if store_plasma_history:
            pp.store_current_step()
        if slice_i > 0:
            pp.evolve(dxi)
    return pp.get_history()
