import numpy as np
from numba import njit

from wake_t.particles.interpolation import gather_sources_qs_baxevanis
from ..psi_and_derivatives import calculate_psi_and_derivatives_at_particles
from ..b_theta import calculate_b_theta_at_particles


def evolve_plasma_rk4(pp, dxi, xi, a2, nabla_a2, b_theta_0, r_fld, xi_fld):
    """
    Evolve the r and pr coordinates of plasma particles to the next xi step
    using a Runge-Kutta method of 4th order.

    Parameters:
    -----------

    pp : PlasmaParticles
        The plasma particles to be evolved.
    dxi : float
        Longitudinal step.
    xi : float
        Current xi position (speed-of-light frame) of the plasma particles.
    a2 : ndarray
        2D array with the square of the laser envelope in the r-z grid.
    nabla_a2 : ndarray
        2D array with the radial derivative of the square of the laser envelope
        in the r-z grid.
    b_theta_0 : ndarray
        2D array with the azimuthal magnetic field of the electron beam(s)
        in the r-z grid.
    r_fld : ndarray
        Array containing the radial position of the grid points.
    xi_fld : ndarray
        Array containing the longitudinal position of the grid points.
    """
    # Calculate derivatives of r and pr at the 4 substeps.
    for i in range(4):
        derivatives_substep(i, pp, xi, a2, nabla_a2, b_theta_0, r_fld, xi_fld)

    dr_arrays, dpr_arrays = pp.get_rk4_arrays()

    # Advance radial position and momentum.
    apply_rk4(pp.r, dxi, *dr_arrays)
    apply_rk4(pp.pr, dxi, *dpr_arrays)

    # If a particle has crossed the axis, mirror it.
    idx_neg = np.where(pp.r < 0.)
    if idx_neg[0].size > 0:
        pp.r[idx_neg] *= -1.
        pp.pr[idx_neg] *= -1.


def derivatives_substep(i, pp, xi, a2, nabla_a2, b_theta_0, r_fld, xi_fld):
    """Calculate r and pr derivatives at the i-th RK4 substep.

    The Runge-Kutta method of 4th order requires knowing the derivative (slope)
    of the variable to advance not only at the current location, but also at
    three other intermediate steps. This method takes care of calculating
    the derivatives of r and pr at the 4 required substeps. This includes
    gathering and computing all the fields at the location of each particle,
    which are required to calculate the derivatives.

    Parameters
    ----------
    i : int
        Index of the RK4 substep (between 0 and 3).
    pp : PlasmaParticles
        The plasma particles being evolved.
    xi : float
        Longitudinal position of the plasma slice at the beginning of the push.
    a2 : ndarray
        2D array with the square of the laser envelope in the r-z grid.
    nabla_a2 : ndarray
        2D array with the radial derivative of the square of the laser envelope
        in the r-z grid.
    b_theta_0 : ndarray
        2D array with the azimuthal magnetic field of the electron beam(s)
        in the r-z grid.
    r_fld : ndarray
        Array containing the radial position of the grid points.
    xi_fld : ndarray
        Array containing the longitudinal position of the grid points.
    """
    # Get arrays that contain the fields at the position of plasma particles.
    # There is one set of arrays for each substep of the Runge-Kutta push.
    (a2_pp, nabla_a2_pp, b_theta_0_pp, b_theta_pp,
     psi_pp, dr_psi_pp, dxi_psi_pp) = pp.get_rk4_field_arrays(i)

    # Get arrays that contain, or will contain, the derivatives of r and pr
    # at each substep.
    dr_arrays, dpr_arrays = pp.get_rk4_arrays()

    # The first substep (at the initial location of the plasma slice) already
    # has all the fields at the plasma particles (they were gathered/calculated
    # in the main loop of the Baxevanis solver). For the other 3 substeps of
    # the Runge-Kutta push, they still need to be obtained:
    if i != 0:
        dxi = xi_fld[1] - xi_fld[0]
        dr = r_fld[1] - r_fld[0]

        if i in [1, 2]:
            mult = 0.5
        else:
            mult = 1

        r = pp.r + dxi * dr_arrays[i-1] * mult
        pr = pp.pr + dxi * dpr_arrays[i-1] * mult
        q = pp.q
        xi = xi - dxi * mult

        # Check for particles with negative radial position and mirror them.
        idx_neg = np.where(r < 0.)
        if idx_neg[0].size > 0:
            r[idx_neg] *= -1.
            pr[idx_neg] *= -1.

        # Gather source terms at position of plasma particles.
        gather_sources_qs_baxevanis(
            a2, nabla_a2, b_theta_0, xi_fld[0], xi_fld[-1],
            r_fld[0], r_fld[-1], dxi, dr, pp.r, xi, a2_pp, nabla_a2_pp,
            b_theta_0_pp)

        # Get sorted particle indices
        idx = np.argsort(r)

        # Calculate wakefield potential and derivatives at plasma particles.
        calculate_psi_and_derivatives_at_particles(
            r, pr, q, idx, pp.r_max_plasma, pp.dr_p, pp.parabolic_coefficient,
            psi_pp, dr_psi_pp, dxi_psi_pp)

        # Calculate gamma of plasma particles
        gamma = (
            1. + pr ** 2 + a2_pp + (1. + psi_pp) ** 2) / (2. * (1. + psi_pp))

        # Calculate azimuthal magnetic field from the plasma at the location of
        # the plasma particles.
        calculate_b_theta_at_particles(
            r, pr, q, gamma, psi_pp, dr_psi_pp, dxi_psi_pp,
            b_theta_0_pp, nabla_a2_pp, idx, pp.dr_p, b_theta_pp)
    else:
        r = pp.r
        pr = pp.pr
        gamma = pp.gamma

    # Using the gathered/calculated fields, compute derivatives of r and pr
    # at the current slice.
    dr_pp = dr_arrays[i]
    dpr_pp = dpr_arrays[i]
    calculate_derivatives(
        pr, gamma, b_theta_0_pp, nabla_a2_pp, b_theta_pp,
        psi_pp, dr_psi_pp, dr_pp, dpr_pp
    )

    if i != 0:
        # For particles which crossed the axis and were inverted, invert now
        # back the sign of the derivatives.
        if idx_neg[0].size > 0:
            dr_pp[idx_neg] *= -1.
            dpr_pp[idx_neg] *= -1.


@njit()
def calculate_derivatives(
        pr, gamma, b_theta_0, nabla_a2, b_theta_bar, psi, dr_psi, dr, dpr):
    """
    Calculate the derivative of the radial position and the radial momentum
    of the plasma particles at the current slice.

    Parameters:
    -----------
    pr, gamma : ndarray
        Arrays containing the radial momentum and Lorentz factor of the
        plasma particles.
    b_theta_0 : ndarray
        Array containing the value of the azimuthal magnetic field from
        the beam distribution at the position of each plasma particle.
    nabla_a2 : ndarray
        Array containing the value of the gradient of the laser normalized
        vector potential at the position of each plasma particle.
    b_theta_bar : ndarray
        Array containing the value of the azimuthal magnetic field from
        the plasma at the position of each plasma particle.
    psi, dr_psi : ndarray
        Arrays containing the wakefield potential and its radial derivative
        at the position of each plasma particle.
    dr, dpr : ndarray
        Arrays where the value of the derivatives of the radial position and
        radial momentum will be stored.
    """
    # Calculate derivatives of r and pr.
    for i in range(pr.shape[0]):
        psi_i = psi[i]
        dpr[i] = (gamma[i] * dr_psi[i] / (1. + psi_i)
                  - b_theta_bar[i]
                  - b_theta_0[i]
                  - nabla_a2[i] / (2. * (1. + psi_i)))
        dr[i] = pr[i] / (1. + psi_i)


@njit()
def apply_rk4(x, dt, kx_1, kx_2, kx_3, kx_4):
    """Apply the Runge-Kutta method of 4th order to evolve `x`

    Parameters
    ----------
    x : ndarray
        Array containing the variable to be advanced.
    dt : _type_
        Discretization step size.
    dx_1, dx_2, dx_3, dx_4 : ndarray
        Arrays containing the derivatives of `x` at the 4 substeps.
    """
    inv_6 = 1. / 6.
    for i in range(x.shape[0]):
        x[i] += dt * (kx_1[i] + 2. * (kx_2[i] + kx_3[i]) + kx_4[i]) * inv_6
