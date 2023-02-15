import numpy as np

from wake_t.utilities.numba import njit_serial
from wake_t.particles.interpolation import gather_sources_qs_baxevanis
from ..psi_and_derivatives import calculate_psi_and_derivatives_at_particles
from ..b_theta import calculate_b_theta_at_particles


@njit_serial()
def evolve_plasma_rk4(
        dxi, dr, xi, r, pr, gamma, q, r_max_plasma, dr_p, pc,
        a2, nabla_a2, b_t_0, r_fld, xi_fld,
        dr_1, dr_2, dr_3, dr_4, dpr_1, dpr_2, dpr_3, dpr_4,
        a2_1, nabla_a2_1, b_t_0_1, b_t_1, psi_1, dr_psi_1, dxi_psi_1,
        a2_2, nabla_a2_2, b_t_0_2, b_t_2, psi_2, dr_psi_2, dxi_psi_2,
        a2_3, nabla_a2_3, b_t_0_3, b_t_3, psi_3, dr_psi_3, dxi_psi_3,
        a2_4, nabla_a2_4, b_t_0_4, b_t_4, psi_4, dr_psi_4, dxi_psi_4):
    """
    Evolve the r and pr coordinates of plasma particles to the next xi step
    using a Runge-Kutta method of 4th order.

    Parameters
    ----------
    dxi, dr : float
        Longitudinal and radial step.
    xi : float
        Current longitudinal position of the plasma slice.
    r, pr, gamma, q : ndarray
        Radial position, radial momentum, Lorentz factor and charge of the
        plasma particles.
    r_max_plasma : float
        Maximum radial extent of the plasma
    dr_p : float
        Initial radial spacing between plasma particles.
    pc : float
        Coefficient for the parabolic radial plasma profile.
    a2, nabla_a2, b_t_0 : ndarray
        Laser and beam source fields.
    xi_fld, r_fld : ndarray
        Grid coordinates
    dr_1, ..., dr_4 : ndarray
        Arrays containing the derivative of the radial position of the
        particles at the current slice and the 3 intermediate steps.
    dpr_1, ..., dpr_4 : ndarray
        Arrays containing the derivative of the radial momentum of the
        particles at the current slice and the 3 intermediate steps.
    a2_i, ..., dxi_psi_i : ndarray
        Arrays where the field values at the particle positions at substep i
        will be stored.
    """
    # Calculate derivatives of r and pr at the current slice.
    calculate_derivatives(
        pr, gamma,
        b_t_0_1, nabla_a2_1, b_t_1, psi_1, dr_psi_1,
        dr_1, dpr_1)

    # Calculate derivatives of r and pr at the three RK4 substeps.
    derivatives_substep(
        xi - dxi * 0.5, r + dxi * dr_1 * 0.5, pr + dxi * dpr_1 * 0.5, q,
        dxi, dr, r_max_plasma, dr_p, pc,
        a2, nabla_a2, b_t_0, r_fld, xi_fld,
        a2_2, nabla_a2_2, b_t_0_2, b_t_2, psi_2, dr_psi_2, dxi_psi_2,
        dr_2, dpr_2)
    derivatives_substep(
        xi - dxi * 0.5, r + dxi * dr_2 * 0.5, pr + dxi * dpr_2 * 0.5, q,
        dxi, dr, r_max_plasma, dr_p, pc,
        a2, nabla_a2, b_t_0, r_fld, xi_fld,
        a2_3, nabla_a2_3, b_t_0_3, b_t_3, psi_3, dr_psi_3, dxi_psi_3,
        dr_3, dpr_3)
    derivatives_substep(
        xi - dxi, r + dxi * dr_3, pr + dxi * dpr_3, q,
        dxi, dr, r_max_plasma, dr_p, pc,
        a2, nabla_a2, b_t_0, r_fld, xi_fld,
        a2_4, nabla_a2_4, b_t_0_4, b_t_4, psi_4, dr_psi_4, dxi_psi_4,
        dr_4, dpr_4)

    # Advance radial position and momentum.
    apply_rk4(r, dxi, dr_1, dr_2, dr_3, dr_4)
    apply_rk4(pr, dxi, dpr_1, dpr_2, dpr_3, dpr_4)

    # If a particle has crossed the axis, mirror it.
    idx_neg = np.where(r < 0.)
    if idx_neg[0].size > 0:
        r[idx_neg] *= -1.
        pr[idx_neg] *= -1.


@njit_serial()
def derivatives_substep(
        xi, r, pr, q, dxi, dr, r_max_plasma, dr_p, pc,
        a2, nabla_a2, b_t_0, r_fld, xi_fld,
        a2_i, nabla_a2_i, b_t_0_i, b_t_i, psi_i, dr_psi_i, dxi_psi_i,
        dr_i, dpr_i):
    """Calculate r and pr derivatives at the i-th RK4 substep.

    The Runge-Kutta method of 4th order requires knowing the derivative (slope)
    of the variable to advance not only at the current location, but also at
    three other intermediate steps. This method takes care of calculating
    the derivatives of r and pr at the 3 required substeps. This includes
    gathering and computing all the fields at the location of each particle,
    which are required to calculate the derivatives.

    Parameters
    ----------
    xi : float
        Current longitudinal position of the plasma slice.
    r, pr, q : ndarray
        Radial position, radial momentum and charge of the plasma particles.
    dxi, dr : float
        Grid spacing.
    r_max_plasma : float
        Maximum radial extent of the plasma
    dr_p : float
        Initial radial spacing between plasma particles.
    pc : float
        Coefficient for the parabolic radial plasma profile.
    a2, nabla_a2, b_t_0 : ndarray
        Laser and beam source fields.
    xi_fld, r_fld : ndarray
        Grid coordinates
    a2_i, ..., dxi_psi_i : ndarray
        Arrays where the field values at the particle positions at substep i
        will be stored.
    dr_i, dpr_i : ndarray
        Arrays that will contain the derivative of the radial position and of
        the radial momentum of the particles at substep i.
    """

    # Check for particles with negative radial position and mirror them.
    idx_neg = np.where(r < 0.)
    if idx_neg[0].size > 0:
        r[idx_neg] *= -1.
        pr[idx_neg] *= -1.

    # Gather source terms at position of plasma particles.
    gather_sources_qs_baxevanis(
        a2, nabla_a2, b_t_0, xi_fld[0], xi_fld[-1],
        r_fld[0], r_fld[-1], dxi, dr, r, xi, a2_i, nabla_a2_i,
        b_t_0_i)

    # Get sorted particle indices
    idx = np.argsort(r)

    # Calculate wakefield potential and derivatives at plasma particles.
    calculate_psi_and_derivatives_at_particles(
        r, pr, q, idx, r_max_plasma, dr_p, pc,
        psi_i, dr_psi_i, dxi_psi_i)

    # Calculate gamma of plasma particles
    gamma = (
        1. + pr ** 2 + a2_i + (1. + psi_i) ** 2) / (2. * (1. + psi_i))

    # Calculate azimuthal magnetic field from the plasma at the location of
    # the plasma particles.
    calculate_b_theta_at_particles(
        r, pr, q, gamma, psi_i, dr_psi_i, dxi_psi_i,
        b_t_0_i, nabla_a2_i, idx, dr_p, b_t_i)

    # Using the gathered/calculated fields, compute derivatives of r and pr
    # at the current slice.
    calculate_derivatives(
        pr, gamma, b_t_0_i, nabla_a2_i, b_t_i,
        psi_i, dr_psi_i, dr_i, dpr_i
    )

    # For particles which crossed the axis and were inverted, invert now
    # back the sign of the derivatives.
    if idx_neg[0].size > 0:
        dr_i[idx_neg] *= -1.
        dpr_i[idx_neg] *= -1.


@njit_serial()
def calculate_derivatives(
        pr, gamma, b_t_0, nabla_a2, b_t_bar, psi, dr_psi, dr, dpr):
    """
    Calculate the derivative of the radial position and the radial momentum
    of the plasma particles at the current slice.

    Parameters
    ----------
    pr, gamma : ndarray
        Arrays containing the radial momentum and Lorentz factor of the
        plasma particles.
    b_t_0 : ndarray
        Array containing the value of the azimuthal magnetic field from
        the beam distribution at the position of each plasma particle.
    nabla_a2 : ndarray
        Array containing the value of the gradient of the laser normalized
        vector potential at the position of each plasma particle.
    b_t_bar : ndarray
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
                  - b_t_bar[i]
                  - b_t_0[i]
                  - nabla_a2[i] / (2. * (1. + psi_i)))
        dr[i] = pr[i] / (1. + psi_i)


@njit_serial()
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
