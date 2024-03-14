""" Contains the 5th order Adams–Bashforth pusher for the plasma particles. """


import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial()
def evolve_plasma_ab2(
        dxi, r, pr, gamma, m, q, r_to_x,
        nabla_a2, b_theta_0, b_theta, psi, dr_psi,
        dr, dpr
        ):
    """
    Evolve the r and pr coordinates of plasma particles to the next xi step
    using an Adams-Bashforth method of 2nd order.

    Parameters
    ----------
    dxi : float
        Longitudinal step.
    r, pr, gamma, m, q, r_to_x : ndarray
        Radial position, radial momentum, Lorentz factor, mass and charge of
        the plasma particles as well an array that keeps track of axis crosses
        to convert from r to x.
    nabla_a2, b_theta_0, b_theta, psi, dr_psi : ndarray
        Arrays with the value of the fields at the particle positions.
    dr, dpr : ndarray
        Arrays containing the derivative of the radial position and momentum
        of the particles at the 2 slices previous to the next step.
    """

    calculate_derivatives(
        pr, gamma, m, q, b_theta_0, nabla_a2, b_theta,
        psi, dr_psi, dr[0], dpr[0]
    )

    # Push radial position.
    apply_ab2(r, dxi, dr)

    # Push radial momentum.
    apply_ab2(pr, dxi, dpr)

    # Shift derivatives for next step (i.e., the derivative at step i will be
    # the derivative at step i+i in the next iteration.)
    dr[1] = dr[0]
    dpr[1] = dpr[0]

    # If a particle has crossed the axis, mirror it.
    check_axis_crossing(r, pr, dr[1], dpr[1], r_to_x)


@njit_serial(fastmath=True, error_model="numpy")
def calculate_derivatives(
        pr, gamma, m, q, b_theta_0, nabla_a2, b_theta_bar, psi, dr_psi, dr, dpr):
    """
    Calculate the derivative of the radial position and the radial momentum
    of the plasma particles at the current slice.

    Parameters
    ----------
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
        q_over_m = q[i] / m[i]
        inv_psi_i = 1. / (1. + psi[i] * q_over_m)
        dpr[i] = (gamma[i] * dr_psi[i] * inv_psi_i
                  - b_theta_bar[i]
                  - b_theta_0[i]
                  - nabla_a2[i] * 0.5 * inv_psi_i * q_over_m) * q_over_m
        dr[i] = pr[i] * inv_psi_i


@njit_serial()
def apply_ab2(x, dt, dx):
    """Apply the Adams-Bashforth method of 2nd order to evolve `x`.

    Parameters
    ----------
    x : ndarray
        Array containing the variable to be advanced.
    dt : float
        Discretization step size.
    dx : ndarray
        Array containing the derivatives of `x` at the two previous steps.
    """
    for i in range(x.shape[0]):
        x[i] += dt * (1.5 * dx[0, i] - 0.5 * dx[1, i])


@njit_serial()
def check_axis_crossing(r, pr, dr, dpr, r_to_x):
    """Check for particles with r < 0 and invert them."""
    for i in range(r.shape[0]):
        if r[i] < 0.:
            r[i] *= -1.
            pr[i] *= -1.
            dr[i] *= -1.
            dpr[i] *= -1.
            r_to_x[i] *= -1
