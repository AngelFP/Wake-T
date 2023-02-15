""" Contains the 5th order Adams–Bashforth pusher for the plasma particles. """


import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial()
def evolve_plasma_ab5(
        dxi, r, pr, gamma,
        nabla_a2_pp, b_theta_0_pp, b_theta_pp, psi_pp, dr_psi_pp,
        dr_1, dr_2, dr_3, dr_4, dr_5, dpr_1, dpr_2, dpr_3, dpr_4, dpr_5):
    """
    Evolve the r and pr coordinates of plasma particles to the next xi step
    using an Adams–Bashforth method of 5th order.

    Parameters
    ----------
    dxi : float
        Longitudinal step.
    r, pr, gamma : ndarray
        Radial position, radial momentum, and Lorentz factor of the plasma
        particles.
    a2_pp, ..., dxi_psi_pp : ndarray
        Arrays where the value of the fields at the particle positions will
        be stored.
    dr_1, ..., dr_5 : ndarray
        Arrays containing the derivative of the radial position of the
        particles at the 5 slices previous to the next one.
    dpr_1, ..., dpr_5 : ndarray
        Arrays containing the derivative of the radial momentum of the
        particles at the 5 slices previous to the next one.
    """

    calculate_derivatives(
        pr, gamma, b_theta_0_pp, nabla_a2_pp, b_theta_pp,
        psi_pp, dr_psi_pp, dr_1, dpr_1
    )

    # Push radial position.
    apply_ab5(r, dxi, dr_1, dr_2, dr_3, dr_4, dr_5)

    # Push radial momentum.
    apply_ab5(pr, dxi, dpr_1, dpr_2, dpr_3, dpr_4, dpr_5)

    # Shift derivatives for next step (i.e., the derivative at step i will be
    # the derivative at step i+i in the next iteration.)
    dr_5[:] = dr_4
    dr_4[:] = dr_3
    dr_3[:] = dr_2
    dr_2[:] = dr_1
    dpr_5[:] = dpr_4
    dpr_4[:] = dpr_3
    dpr_3[:] = dpr_2
    dpr_2[:] = dpr_1

    # If a particle has crossed the axis, mirror it.
    idx_neg = np.where(r < 0.)
    if idx_neg[0].size > 0:
        r[idx_neg] *= -1.
        pr[idx_neg] *= -1.
        dr_1[idx_neg] *= -1.
        dr_2[idx_neg] *= -1.
        dr_3[idx_neg] *= -1.
        dr_4[idx_neg] *= -1.
        dr_5[idx_neg] *= -1.
        dpr_1[idx_neg] *= -1.
        dpr_2[idx_neg] *= -1.
        dpr_3[idx_neg] *= -1.
        dpr_4[idx_neg] *= -1.
        dpr_5[idx_neg] *= -1.


@njit_serial()
def calculate_derivatives(
        pr, gamma, b_theta_0, nabla_a2, b_theta_bar, psi, dr_psi, dr, dpr):
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
        psi_i = psi[i]
        dpr[i] = (gamma[i] * dr_psi[i] / (1. + psi_i)
                  - b_theta_bar[i]
                  - b_theta_0[i]
                  - nabla_a2[i] / (2. * (1. + psi_i)))
        dr[i] = pr[i] / (1. + psi_i)


@njit_serial()
def apply_ab5(x, dt, dx_1, dx_2, dx_3, dx_4, dx_5):
    """Apply the Adams-Bashforth method of 5th order to evolve `x`.

    Parameters
    ----------
    x : ndarray
        Array containing the variable to be advanced.
    dt : _type_
        Discretization step size.
    dx_1, dx_2, dx_3, dx_4, dx_5 : ndarray
        Arrays containing the derivatives of `x` at the five previous steps.
    """
    inv_720 = 1. / 720.
    for i in range(x.shape[0]):
        x[i] += dt * (
            1901. * dx_1[i] - 2774. * dx_2[i] + 2616. * dx_3[i]
            - 1274. * dx_4[i] + 251. * dx_5[i]) * inv_720
