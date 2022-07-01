""" Contains the 5th order Adams–Bashforth pusher for the plasma particles. """


import numpy as np
from numba import njit


def evolve_plasma_ab5(pp, dxi):
    """
    Evolve the r and pr coordinates of plasma particles to the next xi step
    using an Adams–Bashforth method of 5th order.

    Parameters:
    -----------
    pp : PlasmaParticles
        The plasma particles to be evolved.
    dxi : float
        Longitudinal step.
    """
    # Get fields at the position of plasma particles in current slice.
    a2_pp, nabla_a2_pp, b_theta_0_pp, b_theta_pp = pp.get_field_arrays()
    psi_pp, dr_psi_pp, dxi_psi_pp = pp.get_psi_arrays()

    # Using these fields, compute derivatives of r and pr at the current slice.
    dr_arrays, dpr_arrays = pp.get_ab5_arrays()
    dr_pp = dr_arrays[0]
    dpr_pp = dpr_arrays[0]
    calculate_derivatives(
        pp.pr, pp.gamma, b_theta_0_pp, nabla_a2_pp, b_theta_pp,
        psi_pp, dr_psi_pp, dr_pp, dpr_pp
    )
    
    # Get derivatives of r and pr of last 5 steps.
    dr_arrays, dpr_arrays = pp.get_ab5_arrays()

    # Push radial position.
    apply_ab5(pp.r, dxi, *dr_arrays)

    # Push radial momentum.
    apply_ab5(pp.pr, dxi, *dpr_arrays)

    # Shift derivatives for next step (i.e., the derivative at step i will be
    # the derivative at step i+i in the next iteration.)
    for i in reversed(range(len(dr_arrays)-1)):
        dr_arrays[i+1][:] = dr_arrays[i]
        dpr_arrays[i+1][:] = dpr_arrays[i]

    # If a particle has crossed the axis, mirror it.
    idx_neg = np.where(pp.r < 0.)
    if idx_neg[0].size > 0:
        pp.r[idx_neg] *= -1.
        pp.pr[idx_neg] *= -1.
        for i in range(len(dr_arrays)):
            dr_arrays[i][idx_neg] *= -1.
            dpr_arrays[i][idx_neg] *= -1.


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
        x[i] +=  dt * (
            1901. * dx_1[i] - 2774. * dx_2[i] + 2616. * dx_3[i]
            - 1274. * dx_4[i] + 251. * dx_5[i]) * inv_720
