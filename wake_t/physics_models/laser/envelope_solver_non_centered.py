"""
This module contains a non-centered (in time) version of the envelope solver
described in 'An accurate and efficient laser-envelope solver for the modeling
of laser-plasma accelerators', written by C Benedetti et al in 2018.

Authors: Wilbert den Hertog, Ángel Ferran Pousa, Carlo Benedetti
"""

import numpy as np
import scipy.constants as ct

from wake_t.utilities.numba import njit_serial
from .tdma import TDMA
from .utils import unwrap


@njit_serial(fastmath=True)
def evolve_envelope_non_centered(
        a, a_old, chi, k0, kp, zmin, zmax, nz, rmax, nr, dt, nt,
        use_phase=True):
    """
    Solve the 2D envelope equation using a non-centered time discretization.
    (\nabla_tr^2+2i*k0/kp*d/dt+2*d^2/(dzdt)-d^2/dt^2)â = chi*â

    Parameters
    ----------
    a0 : array
        Initial value for â at tau=0. Dimension: nz*nr.
    aold : array
        Initial value for â at tau=-1. Dimension: nz*nr.
    chi : array
        Arrays of the values of the susceptibility. Dimension nz*nr.
    k0 : float
        Laser wave number.
    kp : float
        Plasma skin depth.
    zmin : float
        Minimum value for zeta.
    zmax : float
        Maximum value for zeta.
    nz : int
        Number of grid points in zeta-direction.
    rmax : float
        Maximum value for rho (minimum value is always 0).
    nr : int
        Amount of points in the rho direction.
    dt : float
        Tau step size.
    nt : int
        Number of tau steps.
    use_phase : bool
        Determines whether to take into account the terms related to the
        longitudinal derivative of the complex phase.

    """
    # Preallocate arrays. a_old and a include 2 ghost cells in the z direction.
    rhs = np.empty(nr, dtype=np.complex128)

    # Calculate step sizes.
    dz = (zmax - zmin) * kp / (nz - 1)
    dr = rmax * kp / nr
    dt = dt * ct.c * kp

    # Precalculate common fractions.
    inv_dt = 1 / dt
    inv_dr = 1 / dr
    inv_dz = 1 / dz
    inv_dzdt = inv_dt * inv_dz
    k0_over_kp = k0 / kp

    # Initialize phase difference
    d_theta1 = 0.
    d_theta2 = 0.

    # Calculate C^+ and C^- [Eq. (8)].
    C_minus = (-2. * inv_dr ** 2. * 0.5 - 2j * k0_over_kp * inv_dt
               + 3 * inv_dzdt)
    C_plus = (-2. * inv_dr ** 2. * 0.5 + 2j * k0_over_kp * inv_dt
              - 3 * inv_dzdt)

    # Calculate L^+ and L^-. Change wrt Benedetti - 2018: in Wake-T we use
    # cell-centered nodes in the radial direction.
    L_base = 1. / (2. * (np.arange(nr) + 0.5))
    L_minus_over_2 = (1. - L_base) * inv_dr ** 2. * 0.5
    L_plus_over_2 = (1. + L_base) * inv_dr ** 2. * 0.5

    # Loop over time iterations.
    for n in range(nt):
        # a_new_jp1 is equivalent to a_new[j+1] and a_new_jp2 to a_new[j+2].
        a_new_jp1 = np.zeros(nr, dtype=np.complex128)
        a_new_jp2 = np.zeros(nr, dtype=np.complex128)

        # Getting the phase of the envelope on axis.
        if use_phase:
            phases = unwrap(np.angle(a[:, 0]))

        # Loop over z.
        for j in range(nz - 1, -1, -1):

            # Calculate phase differences between adjacent points.
            if use_phase:
                d_theta1 = phases[j + 1] - phases[j]
                d_theta2 = phases[j + 2] - phases[j + 1]

            # Calculate D factor [Eq. (6)].
            D_jkn = (1.5 * d_theta1 - 0.5 * d_theta2) * inv_dz

            # Calculate right-hand side of Eq (7).
            for k in range(nr):
                rhs_k = (
                    - (C_minus - chi[j, k] * 0.5 - 2j * inv_dt * D_jkn)
                    * a[j, k]
                    - (4 * np.exp(-1j * d_theta1) * inv_dzdt
                       * (a_new_jp1[k] - a[j + 1, k]))
                    + (1 * np.exp(-1j * (d_theta2 + d_theta1)) * inv_dzdt
                       * (a_new_jp2[k] - a[j + 2, k]))
                )
                if k > 0:
                    rhs_k -= L_minus_over_2[k] * a[j, k - 1]
                if k + 1 < nr:
                    rhs_k -= L_plus_over_2[k] * a[j, k + 1]
                rhs[k] = rhs_k

            # Calculate diagonals.
            d_main = C_plus - chi[j] * 0.5 + 2j * inv_dt * D_jkn
            d_upper = L_plus_over_2[:nr - 1]
            d_lower = L_minus_over_2[1:nr]

            # Update a_old and a at j+2 with the current values of a a_new.
            a_old[j + 2] = a[j + 2]
            a[j + 2] = a_new_jp2

            # Shift a_new[j+1] to a_new[j+2] in preparation for the next
            # iteration.
            a_new_jp2[:] = a_new_jp1

            # Compute a_new at the current j using the TDMA method and store
            # result in a_new_jp1 to use it in the next iteration.
            TDMA(d_lower, d_main, d_upper, rhs, a_new_jp1)

        # When the left of the computational domain is reached, paste the last
        # few values in the a_old and a arrays.
        a_old[0:2] = a[0:2]
        a[0] = a_new_jp1
        a[1] = a_new_jp2
