"""
This module contains the envelope solver. This module is strongly based on the
paper 'An accurate and efficient laser-envelope solver for the modeling of
laser-plasma accelerators', written by C Benedetti et al in 2018.

Authors: Wilbert den Hertog, Ángel Ferran Pousa, Carlo Benedetti
"""

import numpy as np
import scipy.constants as ct

from wake_t.utilities.numba import njit_serial


@njit_serial(fastmath=True)
def TDMA(a, b, c, d, p):
    """TriDiagonal Matrix Algorithm: solve a linear system Ax=b,
    where A is a tridiagonal matrix. Source:
    https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-
    tdma-aka-thomas-algorithm-using-python-with-nump

    Parameters
    ----------
    a : array
        Lower diagonal of A. Dimension: nr-1.
    b : array
        Main diagonal of A. Dimension: nr.
    c : array
        Upper diagonal of A. Dimension: nr-1.
    d : array
        Solution vector. Dimension: nr.

    """
    n = len(d)
    w = np.empty(n - 1, dtype=np.complex128)
    g = np.empty(n, dtype=np.complex128)

    w[0] = c[0] / b[0]  # MAKE SURE THAT b[0]!=0
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]


@njit_serial(fastmath=True)
def evolve_envelope(a0, aold, chi, k0, kp, zmin, zmax, nz, rmax, nr, dt, nt,
                    start_outside_plasma=False):
    """
    Solve the 2D envelope equation
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
    start_outside_plasma : bool
        If `True`, it indicates that the laser is outside of the plasma at
        `t=-dt`. This will then force the plasma susceptibility to be zero
        at that time.

    """
    # Preallocate arrays. a_old and a include 2 ghost cells in the z direction.
    a_old = np.empty((nz + 2, nr), dtype=np.complex128)
    a = np.empty((nz + 2, nr), dtype=np.complex128)
    rhs = np.empty(nr, dtype=np.complex128)

    # Fill in a and a_old arrays.
    a_old[0:-2] = aold
    a[0:-2] = a0
    a_old[-2:] = 0.
    a[-2:] = 0.

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

    # Calculate C^+ and C^- [Eq. (8)].
    C_minus = (-2. * inv_dr ** 2. * 0.5 - 1j * k0_over_kp * inv_dt
               + 1.5 * inv_dzdt - inv_dt ** 2.)
    C_plus = (-2. * inv_dr ** 2. * 0.5 + 1j * k0_over_kp * inv_dt
              - 1.5 * inv_dzdt - inv_dt ** 2.)

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
        phases = np.angle(a[:, 0])

        # If laser starts outside plasma, make chi^{n-1} = 0.
        if start_outside_plasma and n == 0:
            chi_nm1 = 0. * chi
        else:
            chi_nm1 = chi

        # Loop over z.
        for j in range(nz - 1, -1, -1):
            # Calculate phase differences between adjacent points.
            d_theta1 = phases[j + 1] - phases[j]
            d_theta2 = phases[j + 2] - phases[j + 1]

            # Prevent phase jumps bigger than 1.5*pi.
            if d_theta1 < -1.5 * np.pi:
                d_theta1 += 2 * np.pi
            if d_theta2 < -1.5 * np.pi:
                d_theta2 += 2 * np.pi
            if d_theta1 > 1.5 * np.pi:
                d_theta1 -= 2 * np.pi
            if d_theta2 > 1.5 * np.pi:
                d_theta2 -= 2 * np.pi

            # Calculate D factor [Eq. (6)].
            D_jkn = (1.5 * d_theta1 - 0.5 * d_theta2) * inv_dz

            # Calculate right-hand side of Eq (7).
            for k in range(nr):
                rhs[k] = (
                    - 2 * inv_dt ** 2 * a[j, k]
                    - ((C_minus - chi_nm1[j, k] * 0.5 - 1j * inv_dt * D_jkn)
                       * a_old[j, k])
                    - (2 * np.exp(-1j * d_theta1) * inv_dzdt
                       * (a_new_jp1[k] - a_old[j + 1, k]))
                    + (0.5 * np.exp(-1j * (d_theta2 + d_theta1)) * inv_dzdt
                       * (a_new_jp2[k] - a_old[j + 2, k]))
                    - L_plus_over_2[k] * a_old[j, k + 1] * (k + 1 < nr)
                    - L_minus_over_2[k] * a_old[j, k - 1] * (k > 0)
                )

            # Calculate diagonals.
            d_main = C_plus - chi[j] * 0.5 + 1j * inv_dt * D_jkn
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
    return a_old, a
