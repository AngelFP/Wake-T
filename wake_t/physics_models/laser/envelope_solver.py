import cmath
import numpy as np
from numba import njit


@njit()
def L(sign, k, dr):
    """
    Calculation of L_k^{+-}. Change wrt Benedetti - 2018: in Wake-T we use cell
    -centered nodes in the rho direction.

    Parameters
    ----------
    sign : int
        1, 0 or 1, which defines the -, 0, + symbol respectively.
    k : int
        The rho grid coordinate: 0<=k<=Np-1.
    dr : float
        Rho step size.
    nr : int
        Amount of grid points in the rho direction.

    """
    if k > 0:
        if sign == 1 or sign == -1:
            return (1 + sign * 1 / (2 * (k + 0.5))) / dr ** 2
        else:
            return -2 / dr ** 2
    else:
        if sign == 1:
            return 2 / dr ** 2
        elif sign == 0:
            return -2 / dr ** 2
        else:
            return 0


@njit()
def C(sign, k, k0p, dt, dz, dr):
    """
    Calculate Equation (8) from Benedetti - 2018.

    Parameters
    ----------
    sign : int
        1 or -1, which is the + or - respectively in Equation (8).
    k : int
        The rho grid coordinate: 0<=k<=Np-1.
    k0p : float
        k0/kp, laser wave number divided by the plasma skin depth.
    dt : float
        Tau step size.
    dz : float
        Zeta step size.
    dr : float
        Rho step size.

    """
    if dr == 0:  # 1D case
        return (L(-1, k, dr) / 2
                + sign * 1j * k0p / dt - sign * 3 / 2
                * 1 / (dt * dz) - 1 / dt ** 2)
    else:  # 2D case
        return (L(0, k, dr) / 2
                + sign * 1j * k0p / dt - sign * 3 / 2
                * 1 / (dt * dz) - 1 / dt ** 2)


@njit()
def D(th, th1, th2, dz):
    """
    Calculate D in Equation (6) from Benedetti - 2018
    To account for the 'jumps' in the theta function,
    we use the phase of the envelope at the radius.

    Parameters
    ----------
    th : float
        Phase of the envelope at the radius, z=j.
    th1 : float
        Phase of the envelope at the radius, z=j+1.
    th2 : float
        Phase of the envelope at the radius, z=j+2.
    dz : float
        Zeta step size.

    """
    # phase difference between adjacent points j, j+1, j+2
    d_theta1 = th1 - th
    d_theta2 = th2 - th1

    # checking for phase jumps
    if d_theta1 < -1.5 * np.pi:
        d_theta1 += 2 * np.pi
    if d_theta2 < -1.5 * np.pi:
        d_theta2 += 2 * np.pi
    if d_theta1 > 1.5 * np.pi:
        d_theta1 -= 2 * np.pi
    if d_theta2 > 1.5 * np.pi:
        d_theta2 -= 2 * np.pi

    return 1.5 * d_theta1 / dz - 0.5 * d_theta2 / dz


@njit()
def chi():
    return 0


@njit()
def rhs(a_old, a, a_new, j, dz, k, dr, nr, dt, k0p, th, th1, th2):
    """
    The right-hand side of equation 7 in Benedetti, 2018.

    Parameters
    ----------
    a_old : array
        The array of the values of â at former time step. Dimension: (nz+2)*nr
    a : array
        The array of the values of â at current time step. Dimension: (nz+2)*nr
    a_new : array
        The array of the values of â at next time step, at z=j+1 and z=j+2.
        Dimension: 2*nr.
    j : int
        The zeta grid coordinate: 0<=j<nz.
    dz : float
        Zeta step size.
    k : int
        The rho grid coordinate: 0<=k<nr.
    dr : float
        Rho step size.
    nr : int
        Amount of points in the rho direction.
    dt : float
        Tau step size.
    k0p : float
        k0/kp, laser wave number divided by the plasma skin depth.
    th : float
        Phase of the envelope at the radius, z=j.
    th1 : float
        Phase of the envelope at the radius, z=j+1.
    th2 : float
        Phase of the envelope at the radius, z=j+2.

    """
    sol = (- 2 / dt ** 2 * a[j, k]
           - (C(-1, k, k0p, dt, dz, dr)
              - chi() / 2
              - 1j / dt * D(th, th1, th2, dz)) * a_old[j, k]
           - 2 * np.exp(1j * (th - th1)) / (dz * dt)
           * (a_new[0, k] - a_old[j + 1, k])
           + np.exp(1j * (th - th2)) / (2 * dz * dt)
           * (a_new[1, k] - a_old[j + 2, k]))
    if k + 1 < nr:
        sol -= L(1, k, dr) / 2 * a_old[j, k + 1]
    if k > 0:
        sol -= L(-1, k, dr) / 2 * a_old[j, k - 1]
    return sol


@njit()
def TDMA(a, b, c, d):
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
    w = np.zeros(n - 1, dtype=np.complex128)
    g = np.zeros(n, dtype=np.complex128)
    p = np.zeros(n, dtype=np.complex128)

    w[0] = c[0] / b[0]  # MAKE SURE THAT b[0]!=0
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p


@njit()
def solve_2d(k0p, zmin, zmax, nz, rmax, nr, dt, nt, a0, aold):
    """
    Solve the 2D envelope equation
    (\nabla_tr^2+2i*k0/kp*d/dt+2*d^2/(dzdt)-d^2/dt^2)â = chi*â

    Parameters
    ----------
    k0p : float
        k0/kp, laser wave number divided by the plasma skin depth.
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
    a0 : array
        Initial value for â at tau=0. Dimension: nz*nr.
    aold : array
        Initial value for â at tau=-1. Dimension: nz*nr.

    """
    # Add 2 rows of ghost points in the zeta direction.
    a_old = np.zeros((nz + 2, nr), dtype=np.complex128)
    a = np.zeros((nz + 2, nr), dtype=np.complex128)

    # a_new is a 2 x nr array to store new values of a. a_new[0] = a_new[j+1]
    # and a_new[1] = a_new[j+2].
    a_new = np.zeros((2, nr), dtype=np.complex128)

    # Declaration of the 4 vectors used for solving the tridiagonal system.
    d_upper = np.zeros(nr - 1, dtype=np.complex128)
    d_lower = np.zeros(nr - 1, dtype=np.complex128)
    d_main = np.zeros(nr, dtype=np.complex128)
    sol = np.zeros(nr, dtype=np.complex128)

    a_old[0:-2] = aold
    a[0:-2] = a0

    dz = (zmax - zmin) / (nz - 1)
    dr = rmax / nr

    for n in range(0, nt):
        if n % 100 == 0:
            print("Time =", n * dt)
        # Getting the phases of the envelope at the radius.
        phases = np.angle(a[:, 0])
        for j in range(nz - 1, -1, -1):
            th = phases[j]
            th1 = phases[j + 1]
            th2 = phases[j + 2]

            # Fill the vectors according to the numerical scheme.
            for k in range(0, nr):
                sol[k] = rhs(a_old, a, a_new, j, dz, k, dr, nr, dt, k0p, th,
                             th1, th2)
                d_main[k] = (C(1, k, k0p, dt, dz, dr)
                             - chi() / 2
                             + 1j / dt * D(th, th1, th2, dz))
                if k < nr - 1:
                    d_upper[k] = L(1, k, dr) / 2
                if k > 0:
                    d_lower[k - 1] = L(-1, k, dr) / 2
            # Update a_old at j+2 with the current value of a at j+2.
            a_old[j + 2] = a[j + 2]
            # This frees up space to replace a[j+2] by the a_new[1] value.
            a[j + 2] = a_new[1]
            # Now we can shift the a_new[0] value to a_new[1], and calculate
            # the solution vector.
            a_new[1] = a_new[0]
            a_new[0] = TDMA(d_lower, d_main, d_upper, sol)
        # When the left of the computational domain is reached, paste the last
        # few values in the a_old and a arrays.
        a_old[0:2] = a[0:2]
        a[0] = a_new[0]
    return a

# @njit()
# def solve_2d_test(k0p, zmin, zmax, nz, rmax, nr, dt, nt, a0, aold):
#     """
#     Test the numerical method
#
#     Parameters
#     ----------
#     k0p : float
#         k0/kp, laser wave number divided by the plasma skin depth.
#     zmin : float
#         Minimum value for zeta.
#     zmax : float
#         Maximum value for zeta.
#     nz : int
#         Number of grid points in zeta-direction.
#     rmax : float
#         Maximum value for rho (minimum value is always 0).
#     nr : int
#         Amount of points in the rho direction.
#     dt : float
#         Tau step size.
#     nt : int
#         Number of tau steps.
#     a0 : array
#         Initial value for â at tau=0. Dimension: nz*nr.
#     aold : array
#         Initial value for â at tau=-1. Dimension: nz*nr.
#
#     """
#     # a_old corresponds to a(z, r, t-1),
#     # a corresponds to a(z, r, t) and a_new corresponds to a(z, r, t+1)
#
#     # add 2 rows of ghost points in the zeta direction
#     a_old = np.zeros((nz + 2, nr), dtype=np.complex128)
#     a = np.zeros((nz + 2, nr), dtype=np.complex128)
#     a_new = np.zeros((nz + 2, nr), dtype=np.complex128)
#
#     a_old[0:-2] = aold
#     a[0:-2] = a0
#
#     dz = (zmax - zmin) / (nz - 1)
#     dr = rmax / (nr - 1)
#
#     for n in range(0, nt):
#         if n % 100 == 0:
#             print("Time =", n * dt)
#         # Solve the tridiagonal system for the solution on the radius
#         phases = np.angle(a[:, 0])
#         for j in range(nz - 1, -1, -1):
#             d_upper = np.zeros(nr - 1, dtype=np.complex128)
#             d_lower = np.zeros(nr - 1, dtype=np.complex128)
#             d_main = np.zeros(nr, dtype=np.complex128)
#             sol = np.zeros(nr, dtype=np.complex128)
#
#             th = phases[j]
#             th1 = phases[j + 1]
#             th2 = phases[j + 2]
#             for k in range(0, nr):
#                 sol[k] = (rhs(a_old, a, a_new, j, dz, k, dr, nr, dt, k0p, th,
#                               th1, th2)
#                           + testfunc(zmin + j * dz, zmax))
#                 d_main[k] = (C(1, k, k0p, dt, dz, dr)
#                              - chi() / 2
#                              + 1j / dt * D(th, th1, th2, dz))
#                 if k < nr - 1:
#                     d_upper[k] = L(1, k, dr) / 2
#                 if k > 0:
#                     d_lower[k - 1] = L(-1, k, dr) / 2
#             a_new[j] = TDMA(d_lower, d_main, d_upper, sol)
#         a_old[:] = a
#         a[:] = a_new
#     return a_new
#
#
# @njit()
# def solve_2d_chi(k0, kp, w0, zmin, zmax, nz, rmax, nr, dt, nt, a0, aold):
#     """
#     Solve the envelope equation using a type of plasma that guides the laser.
#
#     Parameters
#     ----------
#     k0 : float
#         Laser wave number.
#     kp : float
#         Plasma skin depth.
#     w0 : float
#         The spot size of the laser in m.
#     zmin : float
#         Minimum value for zeta.
#     zmax : float
#         Maximum value for zeta.
#     nz : int
#         Number of grid points in zeta-direction.
#     rmax : float
#         Maximum value for rho (minimum value is always 0).
#     nr : int
#         Amount of points in the rho direction.
#     dt : float
#         Tau step size.
#     nt : int
#         Number of tau steps.
#     a0 : array
#         Initial value for â at tau=0. Dimension: nz*nr.
#     aold : array
#         Initial value for â at tau=-1. Dimension: nz*nr.
#
#     """
#     # a_old corresponds to a(z, r, t-1),
#     # a corresponds to a(z, r, t) and a_new corresponds to a(z, r, t+1)
#
#     # add 2 rows of ghost points in the zeta direction
#     a_old = np.zeros((nz + 2, nr), dtype=np.complex128)
#     a = np.zeros((nz + 2, nr), dtype=np.complex128)
#     a_new = np.zeros((nz + 2, nr), dtype=np.complex128)
#
#     a_old[0:-2] = aold
#     a[0:-2] = a0
#
#     dz = (zmax - zmin) / (nz - 1)
#     dr = rmax / (nr - 1)
#
#     k0p = k0 / kp
#
#     for n in range(0, nt):
#         if n % 100 == 0:
#             print("Time =", n * dt)
#         # Solve the tridiagonal system for the solution on the radius
#         phases = np.angle(a[:, 0])
#         for j in range(nz - 1, -1, -1):
#             d_upper = np.zeros(nr - 1, dtype=np.complex128)
#             d_lower = np.zeros(nr - 1, dtype=np.complex128)
#             d_main = np.zeros(nr, dtype=np.complex128)
#             sol = np.zeros(nr, dtype=np.complex128)
#
#             th = phases[j]
#             th1 = phases[j + 1]
#             th2 = phases[j + 2]
#             for k in range(0, nr):
#                 sol[k] = rhs(a_old, a, a_new, j, dz, k, dr, nr, dt, k0p, th,
#                              th1, th2)
#                 d_main[k] = (C(1, k, k0p, dt, dz, dr)
#                              - chi_2(k * dr, w0, kp) / 2
#                              + 1j / dt * D(th, th1, th2, dz))
#                 if k < nr - 1:
#                     d_upper[k] = L(1, k, dr) / 2
#                 if k > 0:
#                     d_lower[k - 1] = L(-1, k, dr) / 2
#             a_new[j] = TDMA(d_lower, d_main, d_upper, sol)
#         a_old[:] = a
#         a[:] = a_new
#     return a_new
#
#
# @njit()
# def testfunc(z, zm):
#     """
#     Function used for testing purposes.
#
#     Parameters
#     ----------
#     z : float
#         Zeta-coordinate.
#     zm : float
#         Zeta_max coordinate.
#
#     """
#     return -4 * (z - zm)
#
#
# @njit()
# def theta1D(a, j):
#     """
#     Calculate the phase of â in the 1D model.
#
#     Parameters
#     ----------
#     a : array
#         1D array â at a specific time t. Dimension: nz.
#     j : int
#         The zeta grid coordinate: 0<=j<nz.
#
#     """
#     return cmath.phase(a[j])
#
#
# @njit()
# def D1D(a, j, dz):
#     """
#     Calculate D in Equation (6) from Benedetti - 2018 (1D case).
#
#     Parameters
#     ----------
#     a : array
#         1D array â at a specific time t. Dimension: nz.
#     j : int
#         The zeta grid coordinate: 0<=j<nz.
#     dz : float
#         Zeta step size.
#
#     """
#     return cmath.phase(np.exp(1j * (-3 * theta1D(a, j)
#                                     + 4 * theta1D(a, j + 1)
#                                     - theta1D(a, j + 2)))) / (2 * dz)
#
#
# @njit()
# def chi_2(r, w_0, k_p):
#     """
#     Plasma susceptibility function used for testing a guided laser.
#
#     Parameters
#     ----------
#     r : float
#         The rho coordinate.
#     w_0 : float
#         The spot size of the laser in m.
#     k_p : float
#         The plasma density.
#
#     """
#     # re = ct.physical_constants['classical electron radius'][0]
#     re = 2.8179403262e-15
#     dnc = 1 / (np.pi * re * w_0 ** 4 * 1e24)
#     return 1 + dnc * (r / k_p) ** 2
#
#
# # @njit()
# def solve_1d(k0p, zmin, zmax, nz, dt, nt, a0):
#     """
#     Solve the 1D envelope equation, without second time derivative and
#     without chi: (2i*k0/kp*d/dt+2*d^2/(dzdt))â = 0
#
#     Parameters
#     ----------
#     k0p : float
#         k0/kp, laser wave number divided by the plasma skin depth.
#     zmin : float
#         Minimum value for zeta.
#     zmax : float
#         Maximum value for zeta.
#     nz : int
#         Number of grid points in zeta-direction.
#     dt : float
#         Tau step size.
#     nt : int
#         Number of tau steps.
#     a0 : array
#         Initial value for â. Dimension: nz.
#
#     """
#     # Reserve 2 rows for the ghost grid points at j = nz and j = nz+1.
#     # a_old corresponds to a(z, t-1), a_current corresponds to a(z, t)
#     # and a_new corresponds to a(z, t+1)
#     a_old = np.zeros(nz + 2, dtype=np.complex128)
#     a_current = np.zeros(nz + 2, dtype=np.complex128)
#
#     a_old[0:-2] = 0.99 * a0
#     a_current[0:-2] = a0
#     a_new = np.zeros(nz + 2, dtype=np.complex128)
#
#     dz = (zmax - zmin) / (nz - 1)
#
#     c0p = C(1, 0, k0p, dt, dz, 0)  # C_0^{0, +}
#     c0m = C(-1, 0, k0p, dt, dz, 0)  # C_0^{0, -}
#
#     for n in range(0, nt):
#         for j in range(nz, 0, -1):
#             factor_lhs = (c0p - chi() / 2
#                           + 1j / dt * D1D(a_current, j - 1, dz))
#             factor_rhs = (-2 / dt ** 2 * a_current[j - 1]
#                           - (c0m - chi() / 2 - 1j / dt
#                              * D1D(a_current, j - 1, dz))
#                           * a_old[j - 1]
#                           - 2 * np.exp(1j * (theta1D(a_current, j - 1)
#                                              - theta1D(a_current, j)))
#                           / (dt * dz)
#                           * (a_new[j] - a_old[j])
#                           + np.exp(1j * (theta1D(a_current, j - 1)
#                                          - theta1D(a_current, j + 1)))
#                           / (2 * dt * dz)
#                           * (a_new[j + 1] - a_old[j + 1]))
#             a_new[j - 1] = factor_rhs / factor_lhs
#         a_old[:] = a_current
#         a_current[:] = a_new
#     return a_new
