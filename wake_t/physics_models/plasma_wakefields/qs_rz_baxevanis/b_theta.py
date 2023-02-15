"""
Contains the method to compute the azimuthal magnetic field from the plasma
according to the paper by P. Baxevanis and G. Stupakov.

"""

import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial()
def calculate_b_theta_at_particles(r, pr, q, gamma, psi, dr_psi, dxi_psi,
                                   b_theta_0, nabla_a2, idx, dr_p, b_theta_pp):
    """
    Calculate the azimuthal magnetic field from the plasma at the location
    of the plasma particles using Eqs. (24), (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    Parameters
    ----------
    r, pr, q, gamma : arrays
        Arrays containing, respectively, the radial position, radial momentum,
        charge and gamma (Lorentz) factor of the plasma particles.
    psi, dr_psi, dxi_psi : arrays
        Arrays with the value of the wakefield potential and its radial and
        longitudinal derivatives at the location of the plasma particles.
    b_theta_0, nabla_a2 : arrays
        Arrays with the value of the source terms. The first one being the
        azimuthal magnetic field due to the beam distribution, and the second
        the gradient of the normalized vector potential of the laser.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.

    """
    # Calculate a_i and b_i, as well as a_0 and the sorted particle indices.
    a_i, b_i, a_0 = calculate_ai_bi_from_axis(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a2, idx)

    # Calculate field at particles as average between neighboring values.
    n_part = r.shape[0]

    # Calculate field value at plasma particles by interpolating between two
    # neighboring values. Same as with psi and its derivaties.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        if i_sort > 0:
            r_im1 = r[idx[i_sort-1]]
            a_im1 = a_i[idx[i_sort-1]]
            b_im1 = b_i[idx[i_sort-1]]
            r_left = (r_im1 + r_i) / 2
            b_theta_left = a_im1 * r_left + b_im1 / r_left
        else:
            b_theta_left = 0.
            r_left = 0.
        if i_sort < n_part - 1:
            r_ip1 = r[idx[i_sort+1]]
        else:
            r_ip1 = r[i] + dr_p / 2
        r_right = (r_i + r_ip1) / 2
        b_theta_right = a_i[i] * r_right + b_i[i] / r_right

        # Do interpolation.
        b = (b_theta_right - b_theta_left) / (r_right - r_left)
        a = b_theta_left - b*r_left
        b_theta_pp[i] = a + b*r_i

        # Near the peak of a strong blowout, very large and unphysical
        # values could appear. This condition makes sure a threshold us not
        # exceeded.
        if b_theta_pp[i] > 3.:
            b_theta_pp[i] = 3.
        if b_theta_pp[i] < -3.:
            b_theta_pp[i] = -3.


@njit_serial()
def calculate_b_theta(r_fld, r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                      nabla_a2, idx, b_theta, k):
    """
    Calculate the azimuthal magnetic field from the plasma at the radial
    locations in r_fld using Eqs. (24), (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov.

    Parameters
    ----------
    r_fld : array
        Array containing the radial positions where psi should be calculated.
    r, pr, q, gamma : arrays
        Arrays containing, respectively, the radial position, radial momentum,
        charge and gamma (Lorentz) factor of the plasma particles.
    psi, dr_psi, dxi_psi : arrays
        Arrays with the value of the wakefield potential and its radial and
        longitudinal derivatives at the location of the plasma particles.
    b_theta_0, nabla_a2 : arrays
        Arrays with the value of the source terms. The first one being the
        azimuthal magnetic field due to the beam distribution, and the second
        the gradient of the normalized vector potential of the laser.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    b_theta : ndarray
        Array where the values of the plasma azimuthal magnetic field will be
        stored.
    k : int
        Index that determines the slice of b_theta where the values will
        be filled in (the index is k+2 due to the guard cells in the array).

    """
    # Calculate a_i and b_i, as well as a_0 and the sorted particle indices.
    a_i, b_i, a_0 = calculate_ai_bi_from_axis(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a2, idx)

    # Calculate fields at r_fld
    n_part = r.shape[0]
    n_points = r_fld.shape[0]
    b_theta_mesh = b_theta[k+2]
    i_last = 0
    for j in range(n_points):
        r_j = r_fld[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        for i_sort in range(i_last, n_part):
            i_p = idx[i_sort]
            r_i = r[i_p]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields.
        if i_last == -1:
            b_theta_mesh[2+j] = a_0 * r_j
            i_last = 0
        else:
            i_p = idx[i_last]
            b_theta_mesh[2+j] = a_i[i_p] * r_j + b_i[i_p] / r_j


@njit_serial()
def calculate_ai_bi_from_axis(r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                              nabla_a2, idx):
    """
    Calculate the values of a_i and b_i which are needed to determine
    b_theta at any r position.

    For details about the input parameters see method 'calculate_b_theta'.

    The values of a_i and b_i are calculated as follows, using Eqs. (26) and
    (27) from the paper of P. Baxevanis and G. Stupakov:

        Write a_i and b_i as linear system of a_0:

            a_i = K_i * a_0_diff + T_i
            b_i = U_i * a_0_diff + P_i


        Where (im1 stands for subindex i-1):

            K_i = (1 + A_i*r_i/2) * K_im1  +  A_i/(2*r_i)     * U_im1
            U_i = (-A_i*r_i**3/2) * K_im1  +  (1 - A_i*r_i/2) * U_im1

            T_i = ( (1 + A_i*r_i/2) * T_im1  +  A_i/(2*r_i)     * P_im1  +
                    (2*Bi + Ai*Ci)/4 )
            P_i = ( (-A_i*r_i**3/2) * T_im1  +  (1 - A_i*r_i/2) * P_im1  +
                    r_i*(4*Ci - 2*Bi*r_i - Ai*Ci*r_i)/4 )

        With initial conditions:

            K_0 = 1
            U_0 = 0
            T_0 = 0
            P_0 = 0

        Then a_0 can be determined by imposing a_N = 0:

            a_N = K_N * a_0_diff + T_N = 0 <=> a_0_diff = - T_N / K_N

        If the precision of a_i and b_i becomes too low, then T_i and P_i are
        recalculated with an initial guess equal to a_i and b_i, as well as a
        new a_0_diff.

    """
    n_part = r.shape[0]

    # Preallocate arrays
    K = np.zeros(n_part)
    U = np.zeros(n_part)
    T = np.zeros(n_part)
    P = np.zeros(n_part)

    # Establish initial conditions (K_0 = 1, U_0 = 0, O_0 = 0, P_0 = 0)
    K_im1 = 1.
    U_im1 = 0.
    T_im1 = 0.
    P_im1 = 0.

    a_0 = 0.

    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]
        psi_i = psi[i]

        a = 1. + psi_i
        b = 1. / (r_i * a)

        A_i = q_i * b

        l_i = (1. + 0.5 * A_i * r_i)
        m_i = 0.5 * A_i / r_i
        n_i = -0.5 * A_i * r_i ** 3
        o_i = (1. - 0.5 * A_i * r_i)

        K_i = l_i * K_im1 + m_i * U_im1
        U_i = n_i * K_im1 + o_i * U_im1

        K[i] = K_i
        U[i] = U_i

        K_im1 = K_i
        U_im1 = U_i

    i_start = 0

    while i_start < n_part:

        # Iterate over particles
        for i_sort in range(i_start, n_part):
            i = idx[i_sort]
            r_i = r[i]
            pr_i = pr[i]
            q_i = q[i]
            gamma_i = gamma[i]
            psi_i = psi[i]
            dr_psi_i = dr_psi[i]
            dxi_psi_i = dxi_psi[i]
            b_theta_0_i = b_theta_0[i]
            nabla_a2_i = nabla_a2[i]

            a = 1. + psi_i
            a2 = a * a
            a3 = a2 * a
            b = 1. / (r_i * a)
            c = 1. / (r_i * a2)
            pr_i2 = pr_i * pr_i

            A_i = q_i * b
            B_i = q_i * (- (gamma_i * dr_psi_i) * c
                         + (pr_i2 * dr_psi_i) / (r_i * a3)
                         + (pr_i * dxi_psi_i) * c
                         + pr_i2 / (r_i * r_i * a2)
                         + b_theta_0_i * b
                         + nabla_a2_i * c * 0.5)
            C_i = q_i * (pr_i2 * c - (gamma_i / a - 1.) / r_i)

            l_i = (1. + 0.5 * A_i * r_i)
            m_i = 0.5 * A_i / r_i
            n_i = -0.5 * A_i * r_i ** 3
            o_i = (1. - 0.5 * A_i * r_i)

            T_i = l_i * T_im1 + m_i * P_im1 + 0.5 * B_i + 0.25 * A_i * C_i
            P_i = n_i * T_im1 + o_i * P_im1 + r_i * (
                    C_i - 0.5 * B_i * r_i - 0.25 * A_i * C_i * r_i)

            T[i] = T_i
            P[i] = P_i

            T_im1 = T_i
            P_im1 = P_i

        # Calculate a_0_diff.
        a_0_diff = - T_im1 / K_im1
        a_0 += a_0_diff

        i_stop = n_part

        # Calculate a_i (in T_i) and b_i (in P_i) as functions of a_0_diff.
        for i_sort in range(i_start, n_part):
            i = idx[i_sort]
            T_old = T[i]
            P_old = P[i]
            K_old = K[i] * a_0_diff
            U_old = U[i] * a_0_diff

            # Test if precision is lost in the sum T_old + K_old.
            # 0.5 roughly corresponds to one lost bit of precision.
            # Also pass test if this is the first number of this iteration
            # to avoid an infinite loop or if this is the last number
            # to computer as that is zero by construction
            if (i_sort == i_start or i_sort == (n_part-1) or
                    abs(T_old + K_old) >= 0.5 * abs(T_old - K_old) and
                    abs(P_old + U_old) >= 0.5 * abs(P_old - U_old)):
                # Calculate a_i and b_i as functions of a_0_diff.
                # Store the result in T and P
                T[i] = T_old + K_old
                P[i] = P_old + U_old
            else:
                # Stop this iteration, go to the next one
                i_stop = i_sort
                break

        if i_stop < n_part:
            # Set T_im1 and T_im1 properly for the next iteration
            T_im1 = T[idx[i_stop-1]]
            P_im1 = P[idx[i_stop-1]]

        # Start the next iteration where this one stopped
        i_start = i_stop

    return T, P, a_0


@njit_serial()
def calculate_ai_bi_from_edge(r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                              nabla_a2, idx):
    """
    Calculate the values of a_i and b_i which are needed to determine
    b_theta at any r position.

    For details about the input parameters see method 'calculate_b_theta'.

    The values of a_i and b_i are calculated, using Eqs. (26) and
    (27) from the paper of P. Baxevanis and G. Stupakov. In this algorithm,
    Eq. (27) is inverted so that we calculate a_i and b_i as a function
    of a_ip1 and b_ip1. Therefore, we start the loop at the boundary and end up
    on axis. This alternative method has shown to be more robust than
    `calculate_ai_bi_from_axis` to numerical precission issues.

        Write a_i and b_i as linear system of b_N_diff:

            a_i = K_i * b_N_diff + T_i
            b_i = U_i * b_N_diff + P_i


        Where (im1 stands for subindex i-1):

            K_i = (1 - A_i*r_i/2) * K_im1  +  (-A_i/(2*r_i))  * U_im1
            U_i = A_i*r_i**3/2    * K_im1  +  (1 + A_i*r_i/2) * U_im1

            T_i = ( (1 - A_i*r_i/2) * T_im1  +  (-A_i/(2*r_i))  * P_im1  +
                    (-2*Bi + Ai*Ci)/4 )
            P_i = ( A_i*r_i**3/2    * T_im1  +  (1 + A_i*r_i/2) * P_im1  -
                    r_i*(4*Ci - 2*Bi*r_i + Ai*Ci*r_i)/4 )

        With initial conditions at i=N:

            K_Np1 = 0
            U_Np1 = 1
            T_Np1 = 0
            P_Np1 = 0

        Then b_N_diff can be determined by imposing b_0 = 0:

            b_0 = U_0 * b_N_diff + P_0 = 0
            <=> b_N_diff = - P_0 / U_0

        If the precision of a_i and b_i becomes too low, then T_i and P_i are
        recalculated with an initial guess equal to a_i and b_i, as well as a
        new b_N_diff.

    """

    n_part = r.shape[0]

    # Preallocate arrays
    K = np.zeros(n_part)
    U = np.zeros(n_part)
    T = np.zeros(n_part)
    P = np.zeros(n_part)

    # Initial conditions at i = N
    K_ip1 = 0.
    U_ip1 = 1.
    T_ip1 = 0.
    P_ip1 = 0.

    # Iterate over particles to get K_i and U_i
    for i_sort in range(n_part):
        i = idx[-1-i_sort]
        r_i = r[i]
        q_i = q[i]
        psi_i = psi[i]

        a = 1. + psi_i

        l_i = (1. - 0.5 * q_i / a)
        m_i = -0.5 * q_i / (a*r_i**2)
        n_i = 0.5 * q_i/a * r_i ** 2
        o_i = (1. + 0.5 * q_i / a)

        K_i = l_i * K_ip1 + m_i * U_ip1
        U_i = n_i * K_ip1 + o_i * U_ip1

        K[i] = K_ip1
        U[i] = U_ip1

        K_ip1 = K_i
        U_ip1 = U_i

    i_start = 0

    while i_start < n_part:

        # Iterate over particles to get T_i and P_i
        for i_sort in range(i_start, n_part):
            i = idx[-1-i_sort]
            r_i = r[i]
            pr_i = pr[i]
            q_i = q[i]
            gamma_i = gamma[i]
            psi_i = psi[i]
            dr_psi_i = dr_psi[i]
            dxi_psi_i = dxi_psi[i]
            b_theta_0_i = b_theta_0[i]
            nabla_a2_i = nabla_a2[i]

            a = 1. + psi_i
            a2 = a * a
            a3 = a2 * a
            b = 1. / (r_i * a)
            c = 1. / (r_i * a2)
            pr_i2 = pr_i * pr_i

            A_i = q_i * b
            B_i = q_i * (- (gamma_i * dr_psi_i) * c
                         + (pr_i2 * dr_psi_i) / (r_i * a3)
                         + (pr_i * dxi_psi_i) * c
                         + pr_i2 / (r_i * r_i * a2)
                         + b_theta_0_i * b
                         + nabla_a2_i * c * 0.5)
            C_i = q_i * (pr_i2 * c - (gamma_i / a - 1.) / r_i)

            l_i = (1. - 0.5 * q_i / a)
            m_i = -0.5 * q_i / (a*r_i**2)
            n_i = 0.5 * q_i/a * r_i ** 2
            o_i = (1. + 0.5 * q_i / a)

            T_i = l_i * T_ip1 + m_i * P_ip1 - 0.5 * B_i + 0.25 * A_i * C_i
            P_i = n_i * T_ip1 + o_i * P_ip1 - r_i * (
                    C_i - 0.5 * B_i * r_i + 0.25 * A_i * C_i * r_i)

            T[i] = T_ip1
            P[i] = P_ip1

            T_ip1 = T_i
            P_ip1 = P_i

        # Calculate b_N_diff.
        b_N_diff = - P_ip1 / U_ip1

        i_stop = n_part

        # Calculate a_i (in T_i) and b_i (in P_i) as functions of b_N_diff.
        for i_sort in range(i_start, n_part):
            i = idx[-1-i_sort]
            T_old = T[i]
            P_old = P[i]
            K_old = K[i] * b_N_diff
            U_old = U[i] * b_N_diff

            if (i_sort == i_start or
                    abs(T_old + K_old) >= 0.5 * abs(T_old - K_old) and
                    abs(P_old + U_old) >= 0.5 * abs(P_old - U_old)):
                T[i] = T_old + K_old
                P[i] = P_old + U_old
            else:
                i_stop = i_sort
                break

        if i_stop < n_part:
            T_ip1 = T[idx[-1-i_stop]] + K[idx[-1-i_stop]] * b_N_diff
            P_ip1 = P[idx[-1-i_stop]] + U[idx[-1-i_stop]] * b_N_diff
        else:
            T_ip1 = T_ip1 + K_ip1 * b_N_diff
            P_ip1 = P_ip1 + U_ip1 * b_N_diff

        i_start = i_stop

    # Return a_i (in T_i), b_i (in P_i) and a_0 (in T_ip1)
    return T, P, T_ip1
