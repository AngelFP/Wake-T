"""
Contains the method to compute the azimuthal magnetic field from the plasma
according to the paper by P. Baxevanis and G. Stupakov.

"""

import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial()
def calculate_b_theta_at_particles(r, a_0, a_i, b_i, r_neighbor, idx, b_theta_pp):
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
    # Calculate field at particles as average between neighboring values.
    n_part = r.shape[0]

    for i_sort in range(n_part):
        i = idx[i_sort]
        im1 = idx[i_sort - 1]
        r_i = r[i]
        
        r_left = r_neighbor[i_sort]
        r_right = r_neighbor[i_sort+1]
        
        if i_sort > 0:
            a_i_left = a_i[im1]
            b_i_left = b_i[im1]
            b_theta_left = a_i_left * r_left + b_i_left / r_left
        else:
            a_i_left = a_0
            b_theta_left = a_i_left * r_left
            
        a_i_right = a_i[i]
        b_i_right = b_i[i]
        b_theta_right = a_i_right * r_right + b_i_right / r_right

        # Do interpolation.
        b = (b_theta_right - b_theta_left) / (r_right - r_left)
        a = b_theta_left - b*r_left
        b_theta_pp[i] = a + b*r_i


    # # Calculate field value at plasma particles by interpolating between two
    # # neighboring values. Same as with psi and its derivaties.
    # for i_sort in range(n_part):
    #     i = idx[i_sort]
    #     r_i = r[i]
    #     if i_sort > 0:
    #         r_im1 = r[idx[i_sort-1]]
    #         a_im1 = a_i[idx[i_sort-1]]
    #         b_im1 = b_i[idx[i_sort-1]]
    #         r_left = (r_im1 + r_i) / 2
    #         b_theta_left = a_im1 * r_left + b_im1 / r_left
    #     else:
    #         b_theta_left = 0.
    #         r_left = 0.
    #     if i_sort < n_part - 1:
    #         r_ip1 = r[idx[i_sort+1]]
    #     else:
    #         r_ip1 = r[i] + dr_p / 2
    #     r_right = (r_i + r_ip1) / 2
    #     b_theta_right = a_i[i] * r_right + b_i[i] / r_right

    #     # Do interpolation.
    #     b = (b_theta_right - b_theta_left) / (r_right - r_left)
    #     a = b_theta_left - b*r_left
    #     b_theta_pp[i] = a + b*r_i

    #     # Near the peak of a strong blowout, very large and unphysical
    #     # values could appear. This condition makes sure a threshold us not
    #     # exceeded.
    #     if b_theta_pp[i] > 3.:
    #         b_theta_pp[i] = 3.
    #     if b_theta_pp[i] < -3.:
    #         b_theta_pp[i] = -3.


@njit_serial()
def calculate_b_theta_at_ions(r_ion, r_elec, a_0, a_i, b_i, idx_ion, idx_elec, b_theta_pp):
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
    # Calculate field at particles as average between neighboring values.
    n_ion = r_ion.shape[0]
    n_elec = r_elec.shape[0]
    i_last = 0
    for i_sort in range(n_ion):

        i = idx_ion[i_sort]
        r_i = r_ion[i]
        
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        for i_sort_e in range(i_last, n_elec):
            i_e = idx_elec[i_sort_e]
            r_elec_i = r_elec[i_e]
            if r_elec_i >= r_i:
                i_last = i_sort_e - 1
                break
        # Calculate fields.
        if i_last == -1:
            b_theta_pp[i] = a_0 * r_i
            i_last = 0
        else:
            i_e = idx_elec[i_last]
            b_theta_pp[i] = a_i[i_e] * r_i + b_i[i_e] / r_i


@njit_serial()
def calculate_b_theta(r_fld, a_0, a_i, b_i, r, idx, b_theta):
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
    # Calculate fields at r_fld
    n_part = r.shape[0]
    n_points = r_fld.shape[0]
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
            b_theta[j] = a_0 * r_j
            i_last = 0
        else:
            i_p = idx[i_last]
            b_theta[j] = a_i[i_p] * r_j + b_i[i_p] / r_j


@njit_serial(error_model='numpy')
def calculate_ai_bi_from_axis(r, A, B, C, K, U, idx, a_0_arr, a_i_arr,
                              b_i_arr):
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

    # Establish initial conditions (T_0 = 0, P_0 = 0)
    T_im1 = 0.
    P_im1 = 0.

    a_0 = 0.

    i_start = 0

    while i_start < n_part:

        # Iterate over particles
        for i_sort in range(i_start, n_part):
            i = idx[i_sort]
            r_i = r[i]
            A_i = A[i]
            B_i = B[i]
            C_i = C[i]

            l_i = (1. + 0.5 * A_i * r_i)
            m_i = 0.5 * A_i / r_i
            n_i = -0.5 * A_i * r_i ** 3
            o_i = (1. - 0.5 * A_i * r_i)

            T_i = l_i * T_im1 + m_i * P_im1 + 0.5 * B_i + 0.25 * A_i * C_i
            P_i = n_i * T_im1 + o_i * P_im1 + r_i * (
                    C_i - 0.5 * B_i * r_i - 0.25 * A_i * C_i * r_i)

            a_i_arr[i] = T_i
            b_i_arr[i] = P_i

            T_im1 = T_i
            P_im1 = P_i

        # Calculate a_0_diff.
        a_0_diff = - T_im1 / K[i]
        a_0 += a_0_diff
        a_0_arr[0] = a_0

        i_stop = n_part

        # Calculate a_i (in T_i) and b_i (in P_i) as functions of a_0_diff.
        for i_sort in range(i_start, n_part):
            i = idx[i_sort]
            T_old = a_i_arr[i]
            P_old = b_i_arr[i]
            K_old = K[i] * a_0_diff
            U_old = U[i] * a_0_diff

            # Test if precision is lost in the sum T_old + K_old.
            # 0.5 roughly corresponds to one lost bit of precision.
            # Also pass test if this is the first number of this iteration
            # to avoid an infinite loop or if this is the last number
            # to computer as that is zero by construction
            if (i_sort == i_start or i_sort == (n_part-1) or
                    abs(T_old + K_old) >= 1e-10 * abs(T_old - K_old) and
                    abs(P_old + U_old) >= 1e-10 * abs(P_old - U_old)):
                # Calculate a_i and b_i as functions of a_0_diff.
                # Store the result in T and P
                a_i_arr[i] = T_old + K_old
                b_i_arr[i] = P_old + U_old
            else:
                # Stop this iteration, go to the next one
                i_stop = i_sort
                break

        if i_stop < n_part:
            # Set T_im1 and T_im1 properly for the next iteration
            T_im1 = a_i_arr[idx[i_stop-1]]
            P_im1 = b_i_arr[idx[i_stop-1]]

        # Start the next iteration where this one stopped
        i_start = i_stop


@njit_serial(error_model='numpy')
def calculate_ABC(r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                  nabla_a2, idx, A, B, C):
    n_part = r.shape[0]

    for i_sort in range(n_part):
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

        A[i] = q_i * b
        B[i] = q_i * (- (gamma_i * dr_psi_i) * c
                      + (pr_i2 * dr_psi_i) / (r_i * a3)
                      + (pr_i * dxi_psi_i) * c
                      + pr_i2 / (r_i * r_i * a2)
                      + b_theta_0_i * b
                      + nabla_a2_i * c * 0.5)
        C[i] = q_i * (pr_i2 * c - (gamma_i / a - 1.) / r_i)


@njit_serial(error_model='numpy')
def calculate_KU(r, A, idx, K, U):
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

    # Establish initial conditions (K_0 = 1, U_0 = 0)
    K_im1 = 1.
    U_im1 = 0.

    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        A_i = A[i]

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
