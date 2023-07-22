"""
Contains the method to compute the azimuthal magnetic field from the plasma
according to the paper by P. Baxevanis and G. Stupakov.

"""

from wake_t.utilities.numba import njit_serial


@njit_serial()
def calculate_b_theta_at_particles(
    r_e, pr_e, q_e, gamma_e,
    r_i,
    i_sort_e, i_sort_i,        
    ion_motion,
    r_neighbor_e,
    psi_e, dr_psi_e, dxi_psi_e,
    b_t_0_e, nabla_a2_e,
    A, B, C,
    K, U,
    a_0, a, b,
    b_t_e, b_t_i
):
    """Calculate the azimuthal magnetic field at the plasma particles.

    To simplify the algorithm, this method considers only the magnetic field
    generated by the electrons, not the ions. This is usually a reasonable
    approximation, even when ion motion is enabled, because the ions are much
    slower than the electrons.

    The value of b_theta at a a radial position r is calculated as
        
        b_theta = a_i * r + b_i / r

    This requires determining the values of the a_i and b_i coefficients,
    which are calculated as follows, using Eqs. (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov:

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

    Parameters
    ----------
    r_e, pr_e, q_e, gamma_e : ndarray
        Radial position, momentum, charge and Lorenz factor of the plasma
        electrons.
    r_i : ndarray
        Radial position of the plasma ions.
    i_sort_e, i_sort_i : ndarray
        Sorted indices of the electrons and ions (from lower to higher radii).
    ion_motion : bool
        Whether the ions can move. If `True`, the magnetic field
        will also be calculated at the ions.
    r_neighbor_e : ndarray
        The radial location of the middle points between
        each electron and its left and right neighbors. This array is already
        sorted and should not be indexed with `i_sort_e`.
    psi_e, dr_psi_e, dxi_psi_e : ndarray
        Value of psi and its derivatives at the location of the plasma
        electrons.
    b_t_0_e, nabla_a2_e : ndarray
        Value of the source terms (magnetic field from bunches and
        ponderomotive force of a laser) at the location of the plasma
        electrons.
    A, B, C : ndarray
        Arrays where the A_i, B_i, C_i terms in Eq. (26) will be stored.
    K, U : ndarray
        Auxiliary arrays where terms for solving the system in Eq. (27) will
        be stored.
    a_0, a, b : ndarray
        Arrays where the a_i and b_i coefficients coming out of solving the
        system in Eq. (27) will be stored.
    b_t_e, b_t_i : ndarray
        Arrays where azimuthal magnetic field at the plasma electrons and ions
        will be stored.
    """
    # Calculate the A_i, B_i, C_i coefficients in Eq. (26).
    calculate_ABC(
        r_e, pr_e, q_e, gamma_e,
        psi_e, dr_psi_e, dxi_psi_e, b_t_0_e,
        nabla_a2_e, i_sort_e, A, B, C
    )

    # Calculate the a_i, b_i coefficients in Eq. (27).
    calculate_KU(r_e, A, i_sort_e, K, U)
    calculate_ai_bi_from_axis(r_e, A, B, C, K, U, i_sort_e, a_0, a, b)

    # Calculate b_theta at plasma particles.
    calculate_b_theta_at_electrons(
        r_e, a_0[0], a, b, r_neighbor_e, i_sort_e, b_t_e
    )
    check_b_theta(b_t_e)
    if ion_motion:
        calculate_b_theta_at_ions(
            r_i, r_e, a_0[0], a, b, i_sort_i, i_sort_e, b_t_i
        )
        check_b_theta(b_t_i)


@njit_serial(error_model='numpy')
def calculate_b_theta_at_electrons(r, a_0, a, b, r_neighbor, idx, b_theta):
    """
    Calculate the azimuthal magnetic field from the plasma at the location
    of the plasma electrons using Eqs. (24), (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    position of each electron presents a discontinuity. To avoid this, the
    at each electron is calculated as a linear interpolation between the two
    values at its left and right neighboring points.

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
            a_i_left = a[im1]
            b_i_left = b[im1]
            b_theta_left = a_i_left * r_left + b_i_left / r_left
        else:
            a_i_left = a_0
            b_theta_left = a_i_left * r_left
            
        a_i_right = a[i]
        b_i_right = b[i]
        b_theta_right = a_i_right * r_right + b_i_right / r_right

        # Do interpolation.
        c2 = (b_theta_right - b_theta_left) / (r_right - r_left)
        c1 = b_theta_left - c2*r_left
        b_theta[i] = c1 + c2*r_i


@njit_serial(error_model='numpy')
def calculate_b_theta_at_ions(r_i, r_e, a_0, a, b, idx_i, idx_e, b_theta):
    """
    Calculate the azimuthal magnetic field at the plasma ions. This method
    is identical to `calculate_b_theta` except in that `r_i` is not
    sorted and thus need the additonal `idx_i` argument.

    """
    # Calculate field at particles as average between neighboring values.
    n_i = r_i.shape[0]
    n_e = r_e.shape[0]
    i_last = 0
    for i_sort in range(n_i):
        i = idx_i[i_sort]
        r_i_i = r_i[i]        
        # Get index of last plasma electron with r_i_e < r_i_i, continuing from
        # last electron found in previous iteration.
        for i_sort_e in range(i_last, n_e):
            i_e = idx_e[i_sort_e]
            r_i_e = r_e[i_e]
            if r_i_e >= r_i_i:
                i_last = i_sort_e - 1
                break
        # Calculate fields.
        if i_last == -1:
            b_theta[i] = a_0 * r_i_i
            i_last = 0
        else:
            i_e = idx_e[i_last]
            b_theta[i] = a[i_e] * r_i_i + b[i_e] / r_i_i


@njit_serial(error_model='numpy')
def calculate_b_theta(r_fld, a_0, a, b, r, idx, b_theta):
    """
    Calculate the azimuthal magnetic field from the plasma at the radial
    locations in `r_fld`.

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
            if r_i >= r_j:
                i_last = i_sort - 1
                break
        # Calculate fields.
        if i_last == -1:
            b_theta[j] = a_0 * r_j
            i_last = 0
        else:
            i_p = idx[i_last]
            b_theta[j] = a[i_p] * r_j + b[i_p] / r_j


@njit_serial(error_model='numpy')
def calculate_ai_bi_from_axis(r, A, B, C, K, U, idx, a_0, a, b):
    """
    Calculate the values of a_i and b_i which are needed to determine
    b_theta at any r position.

    For details about the input parameters and algorithm see method
    'calculate_b_theta_at_particles'.

    """
    n_part = r.shape[0]

    # Establish initial conditions (T_0 = 0, P_0 = 0)
    T_im1 = 0.
    P_im1 = 0.

    a_0[:] = 0.

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

            a[i] = T_i
            b[i] = P_i

            T_im1 = T_i
            P_im1 = P_i

        # Calculate a_0_diff.
        a_0_diff = - T_im1 / K[i]
        a_0 += a_0_diff

        # Calculate a_i (in T_i) and b_i (in P_i) as functions of a_0_diff.
        i_stop = n_part
        im1 = 0
        for i_sort in range(i_start, n_part):
            i = idx[i_sort]
            T_old = a[i]
            P_old = b[i]
            K_old = K[i] * a_0_diff
            U_old = U[i] * a_0_diff

            # Test if precision is lost in the sum T_old + K_old.
            # 0.5 roughly corresponds to one lost bit of precision.
            # Also pass test if this is the first number of this iteration
            # to avoid an infinite loop or if this is the last number
            # to computer as that is zero by construction
            # Angel: if T_old + K_old (small number) is less than 10 orders
            # of magnitude smaller than T_old - K_old (big number), then we
            # have enough precision (from simulation tests).
            if (i_sort == i_start or i_sort == (n_part-1) or
                    abs(T_old + K_old) >= 1e-10 * abs(T_old - K_old) and
                    abs(P_old + U_old) >= 1e-10 * abs(P_old - U_old)):
                # Calculate a_i and b_i as functions of a_0_diff.
                # Store the result in T and P
                a[i] = T_old + K_old
                b[i] = P_old + U_old
            else:
                # If the precision is not sufficient, stop this iteration
                # and rescale T_im1 and P_im1 for the next one.
                i_stop = i_sort
                T_im1 = a[im1]
                P_im1 = b[im1]
                break
            im1 = i

        # Start the next iteration where this one stopped
        i_start = i_stop


@njit_serial(error_model='numpy')
def calculate_ABC(r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                  nabla_a2, idx, A, B, C):
    """Calculate the A_i, B_i and C_i coefficients of the linear system."""
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
    """Calculate the K_i and U_i values of the linear system."""
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


@njit_serial()
def check_b_theta(b_theta):
    """Check that the values of b_theta are within a reasonable range

    This is used to prevent issues at the peak of a blowout wake, for example.
    """
    for i in range(b_theta.shape[0]):
        b_theta_i = b_theta[i]
        if b_theta_i < -3:
            b_theta[i] = -3
        elif b_theta_i > 3:
            b_theta[i] = 3
