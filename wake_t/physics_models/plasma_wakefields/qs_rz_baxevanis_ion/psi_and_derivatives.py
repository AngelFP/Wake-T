"""
Contains the method to compute the wakefield potential and its derivatives
according to the paper by P. Baxevanis and G. Stupakov.

"""

import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial()
def calculate_psi_and_derivatives_at_particles(
    r_e, pr_e, q_e, dr_p_e,
    r_i, pr_i, q_i, dr_p_i,
    i_sort_e, i_sort_i,
    ion_motion, calculate_ion_sums,
    r_neighbor_e, log_r_neighbor_e,
    r_neighbor_i, log_r_neighbor_i,
    sum_1_e, sum_2_e, sum_3_e,
    sum_1_i, sum_2_i, sum_3_i,
    psi_bg_i, dr_psi_bg_i, dxi_psi_bg_i,
    psi_bg_e, dr_psi_bg_e, dxi_psi_bg_e,
    psi_e, dr_psi_e, dxi_psi_e,
    psi_i, dr_psi_i, dxi_psi_i,
    psi_max,
    psi, dxi_psi,
):
    """Calculate wakefield potential and derivatives at the plasma particles.

    Parameters
    ----------
    r_e, pr_e, q_e, dr_p_e, r_i, pr_i, q_i, dr_p_i : ndarray
        Radial position, momentum, charge and width (i.e., initial separation)
        of the plasma electrons (subindex e) and ions (subindex i).
    i_sort_e, i_sort_i : ndarray
        Sorted indices of the particles (from lower to higher radii).
    ion_motion : bool
        Whether the ions can move. If `True`, the potential and its derivatives
        will also be calculated at the ions.
    calculate_ion_sums : _type_
        Whether to calculate sum_1, sum_2 and sum_3 for the ions. When ion
        motion is disabled, this only needs to be done once.
    r_neighbor_e, log_r_neighbor_e, r_neighbor_i, log_r_neighbor_i : ndarray
        The radial location (and its logarithm) of the middle points between
        each particle and it left and right neighbors. This array is already
        sorted and should not be indexed with `i_sort`.
    sum_1_e, sum_2_e, sum_3_e, sum_1_i, sum_2_i, sum_3_i : ndarray
        Arrays where the values of sum_1, sum_2 and sum_3 at each particle
        will be stored.
    psi_bg_i, dr_psi_bg_i, dxi_psi_bg_i : ndarray
        Arrays where the contribution of the ion background (calculated at
        r_neighbor_e) to psi and its derivatives will be stored.
    psi_bg_e, dr_psi_bg_e, dxi_psi_bg_e : ndarray
        Arrays where the contribution of the electron background (calculated at
        r_neighbor_i) to psi and its derivatives will be stored.
    psi_e, dr_psi_e, dxi_psi_e, psi_i, dr_psi_i, dxi_psi_i : ndarray
        Arrays where the value of psi and its derivatives at the plasma
        electrons and ions will be stored.
    psi_max : ndarray
        Array with only one element where the the value of psi after the last
        particle is stored. This value is used to ensure the boundary condition
        that psi should be 0 after the last particle.
    psi, dxi_psi : _type_
        Arrays where the value of psi and its longitudinal derivative at all
        plasma particles is stored.
    """

    # Calculate cumulative sums 1 and 2 (Eqs. (29) and (31)).
    calculate_cumulative_sum_1(q_e, i_sort_e, sum_1_e)
    calculate_cumulative_sum_2(r_e, q_e, i_sort_e, sum_2_e)
    if ion_motion or not calculate_ion_sums:
        calculate_cumulative_sum_1(q_i, i_sort_i, sum_1_i)
        calculate_cumulative_sum_2(r_i, q_i, i_sort_i, sum_2_i)

    # Calculate the psi and dr_psi background at the neighboring points.
    # For the electrons, compute the psi and dr_psi due to the ions at
    # r_neighbor_e. For the ions, compute the psi and dr_psi due to the
    # electrons at r_neighbor_i.
    calculate_psi_and_dr_psi(
        r_neighbor_e, log_r_neighbor_e, r_i, dr_p_i, i_sort_i,
        sum_1_i, sum_2_i, psi_bg_i, dr_psi_bg_i
    )
    if ion_motion:
        calculate_psi_and_dr_psi(
            r_neighbor_i, log_r_neighbor_i, r_e, dr_p_e, i_sort_e,
            sum_1_e, sum_2_e, psi_bg_e, dr_psi_bg_e
        )

    # Calculate psi after the last plasma plasma particle (assumes
    # that the total electron and ion charge are the same).
    # This will be used to ensure the boundary condition (psi=0) after last
    # plasma particle.
    psi_max[:] = - (sum_2_e[i_sort_e[-1]] + sum_2_i[i_sort_i[-1]])

    # Calculate psi and dr_psi at the particles including the contribution
    # from the background.
    calculate_psi_dr_psi_at_particles_bg(
        r_e, sum_1_e, sum_2_e, psi_bg_i,
        r_neighbor_e, log_r_neighbor_e, i_sort_e, psi_e, dr_psi_e
    )
    # Apply boundary condition
    psi_e -= psi_max
    if ion_motion:
        calculate_psi_dr_psi_at_particles_bg(
            r_i, sum_1_i, sum_2_i, psi_bg_e,
            r_neighbor_i, log_r_neighbor_i, i_sort_i, psi_i, dr_psi_i
        )
        # Apply boundary condition
        psi_i -= psi_max

    # Check that the values of psi are within a reasonable range (prevents
    # issues at the peak of a blowout wake, for example).
    check_psi(psi)

    # Calculate cumulative sum 3 (Eq. (32)).
    calculate_cumulative_sum_3(r_e, pr_e, q_e, psi_e, i_sort_e, sum_3_e)
    if ion_motion or not calculate_ion_sums:
        calculate_cumulative_sum_3(r_i, pr_i, q_i, psi_i, i_sort_i, sum_3_i)

    # Calculate the dxi_psi background at the neighboring points.
    # For the electrons, compute the psi and dr_psi due to the ions at
    # r_neighbor_e. For the ions, compute the psi and dr_psi due to the
    # electrons at r_neighbor_i.
    calculate_dxi_psi(r_neighbor_e, r_i, i_sort_i, sum_3_i, dxi_psi_bg_i)
    if ion_motion:
        calculate_dxi_psi(r_neighbor_i, r_e, i_sort_e, sum_3_e, dxi_psi_bg_e)

    # Calculate dxi_psi after the last plasma plasma particle.
    # This will be used to ensure the boundary condition (dxi_psi = 0) after
    # last plasma particle.
    dxi_psi_max = sum_3_e[i_sort_e[-1]] + sum_3_i[i_sort_i[-1]]

    # Calculate dxi_psi at the particles including the contribution
    # from the background.
    calculate_dxi_psi_at_particles_bg(
        r_e, sum_3_e, dxi_psi_bg_i, r_neighbor_e, i_sort_e, dxi_psi_e
    )
    # Apply boundary condition
    dxi_psi_e += dxi_psi_max
    if ion_motion:
        calculate_dxi_psi_at_particles_bg(
            r_i, sum_3_i, dxi_psi_bg_e, r_neighbor_i, i_sort_i, dxi_psi_i
        )
        # Apply boundary condition
        dxi_psi_i += dxi_psi_max

    # Check that the values of dxi_psi are within a reasonable range (prevents
    # issues at the peak of a blowout wake, for example).
    check_dxi_psi(dxi_psi)


@njit_serial(fastmath=True)
def calculate_cumulative_sum_1(q, idx, sum_1_arr):
    """Calculate the cumulative sum in Eq. (29)."""
    sum_1 = 0.
    for i_sort in range(q.shape[0]):
        i = idx[i_sort]
        q_i = q[i]
        sum_1 += q_i
        sum_1_arr[i] = sum_1


@njit_serial(fastmath=True)
def calculate_cumulative_sum_2(r, q, idx, sum_2_arr):
    """Calculate the cumulative sum in Eq. (31)."""
    sum_2 = 0.
    for i_sort in range(r.shape[0]):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]
        sum_2 += q_i * np.log(r_i)
        sum_2_arr[i] = sum_2


@njit_serial(fastmath=True, error_model="numpy")
def calculate_cumulative_sum_3(r, pr, q, psi, idx, sum_3_arr):
    """Calculate the cumulative sum in Eq. (32)."""
    sum_3 = 0.
    for i_sort in range(r.shape[0]):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi[i]
        sum_3 += (q_i * pr_i) / (r_i * (1 + psi_i))
        sum_3_arr[i] = sum_3


@njit_serial(fastmath=True, error_model="numpy")
def calculate_psi_dr_psi_at_particles_bg(
        r, sum_1, sum_2, psi_bg, r_neighbor, log_r_neighbor, idx, psi, dr_psi):
    """
    Calculate the wakefield potential and its radial derivative at the
    position of the plasma eletrons (ions) taking into account the background
    from the ions (electrons).

    The value at the position of each plasma particle is calculated
    by doing a linear interpolation between the two neighboring points, where
    the left point is the middle position between the
    particle and its closest left neighbor, and the same for the right.

    Parameters
    ----------
    r : ndarray
        Radial position of the plasma particles (either electrons or ions).
    sum_1, sum_2 : ndarray
        Value of the cumulative sums 1 and 2 at each of the particles.
    psi_bg : ndarray
        Value of the contribution to psi of the background species (the
        ions if `r` contains electron positions, or the electrons if `r`
        contains ion positions) at the location of the neighboring middle
        points in `r_neighbor`.
    r_neighbor, log_r_neighbor : ndarray
        Location and its logarithm of the middle points between the left and
        right neighbors of each particle in `r`.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    psi, dr_psi : ndarray
        Arrays where psi and dr_psi at the plasma particles will be stored.

    """
    # Initialize arrays.
    n_part = r.shape[0]

    # Get initial values for left neighbors.
    r_left = r_neighbor[0]
    psi_bg_left = psi_bg[0]
    psi_left = psi_bg_left

    # Loop over particles.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]

        # Get sums to calculate psi at right neighbor.
        sum_1_right_i = sum_1[i]
        sum_2_right_i = sum_2[i]

        # Get values at right neighbor.
        r_right = r_neighbor[i_sort + 1]
        log_r_right = log_r_neighbor[i_sort + 1]
        psi_bg_right = psi_bg[i_sort + 1]

        # Calculate psi at right neighbor.
        psi_right = sum_1_right_i*log_r_right - sum_2_right_i + psi_bg_right

        # Interpolate psi between left and right neighbors.
        b_1 = (psi_right - psi_left) / (r_right - r_left)
        a_1 = psi_left - b_1*r_left
        psi[i] = a_1 + b_1*r_i

        # dr_psi is simply the slope used for interpolation.
        dr_psi[i] = b_1

        # Update values of next left neighbor with those of the current right
        # neighbor.
        r_left = r_right
        psi_bg_left = psi_bg_right
        psi_left = psi_right


@njit_serial(fastmath=True, error_model="numpy")
def calculate_dxi_psi_at_particles_bg(
        r, sum_3, dxi_psi_bg, r_neighbor, idx, dxi_psi):
    """
    Calculate the longitudinal derivative of the wakefield potential at the
    position of the plasma eletrons (ions) taking into account the background
    from the ions (electrons).

    The value at the position of each plasma particle is calculated
    by doing a linear interpolation between the two neighboring points, where
    the left point is the middle position between the
    particle and its closest left neighbor, and the same for the right.

    Parameters
    ----------
    r : ndarray
        Radial position of the plasma particles (either electrons or ions).
    sum_3 : ndarray
        Value of the cumulative sum 3 at each of the particles.
    dxi_psi_bg : ndarray
        Value of the contribution to dxi_psi of the background species (the
        ions if `r` contains electron positions, or the electrons if `r`
        contains ion positions) at the location of the neighboring middle
        points in `r_neighbor`.
    r_neighbor : ndarray
        Location of the middle points between the left and right neighbors
        of each particle in `r`.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    dxi_psi : ndarray
        Array where dxi_psi at the plasma particles will be stored.

    """
    # Initialize arrays.
    n_part = r.shape[0]

    # Get initial values for left neighbors.
    r_left = r_neighbor[0]
    dxi_psi_bg_left = dxi_psi_bg[0]
    dxi_psi_left = dxi_psi_bg_left

    # Loop over particles.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]

        # Calculate value at right neighbor.
        r_right = r_neighbor[i_sort + 1]
        dxi_psi_bg_right = dxi_psi_bg[i_sort + 1]
        sum_3_right_i = sum_3[i]
        dxi_psi_right = - sum_3_right_i + dxi_psi_bg_right

        # Interpolate value between left and right neighbors.
        b_1 = (dxi_psi_right - dxi_psi_left) / (r_right - r_left)
        a_1 = dxi_psi_left - b_1*r_left
        dxi_psi[i] = a_1 + b_1*r_i

        # Update values of next left neighbor with those of the current right
        # neighbor.
        r_left = r_right
        dxi_psi_bg_left = dxi_psi_bg_right
        dxi_psi_left = dxi_psi_right


@njit_serial()
def calculate_psi(r_eval, log_r_eval, r, sum_1, sum_2, idx, psi):
    """Calculate psi at the radial positions given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # Calculate fields at r_eval.
    i_last = 0
    sum_1_i = 0.
    sum_2_i = 0.
    for j in range(n_points):
        r_j = r_eval[j]
        log_r_j = log_r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        while i_last < n_part:
            i = idx[i_last]
            r_i = r[i]
            if r_i >= r_j:
                break
            i_last += 1
        if i_last > 0:
            i = idx[i_last - 1]
            sum_1_i = sum_1[i]
            sum_2_i = sum_2[i]
        psi[j] += sum_1_i * log_r_j - sum_2_i


@njit_serial(fastmath=True, error_model="numpy")
def calculate_psi_and_dr_psi(
        r_eval, log_r_eval, r, dr_p, idx, sum_1_arr, sum_2_arr, psi, dr_psi):
    """Calculate psi and dr_psi at the radial positions given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # r_max_plasma = r[idx[-1]] + dr_p[idx[-1]] * 0.5
    # log_r_max_plasma = np.log(r_max_plasma)

    # Calculate fields at r_eval.
    i_last = 0
    sum_1_j = 0.
    sum_2_j = 0.
    for j in range(n_points):
        r_j = r_eval[j]
        log_r_j = log_r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        while i_last < n_part:
            i = idx[i_last]
            r_i = r[i]
            if r_i >= r_j:
                break
            i_last += 1
        if i_last > 0:
            i = idx[i_last - 1]
            sum_1_j = sum_1_arr[i]
            sum_2_j = sum_2_arr[i]
        # Calculate fields at r_j.
        psi[j] = sum_1_j*log_r_j - sum_2_j
        dr_psi[j] = sum_1_j / r_j


@njit_serial()
def calculate_dxi_psi(r_eval, r, idx, sum_3_arr, dxi_psi):
    """Calculate dxi_psi at the radial position given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # Calculate fields at r_eval.
    i_last = 0
    sum_3_j = 0
    for j in range(n_points):
        r_j = r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        while i_last < n_part:
            i = idx[i_last]
            r_i = r[i]
            if r_i >= r_j:
                break
            i_last += 1
        if i_last > 0:
            i = idx[i_last - 1]
            sum_3_j = sum_3_arr[i]
        dxi_psi[j] = - sum_3_j


@njit_serial()
def check_psi(psi):
    """Check that the values of psi are within a reasonable range

    This is used to prevent issues at the peak of a blowout wake, for example).
    """
    for i in range(psi.shape[0]):
        if psi[i] < -0.9:
            psi[i] = -0.9


@njit_serial()
def check_dxi_psi(dxi_psi):
    """Check that the values of dxi_psi are within a reasonable range

    This is used to prevent issues at the peak of a blowout wake, for example).
    """
    for i in range(dxi_psi.shape[0]):
        dxi_psi_i = dxi_psi[i]
        if dxi_psi_i < -3:
            dxi_psi[i] = -3
        elif dxi_psi_i > 3:
            dxi_psi[i] = 3
