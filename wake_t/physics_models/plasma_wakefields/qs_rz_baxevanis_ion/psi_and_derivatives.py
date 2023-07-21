"""
Contains the method to compute the wakefield potential and its derivatives
according to the paper by P. Baxevanis and G. Stupakov.

"""

import numpy as np
import math

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

    # Calculate dxi_psi after the last plasma plasma particle . 
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
def determine_neighboring_points(r, dr_p, idx, r_neighbor):
    """
    Determine the position of the middle points between each particle and
    its left and right neighbors.

    The result is stored in the `r_neighbor` array, which is already sorted.
    That is, as opposed to `r`, it does not need to be iterated by using an
    array of sorted indices.
    """
    # Initialize arrays.
    n_part = r.shape[0]

    r_im1 = 0.
    # Calculate psi and dr_psi.
    # Their value at the position of each plasma particle is calculated
    # by doing a linear interpolation between two values at the left and
    # right of the particle. The left point is the middle position between the
    # particle and its closest left neighbor, and the same for the right.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        dr_p_i = dr_p[i]

        # If this is not the first particle, calculate the left point (r_left)
        # and the field values there (psi_left and dr_psi_left) as usual.
        if i_sort > 0:
            r_left = (r_im1 + r_i) * 0.5
        # Otherwise, take r=0 as the location of the left point.
        else:
            r_left = max(r_i - dr_p_i * 0.5, 0.5 * r_i) 

        r_im1 = r_i
        r_neighbor[i_sort] = r_left

        # If this is the last particle, calculate the r_right as
        if i_sort == n_part - 1:
            r_right = r_i + dr_p_i * 0.5
            r_neighbor[-1] = r_right


@njit_serial()
def calculate_psi(r_eval, log_r_eval, r, sum_1, sum_2, idx, psi):    
    """Calculate psi at the radial positions given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # Calculate fields at r_eval.
    i_last = 0
    for j in range(n_points):
        r_j = r_eval[j]
        log_r_j = log_r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        for i_sort in range(i_last, n_part):
            i = idx[i_sort]
            r_i = r[i]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields at r_j.
        if i_last == -1:
            sum_1_j = 0.
            sum_2_j = 0.
            i_last = 0
        else:
            i = idx[i_last]
            sum_1_j = sum_1[i]
            sum_2_j = sum_2[i]
        psi[j] += sum_1_j*log_r_j - sum_2_j


@njit_serial(fastmath=True, error_model="numpy")
def calculate_psi_and_dr_psi(
        r_eval, log_r_eval, r, dr_p, idx, sum_1_arr, sum_2_arr, psi, dr_psi):
    """Calculate psi and dr_psi at the radial positions given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    r_max_plasma = r[idx[-1]] + dr_p[idx[-1]] * 0.5
    log_r_max_plasma = np.log(r_max_plasma)

    # Calculate fields at r_eval.
    i_last = 0
    for j in range(n_points):
        r_j = r_eval[j]
        log_r_j = log_r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        for i_sort in range(i_last, n_part):
            i = idx[i_sort]
            r_i = r[i]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields at r_j.
        if i_last == -1:
            sum_1_j = 0.
            sum_2_j = 0.
            i_last = 0
        else:
            i = idx[i_last]
            sum_1_j = sum_1_arr[i]
            sum_2_j = sum_2_arr[i]
        if r_j < r_max_plasma:
            psi[j] = sum_1_j*log_r_j - sum_2_j
            dr_psi[j] = sum_1_j / r_j
        else:
            psi_max = sum_1_j*log_r_max_plasma - sum_2_j
            psi[j] = psi_max + sum_1_j * (log_r_j - log_r_max_plasma)
            dr_psi[j] = psi_max / r_j


@njit_serial()
def calculate_dxi_psi(r_eval, r, idx, sum_3_arr, dxi_psi):
    """Calculate dxi_psi at the radial position given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # Calculate fields at r_eval.
    i_last = 0
    for j in range(n_points):
        r_j = r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        for i_sort in range(i_last, n_part):
            i = idx[i_sort]
            r_i = r[i]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields at r_j.
        if i_last == -1:
            sum_3_j = 0.
            i_last = 0
        else:
            i = idx[i_last]
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

# @njit_serial()
# def calculate_psi_and_derivatives_at_particles(
#         r, pr, q, idx, psi_pp, dr_psi_pp, dxi_psi_pp):
#     """
#     Calculate the wakefield potential and its derivatives at the position
#     of the plasma particles. This is done by using Eqs. (29) - (32) in
#     the paper by P. Baxevanis and G. Stupakov.

#     As indicated in the original paper, the value of the fields at the
#     discontinuities (at the exact radial position of the plasma particles)
#     is calculated as the average between the two neighboring values.

#     Parameters
#     ----------
#     r, pr, q : array
#         Arrays containing the radial position, momentum and charge of the
#         plasma particles.
#     idx : ndarray
#         Array containing the (radially) sorted indices of the plasma particles.
#     r_max : float
#         Maximum radial extent of the plasma column.
#     dr_p : float
#         Initial spacing between plasma macroparticles. Corresponds also the
#         width of the plasma sheet represented by the macroparticle.
#     psi_pp, dr_psi_pp, dxi_psi_pp : ndarray
#         Arrays where the value of the wakefield potential and its derivatives
#         at the location of the plasma particles will be stored.

#     """
#     # Initialize arrays.
#     n_part = r.shape[0]

#     # Initialize value of sums.
#     sum_1 = 0.
#     sum_2 = 0.
#     sum_3 = 0.

#     # Calculate psi and dr_psi.
#     # Their value at the position of each plasma particle is calculated
#     # by doing a linear interpolation between two values at the left and
#     # right of the particle. The left point is the middle position between the
#     # particle and its closest left neighbor, and the same for the right.
#     for i_sort in range(n_part):
#         i = idx[i_sort]
#         r_i = r[i]
#         q_i = q[i]

#         # Calculate new sums.
#         sum_1_new = sum_1 + q_i
#         sum_2_new = sum_2 + q_i * np.log(r_i)

#         psi_left = sum_1*np.log(r_i) - sum_2
#         psi_right = sum_1_new*np.log(r_i) - sum_2_new
#         psi_pp[i] = 0.5 * (psi_left + psi_right)
            
#         dr_psi_left = sum_1 / r_i
#         dr_psi_right = sum_1_new / r_i
#         dr_psi_pp[i] = 0.5 * (dr_psi_left + dr_psi_right)

#         # Update value of sums.
#         sum_1 = sum_1_new
#         sum_2 = sum_2_new

#     # Boundary condition for psi (Force potential to be zero either at the
#     # plasma edge or after the last particle, whichever is further away).
#     psi_pp -= sum_1*np.log(r_i) - sum_2

#     # In theory, psi cannot be smaller than -1. However, it has been observed
#     # than in very strong blowouts, near the peak, values below -1 can appear
#     # in this numerical method. In addition, values very close to -1 will lead
#     # to particles with gamma >> 10, which will also lead to problems.
#     # This condition here makes sure that this does not happen, improving
#     # the stability of the solver.
#     for i in range(n_part):
#         # Should only happen close to the peak of very strong blowouts.
#         if psi_pp[i] < -0.90:
#             psi_pp[i] = -0.90

#     # Calculate dxi_psi (also by interpolation).
#     for i_sort in range(n_part):
#         i = idx[i_sort]
#         r_i = r[i]
#         pr_i = pr[i]
#         q_i = q[i]
#         psi_i = psi_pp[i]

#         sum_3_new = sum_3 + (q_i * pr_i) / (r_i * (1 + psi_i))

#         dxi_psi_left = -sum_3
#         dxi_psi_right = -sum_3_new
#         dxi_psi_pp[i] = 0.5 * (dxi_psi_left + dxi_psi_right)
#         sum_3 = sum_3_new

#     # Apply longitudinal derivative of the boundary conditions of psi.
#     dxi_psi_pp += sum_3

#     # Again, near the peak of a strong blowout, very large and unphysical
#     # values could appear. This condition makes sure a threshold us not
#     # exceeded.
#     for i in range(n_part):
#         if dxi_psi_pp[i] > 3.:
#             dxi_psi_pp[i] = 3.
#         if dxi_psi_pp[i] < -3.:
#             dxi_psi_pp[i] = -3.


# # @njit_serial()
# def calculate_psi_old(r_eval, r, q, idx, psi):
#     """
#     Calculate the wakefield potential at the radial
#     positions specified in r_eval. This is done by using Eq. (29) in
#     the paper by P. Baxevanis and G. Stupakov.

#     Parameters
#     ----------
#     r_eval : array
#         Array containing the radial positions where psi should be calculated.
#     r, q : array
#         Arrays containing the radial position, and charge of the
#         plasma particles.
#     idx : ndarray
#         Array containing the (radially) sorted indices of the plasma particles.
#     psi : ndarray
#         1D Array where the values of the wakefield potential will be stored.

#     """
#     # Initialize arrays with values of psi and sums at plasma particles.
#     n_part = r.shape[0]
#     sum_1_arr = np.zeros(n_part)
#     sum_2_arr = np.zeros(n_part)
#     sum_1 = 0.
#     sum_2 = 0.

#     # Calculate sum_1, sum_2 and psi_part.
#     for i_sort in range(n_part):
#         i = idx[i_sort]
#         r_i = r[i]
#         q_i = q[i]

#         sum_1 += q_i
#         sum_2 += q_i * np.log(r_i)
#         sum_1_arr[i] = sum_1
#         sum_2_arr[i] = sum_2
#     r_N = r_i

#     # Initialize array for psi at r_eval locations.
#     n_points = r_eval.shape[0]

#     # Calculate fields at r_eval.
#     i_last = 0
#     for j in range(n_points):
#         r_j = r_eval[j]
#         # Get index of last plasma particle with r_i < r_j, continuing from
#         # last particle found in previous iteration.
#         for i_sort in range(i_last, n_part):
#             i = idx[i_sort]
#             r_i = r[i]
#             i_last = i_sort
#             if r_i >= r_j:
#                 i_last -= 1
#                 break
#         # Calculate fields at r_j.
#         if i_last == -1:
#             sum_1_j = 0.
#             sum_2_j = 0.
#             i_last = 0
#         else:
#             i = idx[i_last]
#             sum_1_j = sum_1_arr[i]
#             sum_2_j = sum_2_arr[i]
#         psi[j] = sum_1_j*np.log(r_j) - sum_2_j

#     # Apply boundary conditions.
#     psi -= sum_1*np.log(r_N) - sum_2


# @njit_serial()
# def calculate_psi_and_derivatives(r_fld, r, pr, q, idx):
#     """
#     Calculate the wakefield potential and its derivatives at the radial
#     positions specified in r_fld. This is done by using Eqs. (29) - (32) in
#     the paper by P. Baxevanis and G. Stupakov.

#     Parameters
#     ----------
#     r_fld : array
#         Array containing the radial positions where psi should be calculated.
#     r, pr, q : array
#         Arrays containing the radial position, momentum and charge of the
#         plasma particles.

#     """
#     # Initialize arrays with values of psi and sums at plasma particles.
#     n_part = r.shape[0]
#     psi_part = np.zeros(n_part)
#     sum_1_arr = np.zeros(n_part)
#     sum_2_arr = np.zeros(n_part)
#     sum_3_arr = np.zeros(n_part)
#     sum_1 = 0.
#     sum_2 = 0.
#     sum_3 = 0.

#     # Calculate sum_1, sum_2 and psi_part.
#     for i_sort in range(n_part):
#         i = idx[i_sort]
#         r_i = r[i]
#         pr_i = pr[i]
#         q_i = q[i]

#         sum_1 += q_i
#         sum_2 += q_i * np.log(r_i)
#         sum_1_arr[i] = sum_1
#         sum_2_arr[i] = sum_2
#         psi_part[i] = sum_1 * np.log(r_i) - sum_2
#     r_N = r_i
#     psi_part -= sum_1 * np.log(r_N) - sum_2

#     # Calculate sum_3.
#     for i_sort in range(n_part):
#         i = idx[i_sort]
#         r_i = r[i]
#         pr_i = pr[i]
#         q_i = q[i]
#         psi_i = psi_part[i]

#         sum_3 += (q_i * pr_i) / (r_i * (1 + psi_i))
#         sum_3_arr[i] = sum_3

#     # Initialize arrays for psi and derivatives at r_fld locations.
#     n_points = r_fld.shape[0]
#     psi = np.zeros(n_points)
#     dr_psi = np.zeros(n_points)
#     dxi_psi = np.zeros(n_points)

#     # Calculate fields at r_fld.
#     i_last = 0
#     for j in range(n_points):
#         r_j = r_fld[j]
#         # Get index of last plasma particle with r_i < r_j, continuing from
#         # last particle found in previous iteration.
#         for i_sort in range(i_last, n_part):
#             i = idx[i_sort]
#             r_i = r[i]
#             i_last = i_sort
#             if r_i >= r_j:
#                 i_last -= 1
#                 break
#         # Calculate fields at r_j.
#         if i_last == -1:
#             psi[j] = 0.
#             dr_psi[j] = 0.
#             dxi_psi[j] = 0.
#             i_last = 0
#         else:
#             i = idx[i_last]
#             psi[j] = sum_1_arr[i] * np.log(r_j) - sum_2_arr[i]
#             dr_psi[j] = sum_1_arr[i] / r_j
#             dxi_psi[j] = - sum_3_arr[i]
#     psi -= sum_1 * np.log(r_N) - sum_2
#     dxi_psi = dxi_psi + sum_3
#     return psi, dr_psi, dxi_psi


# @njit_serial()
# def delta_psi_eq(r, sum_1, sum_2, r_max, pc):
#     """ Adapted equation (29) from original paper. """
#     delta_psi_elec = sum_1*np.log(r) - sum_2
#     if r <= r_max:
#         delta_psi_ion = 0.25*r**2 + pc*r**4/16
#     else:
#         delta_psi_ion = (
#             0.25*r_max**2 + pc*r_max**4/16 +
#             (0.5 * r_max**2 + 0.25*pc*r_max**4) * (
#                 np.log(r)-np.log(r_max)))
#     return delta_psi_elec - delta_psi_ion


# @njit_serial()
# def dr_psi_eq(r, sum_1, r_max, pc):
#     """ Adapted equation (31) from original paper. """
#     dr_psi_elec = sum_1 / r
#     if r <= r_max:
#         dr_psi_ion = 0.5 * r + 0.25 * pc * r ** 3
#     else:
#         dr_psi_ion = (0.5 * r_max**2 + 0.25 * pc * r_max**4) / r
#     return dr_psi_elec - dr_psi_ion
