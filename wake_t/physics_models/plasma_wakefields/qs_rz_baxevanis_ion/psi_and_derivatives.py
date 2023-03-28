"""
Contains the method to compute the wakefield potential and its derivatives
according to the paper by P. Baxevanis and G. Stupakov.

"""

import numpy as np
import math

from wake_t.utilities.numba import njit_serial


@njit_serial()
def calculate_cumulative_sums(r, q, idx, sum_1_arr, sum_2_arr):
    sum_1 = 0.
    sum_2 = 0.
    for i_sort in range(r.shape[0]):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]

        sum_1 += q_i
        sum_2 += q_i * np.log(r_i)
        sum_1_arr[i] = sum_1
        sum_2_arr[i] = sum_2


@njit_serial()
def calculate_cumulative_sum_1(q, idx, sum_1_arr):
    sum_1 = 0.
    for i_sort in range(q.shape[0]):
        i = idx[i_sort]
        q_i = q[i]

        sum_1 += q_i
        sum_1_arr[i] = sum_1


@njit_serial()
def calculate_cumulative_sum_2(r, q, idx, sum_2_arr):
    sum_2 = 0.
    for i_sort in range(r.shape[0]):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]

        sum_2 += q_i * math.log(r_i)
        sum_2_arr[i] = sum_2


@njit_serial()
def calculate_cumulative_sum_3(r, pr, q, psi, idx, sum_3_arr):
    sum_3 = 0.
    for i_sort in range(r.shape[0]):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi[i]

        sum_3 += (q_i * pr_i) / (r_i * (1 + psi_i))
        sum_3_arr[i] = sum_3


# def calculate_dxi_psi_at_particles(r, pr, q, psi, idx, dxi_psi):
#     dxi_psi_i = 0.
#     for i_sort in range(r.shape[0]):
#         i = idx[i_sort]
#         r_i = r[i]
#         pr_i = pr[i]
#         q_i = q[i]
#         psi_i = psi[i]

#         dxi_psi_i_new = dxi_psi_i - (q_i * pr_i) / (r_i * (1 + psi_i))
#         dxi_psi[i] = 0.5 * (dxi_psi_i_new + dxi_psi_i)
#         dxi_psi_i = dxi_psi_i_new
#     dxi_psi -= dxi_psi_i_new


# def calculate_psi_dr_psi_at_particles(r, dr_p, sum_1, sum_2, i_left, i_right, idx, psi, dr_psi):
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
#     n_part = r.shape[0]

#     # Calculate fields at r_eval.
#     for j in range(n_part):
#         r_j = r[j]
#         i_left_j = i_left[j]
#         i_right_j = i_right[j]
#         if i_left_j == -1:
#             sum_1_left_j = 0.
#             sum_2_left_j = 0.
#         else:
#             sum_1_left_j = sum_1[i_left_j]
#             sum_2_left_j = sum_2[i_left_j]
#         sum_1_right_j = sum_1[i_right_j]
#         sum_2_right_j = sum_2[i_right_j]

#         r_j_left = r_j - dr_p / 8
#         r_j_right = r_j + dr_p / 8

#         psi_left = sum_1_left_j*np.log(r_j_left) - sum_2_left_j
#         psi_right = sum_1_right_j*np.log(r_j_right) - sum_2_right_j
#         psi[j] = (psi_left + psi_right) * 0.5
#         # dr_psi[j] = (psi_right - psi_left) / (dr_p / 4) #(sum_1_left_j / r_j_left + sum_1_right_j / r_j_right) * 0.5
#         dr_psi[j] = (sum_1_left_j / r_j_left + sum_1_right_j / r_j_right) * 0.5


#     # Apply boundary conditions.
#     i_N = idx[-1]
#     psi -= sum_1[i_N]*np.log(r[i_N]) - sum_2[i_N]

#     # for i in range(1, n_part-1):
#     #     i_left_i = i_left[i]
#     #     i_right_i = i_right[i]
#     #     psi_left = psi[i_left_i]
#     #     r_left = r[i_left_i]
#     #     psi_right = psi[i_right_i]
#     #     r_right = r[i_right_i]
#     #     dr_psi[i] = (psi_right - psi_left) / (r_right - r_left)
#     # print('1')


@njit_serial(fastmath=True)
def calculate_psi_dr_psi_at_particles_bg(
        r, sum_1, sum_2, psi_bg, r_neighbor, idx, psi, dr_psi):
    """
    Calculate the wakefield potential and its derivatives at the position
    of the plasma particles. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    Parameters
    ----------
    r, pr, q : array
        Arrays containing the radial position, momentum and charge of the
        plasma particles.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    r_max : float
        Maximum radial extent of the plasma column.
    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.
    psi_pp, dr_psi_pp, dxi_psi_pp : ndarray
        Arrays where the value of the wakefield potential and its derivatives
        at the location of the plasma particles will be stored.

    """
    # Initialize arrays.
    n_part = r.shape[0]

    # Calculate psi and dr_psi.
    # Their value at the position of each plasma particle is calculated
    # by doing a linear interpolation between two values at the left and
    # right of the particle. The left point is the middle position between the
    # particle and its closest left neighbor, and the same for the right.
    for i_sort in range(n_part):
        i = idx[i_sort]
        im1 = idx[i_sort - 1]
        r_i = r[i]

        r_left = r_neighbor[i_sort]
        psi_bg_left = psi_bg[i_sort]
        # dr_psi_bg_left = dr_psi_bg[i]
        # If this is not the first particle, calculate the left point (r_left)
        # and the field values there (psi_left and dr_psi_left) as usual.
        if i_sort > 0:
            sum_1_left_i = sum_1[im1]
            sum_2_left_i = sum_2[im1]
            psi_left = sum_1_left_i*np.log(r_left) - sum_2_left_i + psi_bg_left
            # dr_psi_left = sum_1_left_i / r_left + dr_psi_bg_left
        # Otherwise, take r=0 as the location of the left point.
        else:
            psi_left = psi_bg_left
            # dr_psi_left = 0.

        # If this is not the last particle, calculate the r_right as
        # middle point.
        # if i_sort < n_part - 1:
        #     ip1 = idx[i_sort + 1]
        # else:
        #     ip1 = -1
        sum_1_right_i = sum_1[i]
        sum_2_right_i = sum_2[i]
        r_right = r_neighbor[i_sort+1]
        psi_bg_right = psi_bg[i_sort+1]
        # dr_psi_bg_right = dr_psi_bg[ip1]

        # Calculate field values ar r_right.
        psi_right = sum_1_right_i*np.log(r_right) - sum_2_right_i + psi_bg_right
        # dr_psi_right = sum_1_right_i / r_right + dr_psi_bg_right

        # Interpolate psi.
        b_1 = (psi_right - psi_left) / (r_right - r_left)
        a_1 = psi_left - b_1*r_left
        psi[i] = a_1 + b_1*r_i

        # Interpolate dr_psi.
        # b_2 = (dr_psi_right - dr_psi_left) / (r_right - r_left)
        # a_2 = dr_psi_left - b_2*r_left
        # dr_psi[i] = a_2 + b_2*r_i
        dr_psi[i] = b_1

        im1 = i


@njit_serial()
def calculate_dxi_psi_at_particles_bg(
        r, sum_3, dxi_psi_bg, r_neighbor, idx, dxi_psi):
    """
    Calculate the wakefield potential and its derivatives at the position
    of the plasma particles. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    Parameters
    ----------
    r, pr, q : array
        Arrays containing the radial position, momentum and charge of the
        plasma particles.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    r_max : float
        Maximum radial extent of the plasma column.
    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.
    psi_pp, dr_psi_pp, dxi_psi_pp : ndarray
        Arrays where the value of the wakefield potential and its derivatives
        at the location of the plasma particles will be stored.

    """
    # Initialize arrays.
    n_part = r.shape[0]

    # Calculate psi and dr_psi.
    # Their value at the position of each plasma particle is calculated
    # by doing a linear interpolation between two values at the left and
    # right of the particle. The left point is the middle position between the
    # particle and its closest left neighbor, and the same for the right.
    for i_sort in range(n_part):
        i = idx[i_sort]
        im1 = idx[i_sort - 1]
        r_i = r[i]

        r_left = r_neighbor[i_sort]
        dxi_psi_bg_left = dxi_psi_bg[i_sort]
        # dr_psi_bg_left = dr_psi_bg[i]
        # If this is not the first particle, calculate the left point (r_left)
        # and the field values there (psi_left and dr_psi_left) as usual.
        if i_sort > 0:
            sum_3_left_i = sum_3[im1]
            dxi_psi_left = - sum_3_left_i + dxi_psi_bg_left
        else:
            dxi_psi_left = dxi_psi_bg_left

        # If this is not the last particle, calculate the r_right as
        # middle point.
        # if i_sort < n_part - 1:
        #     ip1 = idx[i_sort + 1]
        # else:
        #     ip1 = -1
        sum_3_right_i = sum_3[i]
        r_right = r_neighbor[i_sort+1]
        dxi_psi_bg_right = dxi_psi_bg[i_sort+1]
        # dr_psi_bg_right = dr_psi_bg[ip1]

        # Calculate field values ar r_right.
        dxi_psi_right = - sum_3_right_i + dxi_psi_bg_right
        # dr_psi_right = sum_1_right_i / r_right + dr_psi_bg_right

        # Interpolate psi.
        b_1 = (dxi_psi_right - dxi_psi_left) / (r_right - r_left)
        a_1 = dxi_psi_left - b_1*r_left
        dxi_psi[i] = a_1 + b_1*r_i

        im1 = i

    # Boundary condition for psi (Force potential to be zero either at the
    # plasma edge or after the last particle, whichever is further away).
    # i_N = idx[-1]
    # psi -= sum_1[i_N]*np.log(r[i_N]) - sum_2[i_N] + psi_bg[i_N]
    # psi -= psi[i_N]

    # In theory, psi cannot be smaller than -1. However, it has been observed
    # than in very strong blowouts, near the peak, values below -1 can appear
    # in this numerical method. In addition, values very close to -1 will lead
    # to particles with gamma >> 10, which will also lead to problems.
    # This condition here makes sure that this does not happen, improving
    # the stability of the solver.
    # for i in range(n_part):
    #     # Should only happen close to the peak of very strong blowouts.
    #     if psi[i] < -0.90:
    #         psi[i] = -0.90


@njit_serial()
def determine_neighboring_points(r, dr_p, idx, r_neighbor):
    """
    Calculate the wakefield potential and its derivatives at the position
    of the plasma particles. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    Parameters
    ----------
    r, pr, q : array
        Arrays containing the radial position, momentum and charge of the
        plasma particles.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    r_max : float
        Maximum radial extent of the plasma column.
    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.
    psi_pp, dr_psi_pp, dxi_psi_pp : ndarray
        Arrays where the value of the wakefield potential and its derivatives
        at the location of the plasma particles will be stored.

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

        # If this is not the first particle, calculate the left point (r_left)
        # and the field values there (psi_left and dr_psi_left) as usual.
        if i_sort > 0:
            r_left = (r_im1 + r_i) * 0.5
        # Otherwise, take r=0 as the location of the left point.
        else:
            r_left = r_i - dr_p * 0.5

        r_im1 = r_i
        r_neighbor[i_sort] = r_left

        # If this is the last particle, calculate the r_right as
        if i_sort == n_part - 1:
            r_right = r_i + dr_p * 0.5
            r_neighbor[-1] = r_right
    # r_neighbor is sorted, thus, different order than r


# def calculate_dxi_psi_at_particles(r, dr_p, sum_3, i_left, i_right, idx, dxi_psi):
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
#     n_part = r.shape[0]

#     # Calculate fields at r_eval.
#     for j in range(n_part):
#         r_j = r[j]
#         i_left_j = i_left[j]
#         i_right_j = i_right[j]
#         if i_left_j == -1:
#             sum_3_left_j = 0.
#         else:
#             sum_3_left_j = sum_3[i_left_j]
#         sum_3_right_j = sum_3[i_right_j]

#         r_j_left = r_j - dr_p / 8
#         r_j_right = r_j + dr_p / 8

#         psi_left = sum_1_left_j*np.log(r_j_left) - sum_2_left_j
#         psi_right = sum_1_right_j*np.log(r_j_right) - sum_2_right_j
#         psi[j] = (psi_left + psi_right) * 0.5
#         dr_psi[j] = (sum_1_left_j / r_j_left + sum_1_right_j / r_j_right) * 0.5

#     # Apply boundary conditions.
#     i_N = idx[-1]
#     psi -= sum_1[i_N]*np.log(r[i_N]) - sum_2[i_N]


# def calculate_psi_at_particles(r, dr_p, sum_1, sum_2, idx, psi):
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
#     n_part = r.shape[0]

#     # Calculate fields at r_eval.
#     i_last_left = 0
#     for j in range(n_part):
#         r_j = r[idx[j]]

#         r_j_left = r_j - dr_p / 8
#         r_j_right = r_j + dr_p / 8
#         # Get index of last plasma particle with r_i < r_j_left, continuing from
#         # last particle found in previous iteration.
#         for i_sort in range(i_last_left, n_part):
#             i = idx[i_sort]
#             r_i = r[i]
#             i_last_left = i_sort
#             if r_i >= r_j_left:
#                 i_last_left -= 1
#                 break
#         # Get index of last plasma particle with r_i < r_j_right, continuing from
#         # last particle found on the left.
#         for i_sort in range(i_last_left+1, n_part):
#             i = idx[i_sort]
#             r_i = r[i]
#             i_last_right = i_sort
#             if r_i >= r_j_right:
#                 i_last_right -= 1
#                 break
#         # Calculate fields at r_j.
#         if i_last_left == -1:
#             sum_1_left = 0.
#             sum_2_left = 0.
#             i_last_left = 0
#         else:
#             i_left = idx[i_last_left]
#             sum_1_left = sum_1[i_left]
#             sum_2_left = sum_2[i_left]
#         i_right = idx[i_last_right]
#         sum_1_right = sum_1[i_right]
#         sum_2_right = sum_2[i_right]
#         psi_left = sum_1_left*np.log(r_j_left) - sum_2_left
#         psi_right = sum_1_right*np.log(r_j_right) - sum_2_right
#         psi[idx[j]] = (psi_left + psi_right) * 0.5

#     # Apply boundary conditions.
#     i_N = idx[-1]
#     psi -= sum_1[i_N]*np.log(r[i_N]) - sum_2[i_N]


# def determine_left_right_indices(r, dr_p, idx, i_left_arr, i_right_arr):
#     n_part = r.shape[0]

#     # Calculate fields at r_eval.
#     i_last_left = 0
#     for j in range(n_part):
#         i_j = idx[j]
#         r_j = r[i_j]

#         r_j_left = r_j - dr_p / 8
#         r_j_right = r_j + dr_p / 8
#         # Get index of last plasma particle with r_i < r_j_left, continuing from
#         # last particle found in previous iteration.
#         for i_sort in range(i_last_left, n_part):
#             i = idx[i_sort]
#             r_i = r[i]
#             i_last_left = i_sort
#             if r_i >= r_j_left:
#                 i_last_left -= 1
#                 break
#         # Get index of last plasma particle with r_i < r_j_right, continuing from
#         # last particle found on the left.
#         for i_sort in range(i_last_left+1, n_part):
#             i = idx[i_sort]
#             r_i = r[i]
#             i_last_right = i_sort
#             if r_i >= r_j_right:
#                 i_last_right -= 1
#                 break
#         # Calculate fields at r_j.
#         if i_last_left == -1:
#             i_last_left = 0
#             i_left_arr[i_j] = -1
#         else:
#             i_left_arr[i_j] = idx[i_last_left]
#         i_right_arr[i_j] = idx[i_last_right]


@njit_serial()
def calculate_psi(r_eval, r, sum_1, sum_2, idx, psi):
    """
    Calculate the wakefield potential at the radial
    positions specified in r_eval. This is done by using Eq. (29) in
    the paper by P. Baxevanis and G. Stupakov.

    Parameters
    ----------
    r_eval : array
        Array containing the radial positions where psi should be calculated.
    r, q : array
        Arrays containing the radial position, and charge of the
        plasma particles.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    psi : ndarray
        1D Array where the values of the wakefield potential will be stored.

    """
    n_part = r.shape[0]

    # Initialize array for psi at r_eval locations.
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
            sum_1_j = 0.
            sum_2_j = 0.
            i_last = 0
        else:
            i = idx[i_last]
            sum_1_j = sum_1[i]
            sum_2_j = sum_2[i]
        psi[j] += sum_1_j*np.log(r_j) - sum_2_j


@njit_serial()
def calculate_psi_and_dr_psi(r_eval, r, dr_p, idx, sum_1_arr, sum_2_arr, psi, dr_psi):
    # Initialize arrays with values of psi and sums at plasma particles.
    n_part = r.shape[0]

    # Initialize array for psi at r_eval locations.
    n_points = r_eval.shape[0]

    r_max_plasma = r[idx[-1]] + dr_p / 2

    # Calculate fields at r_eval.
    i_last = 0
    for j in range(n_points):
        r_j = r_eval[j]
        if r_j > 0:
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
                psi[j] = sum_1_j*np.log(r_j) - sum_2_j
                dr_psi[j] = sum_1_j / r_j
            else:
                psi_max = sum_1_j*np.log(r_max_plasma) - sum_2_j
                psi[j] = psi_max + sum_1_j * (np.log(r_j)-np.log(r_max_plasma))
                dr_psi[j] = psi_max / r_j


@njit_serial()
def calculate_dxi_psi(r_eval, r, idx, sum_3_arr, dxi_psi):
    # Initialize arrays with values of psi and sums at plasma particles.
    n_part = r.shape[0]

    # Initialize array for psi at r_eval locations.
    n_points = r_eval.shape[0]

    # Calculate fields at r_eval.
    i_last = 0
    for j in range(n_points):
        r_j = r_eval[j]
        if r_j > 0:
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
