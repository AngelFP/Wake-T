"""
Contains the method to compute the wakefield potential and its derivatives
according to the paper by P. Baxevanis and G. Stupakov.

"""

import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial()
def calculate_psi_and_derivatives_at_particles(
        r, pr, q, idx, r_max, dr_p, pc, psi_pp, dr_psi_pp, dxi_psi_pp):
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

    # Initialize value of sums.
    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate psi and dr_psi.
    # Their value at the position of each plasma particle is calculated
    # by doing a linear interpolation between two values at the left and
    # right of the particle. The left point is the middle position between the
    # particle and its closest left neighbor, and the same for the right.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]

        # Calculate new sums.
        sum_1_new = sum_1 + q_i
        sum_2_new = sum_2 + q_i * np.log(r_i)

        # If this is not the first particle, calculate the left point (r_left)
        # and the field values there (psi_left and dr_psi_left) as usual.
        if i_sort > 0:
            r_im1 = r[idx[i_sort-1]]
            r_left = (r_im1 + r_i) / 2
            psi_left = delta_psi_eq(r_left, sum_1, sum_2, r_max, pc)
            dr_psi_left = dr_psi_eq(r_left, sum_1, r_max, pc)
        # Otherwise, take r=0 as the location of the left point.
        else:
            r_left = 0.
            psi_left = 0.
            dr_psi_left = 0.

        # If this is not the last particle, calculate the r_right as
        # middle point.
        if i_sort < n_part - 1:
            r_ip1 = r[idx[i_sort+1]]
            r_right = (r_i + r_ip1) / 2
        # Otherwise, since the particle represents a charge sheet of width
        # dr_p, take the right point as r_i + dr_p/2.
        else:
            r_right = r_i + dr_p/2
        # Calculate field values ar r_right.
        psi_right = delta_psi_eq(r_right, sum_1_new, sum_2_new, r_max, pc)
        dr_psi_right = dr_psi_eq(r_right, sum_1_new, r_max, pc)

        # Interpolate psi.
        b_1 = (psi_right - psi_left) / (r_right - r_left)
        a_1 = psi_left - b_1*r_left
        psi_pp[i] = a_1 + b_1*r_i

        # Interpolate dr_psi.
        b_2 = (dr_psi_right - dr_psi_left) / (r_right - r_left)
        a_2 = dr_psi_left - b_2*r_left
        dr_psi_pp[i] = a_2 + b_2*r_i

        # Update value of sums.
        sum_1 = sum_1_new
        sum_2 = sum_2_new

    # Boundary condition for psi (Force potential to be zero either at the
    # plasma edge or after the last particle, whichever is further away).
    r_furthest = max(r_right, r_max)
    psi_pp -= delta_psi_eq(r_furthest, sum_1, sum_2, r_max, pc)

    # In theory, psi cannot be smaller than -1. However, it has been observed
    # than in very strong blowouts, near the peak, values below -1 can appear
    # in this numerical method. In addition, values very close to -1 will lead
    # to particles with gamma >> 10, which will also lead to problems.
    # This condition here makes sure that this does not happen, improving
    # the stability of the solver.
    for i in range(n_part):
        # Should only happen close to the peak of very strong blowouts.
        if psi_pp[i] < -0.90:
            psi_pp[i] = -0.90

    # Calculate dxi_psi (also by interpolation).
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi_pp[i]

        sum_3_new = sum_3 + (q_i * pr_i) / (r_i * (1 + psi_i))

        # Check if it is the first particle.
        if i_sort > 0:
            r_im1 = r[idx[i_sort-1]]
            r_left = (r_im1 + r_i) / 2
            dxi_psi_left = -sum_3
        else:
            r_left = 0.
            dxi_psi_left = 0.

        # Check if it is the last particle.
        if i_sort < n_part - 1:
            r_ip1 = r[idx[i_sort+1]]
            r_right = (r_i + r_ip1) / 2
        else:
            r_right = r_i + dr_p/2
        dxi_psi_right = -sum_3_new

        # Do interpolation.
        b = (dxi_psi_right - dxi_psi_left) / (r_right - r_left)
        a = dxi_psi_left - b*r_left
        dxi_psi_pp[i] = a + b*r_i
        sum_3 = sum_3_new

    # Apply longitudinal derivative of the boundary conditions of psi.
    if r_right <= r_max:
        dxi_psi_pp += sum_3
    else:
        dxi_psi_pp += sum_3 - ((sum_1 - r_max**2/2 - pc*r_max/4)
                               * pr_i / (r_right * (1 + psi_i)))

    # Again, near the peak of a strong blowout, very large and unphysical
    # values could appear. This condition makes sure a threshold us not
    # exceeded.
    for i in range(n_part):
        if dxi_psi_pp[i] > 3.:
            dxi_psi_pp[i] = 3.
        if dxi_psi_pp[i] < -3.:
            dxi_psi_pp[i] = -3.


@njit_serial()
def calculate_psi(r_fld, r, q, idx, r_max, pc, psi, k):
    """
    Calculate the wakefield potential at the radial
    positions specified in r_fld. This is done by using Eq. (29) in
    the paper by P. Baxevanis and G. Stupakov.

    Parameters
    ----------
    r_fld : array
        Array containing the radial positions where psi should be calculated.
    r, q : array
        Arrays containing the radial position, and charge of the
        plasma particles.
    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.
    r_max : float
        Maximum radial extent of the plasma column.
    pc : float
        The parabolic density profile coefficient.
    psi : ndarray
        Array where the values of the wakefield potential will be stored.
    k : int
        Index that determines the slice of psi where the values will
        be filled in (the index is k+2 due to the guard cells in the array).

    """
    # Initialize arrays with values of psi and sums at plasma particles.
    n_part = r.shape[0]
    sum_1_arr = np.zeros(n_part)
    sum_2_arr = np.zeros(n_part)
    sum_1 = 0.
    sum_2 = 0.

    # Calculate sum_1, sum_2 and psi_part.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]

        sum_1 += q_i
        sum_2 += q_i * np.log(r_i)
        sum_1_arr[i] = sum_1
        sum_2_arr[i] = sum_2
    r_N = r_i

    # Initialize array for psi at r_fld locations.
    n_points = r_fld.shape[0]
    psi_slice = psi[k+2]

    # Calculate fields at r_fld.
    i_last = 0
    for j in range(n_points):
        r_j = r_fld[j]
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
        psi_slice[2+j] = delta_psi_eq(r_j, sum_1_j, sum_2_j, r_max, pc)

    # Apply boundary conditions.
    r_furthest = max(r_N, r_max)
    psi_slice -= delta_psi_eq(r_furthest, sum_1, sum_2, r_max, pc)


@njit_serial()
def calculate_psi_and_derivatives(r_fld, r, pr, q):
    """
    Calculate the wakefield potential and its derivatives at the radial
    positions specified in r_fld. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    Parameters
    ----------
    r_fld : array
        Array containing the radial positions where psi should be calculated.
    r, pr, q : array
        Arrays containing the radial position, momentum and charge of the
        plasma particles.

    """
    # Initialize arrays with values of psi and sums at plasma particles.
    n_part = r.shape[0]
    psi_part = np.zeros(n_part)
    sum_1_arr = np.zeros(n_part)
    sum_2_arr = np.zeros(n_part)
    sum_3_arr = np.zeros(n_part)
    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate sum_1, sum_2 and psi_part.
    idx = np.argsort(r)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]

        sum_1 += q_i
        sum_2 += q_i * np.log(r_i)
        sum_1_arr[i] = sum_1
        sum_2_arr[i] = sum_2
        psi_part[i] = sum_1 * np.log(r_i) - sum_2 - 0.25 * r_i ** 2
    r_N = r[-1]
    psi_part += - (sum_1 * np.log(r_N) - sum_2 - 0.25 * r_N ** 2)

    # Calculate sum_3.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi_part[i]

        sum_3 += (q_i * pr_i) / (r_i * (1 + psi_i))
        sum_3_arr[i] = sum_3

    # Initialize arrays for psi and derivatives at r_fld locations.
    n_points = r_fld.shape[0]
    psi = np.zeros(n_points)
    dr_psi = np.zeros(n_points)
    dxi_psi = np.zeros(n_points)

    # Calculate fields at r_fld.
    i_last = 0
    for j in range(n_points):
        r_j = r_fld[j]
        # Get index of last plasma particle with r_i < r_j.
        for i_sort in range(n_part):
            i = idx[i_sort]
            r_i = r[i]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields at r_j.
        if i_last == -1:
            psi[j] = -0.25 * r_j ** 2
            dr_psi[j] = -0.5 * r_j
            dxi_psi[j] = 0.
        else:
            i_p = idx[i_last]
            psi[j] = sum_1_arr[i_p] * np.log(r_j) - sum_2_arr[
                i_p] - 0.25 * r_j ** 2
            dr_psi[j] = sum_1_arr[i_p] / r_j - 0.5 * r_j
            dxi_psi[j] = - sum_3_arr[i_p]
    psi = psi - (sum_1 * np.log(r_N) - sum_2 - 0.25 * r_N ** 2)
    dxi_psi = dxi_psi + sum_3
    return psi, dr_psi, dxi_psi


@njit_serial()
def delta_psi_eq(r, sum_1, sum_2, r_max, pc):
    """ Adapted equation (29) from original paper. """
    delta_psi_elec = sum_1*np.log(r) - sum_2
    if r <= r_max:
        delta_psi_ion = 0.25*r**2 + pc*r**4/16
    else:
        delta_psi_ion = (
            0.25*r_max**2 + pc*r_max**4/16 +
            (0.5 * r_max**2 + 0.25*pc*r_max**4) * (
                np.log(r)-np.log(r_max)))
    return delta_psi_elec - delta_psi_ion


@njit_serial()
def dr_psi_eq(r, sum_1, r_max, pc):
    """ Adapted equation (31) from original paper. """
    dr_psi_elec = sum_1 / r
    if r <= r_max:
        dr_psi_ion = 0.5 * r + 0.25 * pc * r ** 3
    else:
        dr_psi_ion = (0.5 * r_max**2 + 0.25 * pc * r_max**4) / r
    return dr_psi_elec - dr_psi_ion
