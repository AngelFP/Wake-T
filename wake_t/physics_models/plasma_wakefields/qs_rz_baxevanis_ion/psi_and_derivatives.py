"""
Contains the method to compute the wakefield potential and its derivatives
according to the paper by P. Baxevanis and G. Stupakov.

"""
from typing import List
import numpy as np

from wake_t.utilities.numba import njit_serial
from .plasma_species import PlasmaSpecies


def calculate_psi_and_derivatives_at_species(species: List[PlasmaSpecies]):
    """Calculate wakefield potential and derivatives at the plasma particles."""
    species = [sp for sp in species if not sp.is_empty]
    psi_max = 0.
    # Calculate cumulative sums 1 and 2 (Eqs. (29) and (31)).
    for s in species:
        if s.can_move or not s.first_iteration_computed:
            calculate_cumulative_sum_1(s.q * s.w, s.i_sort, s._sum_1)
            calculate_cumulative_sum_2(s.r, s.q * s.w, s.i_sort, s._sum_2)
        # Calculate psi after the last plasma plasma particle (assumes
        # that the total electron and ion charge are the same).
        # This will be used to ensure the boundary condition (psi=0) after last
        # plasma particle.
        psi_max -= s._sum_2[s.i_sort[-1]]
    for s in species:
        s._psi_max[:] = psi_max
    
    # Calculate the psi and dr_psi at the neighboring points adding
    # the contribution of each species. Then, use linear interpolation to get
    # the value at the location of each particle.
    for s_gather in species:
        if s_gather.can_move:
            for i, s_deposit in enumerate(species):
                calculate_psi_and_dr_psi(
                    s_gather._r_neighbor, s_gather._log_r_neighbor,
                    s_deposit.r, s_deposit.dr_p, s_deposit.i_sort,
                    s_deposit._sum_1, s_deposit._sum_2,
                    s_gather._psi_bg, s_gather._dr_psi_bg,
                    first=i==0
                )
            interpolate_psi_dr_psi_from_neighbors(
                s_gather.r, s_gather._psi_bg, s_gather._r_neighbor,
                s_gather.i_sort, s_gather._psi, s_gather._dr_psi)            
            # Apply boundary condition.
            s_gather._psi -= psi_max
            # Check that the values of psi are within a reasonable range (prevents
            # issues at the peak of a blowout wake, for example).
            check_psi(s_gather._psi)

    # Calculate cumulative sum 3 (Eq. (32)).
    dxi_psi_max = 0.
    for s in species:
        if s.can_move or not s.first_iteration_computed:
            calculate_cumulative_sum_3(s.r, s.pr, s.q * s.w, s._psi, s.i_sort, s._sum_3)
        # Calculate dxi_psi after the last plasma plasma particle.
        # This will be used to ensure the boundary condition (dxi_psi = 0) after
        # last plasma particle.
        dxi_psi_max += s._sum_3[s.i_sort[-1]]
    
    # Calculate dxi_psi at the neighboring points adding
    # the contribution of each species. Then, use linear interpolation to get
    # the value at the location of each particle.
    for s_gather in species:
        if s_gather.can_move:
            for i, s_deposit in enumerate(species):
                calculate_dxi_psi(
                    s_gather._r_neighbor,
                    s_deposit.r, s_deposit.i_sort, s_deposit._sum_3,
                    s_gather._dxi_psi_bg,
                    first=i==0
                )
            interpolate_dxi_psi_from_neighbors(
                s_gather.r, s_gather._dxi_psi_bg, s_gather._r_neighbor,
                s_gather.i_sort, s_gather._dxi_psi
            )
            # Apply boundary condition
            s_gather._dxi_psi += dxi_psi_max
            # Check that the values of dxi_psi are within a reasonable range (prevents
            # issues at the peak of a blowout wake, for example).
            check_dxi_psi(s_gather._dxi_psi)


def calculate_psi_at_grid(species: List[PlasmaSpecies], r_grid, log_r_grid, psi):
    """Calculate psi at the simulation grid."""
    species = [sp for sp in species if not sp.is_empty]
    for sp in species:
        calculate_psi(
            r_grid, log_r_grid, sp.r, sp._sum_1, sp._sum_2,
            sp.i_sort, psi)
    psi -= sp._psi_max


@njit_serial(fastmath=True)
def calculate_cumulative_sum_1(q, w, w_center, sum_1_arr):
    """Calculate the cumulative sum in Eq. (29)."""
    sum_1 = 0.0
    for i in range(w.shape[0]):
        w_i = w[i]
        w_center_i = w_center[i]
        # Integrate up to particle centers.
        sum_1_arr[i] = sum_1 + q * w_center_i
        # And add all charge for next iteration.
        sum_1 += q * w_i
    # Total sum after last particle.
    sum_1_arr[-1] = sum_1


@njit_serial(fastmath=True)
def calculate_cumulative_sum_2(q, log_r, w, w_center, sum_2_arr):
    """Calculate the cumulative sum in Eq. (31)."""
    sum_2 = 0.0
    for i in range(log_r.shape[0]):
        log_r_i = log_r[i]
        w_i = w[i]
        w_center_i = w_center[i]
        # Integrate up to particle centers.
        sum_2_arr[i] = sum_2 + q * w_center_i * log_r_i
        # And add all charge for next iteration.
        sum_2 += q * w_i * log_r_i
    # Total sum after last particle.
    sum_2_arr[-1] = sum_2


@njit_serial(fastmath=True, error_model="numpy")
def calculate_cumulative_sum_3(q, r, pr, w, w_center, psi, sum_3_arr):
    """Calculate the cumulative sum in Eq. (32)."""
    sum_3 = 0.0
    for i in range(r.shape[0]):
        r_i = r[i]
        pr_i = pr[i]
        w_i = w[i]
        w_center_i = w_center[i]
        psi_i = psi[i]
        # Integrate up to particle centers.
        sum_3_arr[i] = sum_3 + (q * w_center_i * pr_i) / (r_i * (1 + psi_i))
        # And add all charge for next iteration.
        sum_3 += (q * w_i * pr_i) / (r_i * (1 + psi_i))
    # Total sum after last particle.
    sum_3_arr[-1] = sum_3


@njit_serial(fastmath=True, error_model="numpy")
def interpolate_psi_dr_psi_from_neighbors(
        r, psi_bg, r_neighbor, idx, psi, dr_psi):
    """
    Calculate psi and dr_psi at the particles using linear interpolation
    between the left and right neighbors.
    """
    # Initialize arrays.
    n_part = r.shape[0]

    # Get initial values for left neighbors.
    r_left = r_neighbor[0]
    psi_left = psi_bg[0]

    # Loop over particles.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]

        # Get values at right neighbor.
        r_right = r_neighbor[i_sort + 1]
        psi_right = psi_bg[i_sort + 1]

        # Interpolate psi between left and right neighbors.
        b_1 = (psi_right - psi_left) / (r_right - r_left)
        a_1 = psi_left - b_1*r_left
        psi[i] = a_1 + b_1*r_i

        # dr_psi is simply the slope used for interpolation.
        dr_psi[i] = b_1

        # Update values of next left neighbor with those of the current right
        # neighbor.
        r_left = r_right
        psi_left = psi_right


@njit_serial(fastmath=True, error_model="numpy")
def interpolate_dxi_psi_from_neighbors(
        r, dxi_psi_bg, r_neighbor, idx, dxi_psi):
    """
    Calculate dxi_psi at the particles using linear interpolation
    between the left and right neighbors.

    """
    # Initialize arrays.
    n_part = r.shape[0]

    # Get initial values for left neighbors.
    r_left = r_neighbor[0]
    dxi_psi_left = dxi_psi_bg[0]

    # Loop over particles.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]

        # Calculate value at right neighbor.
        r_right = r_neighbor[i_sort + 1]
        dxi_psi_right = dxi_psi_bg[i_sort + 1]

        # Interpolate value between left and right neighbors.
        b_1 = (dxi_psi_right - dxi_psi_left) / (r_right - r_left)
        a_1 = dxi_psi_left - b_1*r_left
        dxi_psi[i] = a_1 + b_1*r_i

        # Update values of next left neighbor with those of the current right
        # neighbor.
        r_left = r_right
        dxi_psi_left = dxi_psi_right


@njit_serial()
def calculate_psi(r_eval, log_r_eval, r, sum_1, sum_2, idx, psi):
    """Calculate psi at the radial positions given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # Calculate psi after the last plasma plasma particle
    # This is used to ensure the boundary condition psi=0, which also
    # assumes that the total electron and ion charge are the same.
    sum_2_max = sum_2_arr[-1]

    # Calculate fields at r_eval.
    i_last = 0
    r_left = 0.0
    sum_1_left = 0.0
    sum_2_left = 0.0
    psi_left = 0.0
    for j in range(n_points):
        r_j = r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        while i_last < n_part:
            r_right = r[i_last]
            if r_right >= r_j:
                break
            r_left = r_right
            i_last += 1
        if i_last < n_part:
            if i_last > 0:
                log_r_left = log_r[i_last - 1]
                sum_1_left = sum_1_arr[i_last - 1]
                sum_2_left = sum_2_arr[i_last - 1]
                psi_left = sum_1_left * log_r_left - sum_2_left
            log_r_right = log_r[i_last]
            sum_1_right = sum_1_arr[i_last]
            sum_2_right = sum_2_arr[i_last]
            psi_right = sum_1_right * log_r_right - sum_2_right

            # Interpolate sums.
            inv_dr = 1.0 / (r_right - r_left)
            slope_2 = (psi_right - psi_left) * inv_dr
            psi_j = psi_left + slope_2 * (r_j - r_left) + sum_2_max
        else:
            sum_1_left = sum_1_arr[-1]
            sum_2_left = sum_2_arr[-1]
            psi_j = sum_1_left * np.log(r_j) - sum_2_left + sum_2_max

        # Calculate fields at r_j.
        if add:
            psi[j] += psi_j
        else:
            psi[j] = psi_j


@njit_serial(fastmath=True, error_model="numpy")
def calculate_psi_and_dr_psi(
        r_eval, log_r_eval, r, dr_p, idx, sum_1_arr, sum_2_arr, psi, dr_psi, first=True):
    """Calculate psi and dr_psi at the radial positions given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # Calculate psi after the last plasma plasma particle
    # This is used to ensure the boundary condition psi=0, which also
    # assumes that the total electron and ion charge are the same.
    sum_2_max = sum_2_arr[-1]

    # Calculate fields at r_eval.
    i_last = 0
    r_left = 0.0
    sum_1_left = 0.0
    sum_2_left = 0.0
    psi_left = 0.0
    dr_psi_left = 0.0
    for j in range(n_points):
        r_j = r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        while i_last < n_part:
            r_right = r[i_last]
            if r_right >= r_j:
                break
            r_left = r_right
            i_last += 1
        if i_last < n_part:
            if i_last > 0:
                log_r_left = log_r[i_last - 1]
                sum_1_left = sum_1_arr[i_last - 1]
                sum_2_left = sum_2_arr[i_last - 1]
                dr_psi_left = sum_1_left / r_left
                psi_left = sum_1_left * log_r_left - sum_2_left
            log_r_right = log_r[i_last]
            sum_1_right = sum_1_arr[i_last]
            sum_2_right = sum_2_arr[i_last]
            dr_psi_right = sum_1_right / r_right
            psi_right = sum_1_right * log_r_right - sum_2_right

            # Interpolate sums.
            inv_dr = 1.0 / (r_right - r_left)
            slope_1 = (dr_psi_right - dr_psi_left) * inv_dr
            slope_2 = (psi_right - psi_left) * inv_dr
            dr_psi_j = dr_psi_left + slope_1 * (r_j - r_left)
            psi_j = psi_left + slope_2 * (r_j - r_left) + sum_2_max
        else:
            sum_1_left = sum_1_arr[-1]
            sum_2_left = sum_2_arr[-1]
            dr_psi_j = sum_1_left / r_j
            psi_j = sum_1_left * np.log(r_j) - sum_2_left + sum_2_max

        # Calculate fields at r_j.
        if first:            
            psi[j] = sum_1_j*log_r_j - sum_2_j
            dr_psi[j] = sum_1_j / r_j
        else:
            psi[j] += sum_1_j*log_r_j - sum_2_j
            dr_psi[j] += sum_1_j / r_j


@njit_serial()
def calculate_dxi_psi(r_eval, r, idx, sum_3_arr, dxi_psi, first=True):
    """Calculate dxi_psi at the radial position given in `r_eval`."""
    # Get number of plasma particles.
    n_part = r.shape[0]

    # Get number of points to evaluate.
    n_points = r_eval.shape[0]

    # Calculate dxi_psi after the last plasma plasma particle
    # This is used to ensure the boundary condition dxi_psi=0, which also
    # assumes that the total electron and ion charge are the same.
    sum_3_max = sum_3_arr[-1]

    # Calculate fields at r_eval.
    i_last = 0
    r_left = 0.0
    dxi_psi_left = 0.0
    for j in range(n_points):
        r_j = r_eval[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        while i_last < n_part:
            r_i = r[i_last]
            if r_i >= r_j:
                break
            i_last += 1
        if i_last > 0:
            i = idx[i_last - 1]
            sum_3_j = sum_3_arr[i]
        if first:
            dxi_psi[j] = - sum_3_j
        else:
            dxi_psi[j] += - sum_3_j


@njit_serial()
def check_psi(psi):
    """Check that the values of psi are within a reasonable range

    This is used to prevent issues at the peak of a blowout wake, for example).
    """
    for i in range(psi.shape[0]):
        psi_i = psi[i]
        if psi_i < -0.99:
            psi[i] = -0.99
        elif psi_i > 0.99:
            psi[i] = 0.99


@njit_serial()
def check_psi_derivative(dxi_psi):
    """Check that the values of dxi_psi are within a reasonable range

    This is used to prevent issues at the peak of a blowout wake, for example).
    """
    for i in range(dxi_psi.shape[0]):
        dxi_psi_i = dxi_psi[i]
        if dxi_psi_i < -3:
            dxi_psi[i] = -3
        elif dxi_psi_i > 3:
            dxi_psi[i] = 3
