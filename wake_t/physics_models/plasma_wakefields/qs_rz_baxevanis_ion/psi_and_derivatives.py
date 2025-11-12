"""
Contains the method to compute the wakefield potential and its derivatives
according to the paper by P. Baxevanis and G. Stupakov.

"""

from typing import List
import numpy as np

from wake_t.utilities.numba import njit_serial
from .plasma_particles import PlasmaParticles
from .b_theta import (
    calculate_b_theta_with_interpolation,
)


def calculate_psi_and_derivatives_at_species(
    species: List[PlasmaParticles],
):
    """Calculate wakefield potential and derivatives at the plasma particles.

    Parameters
    ----------
    r_e, log_r_e, pr_e, w_e, w_center_e : ndarray
        Radial position (and log), momentum, weight (and central weight)
        of the plasma electrons.
    q_e : float
        Charge of the plasma electron species.
    r_i, log_r_i, pr_i, w_i, w_center_i, dr_p_i : ndarray
        Radial position (and log), momentum, weight (and central weight)
        of the plasma ions.
    q_i : float
        Charge of the plasma ion species.
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
    psi_e, dr_psi_e, dxi_psi_e, psi_i, dr_psi_i, dxi_psi_i : ndarray
        Arrays where the value of psi and its derivatives at the plasma
        electrons and ions will be stored.
    psi, dr_psi, dxi_psi : _type_
        Arrays where the value of psi and its derivatives at all
        plasma particles is stored.
    """

    species = [sp for sp in species if not sp.is_empty]
    psi_max = 0.0
    # Calculate cumulative sums 1 and 2 (Eqs. (29) and (31)).
    for s in species:
        if s.ion_motion or not s.ions_computed:
            calculate_cumulative_sum_1(s.q, s.w, s.w_center, s._sum_1)
            calculate_cumulative_sum_2(s.q, s._log_r, s.w, s.w_center, s._sum_2)
        # Calculate the psi and dr_psi background at the neighboring points.
        # For the electrons, compute the psi and dr_psi due to the ions at
        # r_neighbor_e. For the ions, compute the psi and dr_psi due to the
        # electrons at r_neighbor_i.
        psi_max -= s._sum_2[s.i_sort[-1]]
    for s in species:
        s._psi_max[:] = psi_max

    for sp in species:
        if sp.ion_motion:
            for i in range(len(species)):
                calculate_psi_and_dr_psi_with_interpolation(
                    sp.r,
                    species[1 - i].r,
                    species[1 - i]._log_r,
                    species[1 - i]._sum_1,
                    species[1 - i]._sum_2,
                    sp._psi,
                    sp._dr_psi,
                    add=i == 0,
                )
            calculate_psi_and_dr_psi_at_particle_centers(
                sp.r, sp._log_r, sp._sum_1, sp._sum_2, sp._psi, sp._dr_psi
            )
            # Apply boundary condition.
            sp._psi -= psi_max
            # Check that the values of psi are within a reasonable range (prevents
            # issues at the peak of a blowout wake, for example).
            check_psi(sp._psi)

    dxi_psi_max = 0.0
    for s in species:
        if s.ion_motion or not s.ions_computed:
            calculate_cumulative_sum_3(
                s.q, s.r, s.pr, s.w, s.w_center, s._psi, s._sum_3
            )
        # Calculate dxi_psi after the last plasma plasma particle.
        # This will be used to ensure the boundary condition (dxi_psi = 0) after
        # last plasma particle.
        dxi_psi_max += s._sum_3[s.i_sort[-1]]

    # Calculate the dxi_psi background at the neighboring points.
    # For the electrons, compute the psi and dr_psi due to the ions at
    # r_neighbor_e. For the ions, compute the psi and dr_psi due to the
    # electrons at r_neighbor_i.
    for sp in species:
        if sp.ion_motion:
            for i in range(len(species)):
                calculate_dxi_psi_with_interpolation(
                    sp.r,
                    species[1 - i].r,
                    species[1 - i]._sum_3,
                    sp._dxi_psi,
                    add=i == 0,
                )
            calculate_dxi_psi_at_particle_centers(sp.r, sp._sum_3, sp._dxi_psi)
            # Apply boundary condition
            sp._dxi_psi += dxi_psi_max
            # Check that the values of dxi_psi are within a reasonable range (prevents
            # issues at the peak of a blowout wake, for example).
            check_psi_derivative(sp._dxi_psi)


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
def calculate_psi_with_interpolation(
    r_eval, r, log_r, sum_1_arr, sum_2_arr, psi, add=False
):
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
def calculate_psi_and_dr_psi_with_interpolation(
    r_eval, r, log_r, sum_1_arr, sum_2_arr, psi, dr_psi, add=False
):
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
        if add:
            psi[j] += psi_j
            dr_psi[j] += dr_psi_j
        else:
            psi[j] = psi_j
            dr_psi[j] = dr_psi_j


@njit_serial(fastmath=True, error_model="numpy")
def calculate_psi_and_dr_psi_at_particle_centers(
    r,
    log_r,
    sum_1_arr,
    sum_2_arr,
    psi,
    dr_psi,
):
    # Get number of particles.
    n_part = r.shape[0]

    # Calculate psi after the last plasma plasma particle
    # This is used to ensure the boundary condition psi=0, which also
    # assumes that the total electron and ion charge are the same.
    sum_2_max = sum_2_arr[-1]

    # Calculate fields.
    for i in range(n_part):
        r_i = r[i]
        log_r_i = log_r[i]
        sum_1_i = sum_1_arr[i]
        sum_2_i = sum_2_arr[i]
        dr_psi[i] = sum_1_i / r_i
        psi[i] = sum_1_i * log_r_i - sum_2_i + sum_2_max


@njit_serial()
def calculate_dxi_psi_with_interpolation(r_eval, r, sum_3_arr, dxi_psi, add=False):
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
        if i_last < n_part:
            if i_last > 0:
                r_left = r[i_last - 1]
                dxi_psi_left = -sum_3_arr[i_last - 1]
            r_right = r[i_last]
            dxi_psi_right = -sum_3_arr[i_last]
            slope = (dxi_psi_right - dxi_psi_left) / (r_right - r_left)
            dxi_psi_j = dxi_psi_left + slope * (r_j - r_left) + sum_3_max
        else:
            dxi_psi_j = -sum_3_arr[-1] + sum_3_max
        if add:
            dxi_psi[j] += dxi_psi_j
        else:
            dxi_psi[j] = dxi_psi_j


@njit_serial(fastmath=True, error_model="numpy")
def calculate_dxi_psi_at_particle_centers(
    r,
    sum_3_arr,
    dxi_psi,
):
    # Get number of particles.
    n_part = r.shape[0]

    # Calculate dxi_psi after the last plasma plasma particle
    # This is used to ensure the boundary condition dxi_psi=0, which also
    # assumes that the total electron and ion charge are the same.
    sum_3_max = sum_3_arr[-1]

    # Calculate fields.
    for i in range(n_part):
        dxi_psi[i] = -sum_3_arr[i] + sum_3_max


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


def calculate_psi_at_grid(species: List[PlasmaParticles], r_eval, psi):
    """Calculate psi on the current grid slice."""
    species = [sp for sp in species if not sp.is_empty]
    for sp in species:
        calculate_psi_with_interpolation(
            r_eval,
            sp.r,
            sp._log_r,
            sp._sum_1,
            sp._sum_2,
            psi,
        )
    psi -= sp._psi_max


def calculate_b_theta_at_grid(species: List[PlasmaParticles], r_eval, b_theta):
    """Calculate b_theta on the current grid slice."""
    species = [sp for sp in species]
    for sp in species:
        s_e = species[0]
        calculate_b_theta_with_interpolation(
            r_eval, sp._a_0[0], sp._a_i, sp._b_i, s_e.r, b_theta
        )
