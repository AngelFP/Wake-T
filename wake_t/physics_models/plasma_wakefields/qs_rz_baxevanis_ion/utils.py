import math
from wake_t.utilities.numba import njit_serial


@njit_serial(fastmath=True)
def log(input, output):
    """Calculate log of `input` and store it in `output`"""
    for i in range(input.shape[0]):
        output[i] = math.log(input[i])


@njit_serial(error_model="numpy")
def calculate_chi(q, pz, gamma, chi):
    """Calculate the contribution of each particle to `chi`."""
    for i in range(q.shape[0]):
        q_i = q[i]
        pz_i = pz[i]
        inv_gamma_i = 1. / gamma[i]
        chi[i] = q_i / (1. - pz_i * inv_gamma_i) * inv_gamma_i


@njit_serial(error_model="numpy")
def calculate_rho(q, pz, gamma, chi):
    """Calculate the contribution of each particle to `rho`."""
    for i in range(q.shape[0]):
        q_i = q[i]
        pz_i = pz[i]
        inv_gamma_i = 1. / gamma[i]
        chi[i] = q_i / (1. - pz_i * inv_gamma_i)


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
