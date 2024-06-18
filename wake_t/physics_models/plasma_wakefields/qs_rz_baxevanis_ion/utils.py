"""Contains various utility functions for the gridless solver."""

import math

import numpy as np

from wake_t.utilities.numba import njit_serial


@njit_serial(fastmath=True)
def log(input, output):
    """Calculate log of `input` and store it in `output`"""
    for i in range(input.shape[0]):
        output[i] = math.log(input[i])


@njit_serial(error_model="numpy")
def calculate_chi(q, w, pz, gamma, chi):
    """Calculate the contribution of each particle to `chi`."""
    for i in range(w.shape[0]):
        w_i = w[i]
        pz_i = pz[i]
        inv_gamma_i = 1.0 / gamma[i]
        chi[i] = q * w_i / (1.0 - pz_i * inv_gamma_i) * inv_gamma_i


@njit_serial(error_model="numpy")
def calculate_rho(q, w, pz, gamma, rho):
    """Calculate the contribution of each particle to `rho`."""
    for i in range(w.shape[0]):
        w_i = w[i]
        pz_i = pz[i]
        inv_gamma_i = 1.0 / gamma[i]
        rho[i] = q * w_i / (1.0 - pz_i * inv_gamma_i)


@njit_serial()
def longitudinal_gradient(f, dz, dz_f):
    """Calculate the longitudinal gradient of a 2D array.

    This method is equivalent to using `np.gradient` with
    `edge_order=2`, but is several times faster as it is compiled with numba
    and is more specialized.

    Parameters
    ----------
    f : ndarray
        The array from which to calculate the gradient.
    dz : float
        Longitudinal step size.
    dz_f : ndarray
        Array where the longitudinal gradient will be stored.
    """
    nz, nr = f.shape
    inv_dz = 1.0 / dz
    inv_h = 0.5 * inv_dz
    a = -1.5 * inv_dz
    b = 2.0 * inv_dz
    c = -0.5 * inv_dz
    for j in range(nr):
        f_right = f[nz - 1, j]
        for i in range(1, nz - 1):
            f_left = f[i - 1, j]
            f_right = f[i + 1, j]
            dz_f[i, j] = (f_right - f_left) * inv_h
        dz_f[0, j] = a * f[0, j] + b * f[1, j] + c * f[2, j]
        dz_f[-1, j] = -a * f[-1, j] - b * f[-2, j] - c * f[-3, j]


@njit_serial()
def radial_gradient(f, dr, dr_f):
    """Calculate the radial gradient of a 2D array.

    This method is equivalent to using `np.gradient` with
    `edge_order=2`, but is several times faster as it is compiled with numba
    and is more specialized. It takes advantage of the axial symmetry to
    calculate the derivative on axis.

    Parameters
    ----------
    f : ndarray
        The array from which to calculate the gradient.
    dr : float
        Radial step size.
    dr_f : ndarray
        Array where the radial gradient will be stored.
    """
    nz, nr = f.shape
    inv_dr = 1.0 / dr
    inv_h = 0.5 * inv_dr
    a = -1.5 * inv_dr
    b = 2.0 * inv_dr
    c = -0.5 * inv_dr
    for i in range(nz):
        for j in range(1, nr - 1):
            f_left = f[i, j - 1]
            f_right = f[i, j + 1]
            dr_f[i, j] = (f_right - f_left) * inv_h
        dr_f[i, 0] = (f[i, 1] - f[i, 0]) * inv_h
        dr_f[i, -1] = -a * f[i, -1] - b * f[i, -2] - c * f[i, -3]


@njit_serial()
def calculate_laser_a2(a_complex, a2):
    """Calculate the square of the laser complex envelope amplitude.

    Parameters
    ----------
    a_complex : ndarray
        Array of size (nz, nr) containing the complex envelope of the laser.
    a2 : ndarray
        Array of size (nz+4, nr+4) where the result will be stored.
    """
    nz, nr = a_complex.shape
    a_real = a_complex.real
    a_imag = a_complex.imag
    for i in range(nz):
        for j in range(nr):
            ar = a_real[i, j]
            ai = a_imag[i, j]
            a2[2 + i, 2 + j] = ar * ar + ai * ai


@njit_serial(error_model="numpy")
def update_gamma_and_pz(gamma, pz, pr, a2, psi, q, m):
    """
    Update the gamma factor and longitudinal momentum of the plasma particles.

    Parameters
    ----------
    gamma, pz : ndarray
        Arrays containing the current gamma factor and longitudinal momentum
        of the plasma particles (will be modified here).
    pr, a2, psi : ndarray
        Arrays containing the radial momentum of the particles and the
        value of a2 and psi at the position of the particles.
    q, m : float
        Charge and mass of the plasma species.

    """
    q_over_m = q / m
    for i in range(pr.shape[0]):
        psi_i = psi[i] * q_over_m
        pz_i = (1 + pr[i] ** 2 + q_over_m**2 * a2[i] - (1 + psi_i) ** 2) / (
            2 * (1 + psi_i)
        )
        pz[i] = pz_i
        gamma[i] = 1.0 + pz_i + psi_i


@njit_serial()
def check_gamma(gamma, pz, pr, max_gamma):
    """Check that the gamma of particles does not exceed `max_gamma`"""
    for i in range(gamma.shape[0]):
        if gamma[i] > max_gamma:
            gamma[i] = 1.0
            pz[i] = 0.0
            pr[i] = 0.0


@njit_serial()
def sort_particle_arrays(
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, indices
):
    """Sort all the particle arrays with a given sorting order.

    Implementing it like this looks very ugly, but it is much faster than
    repeating `array = array[indices]` for each array. It is also much faster
    than implementing a `sort_array` function that is called on each array.
    This is probably because of the overhead from calling numba functions.
    """
    a1_orig = np.copy(a1)
    a2_orig = np.copy(a2)
    a3_orig = np.copy(a3)
    a4_orig = np.copy(a4)
    a5_orig = np.copy(a5)
    a6_orig = np.copy(a6)
    a7_orig = np.copy(a7)
    a8_orig = np.copy(a8)
    a9_orig = np.copy(a9)
    a10_orig = np.copy(a10)
    a11_orig = np.copy(a11)
    n_part = indices.shape[0]
    for i in range(n_part):
        i_sort = indices[i]
        if i != i_sort:
            a1[i] = a1_orig[i_sort]
            a2[i] = a2_orig[i_sort]
            a3[i] = a3_orig[i_sort]
            a4[i] = a4_orig[i_sort]
            a5[i] = a5_orig[i_sort]
            a6[i] = a6_orig[i_sort]
            a7[i] = a7_orig[i_sort]
            a8[i] = a8_orig[i_sort]
            a9[i] = a9_orig[i_sort]
            a10[:, i] = a10_orig[:, i_sort]
            a11[:, i] = a11_orig[:, i_sort]
