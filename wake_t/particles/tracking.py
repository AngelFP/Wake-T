""" This module contains the numerical trackers and equations of motion """

from numba import njit, prange
import numpy as np
import scipy.constants as ct

from wake_t.physics_models.beam_optics.transfer_matrices import (
    first_order_matrix, second_order_matrix
)


def runge_kutta_4(beam_matrix, WF, t0, dt, iterations, z_injection=None):
    for i in np.arange(iterations):
        t = t0 + i * dt
        A = equations_of_motion(beam_matrix, t, WF, dt,
                                z_injection)
        B = equations_of_motion(beam_matrix + A / 2., t + dt / 2., WF, dt,
                                z_injection)
        C = equations_of_motion(beam_matrix + B / 2., t + dt / 2., WF, dt,
                                z_injection)
        D = equations_of_motion(beam_matrix + C, t + dt, WF, dt,
                                z_injection)
        update_beam_matrix(beam_matrix, A, B, C, D)
    return beam_matrix


def equations_of_motion(beam_matrix, t, WF, dt, z_injection=None):
    K = -ct.e / (ct.m_e * ct.c)
    x, px, y, py, xi, pz, q = beam_matrix
    if z_injection is not None:
        z = xi + ct.c * t
        if max(z) <= z_injection:
            K = 0.
    wx = K * WF.Wx(x, y, xi, px, py, pz, q, t)
    wy = K * WF.Wy(x, y, xi, px, py, pz, q, t)
    wz = K * WF.Wz(x, y, xi, px, py, pz, q, t)
    return calculate_derivatives(px, py, pz, wx, wy, wz, dt)


@njit()
def update_beam_matrix(bm, A, B, C, D):
    inv_6 = 1 / 6.
    for i in prange(bm.shape[0]):
        for j in prange(bm.shape[1]):
            bm[i, j] += (A[i, j] + 2. * (B[i, j] + C[i, j]) + D[i, j]) * inv_6


@njit()
def calculate_derivatives(px, py, pz, wx, wy, wz, dt):
    n_part = px.shape[0]
    der = np.empty((7, n_part))
    for i in prange(n_part):
        px_i = px[i]
        py_i = py[i]
        pz_i = pz[i]
        inv_gamma_i = 1 / np.sqrt(1 + px_i*px_i + py_i*py_i + pz_i*pz_i)
        der[0, i] = dt * px_i * ct.c * inv_gamma_i
        der[1, i] = dt * wx[i]
        der[2, i] = dt * py_i * ct.c * inv_gamma_i
        der[3, i] = dt * wy[i]
        der[4, i] = dt * (pz_i*inv_gamma_i - 1) * ct.c
        der[5, i] = dt * wz[i]
        der[6, i] = 0.
    return der


def track_with_transfer_map(beam_matrix, z, L, theta, k1, k2, gamma_ref,
                            order=2):
    """
    Track beam distribution throwgh beamline element by using a transfer map.
    This function was initially based on code from Ocelot (see
    https://github.com/ocelot-collab/ocelot) written by S. Tomin.

    Parameters:
    -----------
    beam_matrix : array
        6 x N matrix, where N is the number of particles, containing the
        phase-space information of the bunch as (x, x', y, y', xi, dp) in
        units of (m, rad, m, rad, m, -). dp is defined as
        dp = (g-g_ref)/g_ref, while x' = px/p_kin and y' = py/p_kin, where
        p_kin is the kinetic momentum of each particle.

    z : float
        Longitudinal position in which to obtain the bunch distribution

    L : float
        Total length of the beamline element

    theta : float
        Bending angle of the beamline element

    k1 : float
        Quadrupole gradient of the beamline element in units of 1/m^2. A
        positive value implies focusing on the 'x' plane, while a negative
        gradient corresponds to focusing on 'y'

    k2 : float
        Sextupole gradient of the beamline element in units of 1/m^3. A
        positive value implies focusing on the 'x' plane, while a negative
        gradient corresponds to focusing on 'y'

    gamma_ref : float
        Reference energy with respect to which the particle momentum dp is
        calculated.

    order : int
        Indicates the order of the transport map to apply. Tracking up to
        second order is possible.

    """
    R = first_order_matrix(z, L, theta, k1, gamma_ref)
    bm_new = np.dot(R, beam_matrix)
    if order == 2:
        T = second_order_matrix(z, L, theta, k1, k2, gamma_ref)
        bm_new += np.einsum('ijk,j...,k...', T, beam_matrix, beam_matrix).T
    return bm_new
