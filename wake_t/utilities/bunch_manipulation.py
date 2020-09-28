""" This module contains functions for manupulating bunch distributions """

import numpy as np


def convert_to_ocelot_matrix(bunch_matrix, q, gamma_ref=None):
    """
    Produces a matrix with the phase space coordinates
    (x, x', y, y', xi, dp) from a matrix containing (x, px, y, py, xi, pz).
    """
    x = bunch_matrix[0]
    y = bunch_matrix[2]
    xi = bunch_matrix[4]
    px = bunch_matrix[1]
    py = bunch_matrix[3]
    pz = bunch_matrix[5]
    g = np.sqrt(1 + px**2 + py**2 + pz**2)
    if gamma_ref is None:
        gamma_ref = np.average(g, weights=q)
    b_ref = np.sqrt(1 - gamma_ref**(-2))
    dp = (g-gamma_ref)/(gamma_ref*b_ref)
    p_kin = np.sqrt(gamma_ref**2 - 1)
    return np.array([x, px/p_kin, y, py/p_kin, -xi, dp]), gamma_ref


def convert_from_ocelot_matrix(beam_matrix, gamma_ref):
    """
    Produces a matrix with the phase space coordinates
    (x, px, y, py, xi, pz) from a matrix containing (x, x', y, y', xi, dp).

    Parameters:
    -----------
    bunch_matrix : array
        6 x N matrix, where N is the number of particles, containing the
        phase-space information of the bunch as (x, x', y, y', xi, dp) in
        units of (m, rad, m, rad, m, -). dp is defined as
        dp = (g-g_ref)/g_ref, while x' = px/p_kin and y' = py/p_kin, where
        p_kin is the kinetic momentum of each particle.

    gamma_ref : float
        Reference energy with respect to which the particle momentum dp is
        calculated.

    """
    dp = beam_matrix[5]
    b_ref = np.sqrt(1 - gamma_ref**(-2))
    gamma = dp*gamma_ref*b_ref + gamma_ref
    p_kin = np.sqrt(gamma_ref**2 - 1)
    x = beam_matrix[0]
    px = beam_matrix[1] * p_kin
    y = beam_matrix[2]
    py = beam_matrix[3] * p_kin
    xi = -beam_matrix[4]
    pz = np.sqrt(gamma**2 - px**2 - py**2 - 1)
    return np.array([x, px, y, py, xi, pz])


def rotation_matrix_xz(angle):
    """ Returns matrix to rotate the beam in the x-z plane """
    cs = np.cos(angle)
    sn = np.sin(angle)
    return np.array([[cs, 0., 0., 0., sn, 0.],
                     [0., cs, 0., 0., 0., sn],
                     [0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0.],
                     [-sn, 0., 0., 0., cs, 0.],
                     [0., -sn, 0., 0., 0., cs]])
