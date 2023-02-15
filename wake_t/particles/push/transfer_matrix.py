"""Contains the logic for pushing a particle beam with transport matrices."""

import numpy as np

from wake_t.physics_models.beam_optics.transfer_matrices import (
    first_order_matrix, second_order_matrix
)


def track_with_transfer_map(beam_matrix, z, L, theta, k1, k2, gamma_ref,
                            order=2):
    """
    Track beam distribution throwgh beamline element by using a transfer map.
    This function was initially based on code from Ocelot (see
    https://github.com/ocelot-collab/ocelot) written by S. Tomin.

    Parameters
    ----------
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
