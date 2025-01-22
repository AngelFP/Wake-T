"""Defines a method to compute the source terms from a particle bunch."""

import numpy as np
import scipy.constants as ct

from wake_t.particles.deposition import deposit_3d_distribution
from wake_t.utilities.numba import njit_serial


@njit_serial()
def calculate_bunch_source(q_bunch, n_r, n_xi, b_t_bunch):
    """
    Calculate the bunch source term (azimuthal magnetic field) from the
    charge deposited on the grid.

    Parameters
    ----------
    q_bunch : ndarray
        Array containing the bunch charge on the grid.
    n_r, n_xi : int
        Number of grid points along r and xi.
    r_grid : float
        Radial coordinated of the grid points.
    b_t_bunch : ndarray
        A (nz+4, nr+4) array where the magnetic field will be stored.
    """
    for i in range(n_xi):
        cumsum = 0.0
        # Iterate until n_r + 2 to get bunch source also in guard cells.
        for j in range(n_r + 2):
            q_ij = q_bunch[2 + i, 2 + j]
            cumsum += q_ij
            # At each grid cell, calculate integral only until cell center by
            # assuming that half the charge is evenly distributed within the
            # cell (i.e., subtract half the charge)
            b_t_bunch[2 + i, 2 + j] = (cumsum - 0.5 * q_ij) * 1.0 / (0.5 + j)
        # At the first grid point along r, subtract an additional 1/4 of the
        # charge. This comes from assuming that the density has to be zero on
        # axis.
        b_t_bunch[2 + i, 2] -= 0.25 * q_bunch[2 + i, 2] * 2.0


@njit_serial()
def deposit_bunch_charge(
    x,
    y,
    z,
    q,
    n_p,
    n_r,
    n_xi,
    r_grid,
    xi_grid,
    dr,
    dxi,
    p_shape,
    q_bunch,
    r_min_deposit=0.0,
):
    """
    Deposit the charge of particle bunch in a 2D grid.

    Parameters
    ----------
    x, y, z, q : ndarray
        The coordinates and charge of the bunch particles.
    n_p : float
        The plasma density.
    n_r, n_xi : int
        Number of grid points along r and xi.
    r_grid, xi_grid : float
        Coordinates of the grid nodes.
    dr, dxi : float
        Grid spacing in r and xi.
    p_shape : str
        Particle shape. Used to deposit the charge on the grid.
    q_bunch : ndarray
        A (nz+4, nr+4) array where the charge will be deposited.
    r_min_deposit : float
        The minimum radial position that particles must have in order to
        deposit their weight on the grid.

    Returns
    -------
    bool
        Whether all particles managed to deposit on the grid.
    """
    n_part = x.shape[0]
    s_d = ct.c / np.sqrt(ct.e**2 * n_p / (ct.m_e * ct.epsilon_0))
    k = 1.0 / (2 * np.pi * ct.e * dr * dxi * s_d * n_p)
    w = np.empty(n_part)
    for i in range(n_part):
        w[i] = q[i] * k

    return deposit_3d_distribution(
        z,
        x,
        y,
        w,
        xi_grid[0],
        r_grid[0],
        n_xi,
        n_r,
        dxi,
        dr,
        q_bunch,
        p_shape=p_shape,
        use_ruyten=True,
        r_min_deposit=r_min_deposit,
    )
