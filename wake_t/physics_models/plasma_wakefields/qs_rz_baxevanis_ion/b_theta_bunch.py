"""Defines a method to compute the source terms from a particle bunch."""
import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.particles.deposition import deposit_3d_distribution


def calculate_bunch_source(
        bunch, n_p, n_r, n_xi, r_min, xi_min, dr, dxi, p_shape, b_t):
    """
    Return a (nz+4, nr+4) array with the normalized azimuthal magnetic field
    from a particle distribution. This is Eq. (18) in the original paper.

    Parameters
    ----------
    bunch : ParticleBunch
        The bunch from which to compute the magnetic field.
    n_p : float
        The plasma density.
    n_r, n_xi : int
        Number of grid points along r and xi.
    r_min, xi_min : float
        Minimum location og the grip points along r and xi.
    dr, dxi : float
        Grid spacing in r and xi.
    p_shape : str
        Particle shape. Used to deposit the charge on the grid.
    b_t : ndarray
        A (nz+4, nr+4) array where the magnetic field will be stored.
    """
    # Plasma skin depth.
    s_d = ge.plasma_skin_depth(n_p / 1e6)

    # Calculate particle weights.
    w = bunch.q / ct.e / (2 * np.pi * dr * dxi * s_d * n_p)

    # Obtain charge distribution (using cubic particle shape by default).
    q_dist = np.zeros((n_xi + 4, n_r + 4))
    deposit_3d_distribution(bunch.xi, bunch.x, bunch.y, w, xi_min, r_min, n_xi,
                            n_r, dxi, dr, q_dist, p_shape=p_shape,
                            use_ruyten=True)

    # Remove guard cells.
    q_dist = q_dist[2:-2, 2:-2]

    # Radial position of grid points.
    r_grid_g = (0.5 + np.arange(n_r)) * dr

    # At each grid cell, calculate integral only until cell center by
    # assuming that half the charge is evenly distributed within the cell
    # (i.e., subtract half the charge)
    subs = q_dist / 2

    # At the first grid point along r, subtract an additional 1/4 of the
    # charge. This comes from assuming that the density has to be zero on axis.
    subs[:, 0] += q_dist[:, 0] / 4

    # Calculate field by integration.
    b_t[2:-2, 2:-2] += (
        (np.cumsum(q_dist, axis=1) - subs) * dr / np.abs(r_grid_g))
