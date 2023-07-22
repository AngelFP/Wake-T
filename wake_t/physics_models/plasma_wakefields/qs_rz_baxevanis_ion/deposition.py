"""
This module contains the methods for depositing particle weights on a 1D grid.

"""

import math

from wake_t.utilities.numba import njit_serial


@njit_serial()
def deposit_plasma_particles(r, w, r_min, nr, dr, deposition_array,
                             p_shape='cubic'):
    """
    Deposit the the weight of a 1D slice of plasma particles into a 2D
    r-z grid.

    Parameters
    ----------
    r : array
        Arrays containing the radial coordinates of the
        particles.
    w : array
        Weight of the particles (any quantity) which will be deposited into
        the grid.
    r_min : float
        Position of the first field value along r.
    nr : int
        Number of grid cells (excluding guard cells) along the radial
        direction.
    dr : float
        Grid step size along the radial direction.
    deposition_array : array
        The 2D array of size nr+4 (including two guard cells at each
        boundary) into which the weight will be deposited.
    p_shape : str
        Particle shape to be used. Possible values are 'linear' or 'cubic'.

    """
    if p_shape == 'linear':
        return deposit_plasma_particles_linear(
            r, w, r_min, nr, dr, deposition_array)
    elif p_shape == 'cubic':
        return deposit_plasma_particles_cubic(
            r, w, r_min, nr, dr, deposition_array)


@njit_serial(fastmath=True, error_model="numpy")
def deposit_plasma_particles_linear(r, q, r_min, nr, dr, deposition_array):
    """ Calculate charge distribution assuming linear particle shape. """

    r_max = nr * dr

    # Loop over particles.
    for i in range(r.shape[0]):
        # Get particle components.
        r_i = r[i]
        w_i = q[i]

        # Deposit only if particle is within field boundaries.
        if r_i <= r_max:
            # Positions of the particles in cell units.
            r_cell = (r_i - r_min) / dr

            # Indices of lowest cell in which the particle will deposit charge.
            ir_cell = int(math.ceil(r_cell)) + 1

            # u_r: particle position wrt left neighbor gridpoint in r.
            u_r = r_cell + 2 - ir_cell

            # Precalculate quantities.
            rsl_0 = 1. - u_r
            rsl_1 = u_r

            # Add contribution of particle to density array.
            deposition_array[ir_cell + 0] += rsl_0 * w_i
            deposition_array[ir_cell + 1] += rsl_1 * w_i

        # Apply correction on axis (ensures uniform density in a uniform
        # plasma)
        deposition_array[2] -= deposition_array[1]
        deposition_array[1] = 0.

    for i in range(nr):
        deposition_array[i+2] /= (r_min + i * dr) * dr


@njit_serial(fastmath=True, error_model="numpy")
def deposit_plasma_particles_cubic(r, q, r_min, nr, dr, deposition_array):
    """ Calculate charge distribution assuming cubic particle shape. """

    r_max = nr * dr

    # Loop over particles.
    for i in range(r.shape[0]):
        # Get particle components.
        r_i = r[i]
        w_i = q[i]

        # Deposit only if particle is within field boundaries.
        if r_i <= r_max:
            # Positions of the particle in cell units.
            r_cell = (r_i - r_min) / dr

            # Indices of lowest cell in which the particle will deposit charge.
            ir_cell = int(math.ceil(r_cell))

            # Particle position wrt left neighbor gridpoint.
            u_r = r_cell - ir_cell + 1

            # Precalculate quantities for shape coefficients.
            inv_6 = 1. / 6.
            v_r = 1. - u_r

            # Cubic particle shape coefficients in z and r.
            rsc_0 = inv_6 * v_r ** 3
            rsc_1 = inv_6 * (3. * u_r**3 - 6. * u_r**2 + 4.)
            rsc_2 = inv_6 * (3. * v_r**3 - 6. * v_r**2 + 4.)
            rsc_3 = inv_6 * u_r ** 3

            # Add contribution of particle to density array.
            deposition_array[ir_cell + 0] += rsc_0 * w_i
            deposition_array[ir_cell + 1] += rsc_1 * w_i
            deposition_array[ir_cell + 2] += rsc_2 * w_i
            deposition_array[ir_cell + 3] += rsc_3 * w_i

        # Apply correction on axis (ensures uniform density in a uniform
        # plasma)
        deposition_array[2] -= deposition_array[1]
        deposition_array[3] -= deposition_array[0]
        deposition_array[0] = 0.
        deposition_array[1] = 0.

    for i in range(nr):
        deposition_array[i+2] /= (r_min + i * dr) * dr
