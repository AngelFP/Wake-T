"""
This module contains the methods for depositing the charge of a particle
distribution on a regular grid.

The charge deposition is based on an adaptation of the same algorithm that is
implemented in FBPIC (https://github.com/fbpic/fbpic).

"""


import math
import numpy as np
from numba import njit


def charge_distribution_cyl(z, x, y, q, z_min, r_min, nz, nr, dz, dr, charge_dist,
                            p_shape='cubic'):
    """
    Deposit the charge of a partice distribution in a 2D grid (cylindrical
    symmetry) to obtain the spatial charge distribution.

    Parameters:
    -----------
    z, x, y, q : arrays
        Arrays containing the longitudinal and transverse coordinates of the
        particles as well as their charge.

    z_min : float
        Position of the first field value along z.

    r_min : float
        Position of the first field value along r.

    nz, nr : int
        Number of grid cells (excluding guard cells) along the longitudinal
        and radial direction.

    dz, dr : float
        Grid step size along the longitudinal and radial direction.

    charge_dist : array
        The 2D array containing the charge distribution.

    p_shape : str
        Particle shape to be used. Possible values are 'linear' or 'cubic'.

    Returns:
    --------
    A (nz+4)*(nr+4) 2D array (i.e. including 2 guard cells on each side)
    containing the charge distribution.

    """
    if p_shape == 'linear':
        return charge_distribution_cyl_linear(
            z, x, y, q, z_min, r_min, nz, nr, dz, dr, charge_dist)
    elif p_shape == 'cubic':
        return charge_distribution_cyl_cubic(
            z, x, y, q, z_min, r_min, nz, nr, dz, dr, charge_dist)
    else:
        err_string = ("Particle shape '{}' not recognized. ".format(p_shape) +
                      "Possible values are 'linear' or 'cubic'.")
        raise ValueError(err_string)


@njit
def charge_distribution_cyl_linear(z, x, y, q, z_min, r_min, nz, nr, dz, dr, charge_dist):
    """ Calculate charge distribution assuming linear particle shape. """

    # Precalculate particle shape coefficients needed to satisfy charge
    # conservation during deposition (see work by W.M. Ruyten
    # https://doi.org/10.1006/jcph.1993.1070).
    r_grid = (np.arange(nr) + 0.5) * dr  # Assumes cell-centered in r.
    cell_volume = np.pi*dz*((r_grid+0.5*dr)**2 - (r_grid-0.5*dr)**2)
    cell_volume_norm = cell_volume / (2*np.pi*dr**2*dz)
    cell_number = np.arange(nr) + 1
    ruyten_coef = 6./cell_number*(
        np.cumsum(cell_volume_norm) - 0.5*cell_number**2 - 1./24)

    # Pre-allocate charge density array with 2 guard cells on each side.
    rho = np.zeros((nz+4, nr+4))

    # Loop over particles.
    for i in range(z.shape[0]):
        # Get particle components.
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        w_i = q[i]

        # Calculate radius.
        r_i = math.sqrt(x_i**2 + y_i**2)

        # Positions of the particles in cell units.
        r_cell = (r_i - r_min)/dr
        z_cell = (z_i - z_min)/dz

        # Indices of the lowest cell in which the particle will deposit charge.
        ir_cell = min(int(math.ceil(r_cell))+1, nr+2)
        iz_cell = int(math.ceil(z_cell)) + 1

        # Get corresponding coefficient for corrected shape factor.
        ir = min(int(math.ceil(r_cell)), nr)
        rc = ruyten_coef[ir]

        # Particle position wrt left neighbor gridpoint in r.
        u = r_cell - int(math.ceil(r_cell)) + 1

        # Precalculate quantities.
        zsl_0 = math.ceil(z_cell) - z_cell
        zsl_1 = 1 - zsl_0
        rsl_0 = (1.-u) + rc*(1.-u)*u
        rsl_1 = 1 - rsl_0

        # Add contribution of particle to charge distribution.
        rho[iz_cell+0, ir_cell+0] += zsl_0 * rsl_0 * w_i
        rho[iz_cell+0, ir_cell+1] += zsl_0 * rsl_1 * w_i
        rho[iz_cell+1, ir_cell+0] += zsl_1 * rsl_0 * w_i
        rho[iz_cell+1, ir_cell+1] += zsl_1 * rsl_1 * w_i

    return rho + charge_dist


@njit
def charge_distribution_cyl_cubic(z, x, y, q, z_min, r_min, nz, nr, dz, dr, charge_dist):
    """ Calculate charge distribution assuming cubic particle shape. """

    # Precalculate particle shape coefficients needed to satisfy charge
    # conservation during deposition (see work by W.M. Ruyten
    # https://doi.org/10.1006/jcph.1993.1070).
    r_grid = (np.arange(nr) + 0.5) * dr  # Assumes cell-centered in r.
    cell_volume = np.pi*dz*((r_grid+0.5*dr)**2 - (r_grid-0.5*dr)**2)
    cell_volume_norm = cell_volume / (2*np.pi*dr**2*dz)
    cell_number = np.arange(nr) + 1
    ruyten_coef = 6./cell_number*(
        np.cumsum(cell_volume_norm) - 0.5*cell_number**2 - 0.125)

    # Pre-allocate charge density array with 2 guard cells on each side.
    rho = np.zeros((nz+4, nr+4))

    # Loop over particles.
    for i in range(z.shape[0]):
        # Get particle components.
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        w_i = q[i]

        # Calculate radius.
        r_i = math.sqrt(x_i**2 + y_i**2)

        # Positions of the particle in cell units.
        r_cell = (r_i - r_min)/dr
        z_cell = (z_i - z_min)/dz

        # Indices of the lowest cell in which the particle will deposit charge.
        ir_cell = min(int(math.ceil(r_cell)), nr+2)
        iz_cell = int(math.ceil(z_cell))

        # Particle position wrt left neighbor gridpoint.
        u_z = z_cell - int(math.ceil(z_cell)) + 1
        u_r = r_cell - int(math.ceil(r_cell)) + 1

        # Precalculate quantities for shape coefficients.
        inv_6 = 1./6.
        v_z = 1.-u_z
        v_r = 1.-u_r

        # Get corresponding coefficient for corrected shape factor.
        ir = min(int(math.ceil(r_cell)), nr)
        rc = ruyten_coef[ir]

        # Cubic particle shape coefficients in z and r.
        zsc_0 = inv_6*v_z**3
        zsc_1 = inv_6*(3.*u_z**3 - 6.*u_z**2 + 4.)
        zsc_2 = inv_6*(3.*v_z**3 - 6.*v_z**2 + 4.)
        zsc_3 = inv_6*u_z**3
        rsc_0 = inv_6*v_r**3
        rsc_1 = inv_6*(3.*u_r**3 - 6.*u_r**2 + 4.) + rc*v_r*u_r
        rsc_2 = inv_6*(3.*v_r**3 - 6.*v_r**2 + 4.) - rc*v_r*u_r
        rsc_3 = inv_6*u_r**3

        # Add contribution of particle to charge distribution.
        rho[iz_cell+0, ir_cell+0] += zsc_0 * rsc_0 * w_i
        rho[iz_cell+0, ir_cell+1] += zsc_0 * rsc_1 * w_i
        rho[iz_cell+0, ir_cell+2] += zsc_0 * rsc_2 * w_i
        rho[iz_cell+0, ir_cell+3] += zsc_0 * rsc_3 * w_i
        rho[iz_cell+1, ir_cell+0] += zsc_1 * rsc_0 * w_i
        rho[iz_cell+1, ir_cell+1] += zsc_1 * rsc_1 * w_i
        rho[iz_cell+1, ir_cell+2] += zsc_1 * rsc_2 * w_i
        rho[iz_cell+1, ir_cell+3] += zsc_1 * rsc_3 * w_i
        rho[iz_cell+2, ir_cell+0] += zsc_2 * rsc_0 * w_i
        rho[iz_cell+2, ir_cell+1] += zsc_2 * rsc_1 * w_i
        rho[iz_cell+2, ir_cell+2] += zsc_2 * rsc_2 * w_i
        rho[iz_cell+2, ir_cell+3] += zsc_2 * rsc_3 * w_i
        rho[iz_cell+3, ir_cell+0] += zsc_3 * rsc_0 * w_i
        rho[iz_cell+3, ir_cell+1] += zsc_3 * rsc_1 * w_i
        rho[iz_cell+3, ir_cell+2] += zsc_3 * rsc_2 * w_i
        rho[iz_cell+3, ir_cell+3] += zsc_3 * rsc_3 * w_i

    return rho + charge_dist
