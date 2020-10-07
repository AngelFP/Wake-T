import math
import numpy as np
from numba import njit


def charge_histogram(z, x, y, q, z_min, nz, nr, dz, dr, r_grid, p_shape='cubic'):
    if p_shape == 'linear':
        return charge_histogram_linear(z, x, y, q, z_min, nz, nr, dz, dr, r_grid)
    elif p_shape == 'cubic':
        return charge_histogram_cubic(z, x, y, q, z_min, nz, nr, dz, dr, r_grid)


@njit
def charge_histogram_linear(z, x, y, q, z_min, nz, nr, dz, dr, r_grid):
    invdr = 1./dr
    invdz = 1./dz

    # Calculate Ruyten coefficients.
    vol = np.pi*dz*((r_grid+0.5*dr)**2 - (r_grid-0.5*dr)**2)
    norm_vol = vol/(2*np.pi*dr**2*dz)
    nr_vals = np.arange(nr)
    ruyten_linear_coef = 6./(nr_vals+1)*(
        np.cumsum(norm_vol) - 0.5*(nr_vals+1.)**2 - 1./24)

    # Pre-allocate charge density array with 2 guard cells on each side.
    rho = np.zeros((2+nz+2, 2+nr+2))

    # Loop over all particles
    for i in range(z.shape[0]):

        # Position
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]

        # Weights
        w_i = q[i]

        # Cylindrical conversion
        r_i = math.sqrt(x_i**2 + y_i**2)

        # Positions of the particles, in the cell unit
        r_cell = invdr*r_i - 0.5
        z_cell = invdz*(z_i - z_min) - 0.5
        # Index of the lowest cell of `global_array` that gets modified
        # by this particle (note: `global_array` has 2 guard cells)
        # (`min` function avoids out-of-bounds access at high r)
        ir_cell = min(int(math.ceil(r_cell))+1, nr+2)
        iz_cell = int(math.ceil(z_cell)) + 1

        ir = min(int(math.ceil(r_cell)), nr)

        # Ruyten-corrected shape factor coefficient
        bn = ruyten_linear_coef[ir]

        # Precalculate quantities
        zsl_0 = z_shape_linear(z_cell, 0)
        zsl_1 = z_shape_linear(z_cell, 1)
        rsl_0 = r_shape_linear(r_cell, 0, bn)
        rsl_1 = r_shape_linear(r_cell, 1, bn)

        rho[iz_cell+0, ir_cell+0] += zsl_0 * rsl_0 * w_i
        rho[iz_cell+0, ir_cell+1] += zsl_0 * rsl_1 * w_i
        rho[iz_cell+1, ir_cell+0] += zsl_1 * rsl_0 * w_i
        rho[iz_cell+1, ir_cell+1] += zsl_1 * rsl_1 * w_i

    return rho


@njit
def charge_histogram_cubic(z, x, y, q, z_min, nz, nr, dz, dr, r_grid):
    invdr = 1./dr
    invdz = 1./dz

    # Calculate Ruyten coefficients.
    vol = np.pi*dz*((r_grid+0.5*dr)**2 - (r_grid-0.5*dr)**2)
    norm_vol = vol/(2*np.pi*dr**2*dz)
    nr_vals = np.arange(nr)
    ruyten_cubic_coef = 6./(nr_vals+1)*(
        np.cumsum(norm_vol) - 0.5*(nr_vals+1.)**2 - 1./8)

    # Pre-allocate charge density array with 2 guard cells on each side.
    rho = np.zeros((2+nz+2, 2+nr+2))

    # Loop over all particles
    for i in range(z.shape[0]):

        # Position
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]

        # Weights
        w_i = q[i]

        # Cylindrical conversion
        r_i = math.sqrt(x_i**2 + y_i**2)

        # Positions of the particles, in the cell unit
        r_cell = invdr*r_i - 0.5
        z_cell = invdz*(z_i - z_min) - 0.5
        # Index of the lowest cell of `global_array` that gets modified
        # by this particle (note: `global_array` has 2 guard cells)
        # (`min` function avoids out-of-bounds access at high r)
        ir_cell = min(int(math.ceil(r_cell)), nr)
        iz_cell = int(math.ceil(z_cell))

        ir = min(int(math.ceil(r_cell)), nr)

        # Ruyten-corrected shape factor coefficient
        bn = ruyten_cubic_coef[ir]

        # Precalculate quantities
        zsc_0 = z_shape_cubic(z_cell, 0)
        zsc_1 = z_shape_cubic(z_cell, 1)
        zsc_2 = z_shape_cubic(z_cell, 2)
        zsc_3 = z_shape_cubic(z_cell, 3)
        rsc_0 = r_shape_cubic(r_cell, 0, bn)
        rsc_1 = r_shape_cubic(r_cell, 1, bn)
        rsc_2 = r_shape_cubic(r_cell, 2, bn)
        rsc_3 = r_shape_cubic(r_cell, 3, bn)

        # Add contribution of this particle to the global array
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

    return rho


@njit
def z_shape_linear(cell_position, index):
    s = math.ceil(cell_position) - cell_position
    if index == 1:
        s = 1.-s
    return s


@njit
def r_shape_linear(cell_position, index, beta_n):
    # Get radial cell index
    ir = int(math.ceil(cell_position)) - 1
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - ir
    s = (1.-u) + beta_n*(1.-u)*u
    if index == 1:
        s = 1.-s
    return s


@njit
def z_shape_cubic(cell_position, index):
    iz = int(math.ceil(cell_position)) - 2
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - iz - 1
    s = 0.
    if index == 0:
        s = (1./6.)*(1.-u)**3
    elif index == 1:
        s = (1./6.)*(3.*u**3 - 6.*u**2 + 4.)
    elif index == 2:
        s = (1./6.)*(3.*(1.-u)**3 - 6.*(1.-u)**2 + 4.)
    elif index == 3:
        s = (1./6.)*u**3
    return s


@njit
def r_shape_cubic(cell_position, index, beta_n):
    # Get radial cell index
    ir = int(math.ceil(cell_position)) - 2
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - ir - 1
    s = 0.
    if index == 0:
        s = (1./6.)*(1.-u)**3
    elif index == 1:
        s = (1./6.)*(3.*u**3 - 6.*u**2 + 4.)
        s += beta_n*(1.-u)*u # Add Ruyten correction
    elif index == 2:
        s = (1./6.)*(3.*(1.-u)**3 - 6.*(1.-u)**2 + 4.)
        s -= beta_n*(1.-u)*u # Add Ruyten correction
    elif index == 3:
        s = (1./6.)*u**3
    return s

