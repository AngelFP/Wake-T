"""
This module contains methods for performing the interpolation of the fields
into the particle positions.

"""

import math
import numpy as np
from numba import njit, prange


@njit()
def gather_field_cyl_linear(fld, z_fld, r_fld, x, y, z):
    """
    Interpolate a 2D field defined on an r-z grid to the particle positions
    of a 3D distribution.

    The interpolation is performed using an adaptation of the field gathering
    algorithm from FBPIC (https://github.com/fbpic/fbpic) assuming linear
    particle shapes.

    Parameters:
    -----------

    fld : 2darray
        The field to be interpolated.

    z_fld : 1darray
        The position of the field grid points along z.

    r_fld : 1darray
        The position of the field grid points along r.

    x, y, z : 1darray
        Coordinates of the particle distribution.

    Returns:
    --------

    A 1darray with the field values at the location of each particle.

    """
    n_part = x.shape[0]

    # Calculate needed parameters.
    dr = r_fld[1] - r_fld[0]
    dz = z_fld[1] - z_fld[0]
    z_min_grid = z_fld[0]
    r_min_grid = r_fld[0]
    z_max_grid = z_fld[-1]
    r_max_grid = r_fld[-1]

    # Preallocate output array with field values.
    fld_part = np.zeros(n_part)

    # Iterate over all particles.
    for i in prange(n_part):

        # Get particle position.
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        r_i = math.sqrt(x_i**2 + y_i**2)

        # Gather field only if particle is within field boundaries.
        if z_i > z_min_grid and z_i < z_max_grid and r_i < r_max_grid:
            # Position in cell units.
            r_i_cell = (r_i - r_min_grid)/dr
            z_i_cell = (z_i - z_min_grid)/dz

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1
            iz_lower = int(math.floor(z_i_cell))
            iz_upper = iz_lower + 1

            # If lower r cell is below axis, assume same value as first cell.
            if ir_lower < 0:
                ir_lower = 0

            # Get field value at each bounding cell.
            fld_ll = fld[iz_lower, ir_lower]
            fld_lu = fld[iz_lower, ir_upper]
            fld_ul = fld[iz_upper, ir_lower]
            fld_uu = fld[iz_upper, ir_upper]

            # Interpolate in z.
            dz_u = iz_upper - z_i_cell
            dz_l = z_i_cell - iz_lower
            fld_z_1 = dz_u*fld_ll + dz_l*fld_ul
            fld_z_2 = dz_u*fld_lu + dz_l*fld_uu

            # Interpolate in r.
            dr_u = ir_upper - r_i_cell
            dr_l = 1 - dr_u
            fld_part[i] = dr_u*fld_z_1 + dr_l*fld_z_2
    return fld_part


@njit()
def gather_main_fields_cyl_linear(wx, ez, z_fld, r_fld, x, y, z):
    """
    Convenient method for interpolating at once (more efficient) the transverse
    and longitudinal wakefields.

    The interpolation is performed using an adaptation of the field gathering
    algorithm from FBPIC (https://github.com/fbpic/fbpic) assuming linear
    particle shapes.

    Parameters:
    -----------

    wx : 2darray
        The transverse wakefield.

    wx : 2darray
        The longitudinal wakefield.

    z_fld : 1darray
        The position of the field grid points along z.

    r_fld : 1darray
        The position of the field grid points along r.

    x, y, z : 1darray
        Coordinates of the particle distribution.

    Returns:
    --------

    A tuple with three 1darray containing the values of the longitudinal (z)
    and transverse (x and y) fields acting on each particle.

    """
    n_part = x.shape[0]

    # Calculate needed parameters.
    dr = r_fld[1] - r_fld[0]
    dz = z_fld[1] - z_fld[0]
    z_min_grid = z_fld[0]
    r_min_grid = r_fld[0]
    z_max_grid = z_fld[-1]
    r_max_grid = r_fld[-1]

    # Preallocate output arrays with field values.
    wx_part = np.zeros(n_part)
    wy_part = np.zeros(n_part)
    ez_part = np.zeros(n_part)

    # Iterate over all particles.
    for i in prange(n_part):

        # Get particle position.
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        r_i = math.sqrt(x_i**2 + y_i**2)
        inv_r_i = 1./r_i

        # Gather field only if particle is within field boundaries.
        if z_i > z_min_grid and z_i < z_max_grid and r_i < r_max_grid:
            # Position in cell units.
            r_i_cell = (r_i - r_min_grid)/dr
            z_i_cell = (z_i - z_min_grid)/dz

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1
            iz_lower = int(math.floor(z_i_cell))
            iz_upper = iz_lower + 1

            # If lower r cell is below axis, assume same value as first cell
            # for `ez` and sign inverse of the first cell value for `wx`. This
            # assures that wx=0 on axis.
            wx_corr = 1
            if ir_lower < 0:
                ir_lower = 0
                wx_corr = -1

            # Get field value at each bounding cell.
            wx_ll = wx[iz_lower, ir_lower] * wx_corr
            wx_lu = wx[iz_lower, ir_upper]
            wx_ul = wx[iz_upper, ir_lower] * wx_corr
            wx_uu = wx[iz_upper, ir_upper]
            ez_ll = ez[iz_lower, ir_lower]
            ez_lu = ez[iz_lower, ir_upper]
            ez_ul = ez[iz_upper, ir_lower]
            ez_uu = ez[iz_upper, ir_upper]

            # Interpolate in z
            dz_u = iz_upper - z_i_cell
            dz_l = z_i_cell - iz_lower
            wx_z_1 = dz_u*wx_ll + dz_l*wx_ul
            wx_z_2 = dz_u*wx_lu + dz_l*wx_uu
            ez_z_1 = dz_u*ez_ll + dz_l*ez_ul
            ez_z_2 = dz_u*ez_lu + dz_l*ez_uu

            # Interpolate in r
            dr_u = ir_upper - r_i_cell
            dr_l = 1 - dr_u
            w = dr_u*wx_z_1 + dr_l*wx_z_2
            wx_part[i] = w * x_i * inv_r_i
            wy_part[i] = w * y_i * inv_r_i
            ez_part[i] = dr_u*ez_z_1 + dr_l*ez_z_2
    return wx_part, wy_part, ez_part
