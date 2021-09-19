"""
This module contains methods for performing the interpolation of the fields
into the particle positions.

The interpolation methods are based on an adaptation of the field gathering
algorithm from FBPIC (https://github.com/fbpic/fbpic) assuming linear particle
shapes.

"""


import math
import numpy as np
from wake_t.utilities.numba_caching import njit_func
from numba import prange


@njit_func
def gather_field_cyl_linear(fld, z_min, z_max, r_min, r_max, dz, dr, x, y, z):
    """
    Interpolate a 2D field defined on an r-z grid to the particle positions
    of a 3D distribution.

    Parameters:
    -----------

    fld : 2darray
        The field to be interpolated.

    z_min, z_max : float
        Position of the first and last field values along z.

    r_min,r_max : float
        Position of the first and last field values along r.

    dz, dr : float
        Grid step size along the longitudinal and radial direction.

    x, y, z : 1darray
        Coordinates of the particle distribution.

    Returns:
    --------

    A 1darray with the field values at the location of each particle.

    """
    n_part = x.shape[0]

    # Preallocate output array with field values.
    fld_part = np.zeros(n_part)

    # Iterate over all particles.
    for i in range(n_part):
        # Get particle position.
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        r_i = math.sqrt(x_i**2 + y_i**2)

        # Gather field only if particle is within field boundaries.
        if z_i >= z_min and z_i <= z_max and r_i <= r_max:
            # Position in cell units.
            r_i_cell = (r_i - r_min)/dr + 2.
            z_i_cell = (z_i - z_min)/dz + 2.

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1
            iz_lower = int(math.floor(z_i_cell))
            iz_upper = iz_lower + 1

            # If lower r cell is below axis, assume same value as first cell.
            if ir_lower < 2:
                ir_lower = 2

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


@njit_func
def gather_main_fields_cyl_linear(wx, ez, z_min, z_max, r_min, r_max, dz, dr,
                                  x, y, z):
    """
    Convenient method for interpolating at once (more efficient) the transverse
    and longitudinal wakefields.

    Parameters:
    -----------

    wx : 2darray
        The transverse wakefield.

    ez : 2darray
        The longitudinal wakefield.

    z_min, z_max : float
        Position of the first and last field values along z.

    r_min,r_max : float
        Position of the first and last field values along r.

    dz, dr : float
        Grid step size along the longitudinal and radial direction.

    x, y, z : 1darray
        Coordinates of the particle distribution.

    Returns:
    --------

    A tuple with three 1darray containing the values of the longitudinal (z)
    and transverse (x and y) fields acting on each particle.

    """
    n_part = x.shape[0]

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
        if z_i >= z_min and z_i <= z_max and r_i <= r_max:
            # Position in cell units.
            r_i_cell = (r_i - r_min)/dr + 2
            z_i_cell = (z_i - z_min)/dz + 2

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1
            iz_lower = int(math.floor(z_i_cell))
            iz_upper = iz_lower + 1

            # If lower r cell is below axis, assume same value as first cell
            # for `ez` and sign inverse of the first cell value for `wx`. This
            # ensures that wx=0 on axis.
            wx_corr = 1
            if ir_lower < 2:
                ir_lower = 2
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


@njit_func
def gather_sources_qs_baxevanis(fld_1, fld_2, fld_3, z_min, z_max, r_min,
                                r_max, dz, dr, r, z):
    """
    Convenient method for gathering at once the three source fields needed
    by the Baxevanis wakefield model (a2 and nabla_a from the laser, and
    b_theta_0 from the beam) into the plasma particles. This method is also
    catered to cylindrical geometry and assumes all plasma particles have the
    same longitudinal position (which is the case in a quasistatic model,
    where there is only a single column of particles).

    Parameters:
    -----------

    fld_1, fld_2, fld_3 : ndarray
        The three source fields, corresponding respectively to a2, nabla_a2
        and b_theta_0. Each of them is a (nr+4, nr+4) array, including 2 guard
        cells in each boundary.

    z_min, z_max : float
        Position of the first and last field values along z.

    r_min, r_max : float
        Position of the first and last field values along r.

    dz, dr : float
        Grid step size along the longitudinal and radial direction.

    r : 1darray
        Transverse position of the plasma particles.

    z : int
        Longitudinal position of the column of plasma particles.

    Returns:
    --------

    A tuple with three 1darray containing the gathered field values at the
    position of each particle.

    """
    n_part = r.shape[0]

    # Preallocate output arrays with field values.
    fld_1_part = np.zeros(n_part)
    fld_2_part = np.zeros(n_part)
    fld_3_part = np.zeros(n_part)

    # Iterate over all particles.
    for i in prange(n_part):

        # Get particle position.
        z_i = z
        r_i = r[i]

        # Gather field only if particle is within field boundaries.
        if z_i >= z_min and z_i <= z_max and r_i <= r_max:
            # Position in cell units.
            r_i_cell = (r_i - r_min)/dr + 2
            z_i_cell = (z_i - z_min)/dz + 2

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1
            iz_lower = int(math.floor(z_i_cell))
            iz_upper = iz_lower + 1

            # If lower r cell is below axis, assume same value as first cell.
            # For `nabla_a2` and `b_theta_0`, invert the sign to ensure they
            # are `0` on axis.
            sign = 1
            if ir_lower < 2:
                ir_lower = 2
                sign = -1

            # Get field value at each bounding cell.
            fld_1_ll = fld_1[iz_lower, ir_lower]
            fld_1_lu = fld_1[iz_lower, ir_upper]
            fld_1_ul = fld_1[iz_upper, ir_lower]
            fld_1_uu = fld_1[iz_upper, ir_upper]
            fld_2_ll = fld_2[iz_lower, ir_lower] * sign
            fld_2_lu = fld_2[iz_lower, ir_upper]
            fld_2_ul = fld_2[iz_upper, ir_lower] * sign
            fld_2_uu = fld_2[iz_upper, ir_upper]
            fld_3_ll = fld_3[iz_lower, ir_lower] * sign
            fld_3_lu = fld_3[iz_lower, ir_upper]
            fld_3_ul = fld_3[iz_upper, ir_lower] * sign
            fld_3_uu = fld_3[iz_upper, ir_upper]

            # Interpolate in z
            dz_u = iz_upper - z_i_cell
            dz_l = z_i_cell - iz_lower
            fld_1_z_1 = dz_u*fld_1_ll + dz_l*fld_1_ul
            fld_1_z_2 = dz_u*fld_1_lu + dz_l*fld_1_uu
            fld_2_z_1 = dz_u*fld_2_ll + dz_l*fld_2_ul
            fld_2_z_2 = dz_u*fld_2_lu + dz_l*fld_2_uu
            fld_3_z_1 = dz_u*fld_3_ll + dz_l*fld_3_ul
            fld_3_z_2 = dz_u*fld_3_lu + dz_l*fld_3_uu

            # Interpolate in r
            dr_u = ir_upper - r_i_cell
            dr_l = 1 - dr_u
            fld_1_part[i] = dr_u*fld_1_z_1 + dr_l*fld_1_z_2
            fld_2_part[i] = dr_u*fld_2_z_1 + dr_l*fld_2_z_2
            fld_3_part[i] = dr_u*fld_3_z_1 + dr_l*fld_3_z_2
    return fld_1_part, fld_2_part, fld_3_part
