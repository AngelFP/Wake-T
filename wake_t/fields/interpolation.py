""" Contains methods for interpolating fields to a different grid. """

import math

from wake_t.utilities.numba import njit_serial


@njit_serial(fastmath=True)
def interpolate_rz_field(fld, z_min, r_min, dz, dr, z_new, r_new, fld_new):
    """
    Interpolate a field in r-z geometry to a new grid using linear
    interpolation.

    Parameters
    ----------
    fld : 2darray
        The original field to interpolate.
    z_min : float
        Position of the first field value along `z`.
    r_min : float
        Position of the first field value along `r`.
    dz : float
        Grid spacing in `z` of the original field.
    dr : float
        Grid spacing in `r` of the original field.
    z_new : 1darray
        Location of the grid nodes in `z` of the new, interpolated field.
    r_new : 1darray
        Location of the grid nodes in `r` of the new, interpolated field.
    fld_new : 2darray
        Array where to store the new, interpolated field. The shape of this
        array should be (`nz_new`, `nr_new`), here `nz_new` and `nr_new` are
        the length of `z_new` and `r_new`, respectively.
    """
    nz, nr = fld.shape
    for i in range(z_new.shape[0]):
        z_i = z_new[i]

        z_i_cell = (z_i - z_min) / dz
        iz_lower = int(math.floor(z_i_cell))
        iz_upper = iz_lower + 1
        dz_u = iz_upper - z_i_cell
        dz_l = z_i_cell - iz_lower

        iz_upper = min(iz_upper, nz - 1)

        for j in range(r_new.shape[0]):
            r_j = r_new[j]

            r_j_cell = (r_j - r_min) / dr
            jr_lower = max(int(math.floor(r_j_cell)), 0)
            jr_upper = jr_lower + 1
            dr_u = jr_upper - r_j_cell
            dr_l = 1 - dr_u

            jr_upper = min(jr_lower + 1, nr-1)

            # Get field value at each bounding cell.
            fld_ll = fld[iz_lower, jr_lower]
            fld_lu = fld[iz_lower, jr_upper]
            fld_ul = fld[iz_upper, jr_lower]
            fld_uu = fld[iz_upper, jr_upper]

            # Interpolate in z.
            fld_z_1 = dz_u*fld_ll + dz_l*fld_ul
            fld_z_2 = dz_u*fld_lu + dz_l*fld_uu

            # Interpolate in r.
            fld_new[i, j] = dr_u*fld_z_1 + dr_l*fld_z_2
