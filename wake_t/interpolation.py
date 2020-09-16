import math
import numpy as np
from numba import njit, prange


@njit()
def interpolate_cyl_linear(fld, z_fld, r_fld, x, y, z):
    n_part = x.shape[0]
    fld_part = np.zeros(n_part)
    dr = r_fld[1] - r_fld[0]
    dz = z_fld[1] - z_fld[0]
    inv_dr = 1./dr
    inv_dz = 1./dz
    z_min = z_fld[0]
    for i in prange(n_part):
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]

        r_i = np.sqrt(x_i**2 + y_i**2)
        r_i_cell = r_i*inv_dr - 0.5
        z_i_cell = (z_i - z_min)*inv_dz - 0.5
        ir_lower = int(math.floor(r_i_cell))
        ir_upper = ir_lower + 1
        iz_lower = int(math.floor(z_i_cell))
        iz_upper = iz_lower + 1
        if ir_lower < 0:
            ir_lower = 0
        fld_ll = fld[iz_lower, ir_lower]
        fld_lu = fld[iz_lower, ir_upper]
        fld_ul = fld[iz_upper, ir_lower]
        fld_uu = fld[iz_upper, ir_upper]

        # Interpolate in z
        fld_z_1 = (iz_upper - z_i_cell)*fld_ll + (z_i_cell - iz_lower)*fld_ul
        fld_z_2 = (iz_upper - z_i_cell)*fld_lu + (z_i_cell - iz_lower)*fld_uu

        # Interpolate in r
        fld_part[i] = (ir_upper - r_i_cell)*fld_z_1 + (r_i_cell - ir_lower)*fld_z_2
    return fld_part

