import math
import numpy as np
from numba import njit, prange


@njit()
def interpolate_cyl_linear(wx, ez, z_fld, r_fld, x, y, z):
    n_part = x.shape[0]
    wx_part = np.zeros(n_part)
    wy_part = np.zeros(n_part)
    ez_part = np.zeros(n_part)
    dr = r_fld[1] - r_fld[0]
    dz = z_fld[1] - z_fld[0]
    inv_dr = 1./dr
    inv_dz = 1./dz
    z_min = z_fld[0]
    for i in prange(n_part):
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]

        r_i = math.sqrt(x_i**2 + y_i**2)
        inv_r_i = 1./r_i
        r_i_cell = r_i*inv_dr - 0.5
        z_i_cell = (z_i - z_min)*inv_dz - 0.5
        ir_lower = int(math.floor(r_i_cell))
        ir_upper = ir_lower + 1
        iz_lower = int(math.floor(z_i_cell))
        iz_upper = iz_lower + 1
        wx_corr = 1
        if ir_lower < 0:
            ir_lower = 0
            wx_corr = -1

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

