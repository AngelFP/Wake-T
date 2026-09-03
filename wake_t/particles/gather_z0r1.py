"""
Inverse (adjoint-style) of deposit_3d_distribution: gather/interpolate from a
2D (z, r) deposition_array back to particle weights w at particle locations.

Important:
- This is the PIC-consistent "inverse": gather using the SAME shape functions
  (linear/cubic) and SAME indexing/guard-cell conventions.

It is used in SALAME algorithm in physics_model/plasma_wakefields/qs_rz_baxevenis_ion/wakefield.py
"""

import math
import numpy as np

from wake_t.utilities.numba import njit_serial, prange


@njit_serial
def gather_z0r1(
    z,
    x,
    y,
    z_min,
    r_min,
    nz,
    nr,
    dz,
    dr,
    deposition_array,
    use_ruyten=False,
    r_min_deposit=0.0,
):
    """Gather from grid to particles using 0th-order in z and 1st-order in r.

    Adjoint of deposit_3d_distribution_z0r1: uses identical index and
    weight computations, but reads from deposition_array instead of
    accumulating into it.

    Returns (w_out, all_gathered)
    """
    if use_ruyten:
        ruyten_coef = np.zeros(nr + 1)
        r_grid = (np.arange(nr) + 0.5) * dr
        cell_volume = np.pi * dz * ((r_grid + 0.5 * dr) ** 2 - (r_grid - 0.5 * dr) ** 2)
        cell_volume_norm = cell_volume / (2 * np.pi * dr**2 * dz)
        cell_number = np.arange(nr) + 1
        ruyten_coef[1:] = (
            6.0
            / cell_number
            * (np.cumsum(cell_volume_norm) - 0.5 * cell_number**2 - 1.0 / 24)
        )

    z_max = z_min + (nz - 1) * dz
    r_max = nr * dr

    w_out = np.zeros(z.shape[0])
    all_gathered = True

    for i in prange(z.shape[0]):
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        r_i = math.sqrt(x_i**2 + y_i**2)

        if z_i >= z_min and z_i <= z_max and r_i >= r_min_deposit and r_i <= r_max:
            r_cell = (r_i - r_min) / dr
            z_cell = (z_i - z_min) / dz

            # 0th order in z: nearest grid point (same as deposit)
            iz_nearest = int(math.floor(z_cell + 0.5))
            iz_nearest = min(max(iz_nearest, 0), nz - 1)
            iz_cell = iz_nearest + 2

            # 1st order in r: linear split (same as deposit)
            ir_cell = min(int(math.ceil(r_cell)) + 1, nr + 2)

            if r_cell < 0:
                u_r = 1.0
            else:
                u_r = r_cell - int(math.ceil(r_cell)) + 1

            rsl_0 = 1.0 - u_r
            rsl_1 = u_r

            if use_ruyten:
                ir = min(int(math.ceil(r_cell)), nr)
                rc = ruyten_coef[ir]
                rsl_0 += rc * (1.0 - u_r) * u_r
                rsl_1 -= rc * (1.0 - u_r) * u_r

            w_out[i] = (
                rsl_0 * deposition_array[iz_cell, ir_cell + 0]
                + rsl_1 * deposition_array[iz_cell, ir_cell + 1]
            )
        else:
            all_gathered = False
            w_out[i] = 0.0

    return w_out, all_gathered
