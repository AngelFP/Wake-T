import math

from wake_t.utilities.numba import njit_serial


@njit_serial(fastmath=True)
def gather_sources(fld_1, fld_2, fld_3, r_min, r_max, dr, r,
                   fld_1_pp, fld_2_pp, fld_3_pp):
    """
    Convenient method for gathering at once the three source fields needed
    by the Baxevanis wakefield model (a2 and nabla_a from the laser, and
    b_theta_0 from the beam) into the plasma particles. This method is also
    catered to cylindrical geometry and assumes all plasma particles have the
    same longitudinal position (which is the case in a quasistatic model,
    where there is only a single column of particles).

    Parameters
    ----------
    fld_1, fld_2, fld_3 : ndarray
        The three source fields, corresponding respectively to a2, nabla_a2
        and b_theta_0. Each of them is a (nr+4) array, including 2 guard
        cells in each boundary.
    r_min, r_max : float
        Position of the first and last field values along r.
    dz : float
        Grid step size along the radial direction.
    r : 1darray
        Transverse position of the plasma particles.

    """

    # Iterate over all particles.
    for i in range(r.shape[0]):

        # Get particle position.
        r_i = r[i]

        # Gather field only if particle is within field boundaries.
        if r_i <= r_max:
            # Position in cell units.
            r_i_cell = (r_i - r_min) / dr + 2

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1

            # If lower r cell is below axis, assume same value as first cell.
            # For `nabla_a2` and `b_theta_0`, invert the sign to ensure they
            # are `0` on axis.
            sign = 1
            if ir_lower < 2:
                ir_lower = 2
                sign = -1

            # Get field value at each bounding cell.
            fld_1_l = fld_1[ir_lower]
            fld_1_u = fld_1[ir_upper]
            fld_2_l = fld_2[ir_lower] * sign
            fld_2_u = fld_2[ir_upper]
            fld_3_l = fld_3[ir_lower] * sign
            fld_3_u = fld_3[ir_upper]

            # Interpolate in r
            dr_u = ir_upper - r_i_cell
            dr_l = 1 - dr_u
            fld_1_pp[i] = dr_u*fld_1_l + dr_l*fld_1_u
            fld_2_pp[i] = dr_u*fld_2_l + dr_l*fld_2_u
            fld_3_pp[i] = dr_u*fld_3_l + dr_l*fld_3_u
        else:
            fld_1_pp[i] = 0.
            fld_2_pp[i] = 0.
            fld_3_pp[i] = fld_3[-3] * r_max / r_i



@njit_serial()
def gather_psi_bg(sum_1_bg_grid, r_min, r_max, dr, r, sum_1_bg):
    """
    Convenient method for gathering at once the three source fields needed
    by the Baxevanis wakefield model (a2 and nabla_a from the laser, and
    b_theta_0 from the beam) into the plasma particles. This method is also
    catered to cylindrical geometry and assumes all plasma particles have the
    same longitudinal position (which is the case in a quasistatic model,
    where there is only a single column of particles).

    Parameters
    ----------
    fld_1, fld_2, fld_3 : ndarray
        The three source fields, corresponding respectively to a2, nabla_a2
        and b_theta_0. Each of them is a (nr+4) array, including 2 guard
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

    Returns
    --------
    A tuple with three 1darray containing the gathered field values at the
    position of each particle.

    """

    # Iterate over all particles.
    for i in range(r.shape[0]):

        # Get particle position.
        r_i = r[i]

        # Gather field only if particle is within field boundaries.
        if r_i <= r_max:
            # Position in cell units.
            r_i_cell = (r_i - r_min)/dr + 2

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1

            # Get field value at each bounding cell.
            sum_1_bg_grid_l = sum_1_bg_grid[ir_lower]
            sum_1_bg_grid_u = sum_1_bg_grid[ir_upper]

            # Interpolate in r.
            dr_u = ir_upper - r_i_cell
            dr_l = 1 - dr_u
            sum_1_bg[i] = dr_u*sum_1_bg_grid_l + dr_l*sum_1_bg_grid_u


@njit_serial()
def gather_dr_psi_bg(sum_2_bg_grid, r_min, r_max, dr, r, sum_2_bg):
    """
    Convenient method for gathering at once the three source fields needed
    by the Baxevanis wakefield model (a2 and nabla_a from the laser, and
    b_theta_0 from the beam) into the plasma particles. This method is also
    catered to cylindrical geometry and assumes all plasma particles have the
    same longitudinal position (which is the case in a quasistatic model,
    where there is only a single column of particles).

    Parameters
    ----------
    fld_1, fld_2, fld_3 : ndarray
        The three source fields, corresponding respectively to a2, nabla_a2
        and b_theta_0. Each of them is a (nr+4) array, including 2 guard
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

    Returns
    --------
    A tuple with three 1darray containing the gathered field values at the
    position of each particle.

    """

    # Iterate over all particles.
    for i in range(r.shape[0]):

        # Get particle position.
        r_i = r[i]

        # Gather field only if particle is within field boundaries.
        if r_i <= r_max:
            # Position in cell units.
            r_i_cell = (r_i - r_min)/dr + 2

            # Indices of upper and lower cells in r and z.
            ir_lower = int(math.floor(r_i_cell))
            ir_upper = ir_lower + 1

            # If lower r cell is below axis, assume same value as first cell.
            # For `nabla_a2` and `b_theta_0`, invert the sign to ensure they
            # are `0` on axis.
            sign = 1
            if ir_lower < 2:
                ir_lower = 2
                sign = -1

            # Get field value at each bounding cell.
            sum_2_bg_grid_l = sum_2_bg_grid[ir_lower] * sign
            sum_2_bg_grid_u = sum_2_bg_grid[ir_upper]

            # Interpolate in r.
            dr_u = ir_upper - r_i_cell
            dr_l = 1 - dr_u
            sum_2_bg[i] = dr_u*sum_2_bg_grid_l + dr_l*sum_2_bg_grid_u
