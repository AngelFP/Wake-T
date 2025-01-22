import math

from wake_t.utilities.numba import njit_serial


@njit_serial(error_model="numpy")
def gather_laser_sources(a2, nabla_a, r_min, r_max, dr, r, a2_pp, nabla_a_pp):
    """
    Gather the laser source terms (a2 and nabla_a) needed
    by the Baxevanis wakefield model into the plasma particles. This method is
    also catered to cylindrical geometry and assumes all plasma particles have
    the same longitudinal position (which is the case in a quasistatic model,
    where there is only a single column of particles).

    Parameters
    ----------
    a2, nabla_a : ndarray
        The source fields, corresponding respectively to a2 and nabla_a2.
        Each of them is a (nr+4) array, including 2 guard
        cells in each boundary.
    r_min, r_max : float
        Position of the first and last field values along r.
    dz : float
        Grid step size along the radial direction.
    r : 1darray
        Transverse position of the plasma particles.
    a2_pp, nabla_a_pp : ndarray
        Arrays where the gathered sources will be stored.

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
            fld_1_l = a2[ir_lower]
            fld_1_u = a2[ir_upper]
            fld_2_l = nabla_a[ir_lower] * sign
            fld_2_u = nabla_a[ir_upper]

            # Interpolate in r
            dr_u = ir_upper - r_i_cell
            dr_l = 1 - dr_u
            a2_pp[i] = dr_u * fld_1_l + dr_l * fld_1_u
            nabla_a_pp[i] = dr_u * fld_2_l + dr_l * fld_2_u
        else:
            a2_pp[i] = 0.0
            nabla_a_pp[i] = 0.0


@njit_serial(error_model="numpy")
def gather_bunch_sources(b_t, r_min, r_max, dr, r, b_t_pp):
    """
    Gathering the beam source terms (B_theta) needed by the Baxevanis
    wakefield model. This method is also
    catered to cylindrical geometry and assumes all plasma particles have the
    same longitudinal position (which is the case in a quasistatic model,
    where there is only a single column of particles).

    Parameters
    ----------
    b_t : ndarray
        The source term corresponding to the azimuthal magnetic field of a
        particle bunch. Array of size (nr+4) array, including 2 guard
        cells in each boundary.
    r_min, r_max : float
        Position of the first and last field values along r.
    dz : float
        Grid step size along the radial direction.
    r : 1darray
        Transverse position of the plasma particles.
    b_t_pp : ndarray
        Array where the gathered source will be stored.

    """

    # Iterate over all particles.
    for i in range(r.shape[0]):

        # Get particle position.
        r_i = r[i]

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

        # Get field at lower and upper cell.
        # If the particle is within the boundaries of the grid, proceed
        # normally. Otherwise, calculate the lower and upper field
        # at "virtual" grid points. In principle this is not needed, because we
        # could avoid interpolation in this case and simply get the
        # field at the location of the particles using b_t[-1] * r_max / r_i.
        # This would be more exact. However, it makes it more difficult to
        # compare the results of simulations using an adaptive grid, which
        # would never be able to reproduce the exact same result as a case
        # with no adaptive grid due to the different (although more accurate)
        # field values.
        if r_i <= r_max:
            # Get field value at each bounding cell.
            fld_l = b_t[ir_lower] * sign
            fld_u = b_t[ir_upper]
        else:
            r_lower = (0.5 + ir_lower - 2) * dr
            r_upper = (0.5 + ir_upper - 2) * dr
            fld_l = b_t[-1] * r_max / r_lower * sign
            fld_u = b_t[-1] * r_max / r_upper

        # Interpolate in r
        dr_u = ir_upper - r_i_cell
        dr_l = 1 - dr_u
        b_t_pp[i] += dr_u * fld_l + dr_l * fld_u
