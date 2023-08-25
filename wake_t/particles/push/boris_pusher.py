"""
Contains the Boris pusher

Authors: Jorge Ordóñez Carrasco, Ángel Ferran Pousa.
"""
import numpy as np
import scipy.constants as ct
from numba import prange

from wake_t.utilities.numba import njit_parallel
from wake_t.fields.gather import gather_fields


def apply_boris_pusher(bunch, fields, t, dt):
    """Evolve a particle bunch using the Boris pusher.

    Parameters
    ----------
    bunch : ParticleBunch
        The particle bunch to be evolved.
    fields : list
        List of fields within which the particle bunch will be evolved.
    t : float
        The current time.
    dt : float
        Time step by which to push the particles.
    """
    # Calculate particle species constant.
    q_over_mc = bunch.q_species / (bunch.m_species * ct.c)
    # Get the necessary arrays where the fields  will be gathered.
    ex, ey, ez, bx, by, bz = bunch.get_field_arrays()
    # Advance the particles half of one time steps.
    apply_half_position_push(
        bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz, dt)
    # Gather fields at this position.
    gather_fields(fields, bunch.x, bunch.y, bunch.xi, t+dt/2,
                  ex, ey, ez, bx, by, bz)
    # Advances the momentum one time step using the gathered fields.
    push_momentum(bunch.px, bunch.py, bunch.pz, ex, ey, ez, bx, by, bz, dt,
                  q_over_mc)
    # Completes the particles push using the updated momentum.
    apply_half_position_push(
        bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz, dt)


@njit_parallel()
def apply_half_position_push(x, y, xi, px, py, pz, dt):
    for i in prange(x.shape[0]):
        # Get particle momentum
        px_i = px[i]
        py_i = py[i]
        pz_i = pz[i]

        c_over_gamma_i = ct.c / np.sqrt(1 + (px_i**2 + py_i**2 + pz_i**2))

        # Update particle position
        x[i] += 0.5 * px_i * dt * c_over_gamma_i
        y[i] += 0.5 * py_i * dt * c_over_gamma_i
        xi[i] += 0.5 * (pz_i * c_over_gamma_i - ct.c) * dt


@njit_parallel()
def push_momentum(px, py, pz, ex, ey, ez, bx, by, bz, dt, q_over_mc):
    k = q_over_mc * dt / 2

    for i in prange(px.shape[0]):
        # Get particle momentum and fields.
        px_i = px[i]
        py_i = py[i]
        pz_i = pz[i]
        ex_i = ex[i]
        ey_i = ey[i]
        ez_i = ez[i]
        bx_i = bx[i]
        by_i = by[i]
        bz_i = bz[i]

        p_minus_x = px_i + k * ex_i
        p_minus_y = py_i + k * ey_i
        p_minus_z = pz_i + k * ez_i
        c_over_gamma_med = ct.c / \
            np.sqrt(1 + (p_minus_x**2 + p_minus_y**2 + p_minus_z**2))
        t_x = k * c_over_gamma_med * bx_i
        t_y = k * c_over_gamma_med * by_i
        t_z = k * c_over_gamma_med * bz_i
        cons_s = 2/(1 + t_x**2 + t_y**2 + t_z**2)
        s_x = cons_s * t_x
        s_y = cons_s * t_y
        s_z = cons_s * t_z

        # Calculate first cross product
        p_xc1 = p_minus_x + p_minus_y*t_z - p_minus_z*t_y
        p_yc1 = p_minus_y + p_minus_z*t_x - p_minus_x*t_z
        p_zc1 = p_minus_z + p_minus_x*t_y - p_minus_y*t_x

        # Update particle momentum.
        px[i] = p_minus_x + p_yc1*s_z - p_zc1*s_y + k*ex_i
        py[i] = p_minus_y + p_zc1*s_x - p_xc1*s_z + k*ey_i
        pz[i] = p_minus_z + p_xc1*s_y - p_yc1*s_x + k*ez_i
