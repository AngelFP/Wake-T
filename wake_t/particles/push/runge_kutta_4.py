""" Contains the Runge-Kutta pusher of order 4 """

import math
import scipy.constants as ct
from numba import prange

from wake_t.utilities.numba import njit_parallel
from wake_t.fields.gather import gather_fields


def apply_rk4_pusher(bunch, fields, t, dt):
    """Evolve a particle bunch using the RK4 pusher.

    For a variable evolving as:

        dx/dt = f(t, x) ,

    the pusher calculates its value at t = t_n + dt as:

        x += dt * (k1 + 2*k_2 + 2*k_3 + k4) / 6 ,

    where the k_i coefficients are given by:

        k_1 = f(t_n, x_n) ,
        k_2 = f(t_n + dt/2, x_n + dt*k_1/2) ,
        k_3 = f(t_n + dt/2, x_n + dt*k_2/2) ,
        k_4 = f(t_n + dt, x_n + dt*k_3) .

    In order to be more efficient, only a single 1D k_i array per variable is
    allocated. Once k_i is calculated, its contribution to the push is added
    (i.e., x_push += k_i * fac, where fac can be 1/6 or 2/6). The same
    array can then be overwritten with the values of k_i+1, whose contribution
    is also added to x_push. Once the contribution of all four k_i has been
    added, the push is applied (i.e., x += dt * x_push).

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
    # Get the necessary preallocated arrays.
    (x, y, xi, px, py, pz, dx, dy, dxi, dpx, dpy, dpz,
     k_x, k_y, k_xi, k_px, k_py, k_pz) = bunch.get_rk4_arrays()
    ex, ey, ez, bx, by, bz = bunch.get_field_arrays()

    # Particle species constant (currently assumes electrons).
    q_over_mc = bunch.q_species / (bunch.m_species * ct.c)

    # Calculate push.
    for i in range(4):
        t_i = t

        # Calculate factors.
        if i in [0, 3]:
            fac1 = 1.
            fac2 = 1. / 6.
            if i == 3:
                t_i += dt
        else:
            fac1 = 0.5
            fac2 = 2. / 6.
            t_i += dt / 2

        # Calculate contributions the push.
        if i == 0:
            # Gather field at the initial location of the particles.
            gather_fields(fields, bunch.x, bunch.y, bunch.xi, t_i,
                          ex, ey, ez, bx, by, bz)

            # Calculate k_1.
            calculate_k(k_x, k_y, k_xi, k_px, k_py, k_pz,
                        q_over_mc, bunch.px, bunch.py, bunch.pz,
                        ex, ey, ez, bx, by, bz)

            # Initialize push with the contribution from k1.
            initialize_push(dx, k_x, fac2)
            initialize_push(dy, k_y, fac2)
            initialize_push(dxi, k_xi, fac2)
            initialize_push(dpx, k_px, fac2)
            initialize_push(dpy, k_py, fac2)
            initialize_push(dpz, k_pz, fac2)
        else:
            # Update particle coordinates to x = x_n + k_i * fac1
            update_coord(x, bunch.x, dt, k_x, fac1)
            update_coord(y, bunch.y, dt, k_y, fac1)
            update_coord(xi, bunch.xi, dt, k_xi, fac1)
            update_coord(px, bunch.px, dt, k_px, fac1)
            update_coord(py, bunch.py, dt, k_py, fac1)
            update_coord(pz, bunch.pz, dt, k_pz, fac1)

            # Gather field at updated positions.
            gather_fields(fields, x, y, xi, t_i, ex, ey, ez, bx, by, bz)

            # Calculate k_i.
            calculate_k(k_x, k_y, k_xi, k_px, k_py, k_pz,
                        q_over_mc, px, py, pz, ex, ey, ez, bx, by, bz)

            # Add the contribution of k_i to the push.
            update_push(dx, k_x, fac2)
            update_push(dy, k_y, fac2)
            update_push(dxi, k_xi, fac2)
            update_push(dpx, k_px, fac2)
            update_push(dpy, k_py, fac2)
            update_push(dpz, k_pz, fac2)

    # Apply push.
    apply_push(bunch.x, dt, dx)
    apply_push(bunch.y, dt, dy)
    apply_push(bunch.xi, dt, dxi)
    apply_push(bunch.px, dt, dpx)
    apply_push(bunch.py, dt, dpy)
    apply_push(bunch.pz, dt, dpz)


@njit_parallel()
def initialize_coord(x, x_0):
    for i in prange(x.shape[0]):
        x[i] = x_0[i]


@njit_parallel()
def update_coord(x, x_0, dt, k_x, fac):
    for i in prange(x.shape[0]):
        x[i] = x_0[i] + dt * k_x[i] * fac


@njit_parallel()
def initialize_push(dx, k_x, fac):
    for i in prange(dx.shape[0]):
        dx[i] = k_x[i] * fac


@njit_parallel()
def update_push(dx, k_x, fac):
    for i in prange(dx.shape[0]):
        dx[i] += k_x[i] * fac


@njit_parallel()
def apply_push(x, dt, dx):
    for i in prange(x.shape[0]):
        x[i] += dt * dx[i]


@njit_parallel(fastmath=True, error_model='numpy')
def calculate_k(k_x, k_y, k_xi, k_px, k_py, k_pz,
                q_over_mc, px, py, pz, ex, ey, ez, bx, by, bz):
    for i in prange(k_x.shape[0]):
        px_i = px[i]
        py_i = py[i]
        pz_i = pz[i]
        ex_i = ex[i]
        ey_i = ey[i]
        ez_i = ez[i]
        bx_i = bx[i]
        by_i = by[i]
        bz_i = bz[i]

        c_over_gamma_i = ct.c / math.sqrt(1 + px_i**2 + py_i**2 + pz_i**2)
        vx_i = px_i * c_over_gamma_i
        vy_i = py_i * c_over_gamma_i
        vz_i = pz_i * c_over_gamma_i

        k_x[i] = vx_i
        k_y[i] = vy_i
        k_xi[i] = (vz_i - ct.c)
        k_px[i] = q_over_mc * (ex_i + vy_i * bz_i - vz_i * by_i)
        k_py[i] = q_over_mc * (ey_i - vx_i * bz_i + vz_i * bx_i)
        k_pz[i] = q_over_mc * (ez_i + vx_i * by_i - vy_i * bx_i)
