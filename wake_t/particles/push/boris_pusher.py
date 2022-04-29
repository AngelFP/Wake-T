""" Contains the Boris pusher """

from numba import njit


def apply_boris_pusher(bunch, field, dt):
    ex, ey, ez, bx, by, bz = bunch.get_field_arrays()

    apply_half_position_push(
        bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz, dt)

    field.gather(bunch.x, bunch.y, bunch.xi, ex, ey, ez, bx, by, bz)

    push_momentum(bunch.px, bunch.py, bunch.pz, ex, ey, ez, bx, by, bz, dt)

    apply_half_position_push(
        bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz, dt)


@njit()
def apply_half_position_push(x, y, xi, px, py, pz, dt):
    """
    for i in range(x.shape[0]):
        # Get particle momentum
        px_i = px[i]
        py_i = py[i]
        pz_i = pz[i]

        ...

        # Update particle position
        x[i] = ...
        y[i] = ...
        xi[i] = ..
    """
    raise NotImplementedError()


@njit()
def push_momentum(px, py, pz, ex, ey, ez, bx, by, bz, dt):
    """
    for i in range(px.shape[0]):
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

        ...

        # Update particle momentum.
        px[i] = ...
        py[i] = ...
        pz[i] = ...
    """
    raise NotImplementedError()
