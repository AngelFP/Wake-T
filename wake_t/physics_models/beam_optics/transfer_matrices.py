import numpy as np


def first_order_matrix(z, L, theta, k1, gamma_ref):
    """
    Calculate the first order matrix for the transfer map.
    This function is an adaptation of the one found in the particle tracking
    code Ocelot (see https://github.com/ocelot-collab/ocelot) written by
    S. Tomin.

    Parameters:
    -----------
    z : float
        Longitudinal position in which to calculate the transfer matrix

    L : float
        Total length of the beamline element

    theta : float
        Bending angle of the beamline element

    k1 : float
        Quadrupole gradient of the beamline element in units of 1/m^2. A
        positive value implies focusing on the 'x' plane, while a negative
        gradient corresponds to focusing on 'y'

    k2 : float
        Sextupole gradient of the beamline element in units of 1/m^3. A
        positive value implies focusing on the 'x' plane, while a negative
        gradient corresponds to focusing on 'y'

    gamma_ref : float
        Reference energy with respect to which the particle momentum dp is
        calculated.

    """
    gamma = gamma_ref
    hx = theta/L
    kx2 = (k1 + hx*hx)
    ky2 = -k1
    kx = np.sqrt(kx2 + 0.j)
    ky = np.sqrt(ky2 + 0.j)
    cx = np.cos(z*kx).real
    cy = np.cos(z*ky).real
    sy = (np.sin(ky*z)/ky).real if ky != 0 else z
    igamma2 = 0.
    if gamma != 0:
        igamma2 = 1./(gamma*gamma)
    beta = np.sqrt(1. - igamma2)
    if kx != 0:
        sx = (np.sin(kx*z)/kx).real
        dx = hx/kx2*(1. - cx)
        r56 = hx*hx*(z - sx)/kx2/beta**2
    else:
        sx = z
        dx = z*z*hx/2.
        r56 = hx*hx*z**3/6./beta**2
    r56 -= z/(beta*beta)*igamma2
    u_matrix = np.array([[cx, sx, 0., 0., 0., dx/beta],
                         [-kx2*sx, cx, 0., 0., 0., sx*hx/beta],
                         [0., 0., cy, sy, 0., 0.],
                         [0., 0., -ky2*sy, cy, 0., 0.],
                         [hx*sx/beta, dx/beta, 0., 0., 1., r56],
                         [0., 0., 0., 0., 0., 1.]])
    return u_matrix


def second_order_matrix(z, L, theta, k1, k2, gamma_ref):
    """
    Calculate the second order matrix for the transfer map.
    This function is an adaptation of the one found in the particle tracking
    code Ocelot (see https://github.com/ocelot-collab/ocelot) written by
    S. Tomin.

    Parameters:
    -----------
    beam_matrix : array
        6 x N matrix, where N is the number of particles, containing the
        phase-space information of the bunch as (x, x', y, y', xi, dp) in
        units of (m, rad, m, rad, m, -). dp is defined as
        dp = (g-g_ref)/g_ref, while x' = px/p_kin and y' = py/p_kin, where
        p_kin is the reference kinetic momentum.

    z : float
        Longitudinal position in which to obtain the bunch distribution

    L : float
        Total length of the beamline element

    theta : float
        Bending angle of the beamline element

    k1 : float
        Quadrupole gradient of the beamline element in units of 1/m^2. A
        positive value implies focusing on the 'x' plane, while a negative
        gradient corresponds to focusing on 'y'

    k2 : float
        Sextupole gradient of the beamline element in units of 1/m^3. A
        positive value implies focusing on the 'x' plane, while a negative
        gradient corresponds to focusing on 'y'

    gamma_ref : float
        Reference energy with respect to which the particle momentum dp is
        calculated.

    """
    igamma2 = 0.
    if gamma_ref != 0:
        gamma = gamma_ref
        gamma2 = gamma*gamma
        igamma2 = 1./gamma2

    beta = np.sqrt(1. - igamma2)
    beta2 = beta * beta
    beta3 = beta2*beta
    h = theta/L
    L = z  # Easy fix for the calculation below, where L should actually be z.
    h2 = h*h
    h3 = h2*h
    kx2 = (k1 + h*h)
    ky2 = -k1
    kx4 = kx2*kx2
    kx = np.sqrt(kx2 + 0.j)
    ky = np.sqrt(-k1 + 0.j)
    cx = np.cos(kx*L).real
    sx = (np.sin(kx*L)/kx).real if kx != 0 else L
    cy = np.cos(ky*L).real

    sy = (np.sin(ky*L)/ky).real if ky != 0 else L

    sx2 = sx*sx
    sy2 = sy*sy
    L2 = L*L
    L3 = L2*L
    L4 = L3*L
    L5 = L4*L
    dx = (1. - cx)/kx2 if kx != 0. else L*L/2.
    dx2 = dx * dx
    K1 = k1
    K2 = k2

    d2y = 0.5 * sy * sy
    s2y = sy * cy
    c2y = np.cos(2 * ky * L).real
    fx = (L - sx) / kx2 if kx2 != 0 else L3 / 6.
    f2y = (L - s2y) / ky2 if ky2 != 0 else L3 / 6
    J1 = (L - sx) / kx2 if kx2 != 0 else L3 / 6.
    J2 = (3.*L - 4.*sx + sx*cx) / (2*kx4) if kx != 0 else L ** 5 / 20.
    J3 = ((15*L - 22.5*sx + 9*sx*cx - 1.5*sx*cx*cx + kx2*sx2*sx) /
          (6 * kx4 * kx2)) if kx != 0 else L ** 7 / 56.
    # J3 = (15 * L - 22 * sx + 9 * sx * cx - 2 * sx * cx * cx ) / (
    #        6 * kx4 * kx2) if kx != 0 else L ** 7 / 56.
    J_denom = kx2 - 4 * ky2
    Jc = (c2y - cx) / J_denom if J_denom != 0 else 0.5 * L2
    Js = (cy * sy - sx) / J_denom if J_denom != 0 else L3 / 6
    Jd = (d2y - dx) / J_denom if J_denom != 0 else L4 / 24
    Jf = (f2y - fx) / J_denom if J_denom != 0 else L5 / 120

    khk = K2 + 2*h*K1
    T = np.zeros((6, 6, 6))

    T111 = -1/6*khk*(sx2 + dx) - 0.5*h*kx2*sx2
    T112 = -1/6*khk*sx*dx + 0.5*h*sx*cx
    T122 = -1 / 6 * khk * dx2 + 0.5 * h * dx * cx
    T116 = -h/12/beta*khk*(3*sx*J1 - dx2) + 0.5*h2/beta*sx2 + 0.25/beta*K1*L*sx
    T126 = (-h/12/beta*khk*(sx*dx2 - 2*cx*J2) + 0.25*h2/beta*(sx*dx + cx*J1) -
            0.25/beta*(sx + L*cx))
    T166 = (-h2/6/beta2*khk*(dx2*dx - 2*sx*J2) + 0.5*h3/beta2*sx*J1
            - 0.5*h/beta2*L*sx - 0.5*h/(beta2)*igamma2*dx)
    T133 = K1*K2*Jd + 0.5*(K2 + h * K1)*dx
    T134 = 0.5*K2*Js
    T144 = K2*Jd - 0.5*h*dx

    T211 = -1/6*khk*sx*(1 + 2*cx)
    T212 = -1 / 6 * khk * dx * (1 + 2 * cx)
    T222 = -1 / 3 * khk * sx*dx - 0.5*h*sx
    T216 = -h/12/beta*khk*(3*cx * J1 + sx*dx) - 0.25/beta*K1*(sx - L*cx)
    T226 = -h / 12/beta * khk * (3*sx * J1 + dx2) + 0.25/beta * K1 * L*sx
    T266 = (-h2/6/beta2*khk*(sx*dx2 - 2*cx*J2) - 0.5*h/beta2*K1*(cx*J1 - sx*dx)
            - 0.5*h/beta2*igamma2*sx)
    T233 = K1*K2*Js + 0.5*(K2 + h*K1)*sx
    T234 = 0.5*K2*Jc
    T244 = K2 * Js - 0.5*h*sx

    T313 = 0.5*K2*(cy*Jc - 2*K1*sy*Js) + 0.5*h*K1*sx*sy
    T314 = 0.5 * K2 * (sy * Jc - 2 * cy * Js) + 0.5 * h * sx * cy
    T323 = 0.5 * K2 * (cy * Js - 2 * K1 * sy * Jd) + 0.5 * h * K1 * dx * sy
    T324 = 0.5 * K2 * (sy * Js - 2 * cy * Jd) + 0.5 * h * dx * cy
    T336 = 0.5 * h/beta * K2 * \
        (cy * Jd - 2 * K1 * sy * Jf) + 0.5 * h2 / \
        beta * K1 * J1 * sy - 0.25/beta*K1*L*sy
    T346 = 0.5 * h / beta * K2 * \
        (sy * Jd - 2 * cy * Jf) + 0.5 * h2 / \
        beta * J1 * cy - 0.25 / beta * (sy + L * cy)

    T413 = 0.5*K1*K2*(2*cy*Js - sy*Jc) + 0.5*(K2 + h*K1)*sx*cy
    T414 = 0.5 * K2 * (2 * K1*sy * Js - cy * Jc) + \
        0.5 * (K2 + h * K1) * sx * sy
    T423 = 0.5*K1*K2*(2*cy*Jd - sy*Js) + 0.5*(K2 + h*K1)*dx*cy
    T424 = 0.5 * K2 * (2 * K1 * sy * Jd - cy * Js) + \
        0.5 * (K2 + h * K1) * dx * sy
    T436 = 0.5 * h/beta * K1 * K2 * \
        (2 * cy * Jf - sy * Jd) + 0.5 * h/beta * \
        (K2 + h * K1) * J1 * cy + 0.25/beta*K1*(sy - L*cy)
    T446 = 0.5 * h/beta * K2 * (2 * K1 * sy * Jf - cy * Jd) + \
        0.5 * h/beta * (K2 + h * K1) * J1 * sy - 0.25/beta*K1*L*sy

    T511 = h/12/beta*khk*(sx*dx + 3*J1) - 0.25/beta*K1*(L - sx*cx)
    T512 = h / 12 / beta * khk * dx2 + 0.25 / beta * K1 * sx2
    T522 = h/6/beta*khk*J2 - 0.5/beta*sx - 0.25/beta*K1*(J1 - sx*dx)

    T516 = h2/12/beta2*khk*(3*dx*J1 - 4*J2) + 0.25*h / \
        beta2*K1*J1*(1 + cx) + 0.5*h/beta2*igamma2*sx

    T526 = h2 / 12 / beta2 * khk * \
        (dx * dx2 - 2 * sx * J2) + 0.25 * h / beta2 * \
        K1*sx * J1 + 0.5 * h / beta2 * igamma2 * dx
    T566 = h3 / 6 / beta3 * khk * (3*J3 - 2 * dx * J2) + h2/6 / beta3 * K1*(
        sx*dx2 - J2*(1 + 2*cx)) + 1.5 / beta3 * igamma2 * (h2*J1 - L)
    T533 = - h/beta*K1*K2*Jf - 0.5*h/beta * \
        (K2 + h*K1)*J1 + 0.25/beta*K1*(L - cy*sy)
    T534 = - 0.5*h/beta*K2*Jd - 0.25/beta*K1*sy2
    T544 = -h/beta * K2*Jf + 0.5*h2/beta*J1 - 0.25/beta*(L + cy*sy)

    T[0, 0, 0] = T111
    T[0, 0, 1] = T112*2
    T[0, 1, 1] = T122
    T[0, 0, 5] = T116*2
    T[0, 1, 5] = T126*2
    T[0, 5, 5] = T166
    T[0, 2, 2] = T133
    T[0, 2, 3] = T134*2
    T[0, 3, 3] = T144

    T[1, 0, 0] = T211
    T[1, 0, 1] = T212*2
    T[1, 1, 1] = T222
    T[1, 0, 5] = T216*2
    T[1, 1, 5] = T226*2
    T[1, 5, 5] = T266
    T[1, 2, 2] = T233
    T[1, 2, 3] = T234*2
    T[1, 3, 3] = T244

    T[2, 0, 2] = T313*2
    T[2, 0, 3] = T314*2
    T[2, 1, 2] = T323*2
    T[2, 1, 3] = T324*2
    T[2, 2, 5] = T336*2
    T[2, 3, 5] = T346*2

    T[3, 0, 2] = T413*2
    T[3, 0, 3] = T414*2
    T[3, 1, 2] = T423*2
    T[3, 1, 3] = T424*2
    T[3, 2, 5] = T436*2
    T[3, 3, 5] = T446*2

    T[4, 0, 0] = -T511
    T[4, 0, 1] = -T512*2
    T[4, 1, 1] = -T522
    T[4, 0, 5] = -T516*2
    T[4, 1, 5] = -T526*2
    T[4, 5, 5] = -T566
    T[4, 2, 2] = -T533
    T[4, 2, 3] = -T534*2
    T[4, 3, 3] = -T544
    return T
