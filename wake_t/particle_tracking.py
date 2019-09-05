""" This module contains the numerical trackers and equations of motion """

import numpy as np
import scipy.constants as ct
from numba import jit, njit


def runge_kutta_4(beam_matrix, WF, t0, dt, iterations):
    for i in np.arange(iterations):
        t = t0 + i*dt
        A = dt*equations_of_motion(beam_matrix, t, WF)
        B = dt*equations_of_motion(beam_matrix + A/2, t+dt/2, WF)
        C = dt*equations_of_motion(beam_matrix + B/2, t+dt/2, WF)
        D = dt*equations_of_motion(beam_matrix + C, t+dt, WF)
        beam_matrix += 1/6*(A + 2*B + 2*C + D)
    return beam_matrix

def equations_of_motion(beam_matrix, t, WF):
    K = -ct.e/(ct.m_e*ct.c)
    x, px, y, py, xi, pz, q = beam_matrix
    gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    return np.array([px*ct.c/gamma,
                     K*WF.Wx(x, y, xi, px, py, pz, q, gamma, t),
                     py*ct.c/gamma,
                     K*WF.Wy(x, y, xi, px, py, pz, q, gamma, t),
                     (pz/gamma-1)*ct.c,
                     K*WF.Wz(x, y, xi, px, py, pz, q, gamma, t),
                     np.zeros_like(q)])

def track_with_transfer_map(beam_matrix, z, L, theta, k1, k2, gamma_ref,
                            order=2):
    """
    Track beam distribution throwgh beamline element by using a transfer map.
    This function is stronly based on code from Ocelot (see 
    https://github.com/ocelot-collab/ocelot) written by S. Tomin.

    Parameters:
    -----------
    beam_matrix : array
        6 x N matrix, where N is the number of particles, containing the
        phase-space information of the bunch as (x, x', y, y', xi, dp) in
        units of (m, rad, m, rad, m, -). dp is defined as
        dp = (g-g_ref)/g_ref, while x' = px/p_kin and y' = py/p_kin, where
        p_kin is the kinetic momentum of each particle.

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

    order : int
        Indicates the order of the transport map to apply. Tracking up to
        second order is possible.

    """
    R = first_order_matrix(z, L, theta, k1, gamma_ref)
    new_beam_matrix = np.dot(R, beam_matrix)
    if order == 2:
        T = second_order_matrix(z, L, theta, k1, k2, gamma_ref)
        x, xp, y, yp, xi, dp = (beam_matrix[0], beam_matrix[1], beam_matrix[2],
                                beam_matrix[3], beam_matrix[4], beam_matrix[5])
        # pre-calculate products
        x2 = x * x
        xxp = x * xp
        xp2 = xp * xp
        yp2 = yp * yp
        yyp = y * yp
        y2 = y * y
        dp2 = dp * dp
        xdp = x * dp
        xpdp = xp * dp
        xy = x * y
        xyp = x * yp
        yxp = xp * y
        xpyp = xp * yp
        ydp = y * dp
        ypdp = yp * dp
        # Add second order effects
        new_beam_matrix[0] += (T[0,0,0]*x2 + T[0,0,1]*xxp + T[0,0,5]*xdp
                               + T[0,1,1]*xp2 + T[0,1,5]*xpdp + T[0,5,5]*dp2
                               + T[0,2,2]*y2 + T[0,2,3]*yyp + T[0,3,3]*yp2)
        new_beam_matrix[1] += (T[1,0,0]*x2 + T[1,0,1]*xxp + T[1,0,5]*xdp
                               + T[1,1,1]*xp2 + T[1,1,5]*xpdp + T[1,5,5]*dp2
                               + T[1,2,2]*y2 + T[1,2,3]*yyp + T[1,3,3]*yp2)
        new_beam_matrix[2] += (T[2,0,2]*xy + T[2,0,3]*xyp + T[2,1,2]*yxp
                               + T[2,1,3]*xpyp + T[2,2,5]*ydp + T[2,3,5]*ypdp)
        new_beam_matrix[3] += (T[3,0,2]*xy + T[3,0,3]*xyp + T[3,1,2]*yxp
                               + T[3,1,3]*xpyp + T[3,2,5]*ydp + T[3,3,5]*ypdp)
        new_beam_matrix[4] -= (T[4,0,0]*x2 + T[4,0,1]*xxp + T[4,0,5]*xdp
                                + T[4,1,1]*xp2 + T[4,1,5]*xpdp + T[4,5,5]*dp2
                                + T[4,2,2]*y2 + T[4,2,3]*yyp + T[4,3,3]*yp2)
    return new_beam_matrix

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
                         [-hx*sx/beta, -dx/beta, 0., 0., 1., -r56],
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
        p_kin is the kinetic momentum of each particle.

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
    L=z # TODO: test this.
    h = theta/L
    h2 = h*h
    h3 = h2*h
    kx2 = (k1 + h*h)
    ky2 = -k1
    kx4 = kx2*kx2
    ky4 = ky2*ky2
    kx = np.sqrt(kx2 + 0.j)
    ky = np.sqrt(ky2 + 0.j)
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
    dx = h/kx2*(1. - cx) if kx != 0. else L*L*h/2.
    dx_h = (1. - cx)/kx2 if kx != 0. else L*L/2.

    # Integrals
    denom = kx2 - 4.*ky2
    I111 = 1./3.*(sx2 + dx_h)
    I122 = dx_h*dx_h/3.
    I112 = sx*dx_h/3.
    I11  = L*sx/2.
    I10  = dx_h
    I33  = L*sy/2.
    I34  = (sy - L*cy)/(2.*ky2) if ky !=0. else L3/6.
    I211 = sx/3.*(1. + 2.*cx)
    I222 = 2.*dx_h*sx/3.
    I212 = 1./3.*(2*sx2 - dx_h)
    I21  = 1./2.*(L*cx + sx)
    I22  = I11
    I20  = sx
    I43  = 0.5*(L*cy + sy)
    I44  = I33
    I512 = h*dx_h*dx_h/6
    I51  = L*dx/2.

    if kx != 0:
        I116 = h/kx2*(I11 - I111)
        I12  = 0.5/kx2*(sx - L*cx)
        I126 = h/kx2*(I12 - I112)
        I16  = h/kx2*(dx_h - L*sx/2.)
        I166 = h2/kx4*(I10 - 2*I11 + I111)
        I216 = h/kx2*(I21 - I211)
        I226 = h/kx2*(I22 - I212)
        I26  = h /(2.*kx2)*(sx - L*cx)
        I266 = h2/kx4*(I20 - 2.*I21 + I211)

        I511 = h*(3.*L - 2.*sx - sx*cx)/(6.*kx2)
        I522 = h*(3.*L - 4*sx + sx*cx)/(6.*kx4)
        I516 = h/kx2*(I51 - I511)
        I52  =  (2.*dx - h*L*sx)/(2.*kx2)
        I526 = h/kx2*(I52 - I512)
        I50  = h*(L - sx)/kx2
        I566 = h2/kx4*(I50 - 2*I51 + I511)
        I56  = (h2*(L*(1. + cx) - 2.*sx))/(2.*kx4)

    else:
        I116 = h*L4/24.
        I12  = L3/6.
        I126 = h*L5/40.
        I16  = h*L4/24.
        I166 = h2*L5*L/120.
        I216 = h*L3/6.
        I226 = h*L4/8.
        I26  = h*L3/6.
        I266 = h2*L5/20.

        I511 = h*L3/6.
        I522 = h*L5/60.
        I516 = h2*L5/120.
        I52  = h*L4/24.
        I526 = h2*L5*L/240.
        I50  = h*L3/6.
        I566 = h2*h*L5*L2/840.
        I56  = h2*L5/120.

    if kx != 0 and ky != 0:
        I144 = (sy2 - 2.*dx_h)/denom                         
        I133 = dx_h - ky2*(sy2 - 2.*dx_h)/denom              
        I134 = (sy*cy - sx)/denom                            
        I313 = (kx2*cy*dx_h - 2.*ky2*sx*sy)/denom            
        I324 = (2.*cy*dx_h - sx*sy)/denom                    
        I314 = (2.*cy*sx - (1. + cx)*sy)/denom               
        I323 = (sy - cy*sx - 2.*ky2*sy*dx_h)/denom           
        #derivative of Integrals
        I244 = 2.*(cy*sy - sx)/denom
        I233 = sx - 2.*ky2*(cy*sy - sx)/denom
        I234 = (kx2*dx_h - 2.*ky2*sy2)/denom
        I413 = ((kx2 - 2.*ky2)*cy*sx - ky2*sy*(1. + cx))/denom
        I424 = (cy*sx - cx*sy - 2.*ky2*sy*dx_h)/denom
        I414 = ((kx2 - 2.*ky2)*sx*sy - (1. - cx)*cy)/denom
        I423 = (cy*dx_h*(kx2 - 2*ky2) - ky2*sx*sy)/denom

    elif kx != 0 and ky == 0:
        I323 = (L - sx)/kx2
        I324 = 2.*(1. - cx)/kx4 - L*sx/kx2
        I314 = (2.*sx - L*(1. + cx))/kx2
        I313 = (1. - cx)/kx2
        I144 = (-2. + kx2*L2 + 2.*cx)/kx4
        I133 = (1. - cx)/kx2
        I134 = (L - sx)/kx2
        # derivative of Integrals
        I423 = (1. - cx)/kx2
        I424 = (sx - L*cx)/kx2
        I414 = (cx - 1.)/kx2 + L*sx
        I413 = sx
        I244 = (2.*L - 2.*sx)/kx2
        I233 = sx
        I234 = (1. - cx)/kx2
    else:
        I144 = L4/12.                                          
        I133 = L2/2.                                           
        I134 = L3/6.                                           
        I313 = L2/2.                                           
        I324 = L4/12.                                          
        I314 = L3/6.                                           
        I323 = L3/6.                                           
        I244 = L3/3.
        I233 = L
        I234 = L2/2.
        I413 = L
        I424 = L3/3.
        I414 = L2/2.
        I423 = L2/2.

    if kx == 0 and ky != 0:
        I336 = (h*L*(3.*L*cy + (2.*ky2*L2 - 3.)*sy))/(24.*ky2)
        I346 = (h*((3. - 2.*ky2*L2)*L*cy + 3.*(ky2*L2 - 1.)*sy))/(24.*ky4)
        I436 = I346
        I446 = (h*L*(-3.*L*cy + (3. + 2.*ky2*L2)*sy))/(24.*ky2)

        I533 = (h*(3.*L + 2.*ky2*L3 - 3.*sy*cy))/(24.*ky2)
        I534 = (h*(L2 - sy2))/(8.*ky2)
        I544 = (h*(-3.*L + 2.*ky2*L3 + 3.*sy*cy))/(24.*ky4)

    elif kx == 0 and ky == 0:
        I336 = (h*L4)/24.
        I346 = (h*L5)/40.
        I436 = (h*L3)/6.
        I446 = (h*L4)/8.

        I533 = h*L3/6.
        I534 = h*L4/24.
        I544 = h*L5/60.

    else:
        I336 = h/kx2*(I33 - I313)                                  
        I346 = h/kx2*(I34 - I314)                                  
        I436 = h/kx2*(I43 - I413)
        I446 = h/kx2*(I44 - I414)

        I533 = ((h*(denom*L - 2.*(denom + 2.*ky2)*sx + kx2*cy*sy))
                /(2.*denom*kx2))
        I534 = (h*sy2 - 2*dx)/(2*denom)
        I544 = (sy2 - 2*dx_h)/denom

    K2 = k2/2.
    coef1 = 2.*ky2*h - h3 - K2
    coef3 = 2.*h2 - ky2

    t111 =    coef1*I111 + h*kx4*I122/2.
    t112 = 2.*coef1*I112 - h*kx2*I112
    t116 = 2.*coef1*I116 + coef3*I11 - h2*kx2*I122
    t122 =    coef1*I122 + 0.5*h*I111
    t126 = 2.*coef1*I126 + coef3*I12 + h2*I112
    t166 =    coef1*I166 + coef3*I16 + 0.5*h3*I122 - h*I10
    t133 =       K2*I133 - ky2*h*I10/2.
    t134 =    2.*K2*I134
    t144 =       K2*I144 - h*I10/2.

    t211 =    coef1*I211 + h*kx4*I222/2.
    t212 = 2.*coef1*I212 - h*kx2*I212
    t216 = 2.*coef1*I216 + coef3*I21 - h2*kx2*I222
    t222 =    coef1*I222 + 0.5*h*I211
    t226 = 2.*coef1*I226 + coef3*I22 + h2*I212
    t266 =    coef1*I266 + coef3*I26 + 0.5*h3*I222 - h*I20
    t233 =       K2*I233 - ky2*h*I20/2.
    t234 =    2.*K2*I234
    t244 =       K2*I244 - h*I20/2.

    coef2 = 2*(K2 - ky2*h)

    t313 = coef2*I313 + h*kx2*ky2*I324
    t314 = coef2*I314 - h*kx2*I323
    t323 = coef2*I323 - h*ky2*I314
    t324 = coef2*I324 + h*I313
    t336 = coef2*I336 + ky2*I33 - h2*ky2*I324
    t346 = coef2*I346 + h2*I323 + ky2*I34
    t413 = coef2*I413 + h*kx2*ky2*I424
    t414 = coef2*I414 - h*kx2*I423
    t423 = coef2*I423 - h*ky2*I414
    t424 = coef2*I424 + h*I413
    t436 = coef2*I436 - h2*ky2*I424 + ky2*I43
    t446 = coef2*I446 + h2*I423 + ky2*I44
    # Coordinates transformation from Curvilinear to a Cartesian
    cx_1 = -kx2*sx
    sx_1 = cx
    cy_1 = -ky2*sy
    sy_1 = cy
    dx_1 = h*sx
    T = np.zeros((6, 6, 6))
    T[0, 0, 0] = t111
    T[0, 0, 1] = t112 + h*sx
    T[0, 0, 5] = t116
    T[0, 1, 1] = t122
    T[0, 1, 5] = t126/beta
    T[0, 5, 5] = t166
    T[0, 2, 2] = t133
    T[0, 2, 3] = t134
    T[0, 3, 3] = t144

    T[1, 0, 0] = t211 - h*cx*cx_1
    T[1, 0, 1] = t212 + h*sx_1 - h*(sx*cx_1 + cx*sx_1)
    T[1, 0, 5] = t216 - h*(dx*cx_1 + cx*dx_1)
    T[1, 1, 1] = t222 - h*sx*sx_1
    T[1, 1, 5] = t226 - h*(sx*dx_1 + dx*sx_1)
    T[1, 5, 5] = t266 - dx*h*dx_1

    T[1, 2, 2] = t233
    T[1, 2, 3] = t234
    T[1, 3, 3] = t244

    T[2, 0, 2] = t313
    T[2, 0, 3] = t314 + h*sy
    T[2, 1, 2] = t323
    T[2, 1, 3] = t324
    T[2, 2, 5] = t336
    T[2, 3, 5] = t346/beta

    T[3, 0, 2] = t413 - h*cx*cy_1
    T[3, 0, 3] = t414 + (1 - cx)*h*sy_1
    T[3, 1, 2] = t423 - h*sx*cy_1
    T[3, 1, 3] = t424 - h*sx*sy_1
    T[3, 2, 5] = t436 - h*dx*cy_1
    T[3, 3, 5] = t446 - h*dx*sy_1

    t511 =    coef1*I511 + h*kx4*I522/2.
    t512 = 2.*coef1*I512 - h*kx2*I512
    t516 = 2.*coef1*I516 + coef3*I51 - h2*kx2*I522
    t522 =    coef1*I522 + 0.5*h*I511
    t526 = 2.*coef1*I526 + coef3*I52 + h2*I512
    t566 =    coef1*I566 + coef3*I56 + 0.5*h3*I522 - h*I50
    t533 =       K2*I533 - ky2*h*I50/2.
    t534 =    2.*K2*I534
    t544 =       K2*I544 - h*I50/2.
    i566 = h2*(L - sx*cx)/(4.*kx2) if kx != 0 else h2*L3/6.

    T511 = t511 + 1/4.*kx2*(L - cx*sx)

    T512 = t512 - (1/2.)*kx2*sx2 + h*dx
    T516 = t516 + h*(sx*cx - L)/2.
    T522 = t522 + (L + sx*cx)/4.
    T526 = t526 + h*sx2/2.
    T566 = t566 + i566
    T533 = t533 + 1/4.*ky2*(L - sy*cy )
    T534 = t534 - 1/2.*ky2*sy2
    T544 = t544 + (L + sy*cy)/4.

    T[4, 0, 0] = T511/beta
    T[4, 0, 1] = (T512 + h*dx)/beta
    T[4, 0, 5] = T516/beta
    T[4, 1, 1] = T522/beta
    T[4, 1, 5] = T526/beta
    T[4, 5, 5] = T566/beta + 1.5*L/(beta*beta*beta)*igamma2
    T[4, 2, 2] = T533/beta
    T[4, 2, 3] = T534/beta
    T[4, 3, 3] = T544/beta
    return T
