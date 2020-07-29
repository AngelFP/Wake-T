from time import time
from copy import copy
import numpy as np
import scipy.constants as ct
from numba import njit
import matplotlib
# np.seterr(all='raise')
# matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
import scipy.interpolate as scint
import aptools.plasma_accel.general_equations as ge


def calculate_wakefield(laser, beam_part, r_max, xi_min, xi_max, nr, nxi, n_part, n_p, laser_z_foc):
    s_d = ge.plasma_skin_depth(n_p*1e-6)
    r_max = r_max / s_d
    xi_min= xi_min / s_d
    xi_max = xi_max / s_d

    # Initialize plasma particles
    dr = r_max / n_part
    r_part = np.linspace(dr, r_max, n_part)
    pr_part = np.zeros_like(r_part)
    gamma_part = np.ones_like(r_part)
    q_part = dr*r_part

    # iteration steps
    dxi = (xi_max - xi_min) / nxi

    # Initialize field arrays
    psi_mesh = np.zeros((nr+1, nxi))
    dr_psi_mesh = np.zeros((nr+1, nxi))
    dxi_psi_mesh = np.zeros((nr+1, nxi))
    b_theta_bar_mesh = np.zeros((nr+1, nxi))
    b_theta_0_mesh = np.zeros((nr+1, nxi))
    r_mesh = np.linspace(0, r_max, nr+1)
    xi_mesh = np.linspace(xi_min, xi_max, nxi)

    if beam_part is not None:
        beam_profile = get_beam_function(beam_part, r_max, xi_min, xi_max, nr, nxi, n_p)
    else:
        beam_profile = None

    # Main loop
    t0 = time()
    for step in np.arange(nxi):
        # print(step)
        xi = xi_max - dxi * step

        r_part, pr_part = evolve_plasma_rk4(r_part, pr_part, q_part, xi, dxi, laser, beam_profile, laser_z_foc, s_d)
        if step == 0:
            t_comp = time() - t0
        idx_keep = np.where(r_part<=r_max+0.1)
        r_part = r_part[idx_keep]
        pr_part = pr_part[idx_keep]
        gamma_part = gamma_part[idx_keep]
        q_part = q_part[idx_keep]
        
        psi_mesh[:,-1-step], dr_psi_mesh[:,-1-step], dxi_psi_mesh[:,-1-step], b_theta_bar_mesh[:,-1-step], b_theta_0_mesh[:,-1-step] = calculate_fields_at_mesh(xi, r_mesh, r_part, pr_part, q_part, laser, beam_profile, laser_z_foc, s_d)
    dr = r_max / nr
    dr_psi_mesh, dxi_psi_mesh = np.gradient(psi_mesh, dr, dxi)
    dxi_psi_mesh *= -1
    dr_psi_mesh *= -1
    e_r_mesh = b_theta_bar_mesh +  b_theta_0_mesh - dr_psi_mesh
    n_p = np.gradient(np.vstack(r_mesh) * e_r_mesh, dr, axis=0, edge_order=2)/np.vstack(r_mesh) - np.gradient(dxi_psi_mesh, dxi, axis=1) - 1
    k_r_mesh = np.gradient(dr_psi_mesh, dr, axis=0, edge_order=2)
    e_z_p_mesh = np.gradient(dxi_psi_mesh, dxi, axis=1, edge_order=2)
    t_tot = time() - t0
    # print(t_comp)
    # print(t_tot - t_comp)
    # print(t_tot)

    return n_p, dr_psi_mesh, dxi_psi_mesh, e_z_p_mesh, k_r_mesh, psi_mesh, xi_mesh, r_mesh


def evolve_plasma_rk4(r, pr, q, xi, dxi, laser, beam_profile, laser_z_foc, s_d):
    Ar, Apr = equations_of_motion(dxi, xi, r, pr, q, laser, beam_profile, laser_z_foc, s_d)
    Br, Bpr = equations_of_motion(dxi, xi-dxi/2, r + Ar/2, pr + Apr/2, q, laser, beam_profile, laser_z_foc, s_d)
    Cr, Cpr = equations_of_motion(dxi, xi-dxi/2, r + Br/2, pr + Bpr/2, q, laser, beam_profile, laser_z_foc, s_d)
    Dr, Dpr = equations_of_motion(dxi, xi-dxi, r + Cr, pr + Cpr, q, laser, beam_profile, laser_z_foc, s_d)
    return update_particles_rk4(r, pr, Ar, Br, Cr, Dr, Apr, Bpr, Cpr, Dpr)


@njit()
def update_particles_rk4(r, pr, Ar, Br, Cr, Dr, Apr, Bpr, Cpr, Dpr):
    inv_6 = 1./6.
    for i in range(r.shape[0]):
        r[i] += (Ar[i] + 2.*(Br[i] + Cr[i]) + Dr[i]) * inv_6
        pr[i] += (Apr[i] + 2.*(Bpr[i] + Cpr[i]) + Dpr[i]) * inv_6
    idx_neg = np.where(r < 0)
    r[idx_neg] *= -1
    pr[idx_neg] *= -1
    return r, pr


def equations_of_motion(dxi, xi, r_p, pr_p, q_p, laser, beam_profile, laser_z_foc, s_d):
    r_p= r_p.copy()
    pr_p = pr_p.copy()
    idx_neg = np.where(r_p < 0)
    r_p[idx_neg] *= -1
    pr_p[idx_neg] *= -1

    xi_si = xi*s_d
    r_p_si = r_p*s_d
    
    nabla_a = get_nabla_a(xi_si, r_p_si, laser.a_0, laser.l_0, laser.w_0, laser.tau, laser.xi_c, laser_z_foc, laser.polarization) * s_d
    a2 = get_a2(xi_si, r_p_si, laser.a_0, laser.l_0, laser.w_0, laser.tau, laser.xi_c, laser_z_foc, laser.polarization)
    b_theta_0_p = beam_profile(r_p, xi)#, grid=False)
    # psi_p, dr_psi_p, dxi_psi_p = calculate_psi_and_derivatives_at_particles(r_p, pr_p, q_p)
    # gamma_p = (1 + pr_p**2 + a2 + (1+psi_p)**2)/(2*(1+psi_p))
    # b_theta_bar_p = calculate_plasma_fields_at_particles(r_p, pr_p, q_p, gamma_p, psi_p, dr_psi_p, dxi_psi_p, b_theta_0_p, nabla_a)
    return calculate_derivatives(dxi, r_p, pr_p, q_p, b_theta_0_p, nabla_a, a2)


@njit()
def calculate_derivatives(dxi, r_p, pr_p, q_p, b_theta_0_p, nabla_a, a2):
    n_part = r_p.shape[0]
    dr = np.empty(n_part)
    dpr = np.empty(n_part)
    gamma_p = np.empty(n_part)

    psi_p, dr_psi_p, dxi_psi_p = calculate_psi_and_derivatives_at_particles(r_p, pr_p, q_p)

    for i in range(n_part):
        psi_p_i = psi_p[i]
        gamma_p[i] = (1. + pr_p[i]**2 + a2[i] + (1.+psi_p_i)**2) / (2.*(1. + psi_p_i))

    b_theta_bar_p = calculate_plasma_fields_at_particles(r_p, pr_p, q_p, gamma_p, psi_p, dr_psi_p, dxi_psi_p, b_theta_0_p, nabla_a)

    for i in range(n_part):
        psi_p_i = psi_p[i]
        dpr[i] = dxi * (gamma_p[i] * dr_psi_p[i] / (1. + psi_p_i) - b_theta_bar_p[i] - b_theta_0_p[i] - nabla_a[i] / (2. * (1. + psi_p_i)))
        dr[i] = dxi * pr_p[i] / (1. + psi_p_i)

    return dr, dpr


def calculate_fields_at_mesh(xi, r_vals, r_p, pr_p, q_p, laser, beam_profile, laser_z_foc, s_d):
    xi_si = xi*s_d
    r_p_si = r_p*s_d
    nabla_a = get_nabla_a(xi_si, r_p_si, laser.a_0, laser.l_0, laser.w_0, laser.tau, laser.xi_c, laser_z_foc, laser.polarization) * s_d
    # nabla_a = laser.get_nabla_a(r_p*s_d, xi*s_d, dz_foc=laser_z_foc) * s_d
    a2 = get_a2(xi_si, r_p_si, laser.a_0, laser.l_0, laser.w_0, laser.tau, laser.xi_c, laser_z_foc, laser.polarization)
    # a2 = laser.get_a0_profile(r_p*s_d, xi*s_d, dz_foc=laser_z_foc) ** 2
    b_theta_0_p = beam_profile(r_p, xi)#, grid=False)
    b_theta_0_vals = beam_profile(r_vals, xi)#, grid=False)
    psi_p, dr_psi_p, dxi_psi_p = calculate_psi_and_derivatives_at_particles(r_p, pr_p, q_p)
    gamma_p = (1 + pr_p**2 + a2/2 + (1+psi_p)**2)/(2*(1+psi_p))
    psi_vals, dr_psi_vals, dxi_psi_vals = calculate_psi_and_derivatives_at_mesh(r_vals, r_p, pr_p, q_p)
    b_theta_bar_vals = calculate_plasma_fields_at_mesh(r_vals, r_p, pr_p, q_p, gamma_p, psi_p, dr_psi_p, dxi_psi_p, b_theta_0_p, nabla_a)
    
    return psi_vals, dr_psi_vals, dxi_psi_vals, b_theta_bar_vals, b_theta_0_vals


@njit()
def calculate_psi_and_derivatives_at_mesh(r_vals, r_part, pr_part, q_part):
    n_part = r_part.shape[0]
    n_points = r_vals.shape[0]

    psi_part = np.zeros(n_part)
    psi_vals = np.zeros(n_points)
    dr_psi_vals = np.zeros(n_points)
    dxi_psi_vals = np.zeros(n_points)

    sum_1_arr = np.zeros(n_part)
    sum_2_arr = np.zeros(n_part)
    sum_3_arr = np.zeros(n_part)

    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate sum_1, sum_2 and psi_part.
    idx = np.argsort(r_part)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r_part[i]
        pr_i = pr_part[i]
        q_i = q_part[i]

        sum_1 += q_i
        sum_2 += q_i * np.log(r_i)
        sum_1_arr[i] = sum_1
        sum_2_arr[i] = sum_2
        psi_part[i] = sum_1*np.log(r_i) - sum_2 - r_i**2/4.
    r_N = r_part[-1]
    psi_part += - (sum_1*np.log(r_N) - sum_2 - r_N**2/4.)

    # Calculate sum_3.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r_part[i]
        pr_i = pr_part[i]
        q_i = q_part[i]
        psi_i = psi_part[i]

        sum_3 += (q_i * pr_i) / (r_i * (1+psi_i))
        sum_3_arr[i] = sum_3

    # Calculate fields at r_vals
    i_comp = 0
    for i in range(n_points):
        r = r_vals[i]
        for i_sort in range(n_part):
            i_p = idx[i_sort]
            r_i = r_part[i_p]
            i_comp = i_sort
            if r_i >= r:
                i_comp -= 1
                break
        # calculate fields
        if i_comp == -1:
            psi_vals[i] = - r**2/4.
            dr_psi_vals[i] = - r/2.
            dxi_psi_vals[i] = 0.
        else:
            i_p = idx[i_comp]
            psi_vals[i] = sum_1_arr[i_p]*np.log(r) - sum_2_arr[i_p] - r**2/4.
            dr_psi_vals[i] = sum_1_arr[i_p] / r - r/2.
            dxi_psi_vals[i] = - sum_3_arr[i_p]
    psi_vals = psi_vals - (sum_1*np.log(r_N) - sum_2 - r_N**2/4.)
    dxi_psi_vals = dxi_psi_vals + sum_3

    return psi_vals, dr_psi_vals, dxi_psi_vals


@njit()
def calculate_psi_and_derivatives_at_particles(r_part, pr_part, q_part):
    n_part = r_part.shape[0]
    psi_vals = np.zeros(n_part)
    dr_psi_vals = np.zeros(n_part)
    dxi_psi_vals = np.zeros(n_part)

    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate psi and dr_psi.
    idx = np.argsort(r_part)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r_part[i]
        pr_i = pr_part[i]
        q_i = q_part[i]

        sum_1_new = sum_1 + q_i
        sum_2_new = sum_2 + q_i * np.log(r_i)

        psi_vals[i] = ((sum_1 + sum_1_new)*np.log(r_i) - (sum_2+sum_2_new))/2 - r_i**2/4.
        dr_psi_vals[i] = (sum_1 + sum_1_new) / (2.*r_i) - r_i/2.

        sum_1 = sum_1_new
        sum_2 = sum_2_new
    r_N = r_part[-1]
    psi_vals = psi_vals - (sum_1*np.log(r_N) - sum_2 - r_N**2/4.)

    # Calculate dxi_psi.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r_part[i]
        pr_i = pr_part[i]
        q_i = q_part[i]
        psi_i = psi_vals[i]

        sum_3_new = sum_3 + (q_i * pr_i) / (r_i * (1+psi_i))
        dxi_psi_vals[i] = -(sum_3 + sum_3_new) / 2.
        sum_3 = sum_3_new

    dxi_psi_vals = dxi_psi_vals + sum_3
    return psi_vals, dr_psi_vals, dxi_psi_vals


@njit()
def calculate_plasma_fields_at_particles(r_p, pr_p, q_p, gamma_p, psi_p, dr_psi_p, dxi_psi_p, b_driver, nabla_a):
    """
    New calculation of plasma fields.

    Write a_i and b_i as linear system of a_0:

    a_i = K_i * a_0 + O_i
    b_i = U_i * a_0 + P_i


    Where:

    K_i = (1 + A_i*r_i/2) * K_im1  +  A_i/(2*r_i)     * U_im1
    U_i = (-A_i*r_i**3/2) * K_im1  +  (1 - A_i*r_i/2) * U_im1

    O_i = (1 + A_i*r_i/2) * O_im1  +  A_i/(2*r_i)     * P_im1  +  (2*Bi + Ai*Ci)/4
    P_i = (-A_i*r_i**3/2) * O_im1  +  (1 - A_i*r_i/2) * P_im1  +  r_i*(4*Ci - 2*Bi*r_i - Ai*Ci*r_i)/4

    With initial conditions:

    K_0 = 1
    U_0 = 0

    O_0 = 0
    P_0 = 0

    Then a_0 can be determined by imposing a_N = 0:

    a_N = K_N * a_0 + O_N = 0 <=> a_0 = - O_N / K_N

    """
    n_part = r_p.shape[0]

    # Preallocate arrays
    K = np.zeros(n_part)
    U = np.zeros(n_part)
    O = np.zeros(n_part)
    P = np.zeros(n_part)

    # Establish initial conditions (K_0 = 1, U_0 = 0, O_0 = 0, P_0 = 0)
    K_im1 = 1.
    U_im1 = 0.
    O_im1 = 0.
    P_im1 = 0.

    # Iterate over particles
    idx = np.argsort(r_p)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r_p[i]
        pr_i = pr_p[i]
        q_i = q_p[i]
        gamma_i = gamma_p[i]
        psi_i = psi_p[i]
        dr_psi_i = dr_psi_p[i]
        dxi_psi_i = dxi_psi_p[i]
        b_theta_0 = b_driver[i]
        nabla_a_i = nabla_a[i]

        a = 1. + psi_i
        a2 = a*a
        a3 = a2*a
        b = 1. / (r_i * a)
        c = 1. / (r_i * a2)
        pr_i2 = pr_i * pr_i

        A_i = q_i * b
        B_i = q_i * (- (gamma_i * dr_psi_i) * c
                     + (pr_i2 * dr_psi_i) / (r_i * a3)
                     + (pr_i * dxi_psi_i) * c
                     + pr_i2 / (r_i*r_i * a2)
                     + b_theta_0 * b
                     + nabla_a_i * c / 2.)
        C_i = q_i * (pr_i2 * c - (gamma_i/a - 1.)/ r_i)
        
        l_i = (1. + A_i*r_i/2.)
        m_i = A_i/(2.*r_i)
        n_i = (-A_i*r_i**3/2.)
        o_i = (1. - A_i*r_i/2.)

        K_i = l_i*K_im1 + m_i*U_im1
        U_i = n_i*K_im1 + o_i*U_im1
        O_i = l_i*O_im1 + m_i*P_im1 + (2.*B_i + A_i*C_i)/4.
        P_i = n_i*O_im1 + o_i*P_im1 + r_i*(4.*C_i - 2.*B_i*r_i - A_i*C_i*r_i)/4.

        K[i] = K_i
        U[i] = U_i
        O[i] = O_i
        P[i] = P_i

        K_im1 = K_i
        U_im1 = U_i
        O_im1 = O_i
        P_im1 = P_i

    a_0 = - O_im1 / K_im1

    a_i = K * a_0 + O
    b_i = U * a_0 + P

    # b_theta_bar = a_i * r_p + b_i / r_p

    a_im1 = a_0
    b_im1 = 0.
    a_i_avg = np.zeros(n_part)
    b_i_avg = np.zeros(n_part)
    for i_sort in range(n_part):
        i = idx[i_sort]
        a_i_avg[i] = (a_i[i] + a_im1) / 2.
        b_i_avg[i] = (b_i[i] + b_im1) / 2.
        a_im1 = a_i[i]
        b_im1 = b_i[i]

    b_theta_bar = a_i_avg * r_p + b_i_avg / r_p
    return b_theta_bar        


@njit()
def calculate_plasma_fields_at_mesh(r_vals, r_p, pr_p, q_p, gamma_p, psi_p, dr_psi_p, dxi_psi_p, b_driver, nabla_a):
    n_part = r_p.shape[0]
    n_points = r_vals.shape[0]

    # Preallocate arrays
    b_theta_mesh = np.zeros(n_points)
    K = np.zeros(n_part)
    U = np.zeros(n_part)
    O = np.zeros(n_part)
    P = np.zeros(n_part)

    # Establish initial conditions (K_0 = 1, U_0 = 0, O_0 = 0, P_0 = 0)
    K_im1 = 1.
    U_im1 = 0.
    O_im1 = 0.
    P_im1 = 0.

    # Iterate over particles
    idx = np.argsort(r_p)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r_p[i]
        pr_i = pr_p[i]
        q_i = q_p[i]
        gamma_i = gamma_p[i]
        psi_i = psi_p[i]
        dr_psi_i = dr_psi_p[i]
        dxi_psi_i = dxi_psi_p[i]
        b_theta_0 = b_driver[i]
        nabla_a_i = nabla_a[i]

        a = 1. + psi_i
        a2 = a*a
        a3 = a2*a
        b = 1. / (r_i * a)
        c = 1. / (r_i * a2)
        pr_i2 = pr_i * pr_i

        A_i = q_i * b
        B_i = q_i * (- (gamma_i * dr_psi_i) * c
                     + (pr_i2 * dr_psi_i) / (r_i * a3)
                     + (pr_i * dxi_psi_i) * c
                     + pr_i2 / (r_i*r_i * a2)
                     + b_theta_0 * b
                     + nabla_a_i * c / 2.)
        C_i = q_i * (pr_i2 * c - (gamma_i/a - 1.)/ r_i)
        
        l_i = (1. + A_i*r_i/2.)
        m_i = A_i/(2.*r_i)
        n_i = (-A_i*r_i**3/2.)
        o_i = (1. - A_i*r_i/2.)

        K_i = l_i*K_im1 + m_i*U_im1
        U_i = n_i*K_im1 + o_i*U_im1
        O_i = l_i*O_im1 + m_i*P_im1 + (2.*B_i + A_i*C_i)/4.
        P_i = n_i*O_im1 + o_i*P_im1 + r_i*(4.*C_i - 2.*B_i*r_i - A_i*C_i*r_i)/4.

        K[i] = K_i
        U[i] = U_i
        O[i] = O_i
        P[i] = P_i

        K_im1 = K_i
        U_im1 = U_i
        O_im1 = O_i
        P_im1 = P_i

    a_0 = - O_im1 / K_im1

    a_i = K * a_0 + O
    b_i = U * a_0 + P


    # Calculate fields at r_vals
    i_comp = 0
    for i in range(n_points):
        r = r_vals[i]
        for i_sort in range(n_part):
            i_p = idx[i_sort]
            r_i = r_p[i_p]
            i_comp = i_sort
            if r_i >= r:
                i_comp -= 1
                break
        # calculate fields
        if i_comp == -1:
            b_theta_mesh[i] = a_0 * r
        else:
            i_p = idx[i_comp]
            b_theta_mesh[i] = a_i[i_p] * r + b_i[i_p] / r

    return b_theta_mesh


def get_beam_function(beam_part, r_max, xi_min, xi_max, n_r, n_xi, n_p):
    x, y, xi, q = beam_part
    s_d = ge.plasma_skin_depth(n_p/1e6)
    r_part = np.sqrt(x**2 + y**2) / s_d
    x_edges = np.linspace(xi_min, xi_max, n_xi)
    y_edges = np.linspace(0, r_max, n_r)
    bins = [x_edges, y_edges]
    dr = y_edges[1]-y_edges[0]
    dxi = x_edges[1]-x_edges[0]
    bunch_hist, *_ = np.histogram2d(xi / s_d, r_part, bins=bins, weights=q/ct.e/(2*np.pi*dr*dxi*s_d**3*n_p))#*r_part))
    r_b = y_edges[1:] - dr/2
    xi_b = x_edges[1:] - dxi/2
    bunch_rint = np.cumsum(bunch_hist, axis=1)/r_b * dr
    return scint.interp2d(r_b, xi_b, -bunch_rint)
    # return scint.RectBivariateSpline(r_b, xi_b, -bunch_rint.T, kx=1, ky=1)


@njit()
def get_nabla_a(xi, r, a_0, l_0, w_0, tau, xi_c, dz_foc=0, pol='linear'):
    z_r = np.pi * w_0**2 / l_0
    w_fac = np.sqrt(1 + (dz_foc/z_r)**2)
    s_r = w_0 * w_fac / np.sqrt(2)
    s_z = tau * ct.c / (2*np.sqrt(2*np.log(2))) * np.sqrt(2)
    avg_amplitude = a_0
    if pol == 'linear':
        avg_amplitude /= np.sqrt(2)
    return - 2 * avg_amplitude**2 * r / s_r**2 * (
        np.exp(-(r)**2/(s_r**2)) * np.exp(-(xi-xi_c)**2/(s_z**2)))


@njit()
def get_a2(xi, r, a_0, l_0, w_0, tau, xi_c, dz_foc=0, pol='linear'):
    z_r = np.pi * w_0**2 / l_0
    w_fac = np.sqrt(1 + (dz_foc/z_r)**2)
    s_r = w_0 * w_fac / np.sqrt(2)
    s_z = tau * ct.c / (2*np.sqrt(2*np.log(2))) * np.sqrt(2)
    avg_amplitude = a_0
    if pol == 'linear':
        avg_amplitude /= np.sqrt(2)
    return (avg_amplitude/w_fac)**2 * (np.exp(-(r)**2/(s_r**2)) *
                                       np.exp(-(xi-xi_c)**2/(s_z**2)))
