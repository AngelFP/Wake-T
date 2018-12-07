""" This module contains methods for generating particle distributions"""

import numpy as np
import scipy.constants as ct
from scipy.stats import truncnorm
import aptools.plasma_accel.general_equations as ge

from wake_t.driver_witness import ParticleBunch


def get_gaussian_bunch_from_twiss(en_x, en_y, a_x, a_y, b_x, b_y, ene, ene_sp,
                                  x_c, y_c, xi_c, s_t, q_tot, n_part):
    """
    Creates a 6D Gaussian particle bunch with the specified Twiss parameters.

    Parameters:
    -----------
    en_x : float
        Normalized trace-space emittance in the x-plane in units of m*rad.
    en_y : float
        Normalized trace-space emittance in the y-plane in units of m*rad.
    a_x : float
        Alpha parameter in the x-plane.
    a_y : float
        Alpha parameter in the y-plane.
    b_x : float
        Beta parameter in the x-plane in units of m.
    b_y : float
        Beta parameter in the y-plane in units of m.
    ene: float
        Mean bunch energy in non-dimmensional units (beta*gamma).
    ene_sp: float
        Relative energy spread in %.
    x_c: float
        Central bunch position in the x-plane in units of m.
    y_c: float
        Central bunch position in the y-plane in units of m.
    xi_c: float
        Central bunch position in the xi in units of m.
    s_t: float
        Bunch duration (standard deviation) in units of fs.
    q_tot: float
        Total bunch charge in pC.
    n_part: int
        Total number of particles in the bunch.

    Returns:
    --------
    A ParticleBunch object.

    """

    # Calculate necessary values
    n_part = int(n_part)
    ene_sp = ene_sp/100
    ene_sp_abs = ene_sp*ene
    s_z = s_t*1e-15*ct.c
    em_x = en_x/ene
    em_y = en_y/ene
    g_x = (1+a_x**2)/b_x
    g_y = (1+a_y**2)/b_y
    s_x = np.sqrt(em_x*b_x)
    s_y = np.sqrt(em_y*b_y)
    s_xp = np.sqrt(em_x*g_x)
    s_yp = np.sqrt(em_y*g_y)
    p_x = -a_x*em_x/(s_x*s_xp)
    p_y = -a_y*em_y/(s_y*s_yp)
    q_tot = q_tot/1e12
    # Create normalized gaussian distributions
    u_x = np.random.standard_normal(n_part)
    v_x = np.random.standard_normal(n_part)
    u_y = np.random.standard_normal(n_part)
    v_y = np.random.standard_normal(n_part)
    # Calculate transverse particle distributions
    x = s_x*u_x
    xp = s_xp*(p_x*u_x + np.sqrt(1-np.square(p_x))*v_x)
    y = s_y*u_y
    yp = s_yp*(p_y*u_y + np.sqrt(1-np.square(p_y))*v_y)
    # Create longitudinal distributions (truncated at -3 and 3 sigma in xi)
    xi = truncnorm.rvs(-3, 3,loc=xi_c,scale=s_z, size=n_part) # 
    pz = np.random.normal(ene, ene_sp_abs, n_part)
    # Change from slope to momentum
    px = xp*pz
    py = yp*pz
    # Charge
    q = np.ones(n_part)*(q_tot/n_part)
    return ParticleBunch(q, x, y, xi, px, py, pz)

def get_gaussian_bunch_from_size(en_x, en_y, s_x, s_y, ene, ene_sp, x_c,
                                 y_c, xi_c, s_t, q_tot, n_part):
    """
    Creates a Gaussian bunch with the specified emitance and spot size. It is
    assumed to be on its waist (alpha_x = alpha_y = 0)

    Parameters:
    -----------
    en_x : float
        Normalized trace-space emittance in the x-plane in units of m*rad.
    en_y : float
        Normalized trace-space emittance in the y-plane in units of m*rad.
    s_x : float
        Bunch size (standard deviation) in the x-plane in units of m.
    s_y : float
        Bunch size (standard deviation) in the y-plane in units of m.
    ene: float
        Mean bunch energy in non-dimmensional units (beta*gamma).
    ene_sp: float
        Relative energy spread in %.
    x_c: float
        Central bunch position in the x-plane in units of m.
    y_c: float
        Central bunch position in the y-plane in units of m.
    xi_c: float
        Central bunch position in the xi in units of m.
    s_t: float
        Bunch duration (standard deviation) in units of fs.
    q_tot: float
        Total bunch charge in pC.
    n_part: int
        Total number of particles in the bunch.

    Returns:
    --------
    A ParticleBunch object.

    """
    b_x = s_x**2*ene/en_x
    b_y = s_y**2*ene/en_y
    return get_gaussian_bunch_from_twiss(en_x, en_y, 0, 0, b_x, b_y, ene,
                                         ene_sp, x_c, y_c, xi_c, s_t, q_tot,
                                         n_part)

def get_matched_bunch(en_x, en_y, ene, ene_sp, x_c, y_c, xi_c, s_t,
                      q_tot, n_part, n_p=None, k_x=None):
    """
    Creates a Gaussian bunch matched to the plasma focusing fields.

    Parameters:
    -----------
    en_x : float
        Normalized trace-space emittance in the x-plane in units of m*rad.
    en_y : float
        Normalized trace-space emittance in the y-plane in units of m*rad.
    ene: float
        Mean bunch energy in non-dimmensional units (beta*gamma).
    ene_sp: float
        Relative energy spread in %.
    x_c: float
        Central bunch position in the x-plane in units of m.
    y_c: float
        Central bunch position in the y-plane in units of m.
    xi_c: float
        Central bunch position in the xi in units of m.
    s_t: float
        Bunch duration (standard deviation) in units of fs.
    q_tot: float
        Total bunch charge in pC.
    n_part: int
        Total number of particles in the bunch.
    n_p: double
        Plasma density in units of cm^{-3}. This value is used to calculate the
        focusing fields in the plasma assuming blowout regime.
    k_x: int
        Focusing fields in the plasma in units of T/m. Has priority over n_p.

    Returns:
    --------
    A ParticleBunch object.

    """
    b_m = ge.matched_plasma_beta_function(ene, n_p, k_x)
    return get_gaussian_bunch_from_twiss(en_x, en_y, 0, 0, b_m, b_m, ene, 
                                         ene_sp, x_c, y_c, xi_c, s_t, q_tot,
                                         n_part)
