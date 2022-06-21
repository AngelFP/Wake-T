""" This module contains methods for generating particle distributions"""

import numpy as np
import scipy.constants as ct
from scipy.stats import truncnorm
import aptools.plasma_accel.general_equations as ge
import aptools.data_handling.reading as dr
from aptools.data_handling.utilities import get_available_species

from wake_t.particles.particle_bunch import ParticleBunch


def get_gaussian_bunch_from_twiss(
        en_x, en_y, a_x, a_y, b_x, b_y, ene, ene_sp, s_t, xi_c, q_tot, n_part,
        x_off=0, y_off=0, theta_x=0, theta_y=0, name=None):
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
        Mean bunch energy in non-dimensional units (beta*gamma).

    ene_sp: float
        Relative energy spread in %.

    s_t: float
        Bunch duration (standard deviation) in units of fs.

    xi_c: float
        Central bunch position in the xi in units of m.

    q_tot: float
        Total bunch charge in pC.

    n_part: int
        Total number of particles in the bunch.

    x_off: float
        Centroid offset in the x-plane in units of m.

    y_off: float
        Centroid offset in the y-plane in units of m.

    theta_x: float
        Pointing angle in the x-plane in radians.

    theta_y: float
        Pointing angle in the y-plane in radians.

    name: str
        Name of the particle bunch.

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
    p_x_off = theta_x * ene
    p_y_off = theta_y * ene
    q_tot = q_tot/1e12
    # Create normalized gaussian distributions
    u_x = np.random.standard_normal(n_part)
    v_x = np.random.standard_normal(n_part)
    u_y = np.random.standard_normal(n_part)
    v_y = np.random.standard_normal(n_part)
    # Calculate transverse particle distributions
    x = s_x*u_x + x_off
    xp = s_xp*(p_x*u_x + np.sqrt(1-np.square(p_x))*v_x)
    y = s_y*u_y + y_off
    yp = s_yp*(p_y*u_y + np.sqrt(1-np.square(p_y))*v_y)
    # Create longitudinal distributions (truncated at -3 and 3 sigma in xi)
    xi = truncnorm.rvs(-3, 3, loc=xi_c, scale=s_z, size=n_part)
    pz = np.random.normal(ene, ene_sp_abs, n_part)
    # Change from slope to momentum and apply offset
    px = xp*pz + p_x_off
    py = yp*pz + p_y_off
    # Charge
    q = np.ones(n_part)*(q_tot/n_part)
    return ParticleBunch(q, x, y, xi, px, py, pz, name=name)


def get_gaussian_bunch_from_size(
        en_x, en_y, s_x, s_y, ene, ene_sp, s_t, xi_c, q_tot, n_part, x_off=0,
        y_off=0, theta_x=0, theta_y=0, name=None):
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
        Mean bunch energy in non-dimensional units (beta*gamma).

    ene_sp: float
        Relative energy spread in %.

    s_t: float
        Bunch duration (standard deviation) in units of fs.

    xi_c: float
        Central bunch position in the xi in units of m.

    q_tot: float
        Total bunch charge in pC.

    n_part: int
        Total number of particles in the bunch.

    x_off: float
        Centroid offset in the x-plane in units of m.

    y_off: float
        Centroid offset in the y-plane in units of m.

    theta_x: float
        Pointing angle in the x-plane in radians.

    theta_y: float
        Pointing angle in the y-plane in radians.

    name: str
        Name of the particle bunch.

    Returns:
    --------
    A ParticleBunch object.

    """
    b_x = s_x**2*ene/en_x
    b_y = s_y**2*ene/en_y
    return get_gaussian_bunch_from_twiss(en_x, en_y, 0, 0, b_x, b_y, ene,
                                         ene_sp, s_t, xi_c, q_tot, n_part,
                                         x_off, y_off, theta_x, theta_y,
                                         name=name)


def get_matched_bunch(
        en_x, en_y, ene, ene_sp, s_t, xi_c, q_tot, n_part, x_off=0, y_off=0,
        theta_x=0, theta_y=0, n_p=None, k_x=None, name=None):
    """
    Creates a Gaussian bunch matched to the plasma focusing fields.

    Parameters:
    -----------
    en_x : float
        Normalized trace-space emittance in the x-plane in units of m*rad.

    en_y : float
        Normalized trace-space emittance in the y-plane in units of m*rad.

    ene: float
        Mean bunch energy in non-dimensional units (beta*gamma).

    ene_sp: float
        Relative energy spread in %.

    s_t: float
        Bunch duration (standard deviation) in units of fs.

    xi_c: float
        Central bunch position in the xi in units of m.

    q_tot: float
        Total bunch charge in pC.

    n_part: int
        Total number of particles in the bunch.

    x_off: float
        Centroid offset in the x-plane in units of m.

    y_off: float
        Centroid offset in the y-plane in units of m.

    theta_x: float
        Pointing angle in the x-plane in radians.

    theta_y: float
        Pointing angle in the y-plane in radians.

    n_p: double
        Plasma density in units of m^{-3}. This value is used to calculate the
        focusing fields in the plasma assuming blowout regime.

    k_x: int
        Focusing fields in the plasma in units of T/m. Has priority over n_p.

    name: str
        Name of the particle bunch.

    Returns:
    --------
    A ParticleBunch object.

    """
    if n_p is not None:
        n_p *= 1e-6
    b_m = ge.matched_plasma_beta_function(ene, n_p, k_x)
    return get_gaussian_bunch_from_twiss(en_x, en_y, 0, 0, b_m, b_m, ene,
                                         ene_sp, s_t, xi_c, q_tot, n_part,
                                         x_off, y_off, theta_x, theta_y,
                                         name=name)


def get_from_file(file_path, code_name, preserve_prop_dist=False, name=None,
                  species_name=None, **kwargs):
    """Get particle bunch from file.

    Parameters
    ----------
    file_path : str
        Path to the file containing the particle distribution.
    code_name : str
        Name of the code from which the particle distribution originates.
        Possible values are 'astra', 'csrtrack' and 'openpmd'.
    preserve_prop_dist : bool, optional
        Whether to preserve the average longitudinal position of the
        distribution in the `prop_distance` parameter of the `ParticleBunch`,
        by default False.
    name : str, optional
        Name to assign to the ParticleBunch, by default None. If None, when
        reading a particle species from an `openPMD` file, the `name` is set
        to the original name of the species. For other data formats, the
        default name `elec_bunch_i` will be assigned, where `i`>=0 is an
        integer that is increased for each unnamed `ParticleBunch` that is
        generated.
    species_name : str, optional
        Name of the particle species to be read from an `openpmd` file.
        Required only when more than one particle species is present in the
        file.

    Other Parameters
    ----------------
    **kwargs : dict
        Other parameters to be passed to
        `aptools.data_handling.reading.read_beam`.

    Returns
    -------
    ParticleBunch
        A ParticleBunch with the distribution from the specified file.
    """
    # If reading from an openPMD file, use the right `name` and `species_name`.
    if code_name == 'openpmd':
        if species_name is None:
            available_species = get_available_species(file_path)
            n_species = len(available_species)
            if n_species == 0:
                raise ValueError(
                    "No particle species found in '{}'".format(file_path)
                )
            elif n_species == 1:
                species_name = available_species[0]
            else:
                raise ValueError(
                    'More than one particle species is available in' +
                    "'{}'. ".format(file_path) +
                    'Please specify a `species_name`. ' +
                    'Available species are: ' + str(available_species)
                )
        kwargs['species_name'] = species_name
        if name is None:
            name = species_name
    # Read particle species.
    x, y, z, px, py, pz, q = dr.read_beam(code_name, file_path, **kwargs)
    # Center in z.
    z_avg = np.average(z, weights=q)
    xi = z - z_avg
    # Create ParticleBunch
    bunch = ParticleBunch(q, x, y, xi, px, py, pz, name=name)
    # Preserve `z_avg` as the initial propagation distance of the bunch.
    if preserve_prop_dist:
        bunch.prop_distance = z_avg
    return bunch
