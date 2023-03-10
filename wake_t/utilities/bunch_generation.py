""" This module contains methods for generating particle distributions"""

from typing import Optional

import numpy as np
import scipy.constants as ct
from scipy.stats import truncnorm
import aptools.plasma_accel.general_equations as ge
from aptools.particle_distributions.read import read_distribution
from aptools.data_handling.utilities import get_available_species

from wake_t.particles.particle_bunch import ParticleBunch


def get_gaussian_bunch_from_twiss(
    en_x: float,
    en_y: float,
    a_x: float,
    a_y: float,
    b_x: float,
    b_y: float,
    ene: float,
    ene_sp: float,
    s_t: float,
    xi_c: float,
    q_tot: float,
    n_part: int,
    x_off: Optional[float] = 0,
    y_off: Optional[float] = 0,
    theta_x: Optional[float] = 0,
    theta_y: Optional[float] = 0,
    name: Optional[str] = None,
    q_species: Optional[float] = -ct.e,
    m_species: Optional[float] = ct.m_e
) -> ParticleBunch:
    """
    Creates a 6D Gaussian particle bunch with the specified Twiss parameters.

    Parameters
    ----------
    en_x, en_y : float
        Normalized trace-space emittance in x and y (units of m*rad).
    a_x, a_y : float
        Alpha Twiss parameter in x and y.
    b_x, b_y : float
        Beta Twiss parameter in x and y (units of m).
    ene : float
        Mean bunch energy in non-dimensional units (beta*gamma).
    ene_sp : float
        Relative energy spread in %.
    s_t : float
        Bunch duration (standard deviation) in units of fs.
    xi_c : float
        Central bunch position in the xi in units of m.
    q_tot : float
        Total bunch charge in pC.
    n_part : int
        Total number of particles in the bunch.
    x_off, y_off : float
        Centroid offsets in x and y (units of m).
    theta_x, theta_y : float
        Pointing angle in x and y (in radians).
    name : str
        Name of the particle bunch.
    q_species, m_species : float
        Charge and mass of a single particle of the species represented
        by the macroparticles. For an electron bunch (default),
        ``q_species=-e`` and ``m_species=m_e``

    Returns
    -------
    ParticleBunch
        The generated particle bunch.

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
    # Macroparticle weight.
    w = np.abs(np.ones(n_part) * q_tot / (n_part * q_species))
    return ParticleBunch(w, x, y, xi, px, py, pz, name=name,
                         q_species=q_species, m_species=m_species)


def get_gaussian_bunch_from_size(
    en_x: float,
    en_y: float,
    s_x: float,
    s_y: float,
    ene: float,
    ene_sp: float,
    s_t: float,
    xi_c: float,
    q_tot: float,
    n_part: int,
    x_off: Optional[float] = 0,
    y_off: Optional[float] = 0,
    theta_x: Optional[float] = 0,
    theta_y: Optional[float] = 0,
    name: Optional[str] = None,
    q_species: Optional[float] = -ct.e,
    m_species: Optional[float] = ct.m_e
) -> ParticleBunch:
    """
    Creates a Gaussian bunch with the specified emitance and spot size. It is
    assumed to be on its waist (alpha_x = alpha_y = 0)

    Parameters
    ----------
    en_x, en_y : float
        Normalized trace-space emittance in x and y (units of m*rad).
    s_x, s_y : float
        Bunch size (standard deviation) in x and y (units of m).
    ene : float
        Mean bunch energy in non-dimensional units (beta*gamma).
    ene_sp : float
        Relative energy spread in %.
    s_t : float
        Bunch duration (standard deviation) in units of fs.
    xi_c : float
        Central bunch position in the xi in units of m.
    q_tot : float
        Total bunch charge in pC.
    n_part : int
        Total number of particles in the bunch.
    x_off, y_off : float
        Centroid offsets in x and y (units of m).
    theta_x, theta_y : float
        Pointing angle in x and y (in radians).
    name : str
        Name of the particle bunch.
    q_species, m_species : float
        Charge and mass of a single particle of the species represented
        by the macroparticles. For an electron bunch (default),
        ``q_species=-e`` and ``m_species=m_e``

    Returns
    -------
    ParticleBunch
        The generated particle bunch.

    """
    b_x = s_x**2*ene/en_x
    b_y = s_y**2*ene/en_y
    return get_gaussian_bunch_from_twiss(
        en_x, en_y, 0, 0, b_x, b_y, ene, ene_sp, s_t, xi_c, q_tot, n_part,
        x_off, y_off, theta_x, theta_y, name=name, q_species=q_species,
        m_species=m_species)


def get_matched_bunch(
    en_x: float,
    en_y: float,
    ene: float,
    ene_sp: float,
    s_t: float,
    xi_c: float,
    q_tot: float,
    n_part: int,
    x_off: Optional[float] = 0,
    y_off: Optional[float] = 0,
    theta_x: Optional[float] = 0,
    theta_y: Optional[float] = 0,
    n_p: Optional[float] = None,
    k_x: Optional[float] = None,
    name: Optional[str] = None,
    q_species: Optional[float] = -ct.e,
    m_species: Optional[float] = ct.m_e
) -> ParticleBunch:
    """
    Creates a Gaussian bunch matched to the plasma focusing fields.

    Parameters
    ----------
    en_x, en_y : float
        Normalized trace-space emittance in x and y (units of m*rad).
    ene : float
        Mean bunch energy in non-dimensional units (beta*gamma).
    ene_sp : float
        Relative energy spread in %.
    s_t : float
        Bunch duration (standard deviation) in units of fs.
    xi_c : float
        Central bunch position in the xi in units of m.
    q_tot : float
        Total bunch charge in pC.
    n_part : int
        Total number of particles in the bunch.
    x_off, y_off : float
        Centroid offsets in x and y (units of m).
    theta_x, theta_y : float
        Pointing angle in x and y (in radians).
    n_p : double
        Plasma density in units of m^{-3}. This value is used to calculate the
        focusing fields in the plasma assuming blowout regime.
    k_x : int
        Focusing fields in the plasma in units of T/m. Has priority over n_p.
    name : str
        Name of the particle bunch.
    q_species, m_species : float
        Charge and mass of a single particle of the species represented
        by the macroparticles. For an electron bunch (default),
        ``q_species=-e`` and ``m_species=m_e``

    Returns
    -------
    ParticleBunch
        The generated particle bunch.

    """
    if n_p is not None:
        n_p *= 1e-6
    b_m = ge.matched_plasma_beta_function(ene, n_p, k_x)
    return get_gaussian_bunch_from_twiss(
        en_x, en_y, 0, 0, b_m, b_m, ene, ene_sp, s_t, xi_c, q_tot, n_part,
        x_off, y_off, theta_x, theta_y, name=name, q_species=q_species,
        m_species=m_species)


def get_from_file(
    file_path: str,
    data_format: str,
    preserve_prop_dist: Optional[bool] = False,
    name: Optional[str] = None,
    species_name: Optional[str] = None,
    **kwargs
) -> ParticleBunch:
    """Get particle bunch from file.

    Parameters
    ----------
    file_path : str
        Path to the file containing the particle distribution.
    data_format : str
        Internal format of the data.  Possible values
        are 'astra', 'csrtrack' and 'openpmd'.
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
    if data_format == 'openpmd':
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
    distribution = read_distribution(file_path, data_format, **kwargs)
    # Center in z.
    z_avg = np.average(distribution.z, weights=distribution.w)
    xi = distribution.z - z_avg
    # Create ParticleBunch
    bunch = ParticleBunch(
        w=distribution.w,
        x=distribution.x,
        y=distribution.y,
        xi=xi,
        px=distribution.px,
        py=distribution.py,
        pz=distribution.pz,
        q_species=distribution.q_species,
        m_species=distribution.m_species,
        name=name
    )
    # Preserve `z_avg` as the initial propagation distance of the bunch.
    if preserve_prop_dist:
        bunch.prop_distance = z_avg
    return bunch
