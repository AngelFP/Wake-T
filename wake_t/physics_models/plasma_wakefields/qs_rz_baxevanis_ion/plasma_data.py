"""Data structure for plasma particles."""

from typing import Optional, List, Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.constants as ct


@dataclass
class PlasmaParticlesData:
    """
    Data structure containing plasma particle arrays and parameters.
    
    This class holds all the state and configuration for a 1D slice of 
    plasma particles (electrons and ions).
    """
    # Configuration parameters
    r_max: float
    r_max_plasma: float
    radial_density: Callable[[float], float]
    dr: float
    ppc: float
    pusher: str = "ab2"
    shape: str = "linear"
    max_gamma: float = 10.0
    nr: int = 0
    nz: int = 0
    ion_motion: bool = True
    mass: float = ct.m_p
    free_electrons_per_ion: int = 1
    store_history: bool = False
    diags: List[str] = field(default_factory=list)
    
    # Particle counts
    n_elec: int = 0
    n_part: int = 0
    
    # Species properties
    m_elec: float = 0.0
    m_ion: float = 0.0
    q_species_elec: float = 0.0
    q_species_ion: float = 0.0
    
    # Main particle arrays (combined electrons and ions)
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    dr_p: np.ndarray = field(default_factory=lambda: np.array([]))
    pr: np.ndarray = field(default_factory=lambda: np.array([]))
    pz: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    w: np.ndarray = field(default_factory=lambda: np.array([]))
    w_center: np.ndarray = field(default_factory=lambda: np.array([]))
    r_to_x: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    
    # Species views (will be created as views into main arrays)
    r_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    log_r_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    dr_p_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    pr_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    pz_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    w_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    w_center_elec: np.ndarray = field(default_factory=lambda: np.array([]))
    r_to_x_elec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    id_elec: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    
    r_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    log_r_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    dr_p_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    pr_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    pz_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    w_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    w_center_ion: np.ndarray = field(default_factory=lambda: np.array([]))
    r_to_x_ion: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    id_ion: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    
    # Field arrays (private)
    _a2: np.ndarray = field(default_factory=lambda: np.array([]))
    _nabla_a2: np.ndarray = field(default_factory=lambda: np.array([]))
    _b_t_0: np.ndarray = field(default_factory=lambda: np.array([]))
    _b_t: np.ndarray = field(default_factory=lambda: np.array([]))
    _psi: np.ndarray = field(default_factory=lambda: np.array([]))
    _dr_psi: np.ndarray = field(default_factory=lambda: np.array([]))
    _dxi_psi: np.ndarray = field(default_factory=lambda: np.array([]))
    _chi: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_3_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_3_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _a_0: np.ndarray = field(default_factory=lambda: np.array([]))
    _A: np.ndarray = field(default_factory=lambda: np.array([]))
    _B: np.ndarray = field(default_factory=lambda: np.array([]))
    _C: np.ndarray = field(default_factory=lambda: np.array([]))
    _K: np.ndarray = field(default_factory=lambda: np.array([]))
    _U: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Additional auxiliary arrays
    _a_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _b_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_1: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_2: np.ndarray = field(default_factory=lambda: np.array([]))
    _rho: np.ndarray = field(default_factory=lambda: np.array([]))
    _log_r: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Species views for field arrays
    _psi_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _dr_psi_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _dxi_psi_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _psi_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _dr_psi_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _dxi_psi_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _b_t_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _b_t_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _b_t_0_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _b_t_0_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _nabla_a2_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _nabla_a2_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _a2_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _a2_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_1_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_2_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_1_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _sum_2_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _rho_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _rho_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _chi_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _chi_i: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Adams-Bashforth arrays
    _dr: np.ndarray = field(default_factory=lambda: np.array([]))
    _dpr: np.ndarray = field(default_factory=lambda: np.array([]))
    _dr_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _dpr_e: np.ndarray = field(default_factory=lambda: np.array([]))
    _dr_i: np.ndarray = field(default_factory=lambda: np.array([]))
    _dpr_i: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # History arrays
    r_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    log_r_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    xi_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    pr_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    pz_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    w_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    r_to_x_hist: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    id_hist: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    sum_1_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    sum_2_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    a_i_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    b_i_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    a_0_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # History tracking
    i_push: int = 0
    xi_current: float = 0.0
    ions_computed: bool = False
