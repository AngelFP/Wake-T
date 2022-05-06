"""
Basic example demonstrating how to simulate a plasma stage in Wake-T using
the 2d quasistatic wakefield model (designed for blowout regime but also able
to handle all regimes down to linear wakes.).

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this 2d quasistatic model.

"""

from copy import deepcopy
import matplotlib.pyplot as plt
from aptools.plotting.quick_diagnostics import slice_analysis
import scipy.constants as ct
import numpy as np

from wake_t import PlasmaStage, GaussianPulse
from wake_t.utilities.bunch_generation import get_matched_bunch
from wake_t.diagnostics import analyze_bunch_list, analyze_bunch


# Create laser driver.
laser = GaussianPulse(100e-6, l_0=800e-9, w_0=50e-6, a_0=3,
                      tau=30e-15, z_foc=0.)


def get_bunch():
    # Create bunch (matched to a blowout at a density of 10^{23} m^{-3}).
    en = 1e-6  # m
    ene = 200  # units of beta*gamma
    ene_sp = 0.3  # %
    xi_c = laser.xi_c - 55e-6  # m
    s_t = 10  # fs
    q_tot = 100  # pC
    n_part = 1e5
    bunch = get_matched_bunch(en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part,
                            n_p=1e23)
    return bunch

def simulate_plasma(bunch):
    # Plasma density
    n_p = 1e23

    # Plasma length
    L_p  = 1e-2  # m

    # Calculate dt
    # gamma = 200
    # w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
    # max_kx = (ct.m_e/(2*ct.e*ct.c))*w_p**2
    # w_x = np.sqrt(ct.e*ct.c/ct.m_e * max_kx/gamma)
    # period = 1/w_x
    # dt = 0.1*period

    n_steps = 100
    dt = L_p / ct.c / n_steps


    # Create plasma stage.
    plasma = PlasmaStage(
        L_p, n_p, laser=laser, wakefield_model='quasistatic_2d',
        dt_bunch=dt,
        bunch_pusher='boris',
        n_out=3,
        laser_evolution=True, r_max=200e-6, r_max_plasma=120e-6, xi_min=30e-6,
        xi_max=120e-6, n_r=200, n_xi=180, dz_fields=0.5e-3, ppc=5)


    # Do tracking.
    plasma.track(bunch, out_initial=True, opmd_diag=False)


    # Analyze bunch
    bunch_params = analyze_bunch(bunch)
    energy_spread = bunch_params['rel_ene_spread']
    emittance = bunch_params['emitt_x']
    gamma_avg = bunch_params['avg_ene']

    return energy_spread, emittance, gamma_avg


# Get bunch. Debemos usar siempre el mismo bunch, asi que solo podemos llamar a esta funcion una vez.
bunch = get_bunch()

# Hacer loop sobre lo siguiente:
bunch_copy = deepcopy(bunch)
energy_spread, emittance, gamma_avg = simulate_plasma(bunch)

print(energy_spread)
print(emittance)
print(gamma_avg)
