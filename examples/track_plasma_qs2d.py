"""
Basic example demonstrating how to simulate a plasma stage in Wake-T using
the 2d quasistatic wakefield model (designed for blowout regime but also able
to handle all regimes down to linear wakes.).

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this 2d quasistatic model.

"""

import matplotlib.pyplot as plt
from aptools.plotting.quick_diagnostics import slice_analysis

from wake_t import PlasmaStage, GaussianPulse
from wake_t.utilities.bunch_generation import get_matched_bunch
from wake_t.diagnostics import analyze_bunch_list


# Create laser driver.
laser = GaussianPulse(100e-6, l_0=800e-9, w_0=50e-6, a_0=3,
                      tau=30e-15, z_foc=0.)


# Create bunch (matched to a blowout at a density of 10^{23} m^{-3}).
en = 1e-6  # m
ene = 200  # units of beta*gamma
ene_sp = 0.3  # %
xi_c = laser.xi_c - 55e-6  # m
s_t = 10  # fs
q_tot = 100  # pC
n_part = 1e4
bunch = get_matched_bunch(en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part,
                          n_p=1e23)


# Create plasma stage.
plasma = PlasmaStage(
    1e-2, 1e23, laser=laser, wakefield_model='quasistatic_2d', n_out=50,
    laser_evolution=True, r_max=200e-6, r_max_plasma=120e-6, xi_min=30e-6,
    xi_max=120e-6, n_r=200, n_xi=180, dz_fields=0.5e-3, ppc=5)


# Do tracking.
opmd_diag = False  # Set to True to activate openPMD output.
bunch_list = plasma.track(bunch, opmd_diag=opmd_diag)


# Analyze bunch evolution.
params_evolution = analyze_bunch_list(bunch_list)


# Quick plot of results.
z = params_evolution['prop_dist'] * 1e2
fig_1 = plt.figure()
plt.subplot(411)
plt.plot(z, params_evolution['beta_x']*1e3)
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\beta_x$ [mm]")
plt.subplot(412)
plt.plot(z, params_evolution['emitt_x']*1e6)
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\epsilon_{nx}$ [$\\mu$m]")
plt.subplot(413)
plt.plot(z, params_evolution['rel_ene_spread']*100)
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\frac{\\Delta \\gamma}{\\gamma}$ [%]")
plt.subplot(414)
plt.plot(z, params_evolution['avg_ene'])
plt.xlabel("z [mm]")
plt.ylabel("$\\gamma$")
plt.tight_layout()
fig_2 = plt.figure()
slice_analysis(bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz,
               bunch.q, fig=fig_2)
plt.show()
