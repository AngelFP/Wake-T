"""
Plasma beamline simulation
==========================

This tutorial illustrates how to carry out a simulation of a multi-element
plasma-acceleration beamline.

The setup considered is an LPA with external injection followed by an active
plasma lens for beam capturing.

"""

# %%
# Create particle bunch
# ---------------------
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_size

# Beam parameters.
emitt_nx = emitt_ny = 1e-6  # m
s_x = s_y = 3e-6  # m
s_t = 3.  # fs
gamma_avg = 100 / 0.511
gamma_spread = 1.  # %
q_bunch = 30  # pC
xi_avg = 0.  # m
n_part = 1e4

# Create particle bunch.
bunch = get_gaussian_bunch_from_size(
    emitt_nx, emitt_ny, s_x, s_y, gamma_avg, gamma_spread, s_t, xi_avg,
    q_bunch, n_part, name='elec_bunch')

# Show phase space.
bunch.show()

# %%
# Create laser driver
# -------------------
from wake_t import GaussianPulse

# Laser parameters.
laser_xi_c = 60e-6  # m (laser centroid in simulation box)
w_0 = 40e-6  # m
a_0 = 3
tau = 30e-15  # fs (FWHM in intensity)
z_foc = 1e-2  # laser focus at center of LPA.

# Create Gaussian laser pulse.
laser = GaussianPulse(laser_xi_c, w_0=w_0, a_0=a_0, tau=tau, z_foc=z_foc)

# %%
# Create LPA
# ----------
from wake_t import PlasmaStage

plasma_target = PlasmaStage(
    length=2e-2, density=1e23, wakefield_model='quasistatic_2d',
    xi_max=90e-6, xi_min=-40e-6, r_max=200e-6, n_xi=260, n_r=200, ppc=4,
    laser=laser, n_out=10)

# %%
# Create beam capture section
# ---------------------------
from wake_t import Drift, ActivePlasmaLens

drift_1 = Drift(length=0.1, n_out=10)
drift_2 = Drift(length=0.1, n_out=10)

apl = ActivePlasmaLens(
    length=2e-2, foc_strength=400, wakefields=False, n_out=5)

# %%
# Make beamline
# -------------
from wake_t import Beamline

beamline = Beamline([plasma_target, drift_1, apl, drift_2])


# %%
# Track bunch
# -----------
bunch_list = beamline.track(bunch)


# %%
# Analyze beam evolution
# ----------------------
import matplotlib.pyplot as plt
from wake_t.diagnostics import analyze_bunch_list

params_evolution = analyze_bunch_list(bunch_list)

# Quick plot of results.
z = params_evolution['prop_dist'] * 1e2
fig_1 = plt.figure()
plt.subplot(411)
plt.semilogy(z, params_evolution['beta_x'])
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\beta_x$ [m]")
plt.subplot(412)
plt.semilogy(z, params_evolution['gamma_x'])
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\gamma_{x}$ [$m^{-1}$]")
plt.subplot(413)
plt.plot(z, params_evolution['emitt_x']*1e6)
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\epsilon_{nx}$ [$\\mu$m]")
plt.subplot(414)
plt.plot(z, params_evolution['avg_ene']*0.511)
plt.xlabel("z [cm]")
plt.ylabel("E [MeV]")
plt.tight_layout()
