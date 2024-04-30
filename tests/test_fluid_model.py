from pytest import approx
import matplotlib.pyplot as plt
import numpy as np
from aptools.plotting.quick_diagnostics import slice_analysis

from wake_t import PlasmaStage, GaussianPulse
from wake_t.utilities.bunch_generation import get_matched_bunch
from wake_t.diagnostics import analyze_bunch_list


def test_fluid_model(plot=False):
    """Test the 1D fluid wakefield model."""
    # Set numpy random seed to get reproducible results.
    np.random.seed(0)

    # Create laser driver.
    laser = GaussianPulse(100e-6, l_0=800e-9, w_0=70e-6, a_0=0.8,
                          tau=30e-15, z_foc=0.)

    # Create bunch (matched to a focusing strength of 0.13 MT/m).
    en = 1e-6  # m
    ene = 200  # units of beta*gamma
    ene_sp = 0.3  # %
    xi_c = laser.xi_c - 55e-6  # m
    s_t = 3  # fs
    q_tot = 1  # pC
    n_part = 1e4
    bunch = get_matched_bunch(en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part,
                              k_x=0.13e6)

    # Create plasma stage.
    plasma = PlasmaStage(
        1e-2, 1e23, laser=laser, wakefield_model='cold_fluid_1d', n_out=50,
        laser_evolution=True, beam_wakefields=True, dz_fields=0.5e-3,
        r_max=200e-6, xi_min=40e-6, xi_max=120e-6, n_r=200, n_xi=200)

    # Do tracking.
    bunch_list = plasma.track(bunch)

    # Analyze bunch evolution.
    params_evolution = analyze_bunch_list(bunch_list)

    # Check final parameters.
    ene_sp = params_evolution['rel_ene_spread'][-1]
    assert approx(ene_sp, rel=1e-10) == 0.024157374564016194

    # Quick plot of results.
    if plot:
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
        plt.xlabel("z [cm]")
        plt.ylabel("$\\gamma$")
        plt.tight_layout()
        fig_2 = plt.figure()
        slice_analysis(
            bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz,
            bunch.q, fig=fig_2)
        plt.show()


if __name__ == '__main__':
    test_fluid_model(plot=True)
