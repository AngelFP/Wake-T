from pytest import approx
import numpy as np
import matplotlib.pyplot as plt
from aptools.plotting.quick_diagnostics import slice_analysis

from wake_t import PlasmaStage, GaussianPulse
from wake_t.utilities.bunch_generation import get_matched_bunch
from wake_t.diagnostics import analyze_bunch, analyze_bunch_list


def test_custom_blowout_wakefield(make_plots=False):
    """Check that the `'custom_blowout'` wakefield model works as expected.

    A matched bunch with high emittance is propagated through a 3 cm plasma
    stage. Due to the high emittance and Ez slope, the beam develops not only
    a large chirp but also an uncorrelated energy spread. The test checks that
    the final projected energy spread is always the same.
    """
    # Set numpy random seed to get reproducible results
    np.random.seed(1)

    # Create laser driver.
    laser = GaussianPulse(100e-6, l_0=800e-9, w_0=50e-6, a_0=3,
                        tau=30e-15, z_foc=0.)

    # Create bunch (matched to a blowout at a density of 10^{23} m^{-3}).
    en = 10e-6  # m
    ene = 200  # units of beta*gamma
    ene_sp = 0.3  # %
    xi_c = laser.xi_c - 55e-6  # m
    s_t = 10  # fs
    q_tot = 100  # pC
    n_part = 3e4
    k_x = 1e6
    bunch = get_matched_bunch(en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part,
                              k_x=k_x)

    # Create plasma stage.
    plasma = PlasmaStage(
        3e-2, 1e23, laser=laser, wakefield_model='custom_blowout',
        lon_field=-10e9, lon_field_slope=1e15, foc_strength=k_x,
        xi_fields=xi_c, n_out=50, bunch_pusher='boris')

    # Do tracking.
    bunch_list = plasma.track(bunch)

    bunch_params = analyze_bunch(bunch)
    rel_ene_sp = bunch_params['rel_ene_spread']
    assert approx(rel_ene_sp, rel=1e-10) == 0.21192494153185745

    if make_plots:
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
        slice_analysis(
            bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz,
            bunch.q, fig=fig_2)
        plt.show()


if __name__ == "__main__":
    test_custom_blowout_wakefield(make_plots=True)
