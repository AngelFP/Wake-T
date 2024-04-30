import numpy as np
import matplotlib.pyplot as plt
from pytest import approx

from wake_t import PlasmaStage
from wake_t.utilities.bunch_generation import get_matched_bunch
from wake_t.diagnostics import analyze_bunch_list


def test_multibunch_plasma_simulation(plot=False):
    """Test a plasma simulation with multiple electron bunches.

    This test checks that giving multiple bunches as an input to `track` works
    as expected, i.e., that all bunches are taken into account for the
    wakefield calculation and that all bunches are correctly evolved.
    """
    # Set numpy random seed to get reproducible results.
    np.random.seed(1)

    # Plasma density.
    n_p = 1e23

    # Create driver.
    en = 10e-6
    gamma = 2000
    s_t = 10
    ene_sp = 1
    q_tot = 500
    driver = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=0,
        q_tot=q_tot, n_part=1e4, n_p=n_p)

    # Create witness.
    en = 1e-6
    gamma = 200
    s_t = 3
    ene_sp = 0.1
    q_tot = 50
    witness = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=-80e-6,
        q_tot=q_tot, n_part=1e4, n_p=n_p)

    # Create plasma stage.
    plasma = PlasmaStage(
        length=1e-2, density=n_p, wakefield_model='quasistatic_2d', n_out=50,
        xi_max=20e-6, xi_min=-120e-6, r_max=70e-6, n_xi=280, n_r=70)

    # Do tracking.
    output = plasma.track([driver, witness])

    # Analyze evolution.
    driver_params = analyze_bunch_list(output[0])
    witness_params = analyze_bunch_list(output[1])

    # Assert final parameters are correct.
    final_energy_driver = driver_params['avg_ene'][-1]
    final_energy_witness = witness_params['avg_ene'][-1]
    assert approx(final_energy_driver, rel=1e-10) == 1700.3843657635728
    assert approx(final_energy_witness, rel=1e-10) == 636.3260426124102

    if plot:
        z = driver_params['prop_dist'] * 1e2
        plt.subplot(311)
        plt.plot(z, driver_params['emitt_x']*1e6, label='Driver',
                 color='tab:blue')
        plt.plot(z, witness_params['emitt_x']*1e6, label='Witness',
                 color='tab:orange')
        plt.legend()
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.ylabel("$\\epsilon_{nx}$ [$\\mu$m]")
        plt.subplot(312)
        plt.plot(z, driver_params['rel_ene_spread']*100, color='tab:blue')
        plt.plot(z, witness_params['rel_ene_spread']*100, color='tab:orange')
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.ylabel("$\\frac{\\Delta \\gamma}{\\gamma}$ [%]")
        plt.subplot(313)
        plt.plot(z, driver_params['avg_ene'], color='tab:blue')
        plt.plot(z, witness_params['avg_ene'], color='tab:orange')
        plt.xlabel("z [cm]")
        plt.ylabel("$\\gamma$")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_multibunch_plasma_simulation(plot=True)
