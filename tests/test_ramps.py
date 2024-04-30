import numpy as np

from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_twiss
from wake_t import PlasmaRamp
from wake_t.diagnostics import analyze_bunch


def test_downramp():
    """
    Test that a plasma downramp changes the beta function of an initial beam
    as expected. The simplest `focusing_blowout` wakefield model is used in
    order to have a fast test.
    """
    # Set seed fo reproducible results.
    np.random.seed(0)

    # generate bunch
    en = 1e-6
    alpha = 0
    beta = 0.3e-3
    s_t = 3
    ene = 600
    ene_sp = 2
    bunch = get_gaussian_bunch_from_twiss(
        en_x=en, en_y=en, a_x=alpha, a_y=alpha, b_x=beta, b_y=beta, ene=ene,
        ene_sp=ene_sp, s_t=s_t, xi_c=0e-6, q_tot=100, n_part=1e4)

    downramp = PlasmaRamp(
        1e-2, ramp_type='downramp', profile='inverse_square',
        plasma_dens_top=1e23, plasma_dens_down=1e21,
        wakefield_model='focusing_blowout', n_out=3)

    downramp.track(bunch)
    bunch_params = analyze_bunch(bunch)
    beta_x = bunch_params['beta_x']
    assert beta_x == 0.009750308724018872


def test_upramp():
    """
    Test that a plasma upramp changes the beta function of an initial beam
    as expected. The simplest `focusing_blowout` wakefield model is used in
    order to have a fast test.
    """
    # Set seed fo reproducible results.
    np.random.seed(0)

    # generate bunch
    en = 1e-6
    alpha = 0
    beta = 1e-2
    s_t = 3
    ene = 600
    ene_sp = 2
    bunch = get_gaussian_bunch_from_twiss(
        en_x=en, en_y=en, a_x=alpha, a_y=alpha, b_x=beta, b_y=beta, ene=ene,
        ene_sp=ene_sp, s_t=s_t, xi_c=0e-6, q_tot=100, n_part=1e4)

    downramp = PlasmaRamp(
        1e-2, ramp_type='upramp', profile='inverse_square',
        plasma_dens_top=1e23, plasma_dens_down=1e21,
        wakefield_model='focusing_blowout', n_out=3)

    downramp.track(bunch)
    bunch_params = analyze_bunch(bunch)
    beta_x = bunch_params['beta_x']
    assert beta_x == 0.000763155045965493


if __name__ == '__main__':
    test_downramp()
    test_upramp()
