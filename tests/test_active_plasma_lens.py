from pytest import approx
import numpy as np

from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_twiss
from wake_t.beamline_elements.active_plasma_lens import ActivePlasmaLens
from wake_t.diagnostics import analyze_bunch


def test_active_plasma_lens():
    """
    Check that the beam propagation through an active plasma lens without
    wakefields is accurate.
    """
    # Set numpy random seed to get reproducible results
    np.random.seed(1)

    # generate bunch
    en = 1e-6
    alpha = 0
    beta = 1.
    s_t = 10
    ene = 600
    ene_sp = 2
    bunch = get_gaussian_bunch_from_twiss(
        en_x=en, en_y=en, a_x=alpha, a_y=alpha, b_x=beta, b_y=beta, ene=ene,
        ene_sp=ene_sp, s_t=s_t, xi_c=0e-6, q_tot=10, n_part=1e5)
    apl = ActivePlasmaLens(1e-2, 1000, wakefields=False, n_out=3)

    apl.track(bunch)
    bunch_params = analyze_bunch(bunch)
    gamma_x = bunch_params['gamma_x']
    assert approx(gamma_x, rel=1e-10) == 92.38407675999406


def test_active_plasma_lens_with_wakefields():
    """
    Check that the beam propagation through an active plasma lens with
    wakefields is accurate.
    """
    # Set numpy random seed to get reproducible results.
    np.random.seed(1)

    # Generate bunch.
    en = 1e-6
    alpha = 0
    beta = 1.
    s_t = 10
    ene = 600
    ene_sp = 2
    bunch = get_gaussian_bunch_from_twiss(
        en_x=en, en_y=en, a_x=alpha, a_y=alpha, b_x=beta, b_y=beta, ene=ene,
        ene_sp=ene_sp, s_t=s_t, xi_c=0e-6, q_tot=10, n_part=1e5)

    # Define APL.
    apl = ActivePlasmaLens(
        1e-2, 1000, wakefields=True, n_out=3, density=1e23,
        wakefield_model='quasistatic_2d', r_max=200e-6, xi_min=-30e-6,
        xi_max=30e-6, n_r=200, n_xi=120, dz_fields=0.2e-3, ppc=4
    )

    # Do tracking.
    apl.track(bunch)

    # Analyze and check results.
    bunch_params = analyze_bunch(bunch)
    gamma_x = bunch_params['gamma_x']
    assert approx(gamma_x, rel=1e-10) == 77.32021188373825


if __name__ == '__main__':
    test_active_plasma_lens()
    test_active_plasma_lens_with_wakefields()
