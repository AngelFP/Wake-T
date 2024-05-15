from copy import deepcopy

import numpy as np
import scipy.constants as ct

from wake_t import GaussianPulse, PlasmaStage, Beamline
from wake_t.utilities.bunch_generation import get_matched_bunch


def test_variable_parabolic_coefficient():
    """
    Checks that a z-dependent parabolic coefficient works as expected.
    A piecewise function for the coefficient is defined, which should be
    equivalent to tracking multiple plasma stages each having a different
    parabolic coefficient.
    """
    # Plasma density.
    n_p = 1e23

    # Grid and field parameters.
    xi_max = 0
    xi_min = -100e-6
    r_max = 100e-6
    Nxi = 100
    Nr = 100
    dz_fields = 1e-3

    # Laser parameters.
    a0 = 1
    w0 = 30e-6
    tau = 25e-15
    l0 = 0.8e-6

    # Guiding plasma channel
    r_e = ct.e**2 / (4. * np.pi * ct.epsilon_0 * ct.m_e * ct.c**2)
    rel_delta_n_over_w2 = 1. / (np.pi * r_e * w0**4 * n_p)

    # Length and parabolic coefficient of each section (stretch).
    L_stretches = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
    pc_stretches = rel_delta_n_over_w2 * np.array([1, 2, 1, 0.5, 2])

    # Define z-dependent parabolic coefficient.
    def parabolic_coefficient(z):
        z_stretch_end = np.cumsum(L_stretches)
        i = np.sum(np.float32(z_stretch_end) <= np.float32(z))
        if i == len(L_stretches):
            i -= 1
        return pc_stretches[i]

    # Create identical laser pulses for each case.
    laser = GaussianPulse(-50e-6, a0, w0, tau, z_foc=0., l_0=l0)
    laser_1 = deepcopy(laser)
    laser_2 = deepcopy(laser)

    # Create identical bunches for each case.
    bunch = get_matched_bunch(
        1e-6, 1e-6, 200, 1, 3, laser.xi_c - 30e-6, 1e-6, 1e4, n_p=n_p)
    bunch_1 = bunch.copy()
    bunch_2 = bunch.copy()

    # Create single plasma stage (containing all sections).
    plasma_single = PlasmaStage(
        np.sum(L_stretches), n_p, wakefield_model='quasistatic_2d', n_out=10,
        xi_min=xi_min, xi_max=xi_max, r_max=r_max, r_max_plasma=r_max,
        laser=laser_1, laser_evolution=True, n_r=Nr, n_xi=Nxi, ppc=2,
        dz_fields=dz_fields, parabolic_coefficient=parabolic_coefficient)

    # Track single plasma.
    plasma_single.track(bunch_1)

    # Create set of plasma stages, one per section.
    sub_stages = []
    for i, (l, pc) in enumerate(zip(L_stretches, pc_stretches)):
        stage = PlasmaStage(
            l, n_p, wakefield_model='quasistatic_2d', n_out=3,
            xi_min=xi_min, xi_max=xi_max, r_max=r_max, r_max_plasma=r_max,
            laser=laser_2, laser_evolution=True, n_r=Nr, n_xi=Nxi, ppc=2,
            dz_fields=dz_fields, parabolic_coefficient=pc)
        sub_stages.append(stage)
    plasma_multi = Beamline(sub_stages)

    # Track through set of stages.
    plasma_multi.track(bunch_2)

    # Get final envelope of both lasers.
    a_env_1 = laser_1.get_envelope()
    a_mod_1 = np.abs(a_env_1)
    a_phase_1 = np.angle(a_env_1)
    a_env_2 = laser_2.get_envelope()
    a_mod_2 = np.abs(a_env_2)
    a_phase_2 = np.angle(a_env_2)

    # Check that both envelopes are equal.
    np.testing.assert_almost_equal(a_mod_2, a_mod_1)
    np.testing.assert_almost_equal(a_phase_2, a_phase_1)


if __name__ == "__main__":
    test_variable_parabolic_coefficient()
