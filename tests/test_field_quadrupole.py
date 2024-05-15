import copy

import numpy as np
import scipy.constants as ct

from wake_t.beamline_elements import FieldQuadrupole, Quadrupole
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_twiss


def test_field_vs_tm_quadrupole():
    """
    This test checks that the FieldElement-based quadrupole (FieldQuadrupole)
    and the TM quadrupole (Quadrupole) produce similar results.
    """

    emitt_nx = emitt_ny = 1e-6  # m
    beta_x = beta_y = 1.  # m
    s_t = 100.  # fs
    gamma_avg = 1000
    ene_spread = 0.1  # %
    q_bunch = 30  # pC
    xi_avg = 0.  # m
    n_part = 1e4
    bunch_1 = get_gaussian_bunch_from_twiss(
        en_x=emitt_nx, en_y=emitt_ny, a_x=0, a_y=0, b_x=beta_x, b_y=beta_y,
        ene=gamma_avg, ene_sp=ene_spread, s_t=s_t, xi_c=xi_avg,
        q_tot=q_bunch, n_part=n_part, name='elec_bunch')

    bunch_2 = bunch_1.copy()

    foc_strength = 100  # T/m
    quadrupole_length = 0.05  # m
    k1 = foc_strength * ct.e / ct.m_e / ct.c / gamma_avg

    field_quadrupole = FieldQuadrupole(quadrupole_length, foc_strength)
    tm_quadrupole = Quadrupole(quadrupole_length, k1)

    field_quadrupole.track(bunch_1)
    tm_quadrupole.track(bunch_2)

    np.testing.assert_allclose(bunch_1.x, bunch_2.x, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose(bunch_1.y, bunch_2.y, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose(bunch_1.xi, bunch_2.xi, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose(bunch_1.px, bunch_2.px, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(bunch_1.py, bunch_2.py, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(bunch_1.pz, bunch_2.pz, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    test_field_vs_tm_quadrupole()
