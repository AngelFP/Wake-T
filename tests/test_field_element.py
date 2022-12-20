import numpy as np
import scipy.constants as ct
from pytest import approx

from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_twiss
from wake_t.fields.analytical_field import AnalyticalField
from wake_t.beamline_elements import FieldElement
from wake_t.diagnostics import analyze_bunch


def b_x(x, y, z, t, bx, constants):
    """B_x component."""
    k = constants[0]
    bx -= k * y


def b_y(x, y, z, t, by, constants):
    """B_y component."""
    k = constants[0]
    by += k * x


def test_field_element_tracking():
    """Test that tracking in a field element works as expected by tracking
    a bunch through an analytical field (azimuthal magnetic field)"""
    # Set numpy random seed to get reproducible results
    np.random.seed(1)

    # Define field.
    b_theta = AnalyticalField(b_x=b_x, b_y=b_y, constants=[-100])

    # Create bunch.
    emitt_nx = emitt_ny = 1e-6  # m
    beta_x = beta_y = 1.  # m
    s_t = 100.  # fs
    gamma_avg = 1000
    ene_spread = 0.1  # %
    q_bunch = 30  # pC
    xi_avg = 0.  # m
    n_part = 1e4
    bunch = get_gaussian_bunch_from_twiss(
        en_x=emitt_nx, en_y=emitt_ny, a_x=0, a_y=0, b_x=beta_x, b_y=beta_y,
        ene=gamma_avg, ene_sp=ene_spread, s_t=s_t, xi_c=xi_avg,
        q_tot=q_bunch, n_part=n_part, name='elec_bunch')

    # Create field element.
    element = FieldElement(
        length=1,
        dt_bunch=1e-3/ct.c,
        n_out=10,
        fields=[b_theta]
    )

    # Do tracking.
    element.track(bunch, opmd_diag=True)

    # Check that results have not changed.
    bunch_params = analyze_bunch(bunch)
    beta_x = bunch_params['beta_x']
    assert approx(beta_x, rel=1e-10) == 0.054508554263608434


if __name__ == '__main__':
    test_field_element_tracking()