import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_size
from wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis.solver import (
    calculate_beam_source)


def test_sc_gaussian_beam(show=False):
    """Test the space-charge calculation used in the `quasistatic_2d`
    wakefield model.

    A Gaussian beam is generated and the calculated space-charge field is
    compared against the analytical expectation.

    Parameters
    ----------
    show : bool, optional
        Whether to show a lineout of the fields, by default False
    """
    # Set seed fo reproducible results.
    np.random.seed(0)

    # Plasma density.
    n_p = 1e23
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    E0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p*1e-6)

    # Bunch parameters
    s_x = 1e-6
    en = 1e-6
    gamma = 200
    ene_sp = 1e-5
    s_t = 10
    q_tot = 100
    n_part = 1e7
    s_z = s_t * (1e-15 * ct.c)
    s_r = s_x

    # Generate bunch
    bunch = get_gaussian_bunch_from_size(
        en, en, s_x, s_x, gamma, ene_sp, s_t, 0, q_tot, n_part=n_part)

    # Resolutions ans shapes to test
    n_r_test = [20, 100]
    n_z_test = [41, 101]
    p_shape_test = ['linear', 'cubic']

    for n_r, n_z in zip(n_r_test, n_z_test):
        for p_shape in p_shape_test:
            # Generate grid
            z_min = -10e-6 / s_d
            z_max = 10e-6 / s_d
            r_max = 10e-6 / s_d
            dr = r_max / n_r
            dz = (z_max - z_min) / (n_z - 1)
            r_fld = np.linspace(dr / 2, r_max - dr / 2, n_r)
            z_fld = np.linspace(z_min, z_max, n_z)

            b_theta_grid = np.zeros((n_z+4, n_r+4))
            calculate_beam_source(
                bunch, n_p, n_r, n_z,
                r_fld[0], z_fld[0], dr, dz, p_shape, b_theta_grid)

            # Remove guard cells.
            b_theta_grid = b_theta_grid[2:-2, 2:-2]

            # Convert to SI.
            b_theta_grid *= E0 / ct.c

            # Deposited density at beam center.
            i_center = int((n_z - 1) / 2)

            # Calculate azimuthal magnetic field (analytical and from grid)
            b_theta_ana_0 = b_theta_gaussian_beam(
                r_fld*s_d, 0., q_tot, s_r, s_z, gamma)
            b_theta_grid_0 = b_theta_grid[i_center]

            # Relative deviation w.r.t. analytical.
            rel_b_dev = np.abs((b_theta_grid_0 - b_theta_ana_0)/b_theta_ana_0)

            # Check long range fields are accurate.
            assert rel_b_dev[-1] < 3e-3

            # Check reasonable deviation in general. At low res, ~10%
            # differences can occur close to axis.
            assert np.max(rel_b_dev) < 13e-2

            if show:
                plt.plot(r_fld, b_theta_ana_0)
                plt.plot(r_fld, b_theta_grid_0)

                plt.show()


def b_theta_gaussian_beam(r, z, q_tot, s_r, s_z, gamma):
    """ Analytical azimuthal magnetic field for a Gaussian beam. """
    beta = np.sqrt(1 - 1/gamma**2)
    e_r = (1/((2*np.pi)**(3/2) * ct.epsilon_0)
           * (-q_tot) * 1e-12 / s_z * np.exp(-z**2 / (2*s_z**2))
           * (1 - np.exp(-r**2 / (2*s_r**2))) / r)
    b_theta = beta / ct.c * e_r
    return b_theta


if __name__ == '__main__':
    test_sc_gaussian_beam()
