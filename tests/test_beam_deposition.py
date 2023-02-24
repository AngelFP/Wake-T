import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_size
from wake_t.particles.deposition import deposit_3d_distribution


def test_gaussian_beam(show=False):
    """
    This test checks the accuracy of the charge deposition of a Gaussian beam
    into a cylindrical grid.

    It tests:
        (a) that the desposited charge equals the original charge.
        (b) that the charge density in the grid agrees with
            the expected charge density of a Gaussian beam.
        (c) that the resulting azimuthal magnetic field from the charge density
            agrees with the analytical expectations for a Gaussian
            distribution.

    The test is performed 4 times by using a low and a high resolution, and
    both the `'linear'` and `'cubic'` deposition methods.
    """
    # Set seed fo reproducible results.
    np.random.seed(0)

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

    # Resolutions and shapes to test
    n_r_test = [20, 100]
    n_z_test = [41, 101]
    p_shape_test = ['linear', 'cubic']

    for n_r, n_z in zip(n_r_test, n_z_test):
        for p_shape in p_shape_test:
            # Generate grid
            z_min = -10e-6
            z_max = 10e-6
            r_max = 10e-6
            dr = r_max / n_r
            dz = (z_max - z_min) / (n_z - 1)
            r_fld = np.linspace(dr / 2, r_max - dr / 2, n_r)
            z_fld = np.linspace(z_min, z_max, n_z)
            rho_fld = np.zeros((n_z+4, n_r+4))

            # Deposit beam distribution.
            deposit_3d_distribution(
                bunch.xi, bunch.x, bunch.y, bunch.q,
                z_fld[0], r_fld[0], n_z, n_r, dz, dr, rho_fld,
                p_shape, use_ruyten=True)

            # Remove guard cells.
            rho_fld = rho_fld[2:-2, 2:-2]

            # Check charge conservation.
            rel_q_dev = (np.sum(bunch.q) - np.sum(rho_fld)) / np.sum(bunch.q)
            assert rel_q_dev < 1e-12

            # Calculate charge density.
            q_cell = np.arange(n_r)
            v_cell = np.pi * ((q_cell + 1) ** 2 - q_cell**2) * dr ** 2 * dz
            rho_fld /= v_cell

            # Analytical expectation at beam center.
            rho_ana_0 = rho_gaussian_beam(r_fld, 0., -q_tot, s_r, s_z)

            # Deposited density at beam center.
            i_center = int((n_z - 1) / 2)
            rho_grid_0 = rho_fld[i_center]

            # Normalization factor.
            norm_factor = q_tot*1e-12 / ((2*np.pi)**(3/2) * s_r**2 * s_z)

            # Check charge density conservation.
            assert np.max(np.abs(rho_grid_0 - rho_ana_0)) / norm_factor < 0.09

            # Calculate azimuthal magnetic field (analytical and from grid)
            b_theta_ana_0 = b_theta_gaussian_beam(
                r_fld, 0., q_tot, s_r, s_z, gamma)
            b_theta_fld = calculate_b_theta(rho_fld, r_fld, dr)
            b_theta_grid_0 = b_theta_fld[i_center]

            # Relative deviation w.r.t. analytical.
            rel_b_dev = np.abs((b_theta_grid_0 - b_theta_ana_0)/b_theta_ana_0)

            # Check long range fields are accurate.
            assert rel_b_dev[-1] < 3e-3

            # Check reasonable deviation in general. At low res, ~10%
            # differences can occur close to the axis.
            assert np.max(rel_b_dev) < 13e-2

            if show:
                plt.figure(figsize=(8, 4))
                plt.subplot(121)
                plt.plot(r_fld, rho_ana_0)
                plt.plot(r_fld, rho_grid_0)

                plt.subplot(122)
                plt.plot(r_fld, b_theta_ana_0)
                plt.plot(r_fld, b_theta_grid_0)

                plt.show()


def rho_gaussian_beam(r, z, q_tot, s_r, s_z):
    """ Analytical charge density distribution of Gaussian beam. """
    return (q_tot*1e-12 / ((2*np.pi)**(3/2) * s_r**2 * s_z)
            * np.exp(-z**2 / (2*s_z**2))
            * np.exp(-r**2 / (2*s_r**2)))


def b_theta_gaussian_beam(r, z, q_tot, s_r, s_z, gamma):
    """ Analytical azimuthal magnetic field for a Gaussian beam. """
    beta = np.sqrt(1 - 1/gamma**2)
    e_r = (
        1 / ((2 * np.pi) ** (3/2) * ct.epsilon_0)
        * q_tot * 1e-12 / s_z * np.exp(-z**2 / (2*s_z**2))
        * (1 - np.exp(-r**2 / (2*s_r**2))) / r
    )
    b_theta = beta / ct.c * e_r
    return b_theta


def calculate_b_theta(rho, r_fld, dr):
    """ Calculate azimuthal magnetic field from rho. """
    subs = rho*r_fld/2
    subs[:, 0] += rho[:, 0]*r_fld[0]/4
    b_theta = -(
        (np.cumsum(rho*r_fld, axis=1) - subs) * dr
        / np.abs(r_fld)
        / (ct.c * ct.epsilon_0)
    )
    return b_theta


if __name__ == '__main__':
    test_gaussian_beam(show=True)
