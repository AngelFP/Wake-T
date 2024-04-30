import numpy as np
from numpy.testing import assert_almost_equal
import scipy.constants as ct
import matplotlib.pyplot as plt
from wake_t.physics_models.laser.laser_pulse import GaussianPulse
from wake_t.physics_models.laser.utils import unwrap


def test_gaussian_laser_in_vacuum(plot=False):
    """Test evolution of Gaussian laser pulse in vacuum.

    This test evolves a Gaussian laser pulse over a 1cm vacuum. It checks
    that the evolution is in agreement with the analytical expectation and that
    the final value of the sum of `a_mod` has not changed.
    """
    # Grid properties.
    xi_max = 20e-6
    xi_min = -20e-6
    r_max = 400e-6
    nxi = 201
    nr = 400
    dr = r_max / nr

    # Time steps.
    z_tot = 1e-2
    t_tot = z_tot / ct.c
    nt = 1000
    dt = t_tot / nt

    # Plasma susceptibility (zero in vaccum) and normalization density.
    chi = np.zeros((nxi, nr))
    n_p = 1e23

    # Laser parameters in SI units
    tau = 25e-15  # s
    w_0 = 50e-6  # m
    l_0 = 0.8e-6  # m
    z_c = 0.  # m
    a_0 = 3
    z_foc = z_tot / 2

    # Create and initialize laser pulse.
    laser = GaussianPulse(
        z_c, l_0=l_0, w_0=w_0, a_0=a_0, tau=tau, z_foc=z_foc,
        polarization='circular'
    )
    laser.set_envelope_solver_params(xi_min, xi_max, r_max, nxi, nr, dt)
    laser.initialize_envelope()

    # Preallocate arrays
    laser_w = np.zeros(nt + 1)
    laser_a = np.zeros(nt + 1)

    # Store initial laser properties.
    a_env = laser.get_envelope()
    laser_w[0] = calculate_spot_size(a_env, dr)
    laser_a[0] = calculate_a0(a_env)

    # Evolve laser.
    for n in range(nt):
        laser.evolve(chi, n_p)
        a_env = laser.get_envelope()
        laser_w[n+1] = calculate_spot_size(a_env, dr)
        laser_a[n+1] = calculate_a0(a_env)

    # Calculate analytical evolution.
    z = np.linspace(0, z_tot, nt + 1)
    rayleigh_length = ct.pi * w_0**2 / l_0
    laser_w_an = w_0 * np.sqrt(1 + ((z-z_foc)/rayleigh_length)**2)
    laser_a_an = a_0 / np.sqrt(1 + ((z-z_foc)/rayleigh_length)**2)

    # Check that evolution is as expected.
    diff_w = np.max(np.abs(laser_w - laser_w_an) / laser_w_an)
    diff_a = np.max(np.abs(laser_a - laser_a_an) / laser_a_an)

    assert diff_a < 1e-3
    assert diff_w < 1e-3

    # Check that solution hasn't changed.
    a_mod = np.abs(a_env)
    assert_almost_equal(np.sum(a_mod), 7500.380522059235, decimal=10)

    # Make plots.
    if plot:
        plt.figure()
        plt.plot(z, laser_w)
        plt.plot(z, laser_w_an)

        plt.figure()
        plt.plot(z, laser_a)
        plt.plot(z, laser_a_an)

        plt.show()


def test_gaussian_laser_in_vacuum_with_subgrid(plot=False):
    """Test evolution of Gaussian laser pulse in vacuum using subgridding.

    This test evolves a Gaussian laser pulse over a 1cm vacuum. It checks
    that the evolution is in agreement with the analytical expectation and that
    the final value of the sum of `a_mod` has not changed. It also checks that
    the dimensions of the envelope arrays are as expected when using a subgrid.
    """
    # Grid properties.
    xi_max = 20e-6
    xi_min = -20e-6
    r_max = 400e-6
    nxi = 201
    nr = 400
    dr = r_max / nr

    # Subgrid properties
    subgrid_nxi = 400
    subgrid_nr = 900

    # Time steps.
    z_tot = 1e-2
    t_tot = z_tot / ct.c
    nt = 1000
    dt = t_tot / nt

    # Plasma susceptibility (zero in vaccum) and normalization density.
    chi = np.zeros((nxi, nr))
    n_p = 1e23

    # Laser parameters in SI units
    tau = 25e-15  # s
    w_0 = 50e-6  # m
    l_0 = 0.8e-6  # m
    z_c = 0.  # m
    a_0 = 3
    z_foc = z_tot / 2

    # Create and initialize laser pulse.
    laser = GaussianPulse(
        z_c, l_0=l_0, w_0=w_0, a_0=a_0, tau=tau, z_foc=z_foc,
        polarization='circular'
    )
    laser.set_envelope_solver_params(
        xi_min, xi_max, r_max, nxi, nr, dt, subgrid_nz=subgrid_nxi,
        subgrid_nr=subgrid_nr)
    laser.initialize_envelope()

    # Preallocate arrays
    laser_w = np.zeros(nt + 1)
    laser_a = np.zeros(nt + 1)

    # Store initial laser properties.
    a_env = laser.get_envelope()
    laser_w[0] = calculate_spot_size(a_env, dr)
    laser_a[0] = calculate_a0(a_env)

    # Evolve laser.
    for n in range(nt):
        laser.evolve(chi, n_p)
        a_env = laser.get_envelope()
        laser_w[n+1] = calculate_spot_size(a_env, dr)
        laser_a[n+1] = calculate_a0(a_env)

    # Check that the dimensions of the arrays are as expected.
    assert a_env.shape == (nxi, nr)
    assert laser._a_env.shape == (subgrid_nxi + 2, subgrid_nr)

    # Calculate analytical evolution.
    z = np.linspace(0, z_tot, nt + 1)
    rayleigh_length = ct.pi * w_0**2 / l_0
    laser_w_an = w_0 * np.sqrt(1 + ((z-z_foc)/rayleigh_length)**2)
    laser_a_an = a_0 / np.sqrt(1 + ((z-z_foc)/rayleigh_length)**2)

    # Check that evolution is as expected.
    diff_w = np.max(np.abs(laser_w - laser_w_an) / laser_w_an)
    diff_a = np.max(np.abs(laser_a - laser_a_an) / laser_a_an)

    assert diff_a < 1e-3
    assert diff_w < 1e-3

    # Check that solution hasn't changed.
    a_mod = np.abs(a_env)
    assert_almost_equal(np.sum(a_mod), 7500.120208299948, decimal=10)

    # Make plots.
    if plot:
        plt.figure()
        plt.plot(z, laser_w)
        plt.plot(z, laser_w_an)

        plt.figure()
        plt.plot(z, laser_a)
        plt.plot(z, laser_a_an)

        plt.show()


def test_unwrap():
    """Test that the custom function for phase unwrapping implemented for
    numba agrees with the numpy version.
    """
    phase = np.linspace(0, np.pi, num=5)
    phase[3:] += np.pi
    np.testing.assert_allclose(np.unwrap(phase), unwrap(phase))


def calculate_spot_size(a_env, dr):
    # Project envelope to r
    a_proj = np.sum(np.abs(a_env), axis=0)

    # Maximum is on axis
    a_max = a_proj[0]

    # Get first index of value below a_max / e
    i_first = np.where(a_proj <= a_max / np.e)[0][0]

    # Do linear interpolation to get more accurate value of w.
    # We build a line y = a + b*x, where:
    #     b = (y_2 - y_1) / (x_2 - x_1)
    #     a = y_1 - b*x_1
    #
    #     y_1 is the value of a_proj at i_first - 1
    #     y_2 is the value of a_proj at i_first
    #     x_1 and x_2 are the radial positions of y_1 and y_2
    #
    # We can then determine the spot size by interpolating between y_1 and y_2,
    # that is, do x = (y - a) / b, where y = a_max/e
    y_1 = a_proj[i_first - 1]
    y_2 = a_proj[i_first]
    x_1 = (i_first-1) * dr + dr/2
    x_2 = i_first * dr + dr/2
    b = (y_2 - y_1) / (x_2 - x_1)
    a = y_1 - b*x_1
    w = (a_max/np.e - a) / b
    return w


def calculate_a0(a_env):
    a_mod = np.abs(a_env)
    a0 = np.max(a_mod)
    return a0


if __name__ == "__main__":
    test_gaussian_laser_in_vacuum(plot=True)
    test_gaussian_laser_in_vacuum_with_subgrid(plot=True)
    test_unwrap()
