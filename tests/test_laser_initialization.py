import math
import numpy as np
from aptools.plasma_accel.general_equations import (
    laser_radius_at_z_pos, laser_rayleigh_length)
from wake_t import GaussianPulse, FlattenedGaussianPulse


def test_gaussian_init():
    """Test initialization of Gaussian laser pulse.

    This test initializes a Gaussian laser pulse and checks that the spot size
    and a0 of the generated pulse agree with the desired values. This is done
    for a case on focus and a case off focus (z_foc = 1 cm).
    """
    # Grid properties.
    xi_max = 0e-6
    xi_min = -100e-6
    r_max = 100e-6
    nxi = 201
    nr = 200
    dr = r_max / nr

    # Time step (any value).
    dt = 100e-15

    # Laser properties.
    w0 = 30e-6
    a0 = 1
    l0 = 0.8e-6
    z_foc_to_test = [0., 1e-2]

    for z_foc in z_foc_to_test:
        # Create and initialize pulse.
        laser = GaussianPulse(
            xi_c=-50e-6, a_0=a0, w_0=w0, tau=25e-15, z_foc=z_foc, l_0=l0)
        laser.set_envelope_solver_params(xi_min, xi_max, r_max, nxi, nr, dt)
        laser.initialize_envelope()

        # Get complex envelope.
        a_env = laser.get_envelope()

        # Check correct w0.
        w_env = calculate_spot_size(a_env, dr)
        w_analytic = laser_radius_at_z_pos(w0, l0, z_foc)
        assert math.isclose(w_env, w_analytic, rel_tol=1e-4)

        # Check correct a0.
        z_r = laser_rayleigh_length(w0, l0)
        a0_analytic = a0 / np.sqrt(1 + (z_foc/z_r)**2)
        a0_env = np.max(np.abs(a_env))
        assert math.isclose(a0_env, a0_analytic, rel_tol=1e-4)


def test_flattened_gaussian_init():
    """Test initialization of flattened Gaussian laser pulse.

    This test initializes a flattend Gaussian laser pulse and checks that the
    spot size and a0 of the generated pulse agree with the desired values.
    The pulse is initialized with `N=0` so that it has a purely Gaussian shape,
    thus allowing for an easy comparison against theory. This is done
    for a case on focus and a case off focus (z_foc = 1 cm).
    """
    # Grid properties.
    xi_max = 0e-6
    xi_min = -100e-6
    r_max = 100e-6
    nxi = 201
    nr = 200
    dr = r_max / nr

    # Time step (any value).
    dt = 100e-15

    # Laser properties.
    w0 = 30e-6
    a0 = 1
    l0 = 0.8e-6
    z_foc_to_test = [0., 1e-2]

    for z_foc in z_foc_to_test:
        # Create and initialize pulse.
        laser = FlattenedGaussianPulse(
            xi_c=-50e-6, a_0=a0, w_0=w0, tau=25e-15, z_foc=z_foc, l_0=l0, N=0)
        laser.set_envelope_solver_params(xi_min, xi_max, r_max, nxi, nr, dt)
        laser.initialize_envelope()

        # Get complex envelope.
        a_env = laser.get_envelope()

        # Check correct w0.
        w_env = calculate_spot_size(a_env, dr)
        w_analytic = laser_radius_at_z_pos(w0, l0, -z_foc)
        assert math.isclose(w_env, w_analytic, rel_tol=1e-4)

        # Check correct a0.
        z_r = laser_rayleigh_length(w0, l0)
        a0_analytic = a0 / np.sqrt(1 + (z_foc/z_r)**2)
        a0_env = np.max(np.abs(a_env))
        assert math.isclose(a0_env, a0_analytic, rel_tol=1e-4)


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


if __name__ == "__main__":
    test_gaussian_init()
    test_flattened_gaussian_init()
