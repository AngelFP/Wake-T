import numpy as np
import scipy.constants as ct

from wake_t import GaussianPulse
from wake_t.fields.interpolation import interpolate_rz_field


def test_rz_interpolation_one_to_one():
    """
    This test checks that the method `interpolate_rz_field` generates
    an interpolated array that is identical to the original one when both
    arrays use the same grid.
    For this particular test, the original array is the complex envelope of
    a laser pulse.
    """
    # Grid properties.
    xi_max = 20e-6
    xi_min = -20e-6
    r_max = 400e-6
    nxi = 201
    nr = 400
    dr = r_max / nr
    dxi = (xi_max - xi_min) / (nxi - 1)

    # Time steps.
    z_tot = 1e-2
    t_tot = z_tot / ct.c
    nt = 1000
    dt = t_tot / nt

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

    # Get laser envelope.
    a_env = laser.get_envelope()

    # Interpolate complex envelope into the same original grid.
    nxi_new = nxi
    nr_new = nr
    dr_new = r_max / nr_new
    r_min_grid = dr / 2
    xi_new = np.linspace(xi_min, xi_max, nxi_new)
    r_new = np.linspace(dr_new/2, r_max-dr_new/2, nr_new)
    a_env_new = np.zeros((nxi_new, nr_new), dtype=np.complex128)
    interpolate_rz_field(
        a_env, xi_min, r_min_grid, dxi, dr, xi_new, r_new, a_env_new)

    # Both the old and new arrays should be identical. Check that this is the
    # case.
    np.testing.assert_almost_equal(a_env_new, a_env, decimal=14)


def test_rz_interpolation_back_and_forth():
    """
    This test checks that using the `interpolate_rz_field` method to
    interpolate an array into a new grid, and then interpolate the new array
    into the original grid results in an array that is identical to the
    first one.
    This should be the case when the array values vary linearly, due to the
    linear interpolation used by the method.
    """
    # Grid properties.
    xi_max = 200e-6
    xi_min = -200e-6
    r_max = 400e-6
    nxi = 201
    nr = 400
    dr_orig = r_max / nr
    dxi_orig = (xi_max - xi_min) / (nxi - 1)

    # Create original grid.
    r_min_grid_orig = dr_orig / 2
    xi_orig = np.linspace(xi_min, xi_max, nxi)
    r_orig = np.linspace(dr_orig/2, r_max-dr_orig/2, nr)
    xi_grid_orig, r_grid_orig = np.meshgrid(xi_orig, r_orig, indexing='ij')

    # Create array with linear variation in both directions.
    array_orig = xi_grid_orig * 10 + r_grid_orig * 25

    # Interpolate array to a subgrid.
    nxi_new = 600
    nr_new = 200
    dr_new = r_max / nr_new
    dxi_new = (xi_max - xi_min) / (nxi_new - 1)
    r_min_grid_new = dr_new / 2
    xi_new = np.linspace(xi_min, xi_max, nxi_new)
    r_new = np.linspace(dr_new/2, r_max-dr_new/2, nr_new)
    array_new = np.zeros((nxi_new, nr_new))
    interpolate_rz_field(
        array_orig, xi_min, r_min_grid_orig, dxi_orig, dr_orig, xi_new, r_new,
        array_new)

    # Interpolate back to original grid.
    array_orig_interp = np.zeros((nxi, nr))
    interpolate_rz_field(
        array_new, xi_min, r_min_grid_new, dxi_new, dr_new, xi_orig, r_orig,
        array_orig_interp)

    # Since the original array varies linearly, the final array should be equal
    # to the original (within precision). Check that this is true.
    np.testing.assert_almost_equal(array_orig_interp, array_orig, decimal=17)


if __name__ == "__main__":
    test_rz_interpolation_one_to_one()
    test_rz_interpolation_back_and_forth()
