import numpy as np
import scipy.constants as ct
from wake_t.particles.interpolation import (
    gather_field_cyl_linear, gather_main_fields_cyl_linear)
from wake_t.utilities.bunch_generation import get_matched_bunch


def test_gather_field_cyl_linear():
    """
    Test the method `gather_field_cyl_linear`.

    A 2D array is generated and then gathered exactly at the position
    of the field values. If everything works perfectly, both the original
    and the gathered array should be identical.

    """
    # Create field (r: cell centered, z: node centered)
    n_r = 1000
    n_z = 2000
    r_min = 0
    r_max = 10
    z_min = -10
    z_max = 10
    dr = (r_max - r_min) / n_r
    dz = (z_max - z_min) / (n_z - 1)
    r = np.linspace(dr/2, r_max-dr/2, n_r)
    z = np.linspace(z_min, z_max, n_z)
    R, Z = np.meshgrid(r, z)
    f = np.zeros((n_z+4, n_r+4))
    f[2:-2, 2:-2] = np.sin(R) * np.sin(Z)

    # Particles positioned exactly at grid nodes.
    x_part = R.flatten()
    y_part = np.zeros_like(x_part)
    z_part = Z.flatten()

    # Gather field
    f_part = gather_field_cyl_linear(
        f, z_min, z_max, r_min+dr/2, r_max, dz, dr, x_part, y_part, z_part)

    # Check
    f_part = np.reshape(f_part, (n_z, n_r))
    np.testing.assert_array_almost_equal(f_part, f[2:-2, 2:-2])


def test_gather_main_fields_cyl_linear():
    """
    Test the method `gather_main_fields_cyl_linear`.

    Same test idea as `test_gather_field_cyl_linear`. A 2D array is generated
    which is given as the value of the Wr and Ez fields. These fields are then
    gathered at the exact location of the field values. If everything works
    perfectly, both the original and the gathered arrays should be identical.

    """
    # Create field (r: cell centered, z: node centered)
    n_r = 1000
    n_z = 2000
    r_min = 0
    r_max = 10
    z_min = -10
    z_max = 10
    dr = (r_max - r_min) / n_r
    dz = (z_max - z_min) / (n_z - 1)
    r = np.linspace(dr/2, r_max-dr/2, n_r)
    z = np.linspace(z_min, z_max, n_z)
    R, Z = np.meshgrid(r, z)
    f = np.zeros((n_z+4, n_r+4))
    f[2:-2, 2:-2] = np.sin(R) * np.sin(Z)

    # Particles positioned exactly at grid nodes.
    x_part = R.flatten()
    y_part = np.zeros_like(x_part)
    z_part = Z.flatten()

    # Preallocate field arrays
    n_part = x_part.shape[0]
    ex = np.zeros(n_part)
    ey = np.zeros(n_part)
    ez = np.zeros(n_part)
    bx = np.zeros(n_part)
    by = np.zeros(n_part)
    bz = np.zeros(n_part)

    # Gather field
    gather_main_fields_cyl_linear(
        f, f, f, z_min, z_max, r_min+dr/2, r_max, dz, dr,
        x_part, y_part, z_part, ex, ey, ez, bx, by, bz)

    ex = np.reshape(ex, (n_z, n_r))
    ey = np.reshape(ey, (n_z, n_r))
    ez = np.reshape(ez, (n_z, n_r))
    bx = np.reshape(bx, (n_z, n_r))
    by = np.reshape(by, (n_z, n_r))
    bz = np.reshape(bz, (n_z, n_r))

    # Check
    np.testing.assert_array_almost_equal(ex, f[2:-2, 2:-2])
    np.testing.assert_array_almost_equal(ez, f[2:-2, 2:-2])
    np.testing.assert_array_almost_equal(by, f[2:-2, 2:-2])


def test_gather_main_fields_cyl_linear_at_bunch():
    """
    Test `gather_main_fields_cyl_linear` by gathering fields at the location
    of the particles of a bunch. This test checks that the sum of the
    gathered field values has not changed.

    """
    # Bunch parameters.
    en = 1e-6  # m
    ene = 200  # units of beta*gamma
    ene_sp = 0.3  # %
    xi_c = 0.  # m
    s_t = 3  # fs
    q_tot = 1  # pC
    n_part = 1e4

    # Set numpy random seed to get reproducible results
    np.random.seed(1)

    # Generate bunch (in this case, matched to a density of 10^23 cm-3)
    bunch = get_matched_bunch(
        en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part, n_p=1e23)

    # Create field (r: cell centered, z: node centered)
    n_r = 1000
    n_z = 2000
    r_min = 0
    r_max = 30e-6
    z_min = -10e-6
    z_max = 10e-6
    dr = (r_max - r_min) / n_r
    dz = (z_max - z_min) / (n_z - 1)
    r = np.linspace(dr/2, r_max-dr/2, n_r)
    z = np.linspace(z_min, z_max, n_z)
    R, Z = np.meshgrid(r, z)
    f = np.zeros((n_z+4, n_r+4))
    f[2:-2, 2:-2] = np.sin(R/r_max-1) * np.sin(Z/z_max-1)

    # Get preallocated field arrays
    ex, ey, ez, bx, by, bz = bunch.get_field_arrays()
    
    # Gather field
    gather_main_fields_cyl_linear(
        f, f, f, z_min, z_max, r_min+dr/2, r_max, dz, dr,
        bunch.x, bunch.y, bunch.xi, ex, ey, ez, bx, by, bz)

    # Check
    np.testing.assert_almost_equal(np.sum(ex / ct.c + by), 15.709780373078333)
    np.testing.assert_almost_equal(np.sum(ey / ct.c - bx), -75.57014245377374)
    np.testing.assert_almost_equal(np.sum(ez), 6798.124201568496)


if __name__ == '__main__':
    test_gather_field_cyl_linear()
    test_gather_main_fields_cyl_linear()
    test_gather_main_fields_cyl_linear_at_bunch()
