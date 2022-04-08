import numpy as np
from wake_t.particles.deposition import deposit_3d_distribution


def test_uniform_plasma_deposition():
    """
    This test checks that a uniform distribution of plasma particles
    results in a uniform density distribution after being deposited into
    a grid.

    """
    # Grid and particle parameters.
    r_max = 10
    z_min = -5
    z_max = 5
    n_r = 20
    n_z = 21
    dr = r_max / n_r
    dz = (z_max - z_min) / (n_z - 1)
    r_fld = np.linspace(dr / 2, r_max - dr / 2, n_r)
    z_fld = np.linspace(z_min, z_max, n_z)
    parabolic_coefficient = 0.
    ppc = 5
    r_max_plasma = r_max

    # Create plasma particles.
    dr_p = dr / ppc
    n_part = int(np.round(r_max_plasma / dr * ppc))
    r_max_plasma = n_part * dr_p
    r = np.linspace(dr_p / 2, r_max_plasma - dr_p / 2, n_part)
    pz = np.zeros_like(r)
    gamma = np.ones_like(r)
    q = dr_p * r + dr_p * parabolic_coefficient * r**3

    # Possible particle shapes.
    p_shapes = ['linear', 'cubic']

    # Test all shapes.
    for p_shape in p_shapes:

        # Allocate density array.
        rho_fld = np.zeros((n_z+4, n_r+4))

        # Deposit plasma column along the whole grid.
        for step in np.arange(n_z):
            i = -1 - step
            z_i = z_fld[i]
            w_rho = q / (dr * r * (1 - pz/gamma))
            z = np.full_like(r, z_i)
            x = r
            y = np.zeros_like(r)
            deposit_3d_distribution(z, x, y, w_rho, z_min, r_fld[0], n_z, n_r,
                                    dz, dr, rho_fld, p_shape=p_shape)

        # Check array is uniform.
        assert np.sum(np.abs(rho_fld[2:-2, 2:-2] - 1.)) < 1e-12


if __name__ == "__main__":
    test_uniform_plasma_deposition()
