import numpy as np

from wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis_ion.utils import (
    radial_gradient, longitudinal_gradient
)


def test_gradients():
    """Check that the custom gradient methods behave as `np.gradient`."""
    nz = 200
    nr = 100

    z = np.linspace(0, 10, nz)
    r = np.linspace(0, 3, nr)
    dz = z[1] - z[0]
    dr = r[1] - r[0]
    z, r = np.meshgrid(z, r, indexing='ij')
    f = np.cos(z) * np.sin(r)

    dr_f = np.zeros((nz, nr))
    dz_f = np.zeros((nz, nr))

    longitudinal_gradient(f, dz, dz_f)
    radial_gradient(f, dr, dr_f)

    dz_f_np = np.gradient(f, dz, edge_order=2, axis=0)
    dr_f_np = np.gradient(
        np.concatenate((f[:, ::-1], f), axis=1), dr, edge_order=2, axis=1
    )[:, nr:]

    np.testing.assert_array_almost_equal(dz_f, dz_f_np)
    np.testing.assert_array_almost_equal(dr_f, dr_f_np)


if __name__ == "__main__":
    test_gradients()
