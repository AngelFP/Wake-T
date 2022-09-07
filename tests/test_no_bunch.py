from wake_t import PlasmaStage, GaussianPulse


def test_no_bunch_plasma_simulation():
    """Test a plasma simulation with no electron bunches (only a laser).

    This test checks that a plasma simulation can run without giving
    any bunch as input to `track`.
    """
    # Plasma density.
    n_p = 1e23

    # Create driver.
    laser = GaussianPulse(0., l_0=800e-9, w_0=30e-6, a_0=3,
                          tau=30e-15, z_foc=0.)

    # Create plasma stage.
    plasma = PlasmaStage(
        length=1e-2, density=n_p, wakefield_model='quasistatic_2d', n_out=50,
        xi_max=20e-6, xi_min=-120e-6, r_max=200e-6, n_xi=280, n_r=200,
        laser=laser)

    # Do simulation with no electron bunch.
    plasma.track()


if __name__ == "__main__":
    test_no_bunch_plasma_simulation()
