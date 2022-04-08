from wake_t import GaussianPulse, PlasmaRamp, PlasmaStage, Beamline
from wake_t.utilities.bunch_generation import get_matched_bunch


def test_plasma_beamline():
    """
    This test checks that a beamline consisting of several consecutive plasma
    elements driven by the same laser works properly.

    At the moment, it simply checks that the simulation runs. More detailed
    checks should be added.

    """
    # Laser pulse
    laser = GaussianPulse(
        100e-6, l_0=800e-9, w_0=50e-6, a_0=3, tau=30e-15, z_foc=0)

    # Electron bunch
    bunch = get_matched_bunch(
        en_x=1e-6, en_y=1e-6, ene=200, ene_sp=0.1, s_t=1, xi_c=0, q_tot=10,
        n_part=1e4, n_p=1e23)

    # Plasma elements
    wf_params = {
        'n_r': 200,
        'n_xi': 200,
        'dz_fields': 0.5e-3,
        'xi_min': -30e-6,
        'xi_max': 120e-6,
        'r_max': 200e-6,
        'laser': laser,
        'laser_evolution': True
    }
    plateau = PlasmaStage(
        5e-3, 1e23, wakefield_model='quasistatic_2d', n_out=3, **wf_params)
    downramp = PlasmaRamp(
        5e-3, profile='inverse_square', ramp_type='downramp',
        wakefield_model='quasistatic_2d', plasma_dens_top=1e23,
        decay_length=1e-3, n_out=3, **wf_params)
    plasma_beamline = Beamline([plateau, downramp])

    # Do tracking
    plasma_beamline.track(bunch)


if __name__ == '__main__':
    test_plasma_beamline()
