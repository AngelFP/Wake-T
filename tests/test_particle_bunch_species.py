import numpy as np
import scipy.constants as ct
import matplotlib.pyplot as plt

from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.fields.analytical_field import AnalyticalField
from wake_t.tracking.tracker import Tracker


def test_different_species_single_particle(make_plots=False):
    """Test that the evolution of different particle species is correct.

    This test simulates the evolution of electrons, positrons and protons in
    a constant magnetic field. It checks that the numerical results obtained
    with all particle pushers agree with the analytical solution.
    """
    # Define particles with different mass and charge.
    electron = {'m_species': ct.m_e, 'q_species': -ct.e}
    positron = {'m_species': ct.m_e, 'q_species': ct.e}
    proton = {'m_species': ct.m_p, 'q_species': ct.e}
    species_params = [electron, positron, proton]

    # Total tracking time.
    tracking_time = 1e-12

    # Initial longitudinal momentum.
    pz = 100.

    # Magnetic field to apply
    b_y = 10.  # T

    def constant_by(x, y, z, t, by, constants):
        by[:] = constants[0]

    field = AnalyticalField(b_y=constant_by, constants=[b_y])

    # List of particle pushers to test.
    pushers_to_test = ['rk4', 'boris']

    # Test particle evolution with all pushers.
    for pusher in pushers_to_test:
        # Create list with all particles to track.
        particles = []
        for params in species_params:
            particle = ParticleBunch(
                w=np.array([1.]),
                x=np.array([0.]),
                y=np.array([0.]),
                xi=np.array([0.]),
                px=np.array([0.]),
                py=np.array([0.]),
                pz=np.array([pz]),
                **params
            )
            particles.append(particle)

        # Create tracker.
        tracker = Tracker(
            t_final=tracking_time,
            bunches=particles,
            dt_bunches=[0.1e-15, 0.1e-15, 0.1e-15],
            fields=[field],
            n_diags=50,
            bunch_pusher=pusher
        )

        # Track particles.
        particles_history = tracker.do_tracking()

        # Compare with analytical solution.
        for particle, params in zip(particles, species_params):
            gamma = np.sqrt(1 + pz**2)
            vz_0 = pz / gamma * ct.c
            w = - params['q_species'] * b_y / (params['m_species'] * gamma)
            x_an = vz_0 / w * (1. - np.cos(w * tracking_time))
            np.testing.assert_almost_equal(particle.x, x_an, decimal=14)

        # Make plot.
        if make_plots:
            fig, axes = plt.subplots(2, 1)
            for particle_history in particles_history:
                x_history = np.zeros(len(particle_history))
                y_history = np.zeros(len(particle_history))
                xi_history = np.zeros(len(particle_history))
                px_history = np.zeros(len(particle_history))
                pz_history = np.zeros(len(particle_history))
                for i, p in enumerate(particle_history):
                    x_history[i] = p.x
                    y_history[i] = p.y
                    xi_history[i] = p.xi + p.prop_distance
                    px_history[i] = p.px
                    pz_history[i] = p.pz

                axes[0].plot(xi_history, x_history)
                axes[1].plot(xi_history, px_history)
            plt.show()


if __name__ == '__main__':
    test_different_species_single_particle(make_plots=True)
