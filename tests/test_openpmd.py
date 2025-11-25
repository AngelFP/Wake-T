import os

import numpy as np
import matplotlib.pyplot as plt
from wake_t import PlasmaStage, GaussianPulse
from wake_t.utilities.bunch_generation import get_matched_bunch
from openpmd_viewer.addons import LpaDiagnostics
from wake_t.physics_models.laser.laser_pulse import OpenPMDPulse


tests_output_folder = "./tests_output"


def test_openpmd_viewer():
    """
    This test checks that the data generated with Wake-T can be read and
    analyzed with the openPMD viewer. Both the `h5py` and `openpmd-api`
    backends are tested.
    """
    # Diagnostics directory.
    test_dir = os.path.join(tests_output_folder, "openpmd_viewer_test")
    diag_dir = os.path.join(test_dir, "diags")

    # Create laser driver.
    laser = GaussianPulse(100e-6, l_0=800e-9, w_0=50e-6, a_0=3, tau=30e-15, z_foc=0.0)

    # Create bunch (matched to a blowout at a density of 10^{23} m^{-3}).
    en = 1e-6  # m
    ene = 200  # units of beta*gamma
    ene_sp = 0.3  # %
    xi_c = laser.xi_c - 55e-6  # m
    s_t = 10  # fs
    q_tot = 100  # pC
    n_part = 1e4
    np.random.seed(42)
    bunch = get_matched_bunch(en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part, n_p=1e23)

    # Create plasma stage.
    plasma = PlasmaStage(
        1e-2,
        1e23,
        laser=laser,
        wakefield_model="quasistatic_2d",
        n_out=50,
        laser_evolution=True,
        r_max=200e-6,
        r_max_plasma=120e-6,
        xi_min=30e-6,
        xi_max=120e-6,
        n_r=200,
        n_xi=180,
        dz_fields=0.5e-3,
        ppc=5,
    )

    # Do tracking.
    plasma.track(bunch, opmd_diag=True, diag_dir=diag_dir)

    # Check that the data can be analyzed.
    for backend in ["h5py", "openpmd-api"]:
        diags = LpaDiagnostics(os.path.join(diag_dir, "hdf5"), backend=backend)
        its = diags.iterations
        emitt_x = np.zeros(len(its))
        emitt_y = np.zeros(len(its))
        for i, it in enumerate(its):
            diags.get_field("rho", iteration=it, plot=True)
            plt.savefig(os.path.join(diag_dir, "{}_rho_{}.png".format(backend, it)))
            plt.clf()

            diags.get_field("E", "x", iteration=it, plot=True)
            plt.savefig(os.path.join(diag_dir, "{}_Ex_{}.png".format(backend, it)))
            plt.clf()

            emitt_x[i], emitt_y[i] = diags.get_emittance(iteration=it)

        plt.plot(emitt_x)
        plt.plot(emitt_y)
        plt.savefig(os.path.join(diag_dir, "{}_emitt.png".format(backend)))


def test_openpmd_pulse():
    """
    This test checks that the laser envelope generated with Wake-T can be read
    from openPMD files using the OpenPMDPulse class, and that the pulse can be
    used in a Wake-T simulation. The simulation run using this approach is compared
    to a reference simulation where the laser envelope is generated directly with
    Wake-T. Both simulations should yield the same results.
    """
    # Diagnostics directory of the reference simulation
    test_dir = os.path.join(tests_output_folder, "openpmd_viewer_test")
    diag_dir_orig = os.path.join(test_dir, "diags")

    # Load laser envelope from openPMD file
    file_name = os.path.join(diag_dir_orig, "hdf5", "data%T.h5")
    laser = OpenPMDPulse(file_name=file_name, envelope_name="a", iteration=0)

    # Create bunch (matched to a blowout at a density of 10^{23} m^{-3}).
    en = 1e-6  # m
    ene = 200  # units of beta*gamma
    ene_sp = 0.3  # %
    xi_c = 100e-6 - 55e-6  # m
    s_t = 10  # fs
    q_tot = 100  # pC
    n_part = 1e4
    np.random.seed(42)
    bunch = get_matched_bunch(en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part, n_p=1e23)

    # Create plasma stage.
    plasma = PlasmaStage(
        1e-2,
        1e23,
        laser=laser,
        wakefield_model="quasistatic_2d",
        n_out=50,
        laser_evolution=True,
        r_max=200e-6,
        r_max_plasma=120e-6,
        xi_min=30e-6,
        xi_max=120e-6,
        n_r=200,
        n_xi=180,
        dz_fields=0.5e-3,
        ppc=5,
    )

    # Diagnostics directory.
    test_dir = os.path.join(tests_output_folder, "openpmd_pulse_test")
    diag_dir = os.path.join(test_dir, "diags")

    # Do tracking.
    plasma.track(bunch, opmd_diag=True, diag_dir=diag_dir)

    # Analyze data
    backend = "h5py"
    diags = LpaDiagnostics(os.path.join(diag_dir, "hdf5"), backend=backend)
    its = diags.iterations
    emitt_x = np.zeros(len(its))
    emitt_y = np.zeros(len(its))
    for i, it in enumerate(its):
        diags.get_field("rho", iteration=it, plot=True)
        plt.savefig(os.path.join(diag_dir, "{}_rho_{}.png".format(backend, it)))
        plt.clf()

        diags.get_field("E", "x", iteration=it, plot=True)
        plt.savefig(os.path.join(diag_dir, "{}_Ex_{}.png".format(backend, it)))
        plt.clf()

        emitt_x[i], emitt_y[i] = diags.get_emittance(iteration=it)

    plt.plot(emitt_x)
    plt.plot(emitt_y)
    plt.savefig(os.path.join(diag_dir, "{}_emitt.png".format(backend)))

    # Read data from reference simulation
    diags_orig = LpaDiagnostics(os.path.join(diag_dir_orig, "hdf5"), backend=backend)
    its = diags_orig.iterations
    emitt_x_orig = np.zeros(len(its))
    emitt_y_orig = np.zeros(len(its))
    for i, it in enumerate(its):
        emitt_x_orig[i], emitt_y_orig[i] = diags_orig.get_emittance(iteration=it)

    # Check that both simulations yield the same results for the emittance within a one per mil relative tolerance
    np.testing.assert_allclose(emitt_x, emitt_x_orig, rtol=1e-3)
    np.testing.assert_allclose(emitt_y, emitt_y_orig, rtol=1e-3)


if __name__ == "__main__":
    test_openpmd_viewer()
    test_openpmd_pulse()
