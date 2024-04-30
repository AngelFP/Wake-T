import os
from copy import deepcopy
from numpy.testing import assert_array_equal
from wake_t import Beamline, Drift, PlasmaStage
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_size


tests_output_folder = './tests_output'


def test_single_element():
    """
    This test checks that tracking a beamline made up of a single element
    produces the same result as tracking the element in itself.

    """
    output_folder = os.path.join(tests_output_folder, 'beamline_test_single')

    # Define bunch.
    bunch = get_gaussian_bunch_from_size(
        1e-6, 1e-6, 5e-6, 5e-6, 1000, 1, 10, 0, 100, 1e4)

    # Define plasma stage.
    plasma = PlasmaStage(
        1e-2, 1e23, wakefield_model='quasistatic_2d', n_out=5,
        xi_min=-20e-6, xi_max=20e-6, r_max=50e-6, n_xi=80, n_r=100,
        dz_fields=0.5e-3)

    # Define single-element beamline
    bl = Beamline([deepcopy(plasma)])

    # Track plasma.
    bunch_1 = bunch.copy()
    out_dir_1 = os.path.join(output_folder, 'plasma_diags')
    plasma.track(bunch_1, opmd_diag=True, diag_dir=out_dir_1)

    # Track beamline.
    bunch_2 = bunch.copy()
    out_dir_2 = os.path.join(output_folder, 'bl_diags')
    bl.track(bunch_2, opmd_diag=True, diag_dir=out_dir_2)

    # Check that final beams are identical.
    assert_array_equal(bunch_2.x, bunch_1.x)
    assert_array_equal(bunch_2.y, bunch_1.y)
    assert_array_equal(bunch_2.xi, bunch_1.xi)
    assert_array_equal(bunch_2.px, bunch_1.px)
    assert_array_equal(bunch_2.py, bunch_1.py)
    assert_array_equal(bunch_2.pz, bunch_1.pz)

    # Get list of output files.
    plasma_diag_files = os.listdir(os.path.join(out_dir_1, 'hdf5'))
    bl_diag_files = os.listdir(os.path.join(out_dir_2, 'hdf5'))

    # Check that both PlasmaStage and Beamline generate same output files.
    assert plasma_diag_files == bl_diag_files


def test_multiple_element():
    """
    This test checks that tracking a beamline made up of multiple elements
    produces the same result as tracking each element separately.

    """
    output_folder = os.path.join(tests_output_folder, 'beamline_test_multi')

    # Define bunch.
    bunch = get_gaussian_bunch_from_size(
        1e-6, 1e-6, 5e-6, 5e-6, 1000, 1, 10, 0, 100, 1e4)

    # Define drifts.
    d1 = Drift(1e-2, n_out=3)
    d2 = Drift(1e-2, n_out=3)

    # Define plasma stage.
    plasma = PlasmaStage(
        1e-2, 1e23, wakefield_model='quasistatic_2d', n_out=5,
        xi_min=-20e-6, xi_max=20e-6, r_max=50e-6, n_xi=80, n_r=100,
        dz_fields=0.5e-3)

    # Define single-element beamline
    bl = Beamline([deepcopy(d1), deepcopy(plasma), deepcopy(d2)])

    # Track elements individually.
    bunch_1 = bunch.copy()
    out_dir_d1 = os.path.join(output_folder, 'd1_diags')
    out_dir_plasma = os.path.join(output_folder, 'plasma_diags')
    out_dir_d2 = os.path.join(output_folder, 'd2_diags')
    d1.track(bunch_1, opmd_diag=True, diag_dir=out_dir_d1)
    plasma.track(bunch_1, opmd_diag=True, diag_dir=out_dir_plasma)
    d2.track(bunch_1, opmd_diag=True, diag_dir=out_dir_d2)

    # Track beamline.
    bunch_2 = bunch.copy()
    out_dir_bl = os.path.join(output_folder, 'bl_diags')
    bl.track(bunch_2, opmd_diag=True, diag_dir=out_dir_bl)

    # Check that final beams are identical.
    assert_array_equal(bunch_2.x, bunch_1.x)
    assert_array_equal(bunch_2.y, bunch_1.y)
    assert_array_equal(bunch_2.xi, bunch_1.xi)
    assert_array_equal(bunch_2.px, bunch_1.px)
    assert_array_equal(bunch_2.py, bunch_1.py)
    assert_array_equal(bunch_2.pz, bunch_1.pz)

    # Get list of output files.
    d1_diag_files = os.listdir(os.path.join(out_dir_d1, 'hdf5'))
    d2_diag_files = os.listdir(os.path.join(out_dir_d2, 'hdf5'))
    plasma_diag_files = os.listdir(os.path.join(out_dir_plasma, 'hdf5'))
    bl_diag_files = os.listdir(os.path.join(out_dir_bl, 'hdf5'))

    # Check that both PlasmaStage and Beamline generate same output files.
    n_files_1 = len(d1_diag_files + plasma_diag_files + d2_diag_files)
    n_files_2 = len(bl_diag_files)
    assert n_files_1 == n_files_2


if __name__ == '__main__':
    test_single_element()
    test_multiple_element()
