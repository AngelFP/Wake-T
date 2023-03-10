
import os

import pytest
from numpy.testing import assert_array_almost_equal

from wake_t.utilities.bunch_generation import (
    get_from_file, get_gaussian_bunch_from_size)
from wake_t.utilities.bunch_saving import save_bunch_to_file
from wake_t.diagnostics import OpenPMDDiagnostics


tests_output_folder = './tests_output'


def test_bunch_read_save():
    """
    Check that saving/reading a particle bunch to/from a file does not
    have any effect on the data.

    All supported data formats (openpmd, astra and csrtrack) are tested.
    """
    output_folder = os.path.join(tests_output_folder, 'bunch_read_save')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Beam parameters.
    emitt_nx = emitt_ny = 1e-6  # m
    s_x = s_y = 3e-6  # m
    s_t = 3.  # fs
    gamma_avg = 100 / 0.511
    gamma_spread = 1.  # %
    q_bunch = 30  # pC
    xi_avg = 0.  # m
    n_part = 1e4

    # Create particle bunch.
    bunch = get_gaussian_bunch_from_size(
        emitt_nx, emitt_ny, s_x, s_y, gamma_avg, gamma_spread, s_t, xi_avg,
        q_bunch, n_part, name='elec_bunch')

    # Formats to test.
    data_formats = ['astra', 'csrtrack', 'openpmd']
    file_formats = ['txt', 'fmt1', 'h5']

    for data_format, file_format in zip(data_formats, file_formats):
        # Save bunch.
        save_bunch_to_file(
            bunch,
            data_format,
            os.path.join(
                output_folder,
                'bunch_{}.{}'.format(data_format, file_format)
            )
        )
        # Read saved bunch.
        file_path = os.path.join(
            output_folder,
            'bunch_{}.{}'.format(data_format, file_format))
        bunch_saved = get_from_file(file_path, data_format)

        # For astra and csrtrack, remove the generated reference particle.
        if data_format in ['astra', 'csrtrack']:
            bunch_saved.x = bunch_saved.x[1:]
            bunch_saved.y = bunch_saved.y[1:]
            bunch_saved.xi = bunch_saved.xi[1:]
            bunch_saved.px = bunch_saved.px[1:]
            bunch_saved.py = bunch_saved.py[1:]
            bunch_saved.pz = bunch_saved.pz[1:]
            bunch_saved.w = bunch_saved.w[1:]

        # Check saved bunch is the same as original.
        assert_array_almost_equal(bunch_saved.x, bunch.x)
        assert_array_almost_equal(bunch_saved.y, bunch.y)
        assert_array_almost_equal(bunch_saved.xi, bunch.xi)
        assert_array_almost_equal(bunch_saved.px, bunch.px)
        assert_array_almost_equal(bunch_saved.py, bunch.py)
        assert_array_almost_equal(bunch_saved.pz, bunch.pz)
        assert_array_almost_equal(bunch_saved.q, bunch.q)


def test_openpmd_reading():
    """
    Check that the `get_from_file` method behaves as expected when reading
    openPMD data.
    """
    output_folder = os.path.join(tests_output_folder, 'bunch_openpmd_read')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Beam parameters.
    emitt_nx = emitt_ny = 1e-6  # m
    s_x = s_y = 3e-6  # m
    s_t = 3.  # fs
    gamma_avg = 100 / 0.511
    gamma_spread = 1.  # %
    q_bunch = 30  # pC
    xi_avg = 0.  # m
    n_part = 1e4

    # Create particle bunch.
    bunch_1 = get_gaussian_bunch_from_size(
        emitt_nx, emitt_ny, s_x, s_y, gamma_avg, gamma_spread, s_t, xi_avg,
        q_bunch, n_part, name='elec_bunch')
    bunch_2 = get_gaussian_bunch_from_size(
        emitt_nx, emitt_ny, s_x, s_y, gamma_avg, gamma_spread, s_t, xi_avg,
        q_bunch, n_part, name='elec_bunch_2')

    # Create diagnostics in output folder
    diags = OpenPMDDiagnostics(write_dir=output_folder)

    # Test 1.
    # -------

    # Write diagnostics without any species.
    diags.write_diagnostics(time=0., dt=0.)
    # Trying to read a species from this file should fail.
    with pytest.raises(ValueError):
        file_path = os.path.join(output_folder, 'hdf5', 'data00000000.h5')
        bunch_saved = get_from_file(file_path, 'openpmd')

    # Test 2.
    # -------

    # Write diagnostics with one species.
    diags.write_diagnostics(time=0., dt=0., species_list=[bunch_1])
    # Reading this file should raise no exception even if no species name
    # is specified.
    file_path = os.path.join(output_folder, 'hdf5', 'data00000001.h5')
    bunch_saved = get_from_file(file_path, 'openpmd')
    # Since the `name` parameter has not been specified, it must be equal to
    # the original name of the openPMD species.
    assert bunch_saved.name == 'elec_bunch'
    # If a new custom name is given, use this name.
    custom_name = 'bunch'
    bunch_saved = get_from_file(file_path, 'openpmd', name=custom_name)
    assert bunch_saved.name == custom_name

    # Test 3.
    # -------

    # Write diagnostics with two species.
    diags.write_diagnostics(time=0., dt=0., species_list=[bunch_1, bunch_2])
    # Reading a species without specifying a species name should fail.
    with pytest.raises(ValueError):
        file_path = os.path.join(output_folder, 'hdf5', 'data00000002.h5')
        bunch_saved = get_from_file(file_path, 'openpmd')
    # Reading each species indicating the correct species name should not fail
    bunch_saved = get_from_file(
        file_path, 'openpmd', species_name='elec_bunch')
    bunch_saved = get_from_file(
        file_path, 'openpmd', species_name='elec_bunch_2')


if __name__ == '__main__':
    test_bunch_read_save()
    test_openpmd_reading()
