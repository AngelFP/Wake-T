
import os

from numpy.testing import assert_array_almost_equal

from wake_t.utilities.bunch_generation import (
    get_from_file, get_gaussian_bunch_from_size)
from wake_t.utilities.bunch_saving import save_bunch_to_file


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
            bunch, data_format, output_folder, 'bunch_'+data_format)
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
            bunch_saved.q = bunch_saved.q[1:]

        # Check saved bunch is the same as original.
        assert_array_almost_equal(bunch_saved.x, bunch.x)
        assert_array_almost_equal(bunch_saved.y, bunch.y)
        assert_array_almost_equal(bunch_saved.xi, bunch.xi)
        assert_array_almost_equal(bunch_saved.px, bunch.px)
        assert_array_almost_equal(bunch_saved.py, bunch.py)
        assert_array_almost_equal(bunch_saved.pz, bunch.pz)
        assert_array_almost_equal(bunch_saved.q, bunch.q)


if __name__ == '__main__':
    test_bunch_read_save()
