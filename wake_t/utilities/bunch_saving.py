""" This module contains methods for saving particle distributions to files"""

import aptools.data_handling.saving as ds


def save_bunch_to_file(bunch, code_name, folder_path, file_name,
                       species_name=None, reposition=False,
                       avg_pos=[None, None, None], n_part=None):
    bunch_data = bunch.get_bunch_matrix()
    if species_name is None:
        species_name = bunch.name
    ds.save_beam(code_name, bunch_data, folder_path, file_name,
                 reposition=reposition, avg_pos=avg_pos, n_part=n_part,
                 species_name=species_name)
