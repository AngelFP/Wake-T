""" This module contains methods for saving particle distributions to files"""

import aptools.data_handling.saving as ds


def save_bunch_to_file(
        bunch, data_format, folder_path, file_name, species_name=None,
        reposition=False, avg_pos=[None, None, None],
        avg_mom=[None, None, None], n_part=None):
    """Save a particle bunch to file.

    Parameters
    ----------
    bunch : ParticleBunch
        The bunch to save.
    data_format : str
        The output data format. Can be 'astra', 'csrtrack', 'openpmd' or
        'fbpic'.
    folder_path : str
        Path to the folder where the bunch file will be created.
    file_name : str
        Name of the file to be saved, without format extension.
    species_name : str, optional
        Only required if `format='openpmd'`. Name of the particle species
        under which the bunch data will be stored. If None, the name of the
        bunch will be used.
    reposition : bool, optional
        Whether to reposition the particle distribution in space
        and/or momentum centered in the coordinates specified in `avg_pos` and
        `avg_mom`. By default False.
    avg_pos : list, optional
        Only used it `reposition=True`. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.
    avg_pos : list, optional
        Only used it `reposition=True`. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.
    avg_mom : list, optional
        Only used it `reposition=True`. Contains the new average
        momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in units of m_e*c.
        Setting a component as None prevents repositioning in that coordinate.
    n_part : int, optional
        If specified, it allows for saving only a subset of `n_part` bunch
        particles to file while preserving the total charge. Must be lower
        than the original number of particles. Particles to save are chosen
        randomly. The charge of the saved particles will be modified to
        preserve the total charge.
    """
    # Get bunch data.
    bunch_data = bunch.get_bunch_matrix()

    # For openpmd output, save with species name.
    if data_format == 'openpmd':
        if species_name is None:
            species_name = bunch.name
        ds.save_beam(
            data_format, bunch_data, folder_path, file_name,
            reposition=reposition, avg_pos=avg_pos, avg_mom=avg_mom,
            n_part=n_part, species_name=species_name)
    else:
        ds.save_beam(
            data_format, bunch_data, folder_path, file_name,
            reposition=reposition, avg_pos=avg_pos, avg_mom=avg_mom,
            n_part=n_part)
