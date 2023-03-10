""" This module contains methods for saving particle distributions to files"""

from typing import Optional

from aptools.particle_distributions import (
    save_distribution, ParticleDistribution)

from wake_t.particles.particle_bunch import ParticleBunch


def save_bunch_to_file(
    bunch: ParticleBunch,
    data_format: str,
    file_path: str,
    species_name: Optional[str] = None
):
    """Save a particle bunch to file.

    Parameters
    ----------
    bunch : ParticleBunch
        The bunch to save.
    data_format : str
        The output data format. Can be 'astra', 'csrtrack', or 'openpmd'.
    folder_path : str
        Path to the folder where the bunch file will be created.
    file_name : str
        Name of the file to be saved, without format extension.
    species_name : str, optional
        Only required if `data_format='openpmd'`. Name of the particle species
        under which the bunch data will be stored. If None, the name of the
        bunch will be used.
    """
    kwargs = {}

    # For openpmd output, save with species name.
    if data_format == 'openpmd':
        if species_name is None:
            species_name = bunch.name
        kwargs['species_name'] = species_name
    # Create APtools distribution and save it.
    distribution = ParticleDistribution(
        x=bunch.x,
        y=bunch.y,
        z=bunch.xi,
        px=bunch.px,
        py=bunch.py,
        pz=bunch.pz,
        w=bunch.w,
        q_species=bunch.q_species,
        m_species=bunch.m_species
    )
    save_distribution(distribution, file_path, data_format, **kwargs)
