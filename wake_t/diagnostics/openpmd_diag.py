"""
This module contains the OpenPMDDiagnostics class, which generates the
openPMD output.
"""
import os
from typing import Optional, List

import numpy as np
import scipy.constants as ct
from openpmd_api import (Series, Access, Dataset, Mesh_Record_Component,
                         Unit_Dimension, Geometry)

from wake_t import __version__
from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.fields.base import Field


SCALAR = Mesh_Record_Component.SCALAR


class OpenPMDDiagnostics():
    """
    Class in charge of creating and writing the particle and field
    diagnostics following the openPMD standard.

    Parameters
    ----------
    write_dir : str
        Directory to which the diagnostics will be written. By default
        this will be a 'diags' folder in the current working directory.
    """

    def __init__(
        self,
        write_dir: Optional[str] = None
    ) -> None:
        if write_dir is None:
            self.write_dir = os.path.join(os.getcwd(), 'diags')
        else:
            self.write_dir = os.path.abspath(write_dir)
        # Index of the data file to be written. Will be increased after
        # each time step is written.
        self._index_out = 0
        # Longitudinal position at which the current beamline element begins.
        # This is needed in order to keep track of the global z position in the
        # simulation (needed for `globalOffset` and `positionOffset`
        # attributes) since the beamline elements themselves are not aware of
        # each other.
        self._current_z_pos = 0.

    def write_diagnostics(
        self,
        time: float,
        dt: float,
        species_list: Optional[List[ParticleBunch]] = [],
        fields: Optional[List[Field]] = []
    ) -> None:
        """
        Write to disk the diagnostics of a certain time step.

        Parameters
        ----------
        time : float
            Simulation time at the current beamline element.
        dt : float
            Time step used in the current beamline element.
        species_list : list, optional
            List of particle species to be written to file.
        wakefield : list, optional
            List of Fields that should be written to file.

        """
        # Perform checks.
        self.check_species_names(species_list)

        # Create diagnostics folder if it doesn't exist already.
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)

        # Create file and series.
        file_name = 'data%08T.h5'
        file_path = os.path.join(self.write_dir, 'hdf5', file_name)
        opmd_series = Series(file_path, Access.create)

        # Set basic attributes.
        opmd_series.set_software('Wake-T', __version__)
        opmd_series.meshes_path = 'fields'
        opmd_series.particles_path = 'particles'
        opmd_series.openPMD_extension = 1

        # Create current iteration and set time attributes.
        it = opmd_series.iterations[self._index_out]
        it.time = time + self._current_z_pos/ct.c
        it.dt = dt

        # Write particle diagnostics.
        for species in species_list:
            diag_data = species.get_openpmd_diagnostics_data(it.time)
            self._write_species(it, diag_data)

        # Write field diagnostics.
        for field in fields:
            f_data = field.get_openpmd_diagnostics_data(it.time)
            if f_data is not None:
                self._write_fields(it, f_data)
                # Some field models might also have their own species
                # (e.g. plasma particles).
                if 'species' in f_data.keys():
                    for species in f_data['species']:
                        s_data = f_data[species]
                        self._write_species(it, s_data)

        # Flush data and increase counter for next step.
        opmd_series.flush()
        self._index_out += 1

    def increase_z_pos(
        self,
        dist: float
    ) -> None:
        """
        Increase the current z position along the beamline. This should be
        called after tracking in each beamline element is completed.

        Parameters
        ----------
        dist : float
            This distance should be the length of the beamline element in
            which tracking has just finalized.

        """
        self._current_z_pos += dist

    def _write_species(self, it, species_data):
        """ Write all particle diagnostics of a given species. """
        # Create particles for this species.
        particles = it.particles[species_data['name']]

        # TODO: evaluate adding ED-PIC attributes to particles output. This is
        # tricky because Wake-T is not a PIC code and in the TMElements there
        # is no particle shape/smoothing nor charge/current deposition.
        # Could these attributes be only added in the time steps in which they
        # are actually used?

        if species_data['geometry'] == '3d_cartesian':
            components = ['x', 'y', 'z']
        if species_data['geometry'] == 'rz':
            components = ['r', 'z']

        # Create records of position components.
        position_record = False
        for component in components:
            if component in species_data:
                array = np.ascontiguousarray(species_data[component])
                pos_off = species_data['z_off'] if component == 'z' else 0.
                dataset = Dataset(array.dtype, extent=array.shape)
                offset = Dataset(np.dtype('float64'), extent=[1])
                particles['position'][component].reset_dataset(dataset)
                particles['positionOffset'][component].reset_dataset(offset)
                particles['position'][component].store_chunk(array)
                particles['positionOffset'][component].make_constant(pos_off)
                position_record = True
        if position_record:
            particles['position'].unit_dimension = {Unit_Dimension.L: 1}
            particles['positionOffset'].unit_dimension = {Unit_Dimension.L: 1}
            particles['position'].set_attribute('macroWeighted', np.uint32(0))
            particles['positionOffset'].set_attribute(
                'macroWeighted', np.uint32(0))
            particles['position'].set_attribute('weightingPower', 0.)
            particles['positionOffset'].set_attribute('weightingPower', 0.)

        # Create records of momentum components.
        momentum_record = False
        for component in components:
            pc = 'p' + component
            if pc in species_data:
                array = np.ascontiguousarray(species_data[pc])
                dataset = Dataset(array.dtype, extent=array.shape)
                particles['momentum'][component].reset_dataset(dataset)
                particles['momentum'][component].store_chunk(array)
                momentum_record = True
        if momentum_record:
            particles['momentum'].unit_dimension = {
                Unit_Dimension.L: 1,
                Unit_Dimension.M: 1,
                Unit_Dimension.T: -1,
                }
            particles['momentum'].set_attribute('macroWeighted', np.uint32(0))
            particles['momentum'].set_attribute('weightingPower', 1.)

        # Do the same with geometry-independent data.
        if 'w' in species_data:
            w = np.ascontiguousarray(species_data['w'])
            d_w = Dataset(w.dtype, extent=w.shape)
            particles['weighting'][SCALAR].reset_dataset(d_w)
            particles['weighting'][SCALAR].store_chunk(w)
            particles['weighting'][SCALAR].set_attribute(
                'macroWeighted', np.uint32(1))
            particles['weighting'][SCALAR].set_attribute('weightingPower', 1.)
        if 'r_to_x' in species_data:
            r_to_x = np.ascontiguousarray(species_data['r_to_x'])
            d_r_to_x = Dataset(r_to_x.dtype, extent=r_to_x.shape)
            particles['r_to_x'][SCALAR].reset_dataset(d_r_to_x)
            particles['r_to_x'][SCALAR].store_chunk(r_to_x)
            particles['r_to_x'][SCALAR].set_attribute(
                'macroWeighted', np.uint32(0))
            particles['r_to_x'][SCALAR].set_attribute('weightingPower', 1.)
        if 'id' in species_data:
            ids = np.ascontiguousarray(species_data['id'])
            d_id = Dataset(ids.dtype, extent=ids.shape)
            particles['id'][SCALAR].reset_dataset(d_id)
            particles['id'][SCALAR].store_chunk(ids)
            particles['id'][SCALAR].set_attribute(
                'macroWeighted', np.uint32(0))
            particles['id'][SCALAR].set_attribute('weightingPower', 1.)
        q = species_data['q']
        m = species_data['m']
        d_q = Dataset(np.dtype('float64'), extent=[1])
        d_m = Dataset(np.dtype('float64'), extent=[1])
        particles['charge'][SCALAR].reset_dataset(d_q)
        particles['mass'][SCALAR].reset_dataset(d_m)
        particles['charge'][SCALAR].make_constant(q)
        particles['mass'][SCALAR].make_constant(m)
        particles['charge'].unit_dimension = {
            Unit_Dimension.T: 1,
            Unit_Dimension.I: 1,
        }
        particles['mass'].unit_dimension = {Unit_Dimension.M: 1}
        particles['charge'][SCALAR].set_attribute(
            'macroWeighted', np.uint32(0))
        particles['mass'][SCALAR].set_attribute('macroWeighted', np.uint32(0))
        particles['charge'][SCALAR].set_attribute('weightingPower', 1.)
        particles['mass'][SCALAR].set_attribute('weightingPower', 1.)

    def _write_fields(self, it, wf_data):
        """ Write wakefield diagnostics. """
        # Set common field attributes.
        it.meshes.set_attribute(
            'fieldSolver', wf_data['field_solver'])
        it.meshes.set_attribute(
            'fieldSolverParams', wf_data['field_solver_params'])
        it.meshes.set_attribute(
            'fieldBoundary', wf_data['field_boundary'])
        it.meshes.set_attribute(
            'fieldBoundaryParams', wf_data['field_boundary_params'])
        it.meshes.set_attribute(
            'particleBoundary', wf_data['particle_boundary'])
        it.meshes.set_attribute(
            'particleBoundaryParams', wf_data['particle_boundary_params'])
        it.meshes.set_attribute(
            'currentSmoothing', wf_data['current_smoothing'])
        it.meshes.set_attribute(
            'chargeCorrection', wf_data['charge_correction'])

        # Add diagnostics of each field.
        for field in wf_data['fields']:

            fld = it.meshes[field]

            if 'comps' in wf_data[field]:
                for comp in wf_data[field]['comps']:
                    fld_comp = fld[comp]
                    fld_comp_array = wf_data[field]['comps'][comp]['array']
                    # Add extra dimension to mimic thetaMode geometry (mode 0).
                    # Workaround needed for reading the data with
                    # openPMD-viewer until cylindrical geometry is properly
                    # defined in the standard.
                    fld_comp_array = np.expand_dims(fld_comp_array, axis=0)
                    d_fld_comp = Dataset(
                        fld_comp_array.dtype, extent=fld_comp_array.shape)
                    fld_comp.reset_dataset(d_fld_comp)
                    fld_comp.store_chunk(fld_comp_array)
                    fld_comp.set_attribute(
                        'position', wf_data[field]['comps'][comp]['position'])
            else:
                fld_array = wf_data[field]['array']
                # Add extra dimmension to mimic thetaMode geometry (mode 0).
                fld_array = np.expand_dims(fld_array, axis=0)
                d_fld = Dataset(fld_array.dtype, extent=fld_array.shape)
                fld[SCALAR].reset_dataset(d_fld)
                fld[SCALAR].store_chunk(fld_array)
                fld[SCALAR].set_attribute(
                    'position', wf_data[field]['position'])

            if field == 'E':
                fld.unit_dimension = {
                    Unit_Dimension.L: 1,
                    Unit_Dimension.M: 1,
                    Unit_Dimension.T: -3,
                    Unit_Dimension.I: -1
                }
            elif field == 'B':
                fld.unit_dimension = {
                    Unit_Dimension.M: 1,
                    Unit_Dimension.T: -2,
                    Unit_Dimension.I: -1
                }
            elif field in ['rho', 'chi']:
                fld.unit_dimension = {
                    Unit_Dimension.L: -3,
                    Unit_Dimension.T: 1,
                    Unit_Dimension.I: 1
                }
            elif field == 'a':
                fld.unit_dimension = {
                    Unit_Dimension.L: 1,
                    Unit_Dimension.M: 1,
                    Unit_Dimension.T: -2,
                    Unit_Dimension.I: -1
                }

            # Set geometry to thetaMode until cylindrical geometry is
            # properly defined in the openPMD standard.
            fld.geometry = Geometry.thetaMode  # Geometry.cylindrical
            fld.set_attribute('fieldSmoothing', 'none')
            for attr, val in wf_data[field]['attributes'].items():
                fld.set_attribute(attr, val)
            fld.axis_labels = wf_data[field]['grid']['labels']
            fld.grid_spacing = wf_data[field]['grid']['spacing']
            fld.grid_global_offset = wf_data[field]['grid']['global_offset']

    def check_species_names(
        self,
        species_list: Optional[List[ParticleBunch]] = []
    ) -> None:
        """ Check that no species have duplicate names. """
        sp_names = []
        for species in species_list:
            name = species.name
            if name not in sp_names:
                sp_names.append(name)
            else:
                raise ValueError(
                    'Several species share same name {}.'.format(name))
