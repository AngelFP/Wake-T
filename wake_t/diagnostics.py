"""
This module contains the OpenPMDDiagnostics class, which generates the
openPMD output.
"""
import os
from copy import deepcopy
import numpy as np
import scipy.constants as ct
from openpmd_api import (Series, Access, Dataset, Mesh_Record_Component,
                         Unit_Dimension, Geometry)

from wake_t.__version__ import __version__


SCALAR = Mesh_Record_Component.SCALAR


class OpenPMDDiagnostics():
    """
    Class in charge of creating and writing the particle and field
    diagnostics following the openPMD standard.
    """

    def __init__(self, write_dir=None):
        """
        Initialize diagnostics.

        Parameters
        ----------
        write_dir : str
            Directory to which the diagnostics will be written. By default
            this will be a 'diags' folder in the current working directory.

        """
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

    def write_diagnostics(self, time, dt, species_list=[], wakefield=None):
        """
        Write to disk the diagnostics of a certain time step.

        Parameters
        ----------
        time : float
            Simulation time at the current beamline element.

        dt : float
            Time step used in the current beamline element.

        species_list : list
            List of particle species to be written.

        wakefield : Wakefield
            Instance of a wakefield from which the fields should be written
            to the output. It not specified, no fields will be written for
            this time step.

        """
        # Create diagnostics folder if it doesn't exist already.
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)

        # Create file and series.
        file_name = 'data{0:08d}.h5'.format(self._index_out)
        file_path = os.path.join(self.write_dir, 'hdf5', file_name)
        opmd_series = Series(file_path, Access.create)

        # Set basic attributes.
        opmd_series.set_software('Wake-T', __version__)
        opmd_series.set_meshes_path('fields')
        opmd_series.set_particles_path('particles')

        # Create current iteration and set time attributes.
        it = opmd_series.iterations[self._index_out]
        it.set_time(time + self._current_z_pos/ct.c)
        it.set_dt(dt)

        # Write particle diagnostics.
        for species in species_list:
            diag_data = species.get_openpmd_diagnostics_data()
            self._write_species(it, diag_data)

        # Write field diagnostics.
        if wakefield is not None:
            wf_data = wakefield.get_openpmd_diagnostics_data()
            if wf_data is not None:
                self._write_fields(it, wf_data)

        # Flush data and increase counter for next step.
        opmd_series.flush()
        self._index_out += 1

    def increase_z_pos(self, dist):
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

        # Get arrays.
        x = species_data['x']
        y = species_data['y']
        z = species_data['z']
        px = species_data['px']
        py = species_data['py']
        pz = species_data['pz']
        w = species_data['w']
        q = species_data['q']
        m = species_data['m']
        z_off = species_data['z_off']

        # Generate datasets.
        d_x = Dataset(x.dtype, extent=x.shape)
        d_y = Dataset(y.dtype, extent=y.shape)
        d_z = Dataset(z.dtype, extent=z.shape)
        d_px = Dataset(px.dtype, extent=px.shape)
        d_py = Dataset(py.dtype, extent=py.shape)
        d_pz = Dataset(pz.dtype, extent=pz.shape)
        d_w = Dataset(w.dtype, extent=w.shape)
        d_q = Dataset(np.dtype('float64'), extent=[1])
        d_m = Dataset(np.dtype('float64'), extent=[1])
        d_xoff = Dataset(np.dtype('float64'), extent=[1])
        d_yoff = Dataset(np.dtype('float64'), extent=[1])
        d_zoff = Dataset(np.dtype('float64'), extent=[1])

        # Record data.
        particles['position']['x'].reset_dataset(d_x)
        particles['position']['y'].reset_dataset(d_y)
        particles['position']['z'].reset_dataset(d_z)
        particles['positionOffset']['x'].reset_dataset(d_xoff)
        particles['positionOffset']['y'].reset_dataset(d_yoff)
        particles['positionOffset']['z'].reset_dataset(d_zoff)
        particles['momentum']['x'].reset_dataset(d_px)
        particles['momentum']['y'].reset_dataset(d_py)
        particles['momentum']['z'].reset_dataset(d_pz)
        particles['weighting'][SCALAR].reset_dataset(d_w)
        particles['charge'][SCALAR].reset_dataset(d_q)
        particles['mass'][SCALAR].reset_dataset(d_m)

        # Prepare for writting.
        particles['position']['x'].store_chunk(x)
        particles['position']['y'].store_chunk(y)
        particles['position']['z'].store_chunk(z)
        particles['positionOffset']['x'].make_constant(0.)
        particles['positionOffset']['y'].make_constant(0.)
        particles['positionOffset']['z'].make_constant(z_off)
        particles['momentum']['x'].store_chunk(px)
        particles['momentum']['y'].store_chunk(py)
        particles['momentum']['z'].store_chunk(pz)
        particles['weighting'][SCALAR].store_chunk(w)
        particles['charge'][SCALAR].make_constant(q)
        particles['mass'][SCALAR].make_constant(m)

        # Set units.
        particles['position'].unit_dimension = {Unit_Dimension.L: 1}
        particles['positionOffset'].unit_dimension = {Unit_Dimension.L: 1}
        particles['momentum'].unit_dimension = {
            Unit_Dimension.L: 1,
            Unit_Dimension.M: 1,
            Unit_Dimension.T: -1,
            }
        particles['charge'].unit_dimension = {
            Unit_Dimension.T: 1,
            Unit_Dimension.I: 1,
            }
        particles['mass'].unit_dimension = {Unit_Dimension.M: 1}

        # Set weighting attributes.
        particles['position'].set_attribute('macroWeighted', np.uint32(0))
        particles['positionOffset'].set_attribute(
            'macroWeighted', np.uint32(0))
        particles['momentum'].set_attribute('macroWeighted', np.uint32(0))
        particles['weighting'][SCALAR].set_attribute(
            'macroWeighted', np.uint32(1))
        particles['charge'][SCALAR].set_attribute(
            'macroWeighted', np.uint32(0))
        particles['mass'][SCALAR].set_attribute('macroWeighted', np.uint32(0))
        particles['position'].set_attribute('weightingPower', 0.)
        particles['positionOffset'].set_attribute('weightingPower', 0.)
        particles['momentum'].set_attribute('weightingPower', 1.)
        particles['weighting'][SCALAR].set_attribute('weightingPower', 1.)
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
                    d_fld_comp = Dataset(
                        fld_comp_array.dtype, extent=fld_comp_array.shape)
                    fld_comp.reset_dataset(d_fld_comp)
                    fld_comp.store_chunk(fld_comp_array)
                    fld_comp.set_attribute(
                        'position', wf_data[field]['comps'][comp]['position'])
            else:
                fld_array = wf_data[field]['array']
                d_fld = Dataset(fld_array.dtype, extent=fld_array.shape)
                fld[SCALAR].reset_dataset(d_fld)
                fld[SCALAR].store_chunk(fld_array)
                fld[SCALAR].set_attribute(
                    'position', wf_data[field]['position'])

            if field in ['E', 'W']:
                fld.unit_dimension = {
                    Unit_Dimension.L: 1,
                    Unit_Dimension.M: 1,
                    Unit_Dimension.T: -3,
                    Unit_Dimension.I: -1
                    }
            elif field == 'rho':
                fld.unit_dimension = {
                    Unit_Dimension.L: -3,
                    Unit_Dimension.T: 1,
                    Unit_Dimension.I: 1
                    }

            fld.set_geometry(Geometry.cylindrical)
            fld.set_attribute('fieldSmoothing', 'none')
            fld.set_axis_labels(wf_data[field]['grid']['labels'])
            fld.set_grid_spacing(wf_data[field]['grid']['spacing'])
            global_offset = deepcopy(wf_data[field]['grid']['local_offset'])
            global_offset[-1] += self._current_z_pos
            fld.set_grid_global_offset(global_offset)
