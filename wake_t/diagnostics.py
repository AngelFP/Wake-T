import os
import numpy as np
from openpmd_api import (Series, Access, Dataset, Mesh_Record_Component,
                         Unit_Dimension, Geometry)

from wake_t.__version__ import __version__


SCALAR = Mesh_Record_Component.SCALAR


class OpenPMDDiagnostics():
    def __init__(self, write_dir=None):
        if write_dir is None:
            self.write_dir = os.path.join(os.getcwd(), 'diags')
        else:
            self.write_dir = os.path.abspath(write_dir)
        self._index_out = 0

    def write_diagnostics(self, time, dt, species_list=[], wakefield=None):
        # Create diagnostics folder if it doesn't exist already.
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)
            
        file_name = 'data{0:08d}.h5'.format(self._index_out)
        file_path = os.path.join(self.write_dir, 'hdf5', file_name)
        opmd_series = Series(file_path, Access.create)
        opmd_series.set_software('Wake-T', __version__)
        opmd_series.set_meshes_path('fields')
        opmd_series.set_particles_path('particles')

        it = opmd_series.iterations[self._index_out]
        it.set_time(time)
        it.set_dt(dt)

        for species in species_list:
            diag_data = species.get_openpmd_diagnostics_data()
            self._write_species(it, diag_data)

        if wakefield is not None:
            wf_data = wakefield.get_openpmd_diagnostics_data()
            if wf_data is not None:
                self._write_fields(it, wf_data)

        opmd_series.flush()
        self._index_out += 1

    def _write_species(self, it, species_data):
        # Create particles for this species.
        particles = it.particles[species_data['name']]

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
        particles['positionOffset']['z'].make_constant(0.)
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
        it.meshes.set_attribute('fieldSolver', 'other')
        it.meshes.set_attribute('fieldSolverParams', 'todo')
        it.meshes.set_attribute('fieldBoundary', [
            np.string_("other"), np.string_("other"),
            np.string_("other"), np.string_("other")])
        it.meshes.set_attribute('particleBoundary', [
            np.string_("other"), np.string_("other"),
            np.string_("other"), np.string_("other")])
        it.meshes.set_attribute('currentSmoothing', 'none')
        it.meshes.set_attribute('chargeCorrection', 'none')
        for field in wf_data:

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
            fld.set_grid_global_offset(wf_data[field]['grid']['global_offset'])
