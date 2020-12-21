import os
import numpy as np
import openpmd_api as io


class OpenPMDDiagnostics():
    def __init__(self, write_dir=None):
        if write_dir is None:
            self.write_dir = os.path.join(os.getcwd(), 'diags')
        else:
            self.write_dir = os.path.abspath(write_dir)
        self._index_out = 0

    def initialize(self):
        os.makedirs(self.write_dir)

    def write_diagnostics(self, time, species_list=[], wakefield=None):
        file_name = 'data_{0:08d}'.format(self._index_out)
        file_path = os.path.join(self.write_dir, 'hdf5', file_name)
        opmd_series = io.Series(file_path, io.Access.create)
        
        i = opmd_series.iterations[self._index_out]

        for species in species_list:
            diag_data = species.get_diagnostics_data()
            # register data
            # flush
        if wakefield is not None:
            wf_data = wakefield.get_diagnostics_data()
            # register data
            # flush

        self._index_out += 1

