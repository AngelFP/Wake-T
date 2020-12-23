""" Contains other utilities """

import sys
import numpy as np


def print_progress_bar(pre_string, step, total_steps):
    n_dash = int(round(step/total_steps*20))
    n_space = 20 - n_dash
    status = pre_string + '[' + '-'*n_dash + ' '*n_space + '] '
    if step < total_steps:
        status += '\r'
    sys.stdout.write(status)
    sys.stdout.flush()


def generate_field_diag_dictionary(
        fld_names, fld_comps, fld_arrays, fld_comp_pos, grid_labels,
        grid_spacing, grid_local_offset, fld_solver, fld_solver_params,
        fld_boundary, fld_boundary_params, part_boundary,
        part_boundary_params, current_smoothing, charge_correction):
    """
    Generates a dictionary which can be used by the openPMD diagnostics to
    write the field data.

    """
    diag_data = {}
    diag_data['fields'] = fld_names
    fld_zip = zip(fld_names, fld_comps, fld_arrays, fld_comp_pos)
    for fld, comps, arrays, pos in fld_zip:
        diag_data[fld] = {}
        if comps is not None:
            diag_data[fld]['comps'] = {}
            for comp, arr in zip(comps, arrays):
                diag_data[fld]['comps'][comp] = {}
                diag_data[fld]['comps'][comp]['array'] = arr
                diag_data[fld]['comps'][comp]['position'] = pos
        else:
            diag_data[fld]['array'] = arrays[0]
            diag_data[fld]['position'] = pos
        diag_data[fld]['grid'] = {}
        diag_data[fld]['grid']['spacing'] = grid_spacing
        diag_data[fld]['grid']['labels'] = grid_labels
        diag_data[fld]['grid']['local_offset'] = grid_local_offset
    diag_data['field_solver'] = fld_solver
    diag_data['field_solver_params'] = fld_solver_params
    diag_data['field_boundary'] = fld_boundary
    diag_data['field_boundary_params'] = fld_boundary_params
    diag_data['particle_boundary'] = part_boundary
    diag_data['particle_boundary_params'] = part_boundary_params
    diag_data['current_smoothing'] = current_smoothing
    diag_data['charge_correction'] = charge_correction
    return diag_data
