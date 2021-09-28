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
        grid_spacing, grid_global_offset, fld_solver, fld_solver_params,
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
        diag_data[fld]['grid']['global_offset'] = grid_global_offset
    diag_data['field_solver'] = fld_solver
    diag_data['field_solver_params'] = fld_solver_params
    diag_data['field_boundary'] = fld_boundary
    diag_data['field_boundary_params'] = fld_boundary_params
    diag_data['particle_boundary'] = part_boundary
    diag_data['particle_boundary_params'] = part_boundary_params
    diag_data['current_smoothing'] = current_smoothing
    diag_data['charge_correction'] = charge_correction
    return diag_data


def radial_gradient(fld, dr):
    """
    Calculate the radial gradient of a 2D r-z field.

    To obtain an accurate derivative on axis, a wider array which contains
    the initial field and its mirrored view along the axis is created. The
    gradient of this array is computed and only its upper half is returned.

    Parameters:
    -----------
    fld : ndarray
        A 2D array containing the original r-z field.
    dr : float
        Radial separation between grid points.

    """
    n_r = fld.shape[1]
    fld_with_mirror = np.concatenate((fld[:, ::-1], fld), axis=1)
    return np.gradient(fld_with_mirror, dr, axis=1)[:, n_r:]
