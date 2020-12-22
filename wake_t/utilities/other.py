""" Contains other utilities """

import sys


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
        grid_spacing, grid_global_offset):
    """
    Generates a dictionary which can be used by the openPMD diagnostics to
    write the field data.

    """
    diag_data = {}
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
    return diag_data
