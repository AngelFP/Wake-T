""" This module contains methods for data analysis """

import os

import numpy as np
from h5py import File as H5File

import aptools.data_analysis.beam_diagnostics as bd


def analyze_bunch(bunch, n_slices=50, len_slice=None):
    # perform analysis
    dist = bunch.prop_distance
    params_analysis = _get_distribution_parameters(
        bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz, bunch.q,
        n_slices, len_slice)

    # store data
    bunch_params = _store_bunch_parameters_into_dict(dist, *params_analysis)
    return bunch_params


def analyze_bunch_list(bunch_list, n_slices=50, len_slice=None):
    # preallocate arrays
    list_len = len(bunch_list)
    a_x = np.zeros(list_len)
    a_y = np.zeros(list_len)
    b_x = np.zeros(list_len)
    b_y = np.zeros(list_len)
    g_x = np.zeros(list_len)
    g_y = np.zeros(list_len)
    ene = np.zeros(list_len)
    dist = np.zeros(list_len)
    ene_sp = np.zeros(list_len)
    ene_sp_sl_avg = np.zeros(list_len)
    em_x = np.zeros(list_len)
    em_y = np.zeros(list_len)
    em_x_sl_avg = np.zeros(list_len)
    em_y_sl_avg = np.zeros(list_len)
    s_x = np.zeros(list_len)
    s_y = np.zeros(list_len)
    x_avg = np.zeros(list_len)
    y_avg = np.zeros(list_len)
    i_peak = np.zeros(list_len)
    theta_x = np.zeros(list_len)
    theta_y = np.zeros(list_len)
    s_z = np.zeros(list_len)
    q_tot = np.zeros(list_len)

    # perform analysis
    for i, bunch in enumerate(bunch_list):
        dist[i] = bunch.prop_distance
        params_analysis = _get_distribution_parameters(
            bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz, bunch.q,
            n_slices, len_slice)
        (theta_x[i], theta_y[i], x_avg[i], y_avg[i], s_x[i], s_y[i], a_x[i],
         a_y[i], b_x[i], b_y[i], g_x[i], g_y[i], em_x[i], em_y[i],
         em_x_sl_avg[i], em_y_sl_avg[i], ene[i], ene_sp[i], ene_sp_sl_avg[i],
         i_peak[i], s_z[i], q_tot[i]) = params_analysis

    # store into dictionary
    bunch_list_params = _store_bunch_parameters_into_dict(
        dist, theta_x, theta_y, x_avg, y_avg, s_x, s_y, a_x, a_y, b_x, b_y,
        g_x, g_y, em_x, em_y, em_x_sl_avg, em_y_sl_avg, ene, ene_sp,
        ene_sp_sl_avg, i_peak, s_z, q_tot)
    return bunch_list_params


def save_parameters_to_file(bunch_params, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name + '.h5')
    with H5File(file_path, 'w') as h5_file:
        for param_name, param_data in bunch_params.items():
            h5_file.create_dataset(param_name, data=param_data)


def read_parameters_from_file(file_path):
    bunch_params = {}
    with H5File(file_path, 'r') as h5_file:
        for param in h5_file.keys():
            bunch_params[param] = h5_file.get(param).value
    return bunch_params


def save_bunch_to_file(bunch, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name + '.h5')
    with H5File(file_path, 'w') as h5_file:
        h5_file.create_dataset('x', data=bunch.x)
        h5_file.create_dataset('y', data=bunch.y)
        h5_file.create_dataset('xi', data=bunch.xi)
        h5_file.create_dataset('px', data=bunch.px)
        h5_file.create_dataset('py', data=bunch.py)
        h5_file.create_dataset('pz', data=bunch.pz)
        h5_file.create_dataset('q', data=bunch.q)
        h5_file.attrs['prop_dist'] = bunch.prop_distance


def _get_distribution_parameters(x, y, z, px, py, pz, q, n_slices, len_slice):
    a_x, b_x, g_x = bd.twiss_parameters(x, px, pz, py, w=q)
    a_y, b_y, g_y = bd.twiss_parameters(y, py, pz, px, w=q)
    ene = bd.mean_energy(px, py, pz, w=q)
    ene_sp = bd.relative_rms_energy_spread(px, py, pz, w=q)
    ene_sp_slice_params = bd.relative_rms_slice_energy_spread(
        z, px, py, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    ene_sp_sl, sl_w, sl_e, ene_sp_sl_avg = ene_sp_slice_params
    em_x = bd.normalized_transverse_rms_emittance(x, px, py, pz, w=q)
    em_y = bd.normalized_transverse_rms_emittance(y, py, px, pz, w=q)
    emitt_x_slice_params = bd.normalized_transverse_rms_slice_emittance(
        z, x, px, py, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    em_x_sl, sl_w, sl_e, em_x_sl_avg = emitt_x_slice_params
    emitt_y_slice_params = bd.normalized_transverse_rms_slice_emittance(
        z, y, py, px, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    em_y_sl, sl_w, sl_e, em_y_sl_avg = emitt_y_slice_params
    s_z = bd.rms_length(z, w=q)
    s_x = bd.rms_size(x, w=q)
    s_y = bd.rms_size(y, w=q)
    x_avg = np.average(x, weights=q)
    y_avg = np.average(y, weights=q)
    px_avg = np.average(px, weights=q)
    py_avg = np.average(py, weights=q)
    theta_x = px_avg/ene
    theta_y = py_avg/ene
    i_peak = bd.peak_current(z, q, n_slices=n_slices, len_slice=len_slice)
    q_tot = np.sum(q)
    return (theta_x, theta_y, x_avg, y_avg, s_x, s_y, a_x, a_y, b_x, b_y, g_x,
            g_y, em_x, em_y, em_x_sl_avg, em_y_sl_avg, ene, ene_sp,
            ene_sp_sl_avg, i_peak, s_z, q_tot)


def _store_bunch_parameters_into_dict(
        dist, theta_x, theta_y, x_avg, y_avg, s_x, s_y, a_x, a_y, b_x, b_y,
        g_x, g_y, em_x, em_y, em_x_sl_avg, em_y_sl_avg, ene, ene_sp,
        ene_sp_sl_avg, i_peak, s_z, q_tot):
    params_dict = {
        'prop_dist': dist,
        'theta_x': theta_x,
        'theta_y': theta_y,
        'x_avg': x_avg,
        'y_avg': y_avg,
        'sigma_x': s_x,
        'sigma_y': s_y,
        'sigma_z': s_z,
        'alpha_x': a_x,
        'alpha_y': a_y,
        'beta_x': b_x,
        'beta_y': b_y,
        'gamma_x': g_x,
        'gamma_y': g_y,
        'emitt_x': em_x,
        'emitt_y': em_y,
        'avg_slice_emitt_x': em_x_sl_avg,
        'avg_slice_emitt_y': em_y_sl_avg,
        'avg_ene': ene,
        'rel_ene_spread': ene_sp,
        'avg_slice_rel_ene_spread': ene_sp_sl_avg,
        'i_peak': i_peak,
        'q_tot': q_tot
    }
    return params_dict
