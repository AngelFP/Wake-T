import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as ct
import aptools.data_analysis.beam_diagnostics as bd

from wake_t.beamline_elements import (PlasmaStage, Quadrupole, PlasmaLens, 
                                      PlasmaRamp, Drift)
from wake_t.driver_witness import LaserPulse
from wake_t.utilities.bunch_generation import get_matched_bunch, get_from_file


def track_plasma_stage_col_fluid_model():
    # create laser
    laser = LaserPulse(100e-6, l_0=800e-9, w_0=70e-6, a_0=0.8, tau=30e-15)

    # create bunch
    en = 0.3e-6
    ene = 200
    ene_sp = 0.3
    xi_c = laser.xi_c - 50e-6
    s_t = 1
    q_tot = 0.1
    n_part = 1e4
    bunch = get_matched_bunch(en, en, ene, ene_sp, 0, 0, xi_c, s_t, q_tot,
                              n_part, k_x=130000)
    # create plasma stage
    plasma = PlasmaStage(1e17, 1e-2)

    # start tracking
    bunch_list = list()
    bunch_list.append(copy.copy(bunch))
    bunch_list.extend(plasma.track_beam_numerically(laser,
        bunch, mode='cold_fluid_1d', steps=20, laser_evolution=True, 
        laser_z_foc=0, r_max=70e-6, 
        xi_min=40e-6, xi_max=130e-6, n_r=200, n_xi=500))
    analyze_data(bunch_list)


def analyze_data(beam_list):
    print("Running data analysis...   ")
    l = len(beam_list)
    x_part = np.zeros(l)
    g_part = np.zeros(l)
    px_part = np.zeros(l)
    a_x = np.zeros(l)
    a_y = np.zeros(l)
    b_x = np.zeros(l)
    b_y = np.zeros(l)
    g_x = np.zeros(l)
    g_y = np.zeros(l)
    ene = np.zeros(l)
    dist = np.zeros(l)
    chirp = np.zeros(l)
    ene_sp = np.zeros(l)
    ene_sp_sl = np.zeros(l)
    emitt = np.zeros(l)
    em_x = np.zeros(l)
    em_y = np.zeros(l)
    em_sl_x = np.zeros(l)
    em_sl_y = np.zeros(l)
    emitt_3 = np.zeros(l)
    dx = np.zeros(l)
    sx = np.zeros(l)
    sy = np.zeros(l)
    x_centroid = np.zeros(l)
    y_centroid = np.zeros(l)
    px_centroid = np.zeros(l)
    py_centroid = np.zeros(l)
    disp_x = np.zeros(l)
    sz = np.zeros(l)
    for i, beam in enumerate(beam_list):
        dist[i] = beam.prop_distance
        a_x[i], b_x[i], g_x[i] = bd.twiss_parameters(
            beam.x, beam.px, beam.pz, beam.py, w=beam.q)
        a_y[i], b_y[i], g_y[i] = bd.twiss_parameters(
            beam.y, beam.py, beam.pz, beam.px, w=beam.q)
        ene[i] = bd.mean_energy(beam.px, beam.py, beam.pz, w=beam.q)
        ene_sp[i] = bd.relative_rms_energy_spread(beam.px, beam.py, beam.pz,
                                                  w=beam.q)
        enespls, sl_w, _ = bd.relative_rms_slice_energy_spread(
            beam.xi, beam.px, beam.py, beam.pz, w=beam.q, len_slice=0.1e-6)
        ene_sp_sl[i] = np.average(enespls, weights=sl_w)
        em_x[i] = bd.normalized_transverse_rms_emittance(
            beam.x, beam.px, beam.py, beam.pz, w=beam.q)
        em_y[i] = bd.normalized_transverse_rms_emittance(
            beam.y, beam.py, beam.px, beam.pz, w=beam.q)
        emsx, sl_w, _ = bd.normalized_transverse_rms_slice_emittance(
            beam.xi, beam.x, beam.px, beam.py, beam.pz,
            w=beam.q, len_slice=0.1e-6)
        em_sl_x[i] = np.average(emsx, weights=sl_w)
        emsy, sl_w, _ = bd.normalized_transverse_rms_slice_emittance(
            beam.xi, beam.y, beam.py, beam.px, beam.pz, 
            w=beam.q, len_slice=0.1e-6)
        em_sl_y[i] = np.average(emsy, weights=sl_w)
        sz[i] = bd.rms_length(beam.xi, w=beam.q)
        sx[i] = bd.rms_size(beam.x, w=beam.q)
        sy[i] = bd.rms_size(beam.y, w=beam.q)
        x_centroid[i] = np.average(beam.x, weights=beam.q)
        y_centroid[i] = np.average(beam.y, weights=beam.q)
        px_centroid[i] = np.average(beam.px, weights=beam.q)
        py_centroid[i] = np.average(beam.py, weights=beam.q)
        disp_x[i] = bd.dispersion(beam.x, beam.px, beam.pz, beam.py, w=beam.q)
    plt.figure(1)
    plt.subplot(341)
    plt.semilogy(dist*1e3, b_x*1e3)
    plt.semilogy(dist*1e3, b_y*1e3)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\beta_x$ [mm]")
    plt.subplot(342)
    plt.plot(dist*1e3, a_x)
    plt.plot(dist*1e3, a_y)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\alpha_x$")
    plt.subplot(343)
    plt.plot(dist*1e3, g_x)
    plt.plot(dist*1e3, g_y)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\gamma_x$")
    plt.subplot(344)
    plt.plot(dist*1e3, ene)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\gamma$")
    plt.subplot(345)
    plt.semilogy(dist*1e3, ene_sp*100)
    plt.semilogy(dist*1e3, ene_sp_sl*100)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\frac{\\Delta \\gamma_z}{\\gamma}$ [%]")
    plt.subplot(346)
    plt.plot(dist*1e3, em_x*1e6)
    plt.plot(dist*1e3, em_y*1e6)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\epsilon_{nx}$ [$\\mu$m]")
    plt.subplot(347)
    plt.plot(dist*1e3, sx*1e6)
    plt.plot(dist*1e3, sy*1e6)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\sigma_{x,y}$ [$\\mu$m]")
    plt.subplot(348)
    plt.plot(dist*1e3, sz/ct.c*1e15)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\sigma_z$ [fs]")
    plt.subplot(349)
    plt.plot(dist*1e3, x_centroid*1e6)
    plt.plot(dist*1e3, y_centroid*1e6)
    plt.xlabel("z [mm]")
    plt.ylabel("bunch centroid [$\\mu m$]")
    plt.subplot(3,4,10)
    plt.plot(dist*1e3, px_centroid/ene*1e6)
    plt.plot(dist*1e3, py_centroid/ene*1e6)
    plt.xlabel("z [mm]")
    plt.ylabel("pointing angle [$\\mu rad$]")
    plt.subplot(3,4,11)
    plt.plot(dist*1e3, disp_x)
    plt.xlabel("z [mm]")
    plt.ylabel("$D_x$ [m]")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    track_plasma_stage_col_fluid_model()
