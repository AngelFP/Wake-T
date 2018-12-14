import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as ct
import aptools.data_analysis.beam_diagnostics as bd
from wake_t.beamline_elements import PlasmaStage, PlasmaRamp, Drift
from wake_t.driver_witness import LaserPulse
from wake_t.utilities.bunch_generation import get_matched_bunch

def test_tracking():
    # bunch parameters
    en = 1e-6
    ene = 200
    ene_sp = 0.1
    xi_c = 0
    s_t = 1
    q_tot = 10
    n_part = 1e4
    n_p = 1e17
    # create bunch
    bunch = get_matched_bunch(en, en, ene, ene_sp, 0, 0, xi_c, s_t, q_tot,
                              n_part, n_p=0.5e15)
    bunch.x += 0e-6
    # create laser
    laser = LaserPulse(100e-6, l_0=800e-9, w_0=50e-6, a_0=4)
    # create plasma stage and ramp
    upramp = PlasmaRamp(5e-2, 0.5e21, 1e23, ramp_type='upramp',
                      profile='inverse square')
    downramp = PlasmaRamp(5e-2, 0.5e22, 1e23, ramp_type='downramp',
                      profile='inverse square')
    plasma = PlasmaStage(1e17, 8e-2)
    drift = Drift(10e-2)
    # start tracking
    bunch_list = list()
    bunch_list.append(copy.copy(bunch))
    bunch_list.extend(upramp.track_beam_numerically_RK_parallel(bunch, 200))
    bunch_list.extend(plasma.track_beam_numerically_RK_parallel(
        laser, bunch, 'CustomBlowout', 200, lon_field=-3e10,
        lon_field_slope=6.6e14, foc_strength=2.6e6))
    bunch_list.extend(downramp.track_beam_numerically_RK_parallel(bunch, 200))
    bunch_list.extend(drift.track_bunch(bunch, 50))
    #bunch_list.extend(plasma.track_beam_analytically(
    #    laser, bunch, 'CustomBlowout', 200, lon_field=-3e10, 
    #    lon_field_slope=6.6e14, foc_strength=2.6e6))
    # analyze data
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
    x_centroid = np.zeros(l)
    y_centroid = np.zeros(l)
    px_centroid = np.zeros(l)
    py_centroid = np.zeros(l)
    cv0 = np.zeros(l)
    cv1 = np.zeros(l)
    cv2 = np.zeros(l)
    cv3 = np.zeros(l)
    sz = np.zeros(l)
    for i, beam in enumerate(beam_list):
        dist[i] = beam.prop_distance
        ax, bx, _ = bd.twiss_parameters(beam.x, beam.px, beam.pz, w=beam.q)
        ay, by, _ = bd.twiss_parameters(beam.y, beam.py, beam.pz, w=beam.q)
        a_x[i] = ax
        a_y[i] = ay
        b_x[i] = bx
        b_y[i] = by
        ene[i] = bd.mean_energy(beam.px, beam.py, beam.pz, w=beam.q)
        ene_sp[i] = bd.relative_rms_energy_spread(beam.px, beam.py, beam.pz, w=beam.q)
        enespls, sl_w, _ = bd.relative_rms_slice_energy_spread(
            beam.xi, beam.px, beam.py, beam.pz, w=beam.q)
        ene_sp_sl[i] = np.average(enespls, weights=sl_w)
        em_x[i] = bd.normalized_transverse_rms_emittance(
            beam.x, beam.px, w=beam.q)
        em_y[i] = bd.normalized_transverse_rms_emittance(
            beam.y, beam.py, w=beam.q)
        emsx, sl_w, _ = bd.normalized_transverse_rms_slice_emittance(
            beam.xi, beam.x, beam.px, w=beam.q)
        em_sl_x[i] = np.average(emsx, weights=sl_w)
        emsy, sl_w, _ = bd.normalized_transverse_rms_slice_emittance(
            beam.xi, beam.y, beam.py, w=beam.q)
        em_sl_y[i] = np.average(emsy, weights=sl_w)
        sz[i] = bd.rms_length(beam.xi, w=beam.q)
        x_centroid[i] = np.average(beam.x, weights=beam.q)
        y_centroid[i] = np.average(beam.y, weights=beam.q)
        px_centroid[i] = np.average(beam.px, weights=beam.q)
        py_centroid[i] = np.average(beam.py, weights=beam.q)
    plt.figure(1)
    plt.subplot(241)
    plt.semilogy(dist*1e3, b_x*1e3)
    plt.semilogy(dist*1e3, b_y*1e3)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\beta_x$ [mm]")
    plt.subplot(242)
    plt.plot(dist*1e3, a_x)
    plt.plot(dist*1e3, a_y)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\alpha_x$")
    plt.subplot(243)
    plt.plot(dist*1e3, ene)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\gamma$")
    plt.subplot(244)
    plt.semilogy(dist*1e3, ene_sp*100)
    plt.semilogy(dist*1e3, ene_sp_sl*100)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\frac{\\Delta \\gamma_z}{\\gamma}$ [%]")
    plt.subplot(245)
    plt.plot(dist*1e3, em_x*1e6)
    plt.plot(dist*1e3, em_y*1e6)
    # plt.plot(dist*1e3, em_sl_x*1e6)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\epsilon_{nx}$ [$\\mu$m]")
    plt.subplot(246)
    plt.plot(dist*1e3, sz/ct.c*1e15)
    plt.xlabel("z [mm]")
    plt.ylabel("$\\sigma_z$ [fs]")
    plt.subplot(247)
    plt.plot(dist*1e3, x_centroid*1e6)
    plt.plot(dist*1e3, y_centroid*1e6)
    plt.xlabel("z [mm]")
    plt.ylabel("bunch centroid [$\\mu m$]")
    plt.subplot(248)
    plt.plot(dist*1e3, px_centroid/ene*1e6)
    plt.plot(dist*1e3, py_centroid/ene*1e6)
    plt.xlabel("z [mm]")
    plt.ylabel("pointing angle [$\\mu rad$]")

    # beam = beam_list[-1]
    # z = beam.xi
    # pz = beam.pz
    # plt.figure(2)
    # plt.scatter(z, pz)
    plt.show()

if __name__ == '__main__':
    test_tracking()
