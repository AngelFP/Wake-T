import os
import shutil
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

from openpmd_viewer import OpenPMDTimeSeries
from wake_t import ParticleBunch, PlasmaStage, Beamline


tests_output_folder = "./tests_output"


def run_simulation(output_folder):
    # Plasma density profile
    n_p0 = 1.7e23  # On-axis density in the plasma channel (m^-3)
    kp = np.sqrt(
        n_p0 * sc.e**2 / (sc.epsilon_0 * sc.m_e * sc.c**2)
    )  # plasma wavenumber
    l_plateau = 1e-2
    l_ramp = 0
    l_total = l_plateau + 2 * l_ramp

    def density_profile(z, r):
        return n_p0 * np.ones_like(z)

    # Driver beam
    q_tot = 1000e-12  # [C]
    ene_sp = 0.1  # [%]
    n_emitt_x = 0.1e-6
    n_emitt_y = 0.1e-6

    kin_energy_GeV = 100.0
    mc2_GeV = sc.m_e * sc.c**2 / sc.electron_volt / 1e9
    gamma_beam = 1 + kin_energy_GeV / mc2_GeV

    betax0 = np.sqrt(2 * gamma_beam) / kp
    sx0 = np.sqrt(n_emitt_x * betax0 / gamma_beam)
    sy0 = np.sqrt(n_emitt_y * betax0 / gamma_beam)

    l_beam = 5.453219e-6
    n_part = int(1e6)
    np.random.seed(0)

    z = np.random.uniform(-l_beam / 2, l_beam / 2, n_part)
    x = sx0 * np.random.standard_normal(n_part)
    y = sy0 * np.random.standard_normal(n_part)
    s_g = ene_sp * gamma_beam / 100
    gamma = np.random.normal(gamma_beam, s_g, n_part)
    s_ux = n_emitt_x / sx0
    s_uy = n_emitt_y / sy0
    ux = s_ux * np.random.standard_normal(n_part)
    uy = s_uy * np.random.standard_normal(n_part)
    uz = np.sqrt((gamma**2 - 1) - ux**2 - uy**2)
    q = np.ones(n_part) * q_tot / n_part
    w = np.abs(q / sc.e)
    bunch_dri = ParticleBunch(w, x, y, z, ux, uy, uz, name="bunch_dri")

    # Witness beam
    q_tot = 100e-12  # [C]
    kin_energy_GeV = 0.1
    gamma_beam = 1 + kin_energy_GeV / mc2_GeV

    betax0 = np.sqrt(2 * gamma_beam) / kp
    sx0 = np.sqrt(n_emitt_x * betax0 / gamma_beam)
    sy0 = np.sqrt(n_emitt_y * betax0 / gamma_beam)

    l_beam = 5.453219e-6
    d_beam = 62.973740e-6 * 1.2

    np.random.seed(0)
    zc = -d_beam - l_beam / 3
    z = np.random.uniform(zc - l_beam / 2, zc + l_beam / 2, n_part)
    x = sx0 * np.random.standard_normal(n_part)
    y = sy0 * np.random.standard_normal(n_part)
    s_g = ene_sp * gamma_beam / 100
    gamma = np.random.normal(gamma_beam, s_g, n_part)
    s_ux = n_emitt_x / sx0
    s_uy = n_emitt_y / sy0
    ux = s_ux * np.random.standard_normal(n_part)
    uy = s_uy * np.random.standard_normal(n_part)
    uz = np.sqrt((gamma**2 - 1) - ux**2 - uy**2)
    q = np.ones(n_part) * q_tot / n_part
    w = np.abs(q / sc.e)
    bunch_wit = ParticleBunch(w, x, y, z, ux, uy, uz, name="bunch_wit")

    # Simulation box and grid parameters
    r_max = 144e-6
    r_max_plasma = 108e-6
    l_box = 200e-6
    xi_max = 0 + 45e-6
    xi_min = xi_max - l_box
    dz = 1 / kp / 80
    nz = int(l_box / dz)
    dr = 1 / kp / 40
    nr = int(r_max / dr)
    dz_fields = 200e-6

    # Adaptive grid setup
    res_beam_r = 10.0
    adaptive_dr = np.min([sx0, sy0]) / res_beam_r
    sxi = np.std(x)
    syi = np.std(y)
    adaptive_grid_r_max = 4 * np.max([sxi, syi])
    adaptive_grid_nr = int(adaptive_grid_r_max / adaptive_dr)
    ppc = 2
    ppc = [
        [1.5 * adaptive_grid_r_max, ppc * int(dr / adaptive_dr)],
        [r_max_plasma, ppc],
    ]

    plasma_plateau = PlasmaStage(
        length=l_total,
        density=density_profile,
        wakefield_model="quasistatic_2d",
        ion_motion=False,
        n_out=50,
        laser=None,
        laser_evolution=True,
        r_max=r_max,
        r_max_plasma=r_max_plasma,
        xi_min=xi_min,
        xi_max=xi_max,
        n_r=nr,
        n_xi=nz,
        dz_fields=dz_fields,
        ppc=ppc,
        laser_envelope_substeps=4,
        laser_envelope_nxi=nz * 4,
        max_gamma=10,
        field_diags=[
            "rho",
            "E",
            "B",
            "charge_profile",
        ],
        use_adaptive_grids=False,
        adaptive_grid_r_max=adaptive_grid_r_max,
        adaptive_grid_nr=adaptive_grid_nr,
        adaptive_grid_diags=["E", "B"],
    )

    bunch_wit.do_salame = True
    bunch_wit.salame_n_iter = 100
    bunch_wit.salame_relative_tolerance = 1e-6

    shutil.rmtree(output_folder, ignore_errors=True)

    beamline = Beamline([plasma_plateau])
    beamline.track(
        [bunch_dri, bunch_wit],
        opmd_diag=True,
        show_progress_bar=True,
        diag_dir=output_folder,
    )


def test_salame(plot=False):
    """
    This test checks that the SALAME-generated charge profile produces
    a bunch-weighted longitudinal electric field that is sufficiently flat
    across the nonzero part of the bunch.

    The residual is defined as the average relative deviation of the
    bunch-weighted Ez from its value at the bunch tail.
    """
    output_folder = os.path.join(tests_output_folder, "salame_output")
    diag_dir = os.path.join(output_folder, "hdf5")

    run_simulation(output_folder)

    # Load openPMD diagnostics.
    ts = OpenPMDTimeSeries(diag_dir)

    # Select iteration.
    it = ts.iterations[0]

    # Read longitudinal electric field.
    Ez, info_Ez = ts.get_field(iteration=it, field="E", coord="z")
    z = info_Ez.z

    dz = z[1] - z[0]
    nr = Ez.shape[0]

    # Read the deposited beam charge profile (driver + witness).
    q_bunch, _ = ts.get_field(iteration=it, field="charge_profile")

    # Convert to (z, r) ordering for easier processing.
    Ez_zr = Ez.T
    q_bunch_zr = q_bunch.T

    witness_mask = z < -20e-6
    q_bunch_zr = q_bunch_zr * witness_mask[:, np.newaxis]

    # Compute line charge and current profile.
    g_xi = np.sum(q_bunch_zr[:, int(nr / 2) :], axis=1)
    I_z = g_xi / dz * sc.c

    # Compute bunch-weighted <Ez>.
    w = np.abs(q_bunch_zr)
    wsum = np.sum(w, axis=1)

    Ez_bunch_weighted = np.zeros(Ez_zr.shape[0], dtype=float)
    mask = wsum > 0
    Ez_bunch_weighted[mask] = np.sum(Ez_zr[mask, :] * w[mask, :], axis=1) / wsum[mask]

    # Restrict to nonzero bunch region.
    indices = np.flatnonzero(wsum)
    Ez_target = Ez_bunch_weighted[indices[-1]]
    Ez_nonzero = Ez_bunch_weighted[indices]

    # Compute flattening residual.
    residual = np.sum(np.abs(Ez_nonzero - Ez_target)) / (
        np.abs(Ez_target) * len(Ez_nonzero)
    )

    if residual >= 0.1:
        raise AssertionError("SALAME execution wrong.")

    # Optional diagnostic plot.
    if plot:
        fig, ax1 = plt.subplots(figsize=(6, 3), constrained_layout=False)

        ax1.plot(
            z * 1e6,
            I_z,
            "r",
            marker="o",
            markersize=3,
            linewidth=1,
            label="Current [A]",
        )
        ax1.set_xlabel(r"$\xi$ [$\mu$m]")
        ax1.set_ylabel(r"Current [A]", color="red")
        ax1.tick_params(axis="y", colors="red")
        ax1.set_xlim(z[indices[0]] / 1e-6 - 2, z[indices[-1]] / 1e-6 + 2)
        ax1.set_ylim(-22000, 1200)

        ax2 = ax1.twinx()
        ax2.plot(
            z * 1e6,
            Ez_bunch_weighted,
            "b",
            marker="o",
            markersize=3,
            linewidth=1,
            linestyle="--",
            label=r"Weighted $\langle E_z\rangle$ [V/m]",
        )
        ax2.set_ylabel(r"Weighted $\langle E_z\rangle$ [V/m]", color="blue")
        ax2.tick_params(axis="y", colors="blue")
        ax2.set_ylim(-5e10, 3e9)

        plt.show()


if __name__ == "__main__":
    test_salame(plot=False)
