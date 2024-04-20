import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from wake_t import PlasmaStage
from wake_t.utilities.bunch_generation import get_matched_bunch
from wake_t.diagnostics import analyze_bunch_list
from wake_t.physics_models.plasma_wakefields.qs_rz_baxevanis_ion.gather import gather_bunch_sources
from wake_t.fields.gather import gather_fields


def test_adaptive_grid():
    """Test a plasma simulation using adaptive grids.

    This test runs a simulation of a single time step with and
    without using adaptive grids. The adaptive grids have the same
    radial extent and resolution as the base grid. As such, the test
    checks that the simulation results are identical.

    The resolution of the adaptive and the base grids is identical.
    """
    # Set numpy random seed to get reproducible results.
    np.random.seed(1)

    # Plasma density.
    n_p = 1e23

    # Create driver.
    en = 10e-6
    gamma = 2000
    s_t = 10
    ene_sp = 1
    q_tot = 500
    driver = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=0,
        q_tot=q_tot, n_part=1e4, n_p=n_p, name='driver')

    # Create witness.
    en = 1e-6
    gamma = 200
    s_t = 3
    ene_sp = 0.1
    q_tot = 50
    witness = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=-80e-6,
        q_tot=q_tot, n_part=1e4, n_p=n_p, name='witness')

    # Run simulations with and without adaptive grid.
    driver_params, witness_params, plasma = run_simulation(
        deepcopy(driver), deepcopy(witness), n_p, use_ag=False
    )
    driver_params_ag, witness_params_ag, plasma_ag = run_simulation(
        deepcopy(driver), deepcopy(witness), n_p, use_ag=True
    )

    ag_driver = plasma_ag.wakefield.bunch_grids['driver']
    ag_witness = plasma_ag.wakefield.bunch_grids['witness']
    ags = [ag_driver, ag_witness]
    q_bunch_base = plasma.wakefield.q_bunch[2:-2, 2:-2]
    bt_bunch_base = plasma.wakefield.b_t_bunch[2:-2, 2:-2]
    er_base = plasma.wakefield.fld_arrays[4][2:-2, 2:-2]
    ez_base = plasma.wakefield.fld_arrays[5][2:-2, 2:-2]
    bt_base = plasma.wakefield.fld_arrays[6][2:-2, 2:-2]

    for ag in ags:
        q_bunch_ag = ag.q_bunch[2:-2, 2:-2]
        bt_bunch_ag = ag.b_t_bunch[2:-2, 2:-2]
        er_ag = ag.e_r[2:-2, 2:-2]
        ez_ag = ag.e_z[2:-2, 2:-2]
        bt_ag = ag.b_t[2:-2, 2:-2]

        # Check that the shape of the grid fields and properties is consistent.
        assert er_ag.shape[0] == ag.xi_grid.shape[0]
        assert er_ag.shape[1] == ag.r_grid.shape[0]
        assert er_ag.shape[1] == ag.nr

        # Check that the grid spacing is the same as in the base grid.
        assert plasma_ag.wakefield.dr == ag.dr
        assert plasma_ag.wakefield.dxi == ag.dxi

        # Check that the grid coordinate arrays overlap.
        np.testing.assert_array_equal(
            plasma_ag.wakefield.r_fld, ag.r_grid[:-ag.nr_border]
        )
        np.testing.assert_array_equal(
            plasma_ag.wakefield.xi_fld[ag.i_grid], ag.xi_grid
        )

        # Check that bunch charge distribution and space charge agree
        # between the base and the adaptive grid.
        np.testing.assert_allclose(
            q_bunch_base[ag.i_grid], q_bunch_ag[:, :-ag.nr_border], rtol=1e-11
        )
        np.testing.assert_allclose(
            bt_bunch_base[ag.i_grid],
            bt_bunch_ag[:, :-ag.nr_border],
            rtol=1e-11
        )

        # Check that the field in the adaptive grid agree with those
        # of the base grid.
        np.testing.assert_allclose(
            bt_base[ag.i_grid], bt_ag[:, :-ag.nr_border], rtol=1e-8
        )
        np.testing.assert_allclose(
            er_base[ag.i_grid, :-1], er_ag[:, :-ag.nr_border-1], rtol=1e-8
        )
        np.testing.assert_allclose(
            ez_base[ag.i_grid][1:-1], ez_ag[1:-1, :-ag.nr_border], rtol=1e-8
        )


def test_adaptive_grid_undersized():
    """Test a plasma simulation using adaptive grids.

    This test runs a simulation of a single time step with and
    without using adaptive grids. The adaptive grids are undersized,
    meaning that the bunches will not fully fit within them. As such, the
    beams have to deposit to and gather from both the adaptive grids
    as well as the base grid. This test checks that the simulation results
    when the adaptive grids are undersized are identical to when no
    adaptive grids are used.

    The resolution of the adaptive and the base grids is identical.
    """
    # Set numpy random seed to get reproducible results.
    np.random.seed(1)

    # Plasma density.
    n_p = 1e23

    # Create driver.
    en = 10e-6
    gamma = 2000
    s_t = 10
    ene_sp = 1
    q_tot = 10
    driver = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=0,
        q_tot=q_tot, n_part=1e4, n_p=n_p, name='driver')

    # Create witness.
    en = 1e-6
    gamma = 200
    s_t = 3
    ene_sp = 0.1
    q_tot = 5
    witness = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=-60e-6,
        q_tot=q_tot, n_part=1e4, n_p=n_p, name='witness')

    for p_shape in ["linear", "cubic"]:

        # Run simulations with and without adaptive grid.
        driver_params, witness_params, plasma = run_simulation(
            deepcopy(driver), deepcopy(witness), n_p, length=1e-6, use_ag=False,
            p_shape=p_shape
        )

        # Run simulations with fixed (undersized) adaptive grid.
        driver_params_ag, witness_params_ag, plasma_ag = run_simulation(
            deepcopy(driver), deepcopy(witness), n_p, length=1e-6, use_ag=True,
            nr_ag=[3, 3], r_max_ag=[3e-6, 3e-6], r_lim_ag=[None, None],
            p_shape=p_shape
        )
        # plt.show()
        ag_driver = plasma_ag.wakefield.bunch_grids['driver']
        ag_witness = plasma_ag.wakefield.bunch_grids['witness']
        q_bunch_base_ag = plasma_ag.wakefield.q_bunch[2:-2, 2:-2]
        q_bunch_base = plasma.wakefield.q_bunch[2:-2, 2:-2]
        er_base_ag = plasma_ag.wakefield.e_r[2:-2, 2:-2]
        ez_base_ag = plasma_ag.wakefield.e_z[2:-2, 2:-2]
        bt_base_ag = plasma_ag.wakefield.b_t[2:-2, 2:-2]
        er_base = plasma.wakefield.e_r[2:-2, 2:-2]
        ez_base = plasma.wakefield.e_z[2:-2, 2:-2]
        bt_base = plasma.wakefield.b_t[2:-2, 2:-2]

        q_bunch_ag_combined = deepcopy(q_bunch_base_ag)
        ags = [ag_driver, ag_witness]
        for ag in ags:
            q_bunch_ag = ag.q_bunch[2:-2, 2:]

            q_bunch_ag_combined[ag.i_grid, :ag.nr+2] += q_bunch_ag

        # Check that the charge has been correctly deposited across all grids.
        np.testing.assert_allclose(
                q_bunch_base, q_bunch_ag_combined, rtol=1e-12
            )

        r_test = np.linspace(0, 70e-6, 70)
        b_t_test = np.zeros_like(r_test)
        b_t_test_ag = np.zeros_like(r_test)
        slice_i = 180
        gather_bunch_sources(
            plasma.wakefield.b_t_bunch[slice_i + 2],
            plasma.wakefield.r_fld[0],
            plasma.wakefield.r_fld[-1] + 2*plasma.wakefield.dr,
            plasma.wakefield.dr,
            r_test,
            b_t_test
        )

        gather_bunch_sources(
            ag_driver.b_t_bunch[slice_i + 2 - ag_driver.i_grid[0]],
            ag_driver.r_grid[0],
            ag_driver.r_grid[-1] + 2*ag_driver.dr,
            ag_driver.dr,
            r_test,
            b_t_test_ag
        )
        gather_bunch_sources(
            plasma_ag.wakefield.b_t_bunch[slice_i + 2],
            plasma_ag.wakefield.r_fld[0],
            plasma_ag.wakefield.r_fld[-1] + 2*plasma_ag.wakefield.dr,
            plasma_ag.wakefield.dr,
            r_test,
            b_t_test_ag
        )
        np.testing.assert_allclose(b_t_test, b_t_test_ag, rtol=1e-12)

        # Check that the fields in the base grid agree between the cases with
        # and without adaptive grids.
        np.testing.assert_allclose(er_base, er_base_ag, rtol=1e-7)
        np.testing.assert_allclose(ez_base, ez_base_ag, rtol=1e-5)
        np.testing.assert_allclose(bt_base, bt_base_ag, rtol=1e-7)

        # Check that the fields gathered by the bunch agree between the cases
        # with and without adaptive grids.
        field_arrays = deepcopy(driver).get_field_arrays()
        gather_fields([plasma.wakefield], driver.x, driver.y, driver.xi, 0.,
                      *field_arrays, driver.name)

        field_arrays_ag = deepcopy(driver).get_field_arrays()
        gather_fields([plasma_ag.wakefield], driver.x, driver.y, driver.xi, 0.,
                      *field_arrays_ag, driver.name)
        for arr, arr_ag in zip(field_arrays, field_arrays_ag):
            np.testing.assert_allclose(
                        arr, arr_ag, rtol=1e-11
                    )


def test_adaptive_grids_evolution(create_test_data=False, plot=False):
    """Test that the radial evolution of the adaptive grids is as expected."""
    # Set numpy random seed to get reproducible results.
    np.random.seed(1)

    # Plasma density.
    n_p = 1e23

    # Create driver.
    en = 10e-6
    gamma = 2000
    s_t = 10
    ene_sp = 1
    q_tot = 500
    driver = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=0,
        q_tot=q_tot, n_part=1e4, n_p=n_p, name='driver')

    # Create witness.
    en = 1e-6
    gamma = 200
    s_t = 3
    ene_sp = 0.1
    q_tot = 50
    witness = get_matched_bunch(
        en_x=en, en_y=en, ene=gamma, ene_sp=ene_sp, s_t=s_t, xi_c=-80e-6,
        q_tot=q_tot, n_part=1e4, n_p=n_p, name='witness')

    # Run simulation without adaptive grid.
    driver_params, witness_params, plasma = run_simulation(
        deepcopy(driver), deepcopy(witness), n_p, use_ag=False,
        length=1e-2
    )
    # Run simulation with adaptive grid.
    driver_params_ag, witness_params_ag, plasma_ag = run_simulation(
        deepcopy(driver), deepcopy(witness), n_p, length=1e-2, use_ag=True,
        nr_ag=[6, 6], r_max_ag=[None, None], r_lim_ag=[None, None]
    )
    # Run simulation with fixed adaptive grid.
    driver_params_agf, witness_params_agf, plasma_agf = run_simulation(
        deepcopy(driver), deepcopy(witness), n_p, length=1e-2, use_ag=True,
        nr_ag=[6, 6], r_max_ag=[6e-6, 6e-6], r_lim_ag=[None, None]
    )
    # Run simulation with limited adaptive grid.
    driver_params_agl, witness_params_agl, plasma_agl = run_simulation(
        deepcopy(driver), deepcopy(witness), n_p, length=1e-2, use_ag=True,
        nr_ag=[6, 6], r_max_ag=[None, None], r_lim_ag=[15e-6, 15e-6]
    )

    # Check that the fixed grid is indeed fixed
    np.testing.assert_array_equal(
        plasma_agf.wakefield.bunch_grids['driver']._r_max_hist, 6e-6
    )
    np.testing.assert_array_equal(
        plasma_agf.wakefield.bunch_grids['witness']._r_max_hist, 6e-6
    )

    # Save arrays with the evolution of the radial size of the grids.
    if create_test_data:
        np.savetxt(
            os.path.join("resources", "r_max_hist_driver_ag.txt"),
            plasma_ag.wakefield.bunch_grids['driver']._r_max_hist
        )
        np.savetxt(
            os.path.join("resources", "r_max_hist_witness_ag.txt"),
            plasma_ag.wakefield.bunch_grids['witness']._r_max_hist
        )
        np.savetxt(
            os.path.join("resources", "r_max_hist_driver_agl.txt"),
            plasma_agl.wakefield.bunch_grids['driver']._r_max_hist
        )
        np.savetxt(
            os.path.join("resources", "r_max_hist_witness_agl.txt"),
            plasma_agl.wakefield.bunch_grids['witness']._r_max_hist
        )

    # Check that the radial evolution is as expected.
    np.testing.assert_allclose(
        plasma_ag.wakefield.bunch_grids['driver']._r_max_hist,
        np.loadtxt(os.path.join("resources", "r_max_hist_driver_ag.txt")),
        rtol=1e-12
    )
    np.testing.assert_allclose(
        plasma_ag.wakefield.bunch_grids['witness']._r_max_hist,
        np.loadtxt(os.path.join("resources", "r_max_hist_witness_ag.txt")),
        rtol=1e-12
    )
    np.testing.assert_allclose(
        plasma_agl.wakefield.bunch_grids['driver']._r_max_hist,
        np.loadtxt(os.path.join("resources", "r_max_hist_driver_agl.txt")),
        rtol=1e-12
    )
    np.testing.assert_allclose(
        plasma_agl.wakefield.bunch_grids['witness']._r_max_hist,
        np.loadtxt(os.path.join("resources", "r_max_hist_witness_agl.txt")),
        rtol=1e-12
    )

    # Check that the beam evolution is identical for a fixed adaptive grid
    # and no adaptive grid.
    for param in driver_params.keys():
        np.testing.assert_allclose(
            driver_params[param],
            driver_params_agf[param],
            rtol=1e-12
        )
        np.testing.assert_allclose(
            witness_params[param],
            witness_params_agf[param],
            rtol=1e-11
        )

    if plot:
        plt.plot(plasma_ag.wakefield.bunch_grids['driver']._r_max_hist, label="driver grid")
        plt.plot(plasma_ag.wakefield.bunch_grids['witness']._r_max_hist, label="witness grid")
        plt.plot(plasma_agl.wakefield.bunch_grids['driver']._r_max_hist, label="driver grid (limited)")
        plt.plot(plasma_agl.wakefield.bunch_grids['witness']._r_max_hist, label="witness grid (limited)")
        plt.legend()
        plt.show()


def run_simulation(driver, witness, n_p, length=1e-6, use_ag=False,
                   nr_ag=[70, 70], r_max_ag=[70e-6, 70e-6],
                   r_lim_ag=[None, None], p_shape="cubic"):
    model_params = {
        "xi_max": 20e-6,
        "xi_min": -90e-6,
        "r_max": 70e-6,
        "n_xi": 220,
        "n_r": 70,
        "dz_fields": 1e-3,
        "p_shape": p_shape,
    }
    if use_ag:
        model_params['use_adaptive_grids'] = True
        model_params['adaptive_grid_nr'] = nr_ag
        model_params['adaptive_grid_r_max'] = r_max_ag
        model_params['adaptive_grid_r_lim'] = r_lim_ag

    plasma = PlasmaStage(
        length=length,
        density=n_p,
        wakefield_model='quasistatic_2d_ion',
        n_out=3,
        **model_params
    )

    # Do tracking.
    output = plasma.track([driver, witness])

    # Analyze evolution.
    driver_params = analyze_bunch_list(output[0])
    witness_params = analyze_bunch_list(output[1])
    return driver_params, witness_params, plasma


if __name__ == "__main__":
    test_adaptive_grid()
    test_adaptive_grid_undersized()
    test_adaptive_grids_evolution(create_test_data=True, plot=True)
