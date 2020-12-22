""" This module contains the classes for all beamline elements. """
# TODO: unify classes of plasma elements to avoid code repetition (specially in
# tracking)
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import copy

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.particle_tracking import (runge_kutta_4, track_with_transfer_map)
import wake_t.wakefields as wf
from wake_t.driver_witness import ParticleBunch
from wake_t.utilities.other import print_progress_bar
from wake_t.utilities.bunch_manipulation import (
    convert_to_ocelot_matrix, convert_from_ocelot_matrix, rotation_matrix_xz)
from wake_t.csr import get_csr_calculator
from wake_t.diagnostics import OpenPMDDiagnostics


class Beamline():

    """Class for grouping beamline elements and allowing easier tracking."""

    def __init__(self, elements):
        self.elements = elements

    def track(self, bunch, out_initial=True, opmd_diag=False, diag_dir=None):
        """
        Track bunch through beamline.

        Parameters:
        -----------
        bunch : ParticleBunch
            Particle bunch to be tracked.

        out_initial : bool
            Determines whether the initial bunch should be included in the
            output bunch list. This applies only at the beginning and not for
            every beamline element.

        opmd_diag : bool or OpenPMDDiagnostics
            Determines whether to write simulation diagnostics to disk (i.e.
            particle distributions and fields). The output is written to
            HDF5 files following the openPMD standard. The number of outputs
            per beamline element is determined by its `n_out` value. It is also
            possible to provide an already existing OpenPMDDiagnostics
            instance instead of a boolean value.

        diag_dir : str
            Directory into which the openPMD output will be written. By default
            this is a 'diags' folder in the current directory. Only needed if
            `opmd_diag=True`.

        Returns:
        --------
        A list of size 'n_out' containing the bunch distribution at each step.

        """
        bunch_list = []
        if out_initial:
            bunch_list.append(copy(bunch))
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        for element in self.elements:
            bunch_list.extend(
                element.track(bunch, out_initial=False, opmd_diag=opmd_diag))
        return bunch_list


class PlasmaStage():

    """ Defines a plasma stage. """

    def __init__(self, length, n_p, laser=None, tracking_mode='numerical',
                 wakefield_model='simple_blowout', n_out=None, **model_params):
        """
        Initialize plasma stage.

        Parameters:
        -----------
        length : float
            Length of the plasma stage in cm.

        n_p : float
            Plasma density in units of m^{-3}.

        laser : LaserPulse
            Laser driver of the plasma stage.

        tracking_mode : str
            Tracking algorithm used for the bunch particles. Can be 'numerical'
            for the Runge-Kutta solver or 'analytical' to use the model from
            https://doi.org/10.1038/s41598-019-53887-8

        wakefield_model : str
            Wakefield model to be used. Possible values are 'simple_blowout',
            'custom_blowout', 'from_pic_code', 'cold_fluid_1d' and
            'quasistatic_2d'.

        n_out : int
            Number of times along the stage in which the particle distribution
            should be returned (A list with all output bunches is returned
            after tracking).

        **model_params
            Keyword arguments which will be given to the wakefield model. Each
            model requires a different set of parameters which are listed
            below:

        Model 'simple_blowout'
        ----------------------
        field_offset : float
            By default, the zero-crossing of the accelerating field in the
            cavity is assumed to be lambda_p/2 behind the driver. An offset
            value >0 (<0) will give the fields a positive (negative) offset
            towards the front (back) of the bunch.

        Model 'custom_blowout'
        ----------------------
        lon_field : float
            Value of the longitudinal electric field at the bunch center at the
            beginning of the tracking in units of V/m.

        lon_field_slope : float
            Value of the longitudinal electric field slope along z at the bunch
            center at the beginning of the tracking in units of V/m^2.

        foc_strength : float
            Value of the focusing gradient along the bunch in units of T/m.

        xi_fields : float
            Longitudinal position at which the wakefields have the values
            specified by the parameter above. If not specified, this will
            be the bunch center at the beginning of the plasma stage.

        Model 'from_pic_code'
        ---------------------
        simulation_code : string
            Name of the simulation code from which fields should be read.

        simulation_path : string
            Path to the simulation folder where the fields to read are located.

        time_step : int
            Time step at which the fields should be read.

        auto_update_fields : bool
            If True, new fields will be read from the simulation folder
            automatically as the particles travel through the plasma. The first
            time step will be time_step, and new ones will be loaded as the
            time of flight of the bunch reaches the next time step available in
            the simulation folder.

        reverse_tracking : bool
            Whether to reverse-track the particles through the stage.

        laser_pos_in_pic_code : float (deprecated)
            Position of the laser pulse center in the co-moving frame in the
            pic code simulation.

        filter_fields : bool
            If true, a Gaussian filter is applied to smooth the wakefields.
            This can be useful to remove noise.

        filter_sigma : float
            Sigma to be used by the Gaussian filter.

        Model 'cold_fluid_1d'
        ---------------------
        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the beginning of the PlasmaStage. A negative
            value implies that the focal point is located before the
            PlasmaStage.

        beam_wakefields : bool
            Whether to take into account beam-driven wakefields (False by
            default). This should be set to True for any beam-driven case or
            in order to take into account the beam-loading of the witness in
            a laser-driven case.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calculated.

        xi_min : float
            Minimum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        xi_max : float
            Maximum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        n_r : int
            Number of grid elements along r to calculate the wakefields.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic'.

        Model 'quasistatic_2d'
        ---------------------
        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the beginning of the PlasmaStage. A negative
            value implies that the focal point is located before the
            PlasmaStage.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calculated.

        xi_min : float
            Minimum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        xi_max : float
            Maximum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        n_r : int
            Number of grid elements along r to calculate the wakefields.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.

        ppc : int (optional)
            Number of plasma particles per 1d cell along the radial direction.
            By default ppc=2.

        dz_fields : float (optional)
            Determines how often the plasma wakefields should be updated. If
            dz_fields=0 (default value), the wakefields are calculated at every
            step of the Runge-Kutta solver for the beam particle evolution
            (most expensive option). If specified, the wakefields are only
            updated in steps determined by dz_fields. For example, if
            dz_fields=10e-6, the plasma wakefields are only updated every time
            the simulation window advances by 10 micron. If dz_fields=None, the
            wakefields are only computed once (at the start of the plasma) and
            never updated throughout the simulation.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic'.

        """
        self.length = length
        self.n_p = n_p
        self.laser = laser
        self.tracking_mode = tracking_mode
        self.wakefield_model = wakefield_model
        self.wakefield = self._get_wakefield(wakefield_model, model_params)
        self.n_out = n_out
        self.model_params = model_params

    def track(self, bunch, parallel=False, n_proc=None, out_initial=False,
              opmd_diag=False, diag_dir=None):
        """
        Track bunch through plasma stage.

        Parameters:
        -----------
        bunch : ParticleBunch
            Particle bunch to be tracked.

        parallel : float
            Determines whether or not to use parallel computation.

        n_proc : int
            Number of processes to run in parallel. If None, this will equal
            the number of physical cores.

        out_initial : bool
            Determines whether the initial bunch should be included in the
            output bunch list.

        opmd_diag : bool or OpenPMDDiagnostics
            Determines whether to write simulation diagnostics to disk (i.e.
            particle distributions and fields). The output is written to
            HDF5 files following the openPMD standard. The number of outputs
            the `n_out` value. It is also possible to provide an already
            existing OpenPMDDiagnostics instance instead of a boolean value.

        diag_dir : str
            Directory into which the openPMD output will be written. By default
            this is a 'diags' folder in the current directory. Only needed if
            `opmd_diag=True`.

        Returns:
        --------
        A list of size 'n_out' containing the bunch distribution at each step.

        """
        print('')
        print('Plasma stage')
        print('-'*len('Plasma stage'))
        if out_initial:
            initial_bunch = copy(bunch)
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        if self.tracking_mode == 'numerical':
            bunch_list = self._track_numerically(
                bunch, parallel, n_proc, opmd_diag)
        elif self.tracking_mode == 'analytical':
            bunch_list = self._track_analytically(bunch, parallel, n_proc)
        if out_initial:
            bunch_list.insert(0, initial_bunch)
        return bunch_list

    def _get_wakefield(self, wakefield_model, model_params):
        """Create and return corresponding wakefield."""
        if wakefield_model == "simple_blowout":
            WF = wf.SimpleBlowoutWakefield(
                self.n_p, driver=self.laser, **model_params)
        if wakefield_model == "custom_blowout":
            WF = wf.CustomBlowoutWakefield(
                self.n_p, driver=self.laser, **model_params)
        elif wakefield_model == "from_pic_code":
            raise NotImplementedError('Needs to be updated for new VisualPIC')
            # if vpic_installed:
            #    WF = wf.WakefieldFromPICSimulation(self.n_p, **model_params)
            # else:
            #    raise ImportError('VisualPIC is not installed.')
        elif wakefield_model == 'cold_fluid_1d':
            WF = wf.NonLinearColdFluidWakefield(
                self.calculate_density, driver=self.laser, **model_params)
        elif wakefield_model == 'quasistatic_2d':
            WF = wf.Quasistatic2DWakefield(
                self.calculate_density, laser=self.laser, **model_params)
        return WF

    def _gamma(self, px, py, pz):
        return np.sqrt(1 + px**2 + py**2 + pz**2)

    def _track_numerically(self, bunch, parallel, n_proc, opmd_diag):
        # Get 6D matrix
        mat = bunch.get_6D_matrix_with_charge()
        # Plasma length in time
        t_final = self.length/ct.c
        t_step = t_final/self.n_out
        dt = self._get_optimized_dt(bunch, self.wakefield)
        iterations = int(t_final/dt)
        # force at least 1 iteration per step
        it_per_step = max(int(iterations/self.n_out), 1)
        iterations = it_per_step*self.n_out
        dt_adjusted = t_final/iterations
        # initialize list to store the distribution at each step
        bunch_list = list()
        # get start time
        start = time.time()
        if parallel:
            # compute in parallel
            if n_proc is None:
                num_proc = cpu_count()
            else:
                num_proc = n_proc
            print('Parallel computation in {} processes.'.format(num_proc))
            num_part = mat.shape[1]
            part_per_proc = int(np.ceil(num_part/num_proc))
            process_pool = Pool(num_proc)
            matrix_list = list()
            # start computaton
            try:
                for p in np.arange(num_proc):
                    matrix_list.append(
                        mat[:, p*part_per_proc:(p+1)*part_per_proc])

                print('')
                st_0 = "Tracking in {} step(s)... ".format(self.n_out)
                for s in np.arange(self.n_out):
                    print_progress_bar(st_0, s, self.n_out-1)
                    # if auto_update_fields:
                    #    self.wakefield.check_if_update_fields(s*t_step)
                    partial_solver = partial(
                        runge_kutta_4, WF=self.wakefield, dt=dt_adjusted,
                        iterations=it_per_step, t0=s*t_step)
                    matrix_list = process_pool.map(partial_solver, matrix_list)
                    bunch_matrix = np.concatenate(matrix_list, axis=1)
                    x, px, y, py, xi, pz, q = bunch_matrix
                    new_prop_dist = bunch.prop_distance + (s+1)*t_step*ct.c
                    bunch_list.append(
                        ParticleBunch(bunch.q, x, y, xi, px, py, pz,
                                      prop_distance=new_prop_dist)
                    )
                    if opmd_diag is not False:
                        opmd_diag.write_diagnostics(
                            s*t_step, t_step, [bunch_list[-1]], self.wakefield)
            finally:
                process_pool.close()
                process_pool.join()
        else:
            # compute in single process
            print('Serial computation.')
            print('')
            st_0 = "Tracking in {} step(s)... ".format(self.n_out)
            for s in np.arange(self.n_out):
                print_progress_bar(st_0, s, self.n_out-1)
                # if auto_update_fields:
                #    self.wakefield.check_if_update_fields(s*t_step)
                bunch_matrix = runge_kutta_4(
                    mat, WF=self.wakefield, t0=s*t_step,  dt=dt_adjusted,
                    iterations=it_per_step)
                x, px, y, py, xi, pz, q = copy(bunch_matrix)
                new_prop_dist = bunch.prop_distance + (s+1)*t_step*ct.c
                bunch_list.append(
                    ParticleBunch(bunch.q, x, y, xi, px, py, pz,
                                  prop_distance=new_prop_dist)
                )
                if opmd_diag is not False:
                    opmd_diag.write_diagnostics(
                        s*t_step, t_step, [bunch_list[-1]], self.wakefield)
        # print computing time
        end = time.time()
        print("Done ({:1.3f} seconds).".format(end-start))
        print('-'*80)
        # update bunch data
        last_bunch = bunch_list[-1]
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.increase_prop_distance(self.length)
        return bunch_list

    def _track_analytically(self, bunch, parallel, n_proc):
        # Group velocity of driver
        v_w = self.wakefield.laser.get_group_velocity(self.n_p)*ct.c

        # Main bunch quantities [SI units]
        x_0 = bunch.x
        y_0 = bunch.y
        xi_0 = bunch.xi
        px_0 = bunch.px * ct.m_e * ct.c
        py_0 = bunch.py * ct.m_e * ct.c
        pz_0 = bunch.pz * ct.m_e * ct.c

        # Plasma length in time
        t_final = self.length/ct.c

        # Fields
        E_p = -ct.e/(ct.m_e*ct.c) * self.wakefield.Ez_p(
            x_0, y_0, xi_0, pz_0, py_0, pz_0, bunch.q, 0)
        E = -ct.e/(ct.m_e*ct.c) * self.wakefield.Wz(
            x_0, y_0, xi_0, pz_0, py_0, pz_0, bunch.q, 0)
        K = ct.e/ct.m_e * self.wakefield.Kx(
            x_0, y_0, xi_0, pz_0, py_0, pz_0, bunch.q, 0)

        if any(K <= 0):
            raise ValueError(
                'Detected bunch particles in defocusing phase. Defocusing '
                'fields currently not supported by analytical solver.')
        # Some initial values
        p_0 = np.sqrt(np.square(px_0) + np.square(py_0) + np.square(pz_0))
        g_0 = np.sqrt(np.square(p_0)/(ct.m_e*ct.c)**2 + 1)
        w_0 = np.sqrt(K/g_0)

        # Initial velocities
        v_x_0 = px_0/(ct.m_e*g_0)
        v_y_0 = py_0/(ct.m_e*g_0)

        # calculate oscillation amplitude
        A_x = np.sqrt(x_0**2+v_x_0**2/w_0**2)
        A_y = np.sqrt(y_0**2+v_y_0**2/w_0**2)

        # initial phase (x)
        sn_x = -v_x_0/(A_x*w_0)
        cs_x = x_0/A_x
        phi_x_0 = np.arctan2(sn_x, cs_x)

        # initial phase (y)
        sn_y = -v_y_0/(A_y*w_0)
        cs_y = y_0/A_y
        phi_y_0 = np.arctan2(sn_y, cs_y)

        # track bunch in steps
        # print("Tracking plasma stage in {} steps...   ".format(steps))
        start = time.time()
        p = Pool(cpu_count())
        t = t_final/self.n_out*(np.arange(self.n_out)+1)
        part = partial(self._get_beam_at_specified_time_step_analytically,
                       beam=bunch, g_0=g_0, w_0=w_0, xi_0=xi_0, A_x=A_x,
                       A_y=A_y, phi_x_0=phi_x_0, phi_y_0=phi_y_0, E=E, E_p=E_p,
                       v_w=v_w, K=K)
        bunch_list = p.map(part, t)
        end = time.time()
        print("Done ({} seconds)".format(end-start))

        # update bunch data
        last_bunch = bunch_list[-1]
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.increase_prop_distance(self.length)

        # update laser data
        # laser.increase_prop_distance(self.length)
        # laser.xi_c = laser.xi_c + (v_w-ct.c)*t_final

        # return steps
        return bunch_list

    def _get_optimized_dt(self, beam, WF):
        """ Get optimized time step """
        gamma = self._gamma(beam.px, beam.py, beam.pz)
        k_x = ge.plasma_focusing_gradient_blowout(self.n_p*1e-6)
        mean_gamma = np.average(gamma, weights=beam.q)
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * k_x/mean_gamma)
        T_x = 1/w_x
        dt = 0.1*T_x
        return dt

    def calculate_density(self, z):
        return self.n_p

    def _get_beam_at_specified_time_step_analytically(
            self, t, beam, g_0, w_0, xi_0, A_x, A_y, phi_x_0, phi_y_0, E, E_p,
            v_w, K):
        G = 1 + E/g_0*t
        if (G < 1/g_0).any():
            n_part = len(np.where(G < 1/g_0)[0])
            print('Warning: unphysical energy found in {}'.format(n_part)
                  + 'particles due to negative accelerating gradient.')
            # fix unphysical energies (model does not work well when E<=0)
            G = np.where(G < 1/g_0, 1/g_0, G)

        phi = 2*np.sqrt(K*g_0)/E*(G**(1/2) - 1)
        if (E == 0).any():
            # apply limit when E->0
            idx_0 = np.where(E == 0)[0]
            phi[idx_0] = np.sqrt(K[idx_0]/g_0[idx_0])*t[idx_0]
        A_0 = np.sqrt(A_x**2 + A_y**2)

        x = A_x*G**(-1/4)*np.cos(phi + phi_x_0)
        v_x = -w_0*A_x*G**(-3/4)*np.sin(phi + phi_x_0)
        p_x = G*g_0*v_x/ct.c

        y = A_y*G**(-1/4)*np.cos(phi + phi_y_0)
        v_y = -w_0*A_y*G**(-3/4)*np.sin(phi + phi_y_0)
        p_y = G*g_0*v_y/ct.c

        delta_xi = (ct.c/(2*E*g_0)*(G**(-1) - 1)
                    + A_0**2*K/(2*ct.c*E)*(G**(-1/2) - 1))
        xi = xi_0 + delta_xi

        delta_xi_max = -1/(2*E)*(ct.c/g_0 + A_0**2*K/ct.c)

        g = (g_0 + E*t + E_p*delta_xi_max*t + E_p/2*(ct.c-v_w)*t**2
             + ct.c*E_p/(2*E**2)*np.log(G)
             + E_p*A_0**2*K*g_0/(ct.c*E**2)*(G**(1/2) - 1))
        p_z = np.sqrt(g**2-p_x**2-p_y**2)

        beam_step = ParticleBunch(beam.q, x, y, xi, p_x, p_y, p_z,
                                  prop_distance=beam.prop_distance+t*ct.c)

        return beam_step


class PlasmaRamp():

    """Defines a plasma ramp."""

    def __init__(self, length, plasma_dens_top, plasma_dens_down=None,
                 position_down=None, ramp_type='upramp',
                 profile='inverse square', wakefield_model='blowout',
                 n_out=None, **model_params):
        """
        Initialize plasma ramp.

        Parameters:
        -----------
        length : float
            Length of the plasma stage in cm.

        plasma_dens_top : float
            Plasma density at the beginning (end) of the downramp (upramp) in
            units of m^{-3}.

        plasma_dens_down : float
            Plasma density at the position 'position_down' in units of
            m^{-3}.

        position_down : float
            Position where the plasma density will be equal to
            'plasma_dens_down' measured from the beginning (end) of the
            downramp (upramp).

        ramp_type : string
            Possible types are 'upramp' and 'downramp'.

        profile : string
            Longitudinal density profile of the ramp. Possible values are
            'linear', 'inverse square' and 'exponential'.

        wakefield_model : str
            Wakefield model to be used. Possible values are 'blowout',
            'cold_fluid_1d' and 'quasistatic_2d'.

        n_out : int
            Number of times along the stage in which the particle distribution
            should be returned (A list with all output bunches is returned
            after tracking).

        **model_params
            Keyword arguments which will be given to the wakefield model. Each
            model requires a different set of parameters which are listed
            below:

        Model 'blowout'
        ----------------------
        No additional parameters required.

        Model 'cold_fluid_1d'
        ---------------------
        laser : LaserPulse
            Laser driver of the plasma stage.

        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the beginning of the PlasmaStage. A negative
            value implies that the focal point is located before the
            PlasmaStage. Required only if laser_evolution=True.

        beam_wakefields : bool
            Whether to take into account beam-driven wakefields (False by
            default). This should be set to True for any beam-driven case or
            in order to take into account the beam-loading of the witness in
            a laser-driven case.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calculated. Required only if mode='cold_fluid_1d'.

        xi_min : float
            Minimum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        xi_max : float
            Maximum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        n_r : int
            Number of grid elements along r to calculate the wakefields.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic'.

        Model 'quasistatic_2d'
        ---------------------
        laser : LaserPulse
            Laser driver of the plasma stage.

        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the beginning of the PlasmaStage. A negative
            value implies that the focal point is located before the
            PlasmaStage.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calculated.

        xi_min : float
            Minimum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        xi_max : float
            Maximum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        n_r : int
            Number of grid elements along r to calculate the wakefields.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.

        n_part : int (optional)
            Number of plasma particles along the radial direction. By default
            n_part=1000.

        dz_fields : float (optional)
            Determines how often the plasma wakefields should be updated. If
            dz_fields=0 (default value), the wakefields are calculated at every
            step of the Runge-Kutta solver for the beam particle evolution
            (most expensive option). If specified, the wakefields are only
            updated in steps determined by dz_fields. For example, if
            dz_fields=10e-6, the plasma wakefields are only updated every time
            the simulation window advances by 10 micron. If dz_fields=None, the
            wakefields are only computed once (at the start of the plasma) and
            never updated throughout the simulation.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic'.

        """
        self.length = length
        self.plasma_dens_down = plasma_dens_down
        if position_down is None:
            self.position_down = length
        else:
            self.position_down = position_down
        self.plasma_dens_top = plasma_dens_top
        self.ramp_type = ramp_type
        self.profile = profile
        self.n_out = n_out
        self.wakefield = self._get_wakefield(wakefield_model, model_params)

    def track(self, bunch, parallel=False, n_proc=None, out_initial=False,
              opmd_diag=False, diag_dir=None):
        """
        Track bunch through plasma ramp.

        Parameters:
        -----------
        bunch : ParticleBunch
            Particle bunch to be tracked.

        parallel : float
            Determines whether or not to use parallel computation.

        n_proc : int
            Number of processes to run in parallel. If None, this will equal
            the number of physical cores.

        out_initial : bool
            Determines whether the initial bunch should be included in the
            output bunch list.

        opmd_diag : bool or OpenPMDDiagnostics
            Determines whether to write simulation diagnostics to disk (i.e.
            particle distributions and fields). The output is written to
            HDF5 files following the openPMD standard. The number of outputs
            the `n_out` value. It is also possible to provide an already
            existing OpenPMDDiagnostics instance instead of a boolean value.

        diag_dir : str
            Directory into which the openPMD output will be written. By default
            this is a 'diags' folder in the current directory. Only needed if
            `opmd_diag=True`.

        Returns:
        --------
        A list of size 'n_out' containing the bunch distribution at each step.

        """
        print('')
        print('Plasma ramp')
        print('-'*len('Plasma ramp'))
        # Main beam quantities
        mat = bunch.get_6D_matrix_with_charge()
        # Plasma length in time
        t_final = self.length/ct.c
        t_step = t_final/self.n_out
        dt = self._get_optimized_dt(bunch, self.wakefield)
        iterations = int(t_final/dt)
        # force at least 1 iteration per step
        it_per_step = max(int(iterations/self.n_out), 1)
        iterations = it_per_step*self.n_out
        dt_adjusted = t_final/iterations
        bunch_list = list()
        if out_initial:
            bunch_list.append(copy(bunch))
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        start = time.time()

        if parallel:
            if n_proc is None:
                num_proc = cpu_count()
            else:
                num_proc = n_proc
            print('Parallel computation in {} processes.'.format(num_proc))
            num_part = mat.shape[1]
            part_per_proc = int(np.ceil(num_part/num_proc))
            process_pool = Pool(num_proc)
            matrix_list = list()
            try:
                for p in np.arange(num_proc):
                    matrix_list.append(
                        mat[:, p*part_per_proc:(p+1)*part_per_proc])
                print('')
                st_0 = "Tracking in {} step(s)... ".format(self.n_out)
                for s in np.arange(self.n_out):
                    print_progress_bar(st_0, s, self.n_out-1)
                    partial_solver = partial(
                        runge_kutta_4, WF=self.wakefield, dt=dt_adjusted,
                        iterations=it_per_step, t0=s*t_step)
                    matrix_list = process_pool.map(partial_solver, matrix_list)
                    bunch_matrix = np.concatenate(matrix_list, axis=1)
                    x, px, y, py, xi, pz, q = bunch_matrix
                    new_prop_dist = bunch.prop_distance + (s+1)*t_step*ct.c
                    bunch_list.append(
                        ParticleBunch(bunch.q, x, y, xi, px, py, pz,
                                      prop_distance=new_prop_dist)
                    )
                    if opmd_diag is not False:
                        opmd_diag.write_diagnostics(
                            s*t_step, t_step, [bunch_list[-1]], self.wakefield)
            finally:
                process_pool.close()
                process_pool.join()
        else:
            print('Serial computation.')
            print('')
            st_0 = "Tracking in {} step(s)... ".format(self.n_out)
            for s in np.arange(self.n_out):
                print_progress_bar(st_0, s, self.n_out-1)
                bunch_matrix = runge_kutta_4(
                    mat, WF=self.wakefield, t0=s*t_step, dt=dt_adjusted,
                    iterations=it_per_step)
                x, px, y, py, xi, pz, q = copy(bunch_matrix)
                new_prop_dist = bunch.prop_distance + (s+1)*t_step*ct.c
                bunch_list.append(
                    ParticleBunch(bunch.q, x, y, xi, px, py, pz,
                                  prop_distance=new_prop_dist)
                )
                if opmd_diag is not False:
                    opmd_diag.write_diagnostics(
                        s*t_step, t_step, [bunch_list[-1]], self.wakefield)
        end = time.time()
        print("Done ({:1.3f} seconds).".format(end-start))
        print('-'*80)
        # update bunch data
        last_bunch = bunch_list[-1]
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.increase_prop_distance(self.length)

        return bunch_list

    def _get_wakefield(self, wakefield_model, model_params):
        """Create and return corresponding wakefield."""
        if wakefield_model == "blowout":
            WF = wf.PlasmaRampBlowoutField(self.calculate_density)
        elif wakefield_model == 'cold_fluid_1d':
            if self.ramp_type == 'upramp' and 'laser_z_foc' in model_params:
                model_params['laser_z_foc'] = (
                    self.length - model_params['laser_z_foc'])
            WF = wf.NonLinearColdFluidWakefield(
                self.calculate_density, **model_params)
        elif wakefield_model == 'quasistatic_2d':
            WF = wf.Quasistatic2DWakefield(
                self.calculate_density, **model_params)
        return WF

    def _get_optimized_dt(self, beam, wakefield):
        gamma = self._gamma(beam.px, beam.py, beam.pz)
        mean_gamma = np.average(gamma, weights=beam.q)
        # calculate maximum focusing on ramp.
        z = np.linspace(0, self.length, 100)
        n_p = self.calculate_density(z)
        w_p = np.sqrt(max(n_p)*ct.e**2/(ct.m_e*ct.epsilon_0))
        max_kx = (ct.m_e/(2*ct.e*ct.c))*w_p**2
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * max_kx/mean_gamma)
        period_x = 1/w_x
        dt = 0.1*period_x
        return dt

    def _gamma(self, px, py, pz):
        return np.sqrt(1 + px**2 + py**2 + pz**2)

    def calculate_density(self, z):
        if self.ramp_type == 'upramp':
            z = self.length - z
        if self.profile == 'linear':
            b = -((self.plasma_dens_top - self.plasma_dens_down)
                  / self.position_down)
            a = self.plasma_dens_top
            n_p = a + b*z
            # make negative densities 0
            n_p[n_p < 0] = 0
        elif self.profile == 'inverse square':
            a = np.sqrt(self.plasma_dens_top/self.plasma_dens_down) - 1
            b = self.position_down/a
            n_p = self.plasma_dens_top/np.square(1+z/b)
        elif self.profile == 'exponential':
            a = self.plasma_dens_top
            if self.plasma_dens_down is None:
                # use length as total length for 99 phase advance
                b = 4*np.log(10)/self.length
            else:
                b = (np.log(self.plasma_dens_top / self.plasma_dens_down)
                     / self.position_down)
            n_p = a*np.exp(-b*z)
        elif self.profile == 'gaussian':
            s_z = self.position_down / np.sqrt(2*np.log(self.plasma_dens_top /
                                                        self.plasma_dens_down))
            n_p = self.plasma_dens_top * np.exp(-z**2/(2*s_z**2))
        return n_p


class PlasmaLens():

    """Defines an active plasma lens (APL)"""

    def __init__(self, length, foc_strength, relativistic=True,
                 wakefields=False, wakefield_model='quasistatic_2d', n_p=None,
                 n_out=None, **model_params):
        """
        Initialize plasma lens.

        Parameters:
        -----------
        length : float
            Length of the plasma lens in cm.

        foc_strength : float
            Focusing strength of the plasma lens in T/m.

        relativistic : bool
            Determines whether to use the relativistic approximation of the
            fields experienced by the bunch.

        wakefields : bool
            If True, the beam-induced wakefields in the plasma lens will be
            computed using the model specified in 'wakefield_model' and
            taken into account for the beam evolution.

        wakefield_model : str
            Name of the model which should be used for computing the
            beam-induced wakefields. Possible values are 'cold_fluid_1d' and
            'quasistatic_2d'.

        n_p : float
            Plasma density in the APL in units of m^{-3}. Required only if
            wakefields=True.

        n_out : int
            Number of times along the lens in which the particle distribution
            should be returned (A list with all output bunches is returned
            after tracking).

        Model 'cold_fluid_1d'
        ---------------------
        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the beginning of the PlasmaStage. A negative
            value implies that the focal point is located before the
            PlasmaStage. Required only if laser_evolution=True.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calculated. Required only if mode='cold_fluid_1d'.

        xi_min : float
            Minimum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        xi_max : float
            Maximum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        n_r : int
            Number of grid elements along r to calculate the wakefields.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic'.

        Model 'quasistatic_2d'
        ---------------------
        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the beginning of the PlasmaStage. A negative
            value implies that the focal point is located before the
            PlasmaStage.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calculated.

        xi_min : float
            Minimum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        xi_max : float
            Maximum longitudinal (speed of light frame) position up to which
            plasma wakefield will be calculated.

        n_r : int
            Number of grid elements along r to calculate the wakefields.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.

        n_part : int (optional)
            Number of plasma particles along the radial direction. By default
            n_part=1000.

        dz_fields : float (optional)
            Determines how often the plasma wakefields should be updated. If
            dz_fields=0 (default value), the wakefields are calculated at every
            step of the Runge-Kutta solver for the beam particle evolution
            (most expensive option). If specified, the wakefields are only
            updated in steps determined by dz_fields. For example, if
            dz_fields=10e-6, the plasma wakefields are only updated every time
            the simulation window advances by 10 micron. If dz_fields=None, the
            wakefields are only computed once (at the start of the plasma) and
            never updated throughout the simulation.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic'.

        """
        self.length = length
        self.foc_strength = foc_strength
        self.n_out = n_out
        self.n_p = n_p
        self.field = self._get_wakefield(
            relativistic, wakefields, wakefield_model, model_params)

    def _get_wakefield(self, relativistic, wakefields, wakefield_model,
                       model_params):
        if relativistic:
            lens_field = wf.PlasmaLensFieldRelativistic(self.foc_strength)
        else:
            lens_field = wf.PlasmaLensField(self.foc_strength)
        if wakefields:
            if wakefield_model == 'cold_fluid_1d':
                plasma_wf = wf.NonLinearColdFluidWakefield(
                    self.calculate_density, beam_wakefields=True,
                    **model_params)
            elif wakefield_model == 'quasistatic_2d':
                plasma_wf = wf.Quasistatic2DWakefield(
                    self.calculate_density, **model_params)
            else:
                raise ValueError
            WF = wf.CombinedWakefield([lens_field, plasma_wf])
        else:
            WF = lens_field
        return WF

    def calculate_density(self, z):
        return self.n_p

    def track(self, bunch, parallel=False, n_proc=None, out_initial=False,
              opmd_diag=False, diag_dir=None):
        """
        Track bunch through plasma lens.

        Parameters:
        -----------
        bunch : ParticleBunch
            Particle bunch to be tracked.

        parallel : float
            Determines whether or not to use parallel computation.

        n_proc : int
            Number of processes to run in parallel. If None, this will equal
            the number of physical cores.

        out_initial : bool
            Determines whether the initial bunch should be included in the
            output bunch list.

        opmd_diag : bool or OpenPMDDiagnostics
            Determines whether to write simulation diagnostics to disk (i.e.
            particle distributions and fields). The output is written to
            HDF5 files following the openPMD standard. The number of outputs
            the `n_out` value. It is also possible to provide an already
            existing OpenPMDDiagnostics instance instead of a boolean value.

        diag_dir : str
            Directory into which the openPMD output will be written. By default
            this is a 'diags' folder in the current directory. Only needed if
            `opmd_diag=True`.

        Returns:
        --------
        A list of size 'n_out' containing the bunch distribution at each step.

        """
        print('')
        print('Plasma lens')
        print('-'*len('Plasma lens'))
        # Main beam quantities
        mat = bunch.get_6D_matrix_with_charge()

        # Plasma length in time
        t_final = self.length/ct.c
        t_step = t_final/self.n_out
        dt = self._get_optimized_dt(bunch, self.field)
        iterations = int(t_final/dt)
        # force at least 1 iteration per step
        it_per_step = max(int(iterations/self.n_out), 1)
        iterations = it_per_step*self.n_out
        dt_adjusted = t_final/iterations
        bunch_list = list()
        if out_initial:
            bunch_list.append(copy(bunch))
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        start = time.time()
        if parallel:
            if n_proc is None:
                num_proc = cpu_count()
            else:
                num_proc = n_proc
            print('Parallel computation in {} processes.'.format(num_proc))
            num_part = mat.shape[1]
            part_per_proc = int(np.ceil(num_part/num_proc))
            process_pool = Pool(num_proc)
            matrix_list = list()
            try:
                for p in np.arange(num_proc):
                    matrix_list.append(
                        mat[:, p*part_per_proc:(p+1)*part_per_proc])
                print('')
                st_0 = "Tracking in {} step(s)... ".format(self.n_out)
                for s in np.arange(self.n_out):
                    print_progress_bar(st_0, s, self.n_out-1)
                    partial_solver = partial(
                        runge_kutta_4, WF=self.field, dt=dt_adjusted,
                        iterations=it_per_step, t0=s*t_step)
                    matrix_list = process_pool.map(partial_solver, matrix_list)
                    bunch_matrix = np.concatenate(matrix_list, axis=1)
                    x, px, y, py, xi, pz, q = bunch_matrix
                    new_prop_dist = bunch.prop_distance + (s+1)*t_step*ct.c
                    bunch_list.append(
                        ParticleBunch(bunch.q, x, y, xi, px, py, pz,
                                      prop_distance=new_prop_dist))
                    if opmd_diag is not False:
                        opmd_diag.write_diagnostics(
                            s*t_step, t_step, [bunch_list[-1]], self.field)
            finally:
                process_pool.close()
                process_pool.join()
        else:
            # compute in single process
            print('Serial computation.')
            print('')
            st_0 = "Tracking in {} step(s)... ".format(self.n_out)
            for s in np.arange(self.n_out):
                print_progress_bar(st_0, s, self.n_out-1)
                bunch_matrix = runge_kutta_4(
                    mat, WF=self.field, t0=s*t_step, dt=dt_adjusted,
                    iterations=it_per_step)
                x, px, y, py, xi, pz, q = copy(bunch_matrix)
                new_prop_dist = bunch.prop_distance + (s+1)*t_step*ct.c
                bunch_list.append(
                    ParticleBunch(bunch.q, x, y, xi, px, py, pz,
                                  prop_distance=new_prop_dist)
                )
                if opmd_diag is not False:
                    opmd_diag.write_diagnostics(
                        s*t_step, t_step, [bunch_list[-1]], self.field)
        end = time.time()
        print("Done ({:1.3f} seconds).".format(end-start))
        print('-'*80)
        # update bunch data
        last_bunch = bunch_list[-1]
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.increase_prop_distance(self.length)
        return bunch_list

    def _get_optimized_dt(self, beam, WF):
        gamma = self._gamma(beam.px, beam.py, beam.pz)
        mean_gamma = np.average(gamma, weights=beam.q)
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * self.foc_strength/mean_gamma)
        T_x = 1/w_x
        dt = 0.1*T_x
        return dt

    def _gamma(self, px, py, pz):
        return np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))


class TMElement():
    # TODO: fix backtracking issues.
    """Defines an element to be tracked using transfer maps."""

    def __init__(self, length=0, theta=0, k1=0, k2=0, gamma_ref=None,
                 csr_on=False, n_out=None, order=2):
        self.length = length
        self.theta = theta
        self.k1 = k1
        self.k2 = k2
        self.gamma_ref = gamma_ref
        self.csr_on = csr_on
        self.n_out = n_out
        self.order = order
        self.csr_calculator = get_csr_calculator()
        self.element_name = ''

    def track(self, bunch, backtrack=False, out_initial=False, opmd_diag=False,
              diag_dir=None):
        # Convert bunch to ocelot units and reference frame
        bunch_mat, g_avg = self._get_beam_matrix_for_tracking(bunch)
        if self.gamma_ref is None:
            self.gamma_ref = g_avg

        # Add element to CSR calculator
        if self.csr_on:
            self.csr_calculator.add_lattice_element(self)
        else:
            self.csr_calculator.clear()
        # Determine track and output steps
        l_step, track_steps, output_steps = self._determine_steps()

        # Create diagnostics if needed
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)

        # Print output header
        print('')
        print(self.element_name.capitalize())
        print('-'*len(self.element_name))
        self._print_element_properties()
        csr_string = 'on' if self.csr_on else 'off'
        print('CSR {}.'.format(csr_string))
        print('')
        n_steps = len(track_steps)
        st_0 = 'Tracking in {} step(s)... '.format(n_steps)

        # Start tracking
        start_time = time.time()
        output_bunch_list = list()
        if out_initial:
            output_bunch_list.append(copy(bunch))
        for i in track_steps:
            print_progress_bar(st_0, i+1, n_steps)
            l_curr = (i+1) * l_step * (1-2*backtrack)
            # Track with transfer matrix
            bunch_mat = track_with_transfer_map(
                bunch_mat, l_step, self.length, -self.theta, self.k1, self.k2,
                self.gamma_ref, order=self.order)
            # Apply CSR
            if self.csr_on:
                self.csr_calculator.apply_csr(bunch_mat, bunch.q,
                                              self.gamma_ref, self, l_curr)
            # Add bunch to output list
            if i in output_steps:
                new_bunch_mat = convert_from_ocelot_matrix(
                    bunch_mat, self.gamma_ref)
                new_bunch = self._create_new_bunch(
                    bunch, new_bunch_mat, l_curr)
                output_bunch_list.append(new_bunch)
                if opmd_diag is not False:
                    opmd_diag.write_diagnostics(
                        l_curr/ct.c, l_step/ct.c, [output_bunch_list[-1]])

        # Update bunch data
        self._update_input_bunch(bunch, bunch_mat, output_bunch_list)

        # Finalize
        tracking_time = time.time() - start_time
        print('Done ({} s).'.format(tracking_time))
        print('-'*80)
        return output_bunch_list

    def _get_beam_matrix_for_tracking(self, bunch):
        bunch_mat = bunch.get_6D_matrix()
        # obtain with respect to reference displacement
        bunch_mat[0] -= bunch.x_ref
        # rotate by the reference angle so that it enters normal to the element
        if bunch.theta_ref != 0:
            rot = rotation_matrix_xz(-bunch.theta_ref)
            bunch_mat = np.dot(rot, bunch_mat)
        return convert_to_ocelot_matrix(bunch_mat, bunch.q, self.gamma_ref)

    def _determine_steps(self):
        if self.n_out is not None:
            l_step = self.length/self.n_out
            n_track = self.n_out
            frac_out = 1
        else:
            l_step = self.length
            n_track = 1
            frac_out = 0
        if self.csr_on:
            csr_track_step = self.csr_calculator.get_csr_step(self)
            n_csr = int(self.length / csr_track_step)
            if self.n_out is not None:
                frac_out = int(n_csr / self.n_out)
            l_step = csr_track_step
            n_track = n_csr
        track_steps = np.arange(0, n_track)
        if frac_out != 0:
            output_steps = track_steps[::-frac_out]
        else:
            output_steps = []
        return l_step, track_steps, output_steps

    def _update_input_bunch(self, bunch, bunch_mat, output_bunch_list):
        if len(output_bunch_list) == 0:
            new_bunch_mat = convert_from_ocelot_matrix(
                bunch_mat, self.gamma_ref)
            last_bunch = self._create_new_bunch(bunch, new_bunch_mat,
                                                self.length)
        else:
            last_bunch = output_bunch_list[-1]
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.prop_distance = last_bunch.prop_distance
        bunch.theta_ref = last_bunch.theta_ref
        bunch.x_ref = last_bunch.x_ref

    def _create_new_bunch(self, old_bunch, new_bunch_mat, prop_dist):
        q = old_bunch.q
        if self.theta != 0:
            # angle rotated for prop_dist
            theta_step = self.theta*prop_dist/self.length
            # magnet bending radius
            rho = abs(self.length/self.theta)
            # new reference angle and transverse displacement
            new_theta_ref = old_bunch.theta_ref + theta_step
            sign = -theta_step/abs(theta_step)
            new_x_ref = (
                old_bunch.x_ref +
                sign*rho*(np.cos(new_theta_ref)-np.cos(old_bunch.theta_ref)))
        else:
            # new reference angle and transverse displacement
            new_theta_ref = old_bunch.theta_ref
            new_x_ref = (old_bunch.x_ref + prop_dist *
                         np.sin(old_bunch.theta_ref))
        # new prop. distance
        new_prop_dist = old_bunch.prop_distance + prop_dist
        # rotate distribution if reference angle != 0
        if new_theta_ref != 0:
            rot = rotation_matrix_xz(new_theta_ref)
            new_bunch_mat = np.dot(rot, new_bunch_mat)
        new_bunch_mat[0] += new_x_ref
        # create new bunch
        new_bunch = ParticleBunch(q, bunch_matrix=new_bunch_mat,
                                  prop_distance=new_prop_dist)
        new_bunch.theta_ref = new_theta_ref
        new_bunch.x_ref = new_x_ref
        return new_bunch

    def _print_element_properties(self):
        "To be implemented by each element. Prints the element properties"
        raise NotImplementedError


class Drift(TMElement):
    def __init__(self, length=0, gamma_ref=None, csr_on=False, n_out=None,
                 order=2):
        super().__init__(length, 0, 0, 0, gamma_ref, csr_on, n_out, order)
        self.element_name = 'drift'

    def _print_element_properties(self):
        print('Length = {:1.4f} m'.format(self.length))


class Dipole(TMElement):
    def __init__(self, length=0, theta=0, gamma_ref=None, csr_on=False,
                 n_out=None, order=2):
        super().__init__(length, theta, 0, 0, gamma_ref, csr_on, n_out, order)
        self.element_name = 'dipole'

    def _print_element_properties(self):
        ang_deg = self.theta * 180/ct.pi
        b_field = (ct.m_e*ct.c/ct.e) * self.theta*self.gamma_ref/self.length
        print('Bending angle = {:1.4f} rad ({:1.4f} deg)'.format(
            self.theta, ang_deg))
        print('Dipole field = {:1.4f} T'.format(b_field))


class Quadrupole(TMElement):
    def __init__(self, length=0, k1=0, gamma_ref=None, csr_on=False,
                 n_out=None, order=2):
        super().__init__(length, 0, k1, 0, gamma_ref, csr_on, n_out, order)
        self.element_name = 'quadrupole'

    def _print_element_properties(self):
        g = self.k1 * self.gamma_ref*(ct.m_e*ct.c/ct.e)
        print('Quadrupole gradient = {:1.4f} T/m'.format(g))


class Sextupole(TMElement):
    def __init__(self, length=0, k2=0, gamma_ref=None, csr_on=False,
                 n_out=None, order=2):
        super().__init__(length, 0, 0, k2, gamma_ref, csr_on, n_out, order)
        self.element_name = 'sextupole'

    def _print_element_properties(self):
        g = self.k2 * self.gamma_ref*(ct.m_e*ct.c/ct.e)
        print('Sextupole gradient = {:1.4f} T/m^2'.format(g))
