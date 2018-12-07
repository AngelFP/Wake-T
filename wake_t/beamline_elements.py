""" This module contains the classes for all beamline elements. """

import time
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.particle_tracking import runge_kutta_4
from wake_t.wakefields import *
from wake_t.driver_witness import ParticleBunch


class PlasmaStage(object):

    """ Defines a plasma stage. """

    def __init__(self, n_p, length):
        """
        Initialize plasma stage.

        Parameters:
        -----------
        n_p : float
            Plasma density in units of cm^{-3}.

        length : float
            Length of the plasma stage in cm.

        """
        self.n_p = n_p
        self.length = length

    def get_matched_beta(self, mode, ene, xi=None, foc_strength=None,
                         laser=None):
        """
        Calculate the matched beta function at the plasma for a beam energy.

        Parameters:
        -----------
        mode : string
            Mode used to calculate fields. Possible values are 'Blowout',
            'CustomBlowout', 'FromGivenFields'(deprecated) and 'Linear'.

        ene : float
            Mean beam energy in non-dimensional units (beta*gamma).

        xi : float
            Longitudinal position of the bunch center in the comoving frame.
            Only used if mode='Linear'.

        foc_strength : float
            Focusing gradient in the plasma in units of T/m. Onnly used for
            modes 'CustomBlowout' and 'FromGivenFields'

        laser : LaserPulse
            Laser used in the plasma stage. Only used if mode='Linear'.

        """
        if mode == "Blowout":
            return ge.matched_plasma_beta_function(ene, n_p=self.n_p,
                                                   regime='Blowout')

        elif mode in ["CustomBlowout", "FromGivenFields"]:
            return ge.matched_plasma_beta_function(ene, k_x=foc_strength)

        elif mode == "Linear":
            dist_l_b = -(laser.get_lon_center()-xi)
            return ge.matched_plasma_beta_function(
                ene, n_p=self.n_p, regime='Blowout', dist_from_driver=dist_l_b,
                a_0=laser.a_0, w_0=laser.w_0)

    def _gamma(self, px, py, pz):
        return np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    
    def track_beam_numerically_RK_parallel(
            self, laser, beam, mode, steps, simulation_code=None,
            simulation_path=None, time_step=None, auto_update_fields=False,
            laser_pos_in_pic_code=None, lon_field=None, lon_field_slope=None,
            foc_strength=None, filter_fields=False, filter_sigma=20,
            n_proc=None):
        """
        Track the beam through the plasma using a 4th order Runge-Kutta method.
        
        Parameters:
        -----------
        laser : LaserPulse
            Laser used in the plasma stage.

        beam : ParticleBunch
            Particle bunch to track.

        mode : string
            Mode used to determine the wakefields. Possible values are 
            'Blowout', 'CustomBlowout', 'FromPICCode'.

        steps : int
            Number of steps in which output should be given.

        simulation_code : string
            Name of the simulation code from which fields should be read. Only
            used if mode='FromPICCode'.

        simulation_path : string
            Path to the simulation folder where the fields to read are located.
            Only used if mode='FromPICCode'.

        time_step : int
            Time step at which the fields should be read.

        auto_update_fields : bool
            If True, new fields will be read from the simulation folder
            automatically as the particles travel through the plasma. The first
            time step will be time_step, and new ones will be loaded as the
            time of flight of the bunch reaches the next time step available in
            the simulation folder.

        laser_pos_in_pic_code : float (deprecated)
            Position of the laser pulse center in the comoving frame in the pic
            code simulation.

        lon_field : float
            Value of the longitudinal electric field at the bunch center at the
            beginning of the tracking in units of V/m. Only used if
            mode='CustomBlowout'.

        lon_field_slope : float
            Value of the longitudinal electric field slope along z at the bunch
            center at the beginning of the tracking in units of V/m^2. Only
            used if mode='CustomBlowout'.

        foc_strength : float
            Value of the focusing gradient along the bunch in units of T/m. 
            Only used if mode='CustomBlowout'.

        filter_fields : bool
            If true, a Gaussian filter is applied to smooth the wakefields.
            This can be useful to remove noise. Only used if
            mode='FromPICCode'.

        filter_sigma : float
            Sigma to be used by the Gaussian filter. 

        n_proc : int
            Number of processes to run in parallel. If None, this will equal
            the number of physical cores.

        Returns:
        --------
        A list of size 'steps' containing the beam distribution at each step.

        """
        if mode == "Blowout":
            WF = BlowoutWakefield(self.n_p, laser)
        if mode == "CustomBlowout":
            WF = CustomBlowoutWakefield(
                self.n_p, laser, np.average(beam.xi, weights=beam.q), 
                lon_field, lon_field_slope, foc_strength)
        elif mode == "FromPICCode":
            WF = WakefieldFromPICSimulation(
                simulation_code, simulation_path, laser, time_step, self.n_p,
                filter_fields, filter_sigma)
        # Get 6D matrix
        mat = beam.get_6D_matrix()
        # Plasma length in time
        t_final = self.length/ct.c
        t_step = t_final/steps
        dt = self._get_optimized_dt(beam, WF)
        iterations = int(t_final/dt)
        # force at least 1 iteration per step
        it_per_step = max(int(iterations/steps), 1)
        iterations = it_per_step*steps
        dt_adjusted = t_final/iterations
        # initialize list to store the distribution at each step
        beam_list = list()
        # get start time
        start = time.time()
        if n_proc is None:
            num_proc = cpu_count()
        else:
            num_proc = n_proc
        num_part = mat.shape[1]
        part_per_proc = int(np.ceil(num_part/num_proc))
        process_pool = Pool(num_proc)
        t_s = 0
        matrix_list = list()
        # start computaton
        try:
            for p in np.arange(num_proc):
                matrix_list.append(mat[:,p*part_per_proc:(p+1)*part_per_proc])

            for s in np.arange(steps):
                print(s)
                if auto_update_fields:
                    WF.check_if_update_fields(s*t_step)
                partial_solver = partial(
                    runge_kutta_4, WF=WF, dt=dt_adjusted,
                    iterations=it_per_step, t0=s*t_step)
                matrix_list = process_pool.map(partial_solver, matrix_list)
                beam_matrix = np.concatenate(matrix_list, axis=1)
                x, px, y, py, xi, pz = beam_matrix
                new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                beam_list.append(
                    ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                  prop_distance=new_prop_dist)
                    )
        finally:
            process_pool.close()
            process_pool.join()
        # print computing time
        end = time.time()
        print("Done ({} seconds)".format(end-start))
        # update beam data
        last_beam = beam_list[-1]
        beam.set_phase_space(last_beam.x, last_beam.y, last_beam.xi,
                             last_beam.px, last_beam.py, last_beam.pz)
        beam.increase_prop_distance(self.length)
        return beam_list
    
    def _get_optimized_dt(self, beam, WF):
        """ Get optimized time step """ 
        gamma = self._gamma(beam.px, beam.py, beam.pz)
        Kx = WF.Kx(
            beam.x, beam.y, beam.xi, beam.px, beam.py, beam.pz, gamma, 0)
        mean_Kx = np.average(Kx, weights=beam.q)
        mean_gamma = np.average(gamma, weights=beam.q)
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * mean_Kx/mean_gamma)
        T_x = 1/w_x
        dt = 0.1*T_x
        return dt


class PlasmaUpramp(object):

    """Defines a plasma upramp."""

    def __init__(self, length, initial_dens, final_dens, initial_dens_pos=0):
        """
        Initialize plasma ramp.

        Parameters:
        -----------
        length : float
            Length of the plasma stage in cm.
            
        initial_dens : float
            Plasma density at the position 'initial_dens_pos' in units of
            cm^{-3}.

        final_dens : float
            Plasma density at the end of the ramp in units of cm^{-3}.

        initial_dens_pos : float
            Position where the plasma density will be equal to 'initial_dens'.
            If 0, this corresponds to the start of the ramp.

        """
        self.length = length
        self.initial_dens = initial_dens
        self.initial_dens_pos = initial_dens_pos
        self.final_dens = final_dens
        
    def track_beam_numerically_RK_parallel(self, beam, steps, non_rel=False, 
                                           n_proc=None):
        """
        Track the beam through the plasma using a 4th order Runge-Kutta method.
        
        Parameters:
        -----------
        beam : ParticleBunch
            Particle bunch to track.

        steps : int
            Number of steps in which output should be given.

        non_rel : bool
            If True, the relativistic assumplion is not used for the equations
            of motion.

        n_proc : int
            Number of processes to run in parallel. If None, this will equal
            the number of physical cores.

        Returns:
        --------
        A list of size 'steps' containing the beam distribution at each step.

        """
        if non_rel:
            raise NotImplementedError()
        else:
            field = PlasmaUprampBlowoutField(self.length, self.initial_dens,
                                             self.final_dens,
                                             self.initial_dens_pos)
        # Main beam quantities
        mat = beam.get_6D_matrix()
        # Plasma length in time
        t_final = self.length/ct.c
        t_step = t_final/steps
        dt = self._get_optimized_dt(beam, field)
        iterations = int(t_final/dt)
        # force at least 1 iteration per step
        it_per_step = max(int(iterations/steps), 1)
        iterations = it_per_step*steps
        dt_adjusted = t_final/iterations
        beam_list = list()

        start = time.time()
        if n_proc is None:
            num_proc = cpu_count()
        else:
            num_proc = n_proc
        num_part = mat.shape[1]
        part_per_proc = int(np.ceil(num_part/num_proc))
        process_pool = Pool(num_proc)
        t_s = 0
        matrix_list = list()
        try:
            for p in np.arange(num_proc):
                matrix_list.append(mat[:,p*part_per_proc:(p+1)*part_per_proc])

            for s in np.arange(steps):
                print(s)
                partial_solver = partial(
                    runge_kutta_4, WF=field, dt=dt_adjusted,
                    iterations=it_per_step, t0=s*t_step)
                matrix_list = process_pool.map(partial_solver, matrix_list)
                beam_matrix = np.concatenate(matrix_list, axis=1)
                x, px, y, py, xi, pz = beam_matrix
                new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                beam_list.append(
                    ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                  prop_distance=new_prop_dist)
                    )
        finally:
            process_pool.close()
            process_pool.join()

        end = time.time()
        print("Done ({} seconds)".format(end-start))

        # update beam data
        last_beam = beam_list[-1]
        beam.set_phase_space(last_beam.x, last_beam.y, last_beam.xi,
                             last_beam.px, last_beam.py, last_beam.pz)
        beam.increase_prop_distance(self.length)

        return beam_list
    
    def _get_optimized_dt(self, beam, WF):
        gamma = self._gamma(beam.px, beam.py, beam.pz)
        mean_gamma = np.average(gamma, weights=beam.q)
        # calculate focusing at the end of ramp, where n_p is higher.
        Kx = WF.Kx(beam.x, beam.y, beam.xi, beam.px, beam.py, beam.pz, gamma,
                   self.length/ct.c)
        mean_Kx = np.average(Kx, weights=beam.q)
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * mean_Kx/mean_gamma)
        T_x = 1/w_x
        dt = 0.1*T_x
        return dt

    def _gamma(self, px, py, pz):
        return np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    
