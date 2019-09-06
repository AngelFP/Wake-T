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
from wake_t.wakefields import *
from wake_t.driver_witness import ParticleBunch
from wake_t.utilities.bunch_manipulation import (convert_to_ocelot_matrix,
                                                 convert_from_ocelot_matrix,
                                                 rotation_matrix_xz)


class PlasmaStage():

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
        return np.sqrt(1 + px**2 + py**2 + pz**2)
    
    def track_beam_numerically(
            self, laser, beam, mode, steps, simulation_code=None,
            simulation_path=None, time_step=None, auto_update_fields=False,
            reverse_tracking=False, laser_pos_in_pic_code=None, lon_field=None,
            lon_field_slope=None, foc_strength=None, field_offset=0,
            filter_fields=False, filter_sigma=20, laser_evolution=False,
            laser_z_foc=0, r_max=None, xi_min=None, xi_max=None, n_r=100, 
            n_xi=100, parallel=False, n_proc=None):
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
            'Blowout', 'CustomBlowout', 'FromPICCode' or 'cold_fluid_1d'.

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

        reverse_tracking : bool
            Whether to reverse-track the particles through the stage. Currenly
            only available for mode= 'FromPICCode'. 

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

        field_offset : float
            If 0, the values of 'lon_field', 'lon_field_slope' and
            'foc_strength' will be applied at the bunch center. A value >0 (<0)
            gives them a positive (negative) offset towards the front (back) of
            the bunch. Only used if mode='CustomBlowout'.

        filter_fields : bool
            If true, a Gaussian filter is applied to smooth the wakefields.
            This can be useful to remove noise. Only used if
            mode='FromPICCode'.

        filter_sigma : float
            Sigma to be used by the Gaussian filter. 
        
        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation. Used only if mode='cold_fluid_1d'.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the beginning of the PlasmaStage. A negative
            value implies that the focal point is located before the
            PlasmaStage. Required only if laser_evolution=True and
            mode='cold_fluid_1d'.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calulated. Required only if mode='cold_fluid_1d'.

        xi_min : float
            Minimum longiudinal (speed of light frame) position up to which
            plasma wakefield will be calulated. Required only if
            mode='cold_fluid_1d'.

        xi_max : float
            Maximum longiudinal (speed of light frame) position up to which
            plasma wakefield will be calulated. Required only if
            mode='cold_fluid_1d'.

        n_r : int
            Number of grid elements along r to calculate the wakefields.
            Required only if mode='cold_fluid_1d'.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.
            Required only if mode='cold_fluid_1d'.

        parallel : float
            Determines whether or not to use parallel computation.

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
                lon_field, lon_field_slope, foc_strength, field_offset)
        elif mode == "FromPICCode":
            if vpic_installed:
                WF = WakefieldFromPICSimulation(
                    simulation_code, simulation_path, laser, time_step,
                    self.n_p, filter_fields, filter_sigma, reverse_tracking)
            else:
                return []
        elif mode == 'cold_fluid_1d':
            WF = NonLinearColdFluidWakefield(self.calculate_density, laser,
                                             laser_evolution, laser_z_foc,
                                             r_max, xi_min, xi_max, n_r, n_xi)
        # Get 6D matrix
        mat = beam.get_6D_matrix_with_charge()
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
        if parallel:
            # compute in parallel
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
                    matrix_list.append(
                        mat[:,p*part_per_proc:(p+1)*part_per_proc])

                for s in np.arange(steps):
                    print(s)
                    if auto_update_fields:
                        WF.check_if_update_fields(s*t_step)
                    partial_solver = partial(
                        runge_kutta_4, WF=WF, dt=dt_adjusted,
                        iterations=it_per_step, t0=s*t_step)
                    matrix_list = process_pool.map(partial_solver, matrix_list)
                    beam_matrix = np.concatenate(matrix_list, axis=1)
                    x, px, y, py, xi, pz, q = beam_matrix
                    new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                    beam_list.append(
                        ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                      prop_distance=new_prop_dist)
                        )
            finally:
                process_pool.close()
                process_pool.join()
        else:
            # compute in single process
            for s in np.arange(steps):
                print(s)
                beam_matrix = runge_kutta_4(mat, WF=WF, t0=s*t_step,
                                            dt=dt_adjusted,
                                            iterations=it_per_step)
                x, px, y, py, xi, pz, q = copy(beam_matrix)
                new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                beam_list.append(
                    ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                    prop_distance=new_prop_dist)
                    )
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
        k_x = ge.plasma_focusing_gradient_blowout(self.n_p)
        mean_gamma = np.average(gamma, weights=beam.q)
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * k_x/mean_gamma)
        T_x = 1/w_x
        dt = 0.1*T_x
        return dt

    def calculate_density(self, z):
        return self.n_p*1e6

    def track_beam_analytically(
        self, laser, beam, mode, steps, simulation_code=None,
        simulation_path=None, time_step=None, laser_pos_in_pic_code=None,
        lon_field=None, lon_field_slope=None, foc_strength=None,
        field_offset=0):
        """
        Track the beam through the plasma using a the analytical model from
        https://arxiv.org/abs/1804.10966.
        
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

        field_offset : float
            If 0, the values of 'lon_field', 'lon_field_slope' and
            'foc_strength' will be applied at the bunch center. A value >0 (<0)
            gives them a positive (negative) offset towards the front (back) of
            the bunch. Only used if mode='CustomBlowout'.

        Returns:
        --------
        A list of size 'steps' containing the beam distribution at each step.

        """
        # Main laser quantities
        l_c = laser.xi_c
        v_w = laser.get_group_velocity(self.n_p)*ct.c
        w_0_l = laser.w_0

        # Main beam quantities [SI units]
        x_0 = beam.x
        y_0 = beam.y
        xi_0 = beam.xi
        px_0 = beam.px * ct.m_e * ct.c
        py_0 = beam.py * ct.m_e * ct.c
        pz_0 = beam.pz * ct.m_e * ct.c

        # Distance between laser and beam particle
        dist_l_b = -(l_c-xi_0)

        # Plasma length in time
        t_final = self.length/ct.c

        # Fields
        if mode == "Blowout":
            """Bubble center is assumed at lambda/2"""
            w_p = np.sqrt(self.n_p*1e6*ct.e**2/(ct.m_e*ct.epsilon_0))
            l_p = 2*np.pi*ct.c/w_p
            E_p = -w_p**2/(2*ct.c) * np.ones_like(xi_0)
            K = w_p**2/2 * np.ones_like(xi_0)
            E = E_p*(l_p/2+dist_l_b)

        elif mode == "CustomBlowout":
            E_p = -lon_field_slope*ct.e/(ct.m_e*ct.c) * np.ones_like(xi_0)
            K = foc_strength*ct.c*ct.e/ct.m_e * np.ones_like(xi_0)
            E = -lon_field*ct.e/(ct.m_e*ct.c) + E_p*(xi_0 - np.mean(xi_0))

        elif mode == "Linear":
            a0 = laser.a_0
            n_p_SI = self.n_p*1e6
            w_p = np.sqrt(n_p_SI*ct.e**2/(ct.m_e*ct.epsilon_0))
            k_p = w_p/ct.c
            E0 = ct.m_e*ct.c*w_p/ct.e
            K = (8*np.pi/np.e)**(1/4)*a0/(k_p*w_0_l)

            A = E0*np.sqrt(np.pi/(2*np.e))*a0**2
            E_z = A*np.cos(k_p*(dist_l_b))
            E_z_p = -A*k_p*np.sin(k_p*(dist_l_b))
            g_x = -E0*K**2*k_p*np.sin(k_p*dist_l_b)/ct.c
            g_x_slope = -E0*K**2*k_p**2*np.cos(k_p*dist_l_b)/ct.c

            E = -ct.e/(ct.m_e*ct.c)*E_z
            E_p = -ct.e/(ct.m_e*ct.c)*E_z_p
            K = g_x*ct.c*ct.e/ct.m_e

        elif mode == "Linear2":
            a0 = laser.a_0
            n_p_SI = self.n_p*1e6
            w_p = np.sqrt(n_p_SI*ct.e**2/(ct.m_e*ct.epsilon_0))
            k_p = w_p/ct.c
            E0 = ct.m_e*ct.c*w_p/ct.e

            nb0 = a0**2/2
            L  = np.sqrt(2)
            sz = L/np.sqrt(2)
            sx = w_0_l/2

            A = E0 * nb0 * np.sqrt(2*np.pi) * sz * np.exp(-(sz)**2/2)
            E_z =  A*np.cos(k_p*dist_l_b)
            E_z_p = -A*k_p*np.sin(k_p*(dist_l_b)) # [V/m^2]
            g_x = -(nb0 * np.sqrt(2*np.pi) * sz * np.exp(-(sz)**2/2)
                    * ( 1/ (k_p*sx)**2) * np.sin(k_p*(dist_l_b)) * k_p*E0/ct.c)

            E = -ct.e/(ct.m_e*ct.c)*E_z
            E_p = -ct.e/(ct.m_e*ct.c)*E_z_p
            K = g_x*ct.c*ct.e/ct.m_e

        elif mode == "FromOsiris2D":
            raise NotImplementedError()
            #(E_z, E_z_p, g_x) = self.get_fields_from_osiris_2D(
            #    simulation_path, time_step, laser, laser_pos_in_osiris,
            #    dist_l_b)
            #E = -ct.e/(ct.m_e*ct.c)*E_z
            #E_p = -ct.e/(ct.m_e*ct.c)*E_z_p
            #K = g_x*ct.c*ct.e/ct.m_e

        elif mode == "FromOsiris3D":
            raise NotImplementedError()
            #(E_z, E_z_p, g_x) = self.get_fields_from_osiris_3D(
            #    simulation_path, time_step, laser, laser_pos_in_osiris,
            #    dist_l_b)
            #E = -ct.e/(ct.m_e*ct.c)*E_z
            #E_p = -ct.e/(ct.m_e*ct.c)*E_z_p
            #K = g_x*ct.c*ct.e/ct.m_e
        

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

        # track beam in steps
        #print("Tracking plasma stage in {} steps...   ".format(steps))
        start = time.time()
        p = Pool(cpu_count())
        t = t_final/steps*(np.arange(steps)+1)
        part = partial(self._get_beam_at_specified_time_step_analytically,
                       beam=beam, g_0=g_0, w_0=w_0, xi_0=xi_0, A_x=A_x,
                       A_y=A_y, phi_x_0=phi_x_0, phi_y_0=phi_y_0, E=E, E_p=E_p,
                       v_w=v_w, K=K)
        beam_steps_list = p.map(part, t)
        end = time.time()
        print("Done ({} seconds)".format(end-start))

        # update beam data
        last_beam = beam_steps_list[-1]
        beam.set_phase_space(last_beam.x, last_beam.y, last_beam.xi,
                             last_beam.px, last_beam.py, last_beam.pz)
        beam.increase_prop_distance(self.length)

        # update laser data
        laser.increase_prop_distance(self.length)
        laser.xi_c = laser.xi_c + (v_w-ct.c)*t_final

        # return steps
        return beam_steps_list

    def _get_beam_at_specified_time_step_analytically(
        self, t, beam, g_0, w_0, xi_0, A_x, A_y, phi_x_0, phi_y_0, E, E_p, v_w,
        K):
        G = 1 + E/g_0*t
        if (G < 1/g_0).any():
            n_part = len(np.where(G<1/g_0)[0])
            print('Warning: unphysical energy found in {}'.format(n_part) 
                  + 'particles due to negative accelerating gradient.')
            # fix unphysical energies (model does not work well when E<=0)
            G = np.where(G<1/g_0, 1/g_0, G)

        phi = 2*np.sqrt(K*g_0)/E*(G**(1/2) - 1)
        if (E == 0).any():
            # apply limit when E->0
            idx_0 = np.where(E==0)[0]
            phi[idx_0] =  np.sqrt(K[idx_0]/g_0[idx_0])*t[idx_0]
        A_0 = np.sqrt(A_x**2 + A_y**2)

        x = A_x*G**(-1/4)*np.cos(phi + phi_x_0)
        v_x = -w_0*A_x*G**(-3/4)*np.sin(phi + phi_x_0)
        p_x = G*g_0*v_x/ct.c

        y = A_y*G**(-1/4)*np.cos(phi + phi_y_0)
        v_y = -w_0*A_y*G**(-3/4)*np.sin(phi + phi_y_0)
        p_y = G*g_0*v_y/ct.c

        delta_xi = (ct.c/(2*E*g_0)*(G**(-1)- 1)
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
                 profile='inverse square'):
        """
        Initialize plasma ramp.

        Parameters:
        -----------
        length : float
            Length of the plasma stage in cm.
            
        plasma_dens_top : float
            Plasma density at the beginning (end) of the downramp (upramp) in
            units of cm^{-3}.

        plasma_dens_down : float
            Plasma density at the position 'position_down' in units of
            cm^{-3}.        

        position_down : float
            Position where the plasma density will be equal to 
            'plasma_dens_down' measured from the beginning (end) of the 
            downramp (upramp).

        ramp_type : string
            Possible types are 'upramp' and 'downramp'.

        profile : string
            Longitudinal density profile of the ramp. Possible values are
            'linear', 'inverse square' and 'exponential'.

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
        
    def track_beam_numerically(self, beam, steps, mode='blowout', laser=None,
                               laser_evolution=False, laser_z_foc=0, 
                               r_max=None, xi_min=None, xi_max=None, n_r=100,
                               n_xi=100, parallel=False, n_proc=None):
        """
        Track the beam through the plasma using a 4th order Runge-Kutta method.
        
        Parameters:
        -----------
        beam : ParticleBunch
            Particle bunch to track.

        steps : int
            Number of steps in which output should be given.

        mode: str
            Mode used to determine the wakefields. Possible values are 
            'blowout', 'blowout_non_rel' and 'cold_fluid_1d'.

        laser : LaserPulse
            Laser used in the plasma stage. Required only if
            mode='cold_fluid_1d'.

        laser_evolution : bool
            If True, the laser pulse transverse profile evolves as a Gaussian
            in vacuum. If False, the pulse envelope stays fixed throughout
            the computation.

        laser_z_foc : float
            Focal position of the laser along z in meters. It is measured as
            the distance from the position where the plasma density is
            n=plasma_dens_top, i.e, as the distance from the beginning (end)
            of the downramp (upramp). Required only if laser_evolution=True.

        r_max : float
            Maximum radial position up to which plasma wakefield will be
            calulated. Required only if mode='cold_fluid_1d'.

        xi_min : float
            Minimum longiudinal (speed of light frame) position up to which
            plasma wakefield will be calulated. Required only if
            mode='cold_fluid_1d'.

        xi_max : float
            Maximum longiudinal (speed of light frame) position up to which
            plasma wakefield will be calulated. Required only if
            mode='cold_fluid_1d'.

        n_r : int
            Number of grid elements along r to calculate the wakefields.
            Required only if mode='cold_fluid_1d'.

        n_xi : int
            Number of grid elements along xi to calculate the wakefields.
            Required only if mode='cold_fluid_1d'.

        parallel : float
            Determines whether or not to use parallel computation.

        n_proc : int
            Number of processes to run in parallel. If None, this will equal
            the number of physical cores. Required only if parallel=True.

        Returns:
        --------
        A list of size 'steps' containing the beam distribution at each step.

        """
        if mode == 'blowout':
            field = PlasmaRampBlowoutField(self.calculate_density)
        elif mode == 'blowout_non_rel':
            raise NotImplementedError()
        elif mode == 'cold_fluid_1d':
            if self.ramp_type == 'upramp':
                laser_z_foc = self.length - laser_z_foc
            field = NonLinearColdFluidWakefield(self.calculate_density,
                                                laser, laser_evolution,
                                                laser_z_foc, r_max, xi_min,
                                                xi_max, n_r, n_xi)
        # Main beam quantities
        mat = beam.get_6D_matrix_with_charge()
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

        if parallel:
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
                    matrix_list.append(
                        mat[:,p*part_per_proc:(p+1)*part_per_proc])

                for s in np.arange(steps):
                    print(s)
                    partial_solver = partial(
                        runge_kutta_4, WF=field, dt=dt_adjusted,
                        iterations=it_per_step, t0=s*t_step)
                    matrix_list = process_pool.map(partial_solver, matrix_list)
                    beam_matrix = np.concatenate(matrix_list, axis=1)
                    x, px, y, py, xi, pz, q = beam_matrix
                    new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                    beam_list.append(
                        ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                      prop_distance=new_prop_dist)
                        )
            finally:
                process_pool.close()
                process_pool.join()
        else:
            for s in np.arange(steps):
                print(s)
                beam_matrix = runge_kutta_4(mat, WF=field, t0=s*t_step,
                                            dt=dt_adjusted,
                                            iterations=it_per_step)
                x, px, y, py, xi, pz, q = copy(beam_matrix)
                new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                beam_list.append(
                    ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                    prop_distance=new_prop_dist)
                    )
        end = time.time()
        print("Done ({} seconds)".format(end-start))

        # update beam data
        last_beam = beam_list[-1]
        beam.set_phase_space(last_beam.x, last_beam.y, last_beam.xi,
                             last_beam.px, last_beam.py, last_beam.pz)
        beam.increase_prop_distance(self.length)

        return beam_list
    
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
                 /self.position_down)
            a = self.plasma_dens_top
            n_p = a + b*z
            # make negative densities 0
            n_p[n_p<0] = 0
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
                     /self.position_down)
            n_p = a*np.exp(-b*z)
        elif self.profile == 'gaussian':
            s_z = self.position_down / np.sqrt(2*np.log(self.plasma_dens_top /
                                                        self.plasma_dens_down))
            n_p = self.plasma_dens_top * np.exp(-z**2/(2*s_z**2))
        return n_p


class Drift():

    """Defines a drift space"""

    def __init__(self, length):
        self.length = length

    def track_bunch(self, bunch, steps, backtrack=False):
        print("Tracking drift in {} step(s)...   ".format(steps))
        l_step = self.length/steps
        bunch_list = list()
        for i in np.arange(0, steps):
            l = (i+1)*l_step*(1-2*backtrack)
            (x, y, xi, px, py, pz) = self._track_step(bunch, l)
            new_prop_dist = bunch.prop_distance + l
            new_bunch = ParticleBunch(bunch.q, x, y, xi, px, py, pz,
                                      prop_distance=new_prop_dist)
            new_bunch.x_ref = bunch.x_ref + l*np.sin(bunch.theta_ref)
            new_bunch.theta_ref = bunch.theta_ref
            bunch_list.append(new_bunch)
        # update bunch data
        last_bunch = bunch_list[-1]
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.prop_distance = last_bunch.prop_distance
        bunch.theta_ref = last_bunch.theta_ref
        bunch.x_ref = last_bunch.x_ref
        print("Done.")
        return bunch_list

    def _track_step(self, bunch, length=None):
        x_0 = bunch.x
        y_0 = bunch.y
        xi_0 = bunch.xi
        px_0 = bunch.px
        py_0 = bunch.py
        pz_0 = bunch.pz
        if length is None:
            t = self.length/ct.c
        else:
            t = length/ct.c
        g = np.sqrt(1 + px_0**2 + py_0**2 + pz_0**2)
        vx = px_0*ct.c/g
        vy = py_0*ct.c/g
        vz = pz_0*ct.c/g
        x = x_0 + vx*t
        y = y_0 + vy*t
        xi = xi_0 + (vz-ct.c)*t
        return (x, y, xi, px_0, py_0, pz_0)


class TMElement():
    # TODO: fix backtracking issues.
    """Defines an element to be tracked using transfer maps."""

    def __init__(self, length=0, angle=0, k1=0, k2=0, gamma_ref=None):
        self.length = length
        self.angle = angle
        self.k1 = k1
        self.k2 = k2
        self.gamma_ref = gamma_ref
        self.element_name = ""

    def track_bunch(self, bunch, steps, backtrack=False, order=2):
        print('')
        print('-'*80)
        print(self.element_name.capitalize())
        print('-'*80)
        l_step = self.length/steps
        bunch_list = list()
        bunch_mat, g_avg = self.get_aligned_beam_matrix_for_tracking(bunch)
        if self.gamma_ref is None:
            self.gamma_ref = g_avg
        self.print_element_properties()
        print('-'*80)
        print("Tracking in {} step(s)... ".format(steps), end = '')
        for i in np.arange(0, steps):
            l = (i+1)*l_step*(1-2*backtrack)
            new_prop_dist = bunch.prop_distance + l
            #bunch_mat, g_avg = bunch.get_alternative_6D_matrix()
            new_bunch_mat = track_with_transfer_map(bunch_mat, l, self.length,
                                                    -self.angle, self.k1,
                                                    self.k2, self.gamma_ref,
                                                    order=order)
            new_bunch_mat = convert_from_ocelot_matrix(new_bunch_mat,
                                                       self.gamma_ref)
            new_bunch = self.create_new_bunch(bunch, new_bunch_mat, l)
            bunch_list.append(new_bunch)
        # update bunch data
        last_bunch = bunch_list[-1]
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.prop_distance = last_bunch.prop_distance
        bunch.theta_ref = last_bunch.theta_ref
        bunch.x_ref = last_bunch.x_ref
        print("Done.")
        print('-'*80)
        return bunch_list

    def get_aligned_beam_matrix_for_tracking(self, bunch):
        bunch_mat = bunch.get_6D_matrix()
        # obtain with respect to reference displacement
        bunch_mat[0] -= bunch.x_ref
        # rotate by the reference angle so that it entern normal to the element
        if bunch.theta_ref != 0:
            rot = rotation_matrix_xz(-bunch.theta_ref)
            bunch_mat = np.dot(rot, bunch_mat)
        return convert_to_ocelot_matrix(bunch_mat, bunch.q, self.gamma_ref)

    def create_new_bunch(self, old_bunch, new_bunch_mat, prop_dist):
        q = old_bunch.q
        if self.angle != 0:
            # angle rotated for prop_dist
            theta_step = self.angle*prop_dist/self.length
            # magnet bending radius
            rho = abs(self.length/self.angle)
            # new reference angle and transverse displacement
            new_theta_ref = old_bunch.theta_ref + theta_step
            sign = -theta_step/abs(theta_step)
            new_x_ref = (old_bunch.x_ref
                         + sign*rho*(np.cos(new_theta_ref)-np.cos(old_bunch.theta_ref)))
        else:
            # new reference angle and transverse displacement
            new_theta_ref = old_bunch.theta_ref
            new_x_ref = (old_bunch.x_ref + self.length*np.sin(old_bunch.theta_ref))
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

    def print_element_properties(self):
        "To be implemented by each element. Prints the element properties"
        raise NotImplementedError


class Dipole(TMElement):
    def __init__(self, length=0, angle=0, gamma_ref = None):
        super().__init__(length, angle, 0, 0, gamma_ref)
        self.element_name = 'dipole'

    def print_element_properties(self):
        ang_deg = self.angle * 180/ct.pi
        b_field = (ct.m_e*ct.c/ct.e) * self.angle*self.gamma_ref/self.length
        print('Bending angle = {:1.4f} rad ({:1.4f} deg)'.format(self.angle,
                                                                 ang_deg))
        print('Dipole field = {:1.4f} T'.format(b_field))


class Quadrupole(TMElement):
    def __init__(self, length=0, k1=0, gamma_ref = None):
        super().__init__(length, 0, k1, 0, gamma_ref)
        self.element_name = 'quadrupole'

    def print_element_properties(self):
        g = self.k1 * self.gamma_ref*(ct.m_e*ct.c/ct.e)
        print('Quadrupole gradient = {:1.4f} T/m'.format(g))


class Sextupole(TMElement):
    def __init__(self, length=0, k2=0, gamma_ref = None):
        super().__init__(length, 0, 0, k2, gamma_ref)
        self.element_name = 'sextupole'

    def print_element_properties(self):
        g = self.k2 * self.gamma_ref*(ct.m_e*ct.c/ct.e)
        print('Sextupole gradient = {:1.4f} T/m^2'.format(g))


class PlasmaLens(object):

    """Defines a plasma lens"""

    def __init__(self, length, foc_strength):
        self.length = length
        self.foc_strength = foc_strength

    def get_matched_beta(self, ene):
        return ge.matched_plasma_beta_function(ene, k_x=self.foc_strength)

    def track_beam_numerically(self, beam, steps, non_rel=False,
                               parallel=False, n_proc=None):
        """Tracks the beam through the plasma lens and returns the final
        phase space"""
        if non_rel:
            field = PlasmaLensField(self.foc_strength)
        else:
            field = PlasmaLensFieldRelativistic(self.foc_strength)
        # Main beam quantities
        mat = beam.get_6D_matrix_with_charge()

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
        if parallel:
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
                    x, px, y, py, xi, pz, q = beam_matrix
                    new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                    beam_list.append(ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                                   prop_distance=new_prop_dist))
            finally:
                process_pool.close()
                process_pool.join()
        else:
            # compute in single process
            for s in np.arange(steps):
                print(s)
                beam_matrix = runge_kutta_4(mat, WF=field, t0=s*t_step,
                                            dt=dt_adjusted,
                                            iterations=it_per_step)
                x, px, y, py, xi, pz, q = copy(beam_matrix)
                new_prop_dist = beam.prop_distance + (s+1)*t_step*ct.c
                beam_list.append(
                    ParticleBunch(beam.q, x, y, xi, px, py, pz,
                                    prop_distance=new_prop_dist)
                    )
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
        Kx = WF.Kx(
            beam.x, beam.y, beam.xi, beam.px, beam.py, beam.pz, beam.q, gamma,
            0)
        mean_Kx = np.average(Kx, weights=beam.q)
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * mean_Kx/mean_gamma)
        T_x = 1/w_x
        dt = 0.1*T_x
        return dt

    def _gamma(self, px, py, pz):
        return np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
