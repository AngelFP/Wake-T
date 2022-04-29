""" This module contains the definition of the PlasmaStage class """

import time
from copy import deepcopy

import numpy as np
import scipy.constants as ct
from wake_t.fields.numerical_field import NumericalField

import wake_t.physics_models.plasma_wakefields as wf
from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.diagnostics import OpenPMDDiagnostics


wakefield_models = {
    'simple_blowout': wf.SimpleBlowoutWakefield,
    'custom_blowout': wf.CustomBlowoutWakefield,
    'focusing_blowout': wf.FocusingBlowoutField,
    'cold_fluid_1d': wf.NonLinearColdFluidWakefield,
    'quasistatic_2d': wf.Quasistatic2DWakefield
}


class PlasmaStage():

    """ Generic class for defining a plasma acceleration stage. """

    def __init__(self, length, density, wakefield_model='simple_blowout',
                 bunch_pusher='rk4', n_out=1, **model_params):
        """
        Initialize plasma stage.

        Parameters
        ----------
        length : float
            Length of the plasma stage in m.

        density : float
            Plasma density in units of m^{-3}.

        wakefield_model : str
            Wakefield model to be used. Possible values are 'blowout',
            'custom_blowout', 'focusing_blowout', 'cold_fluid_1d' and
            'quasistatic_2d'.

        bunch_pusher : str
            The pusher used to evolve the particle bunches in time within
            the specified fields. Possible values are 'rk4' (Runge-Kutta
            method of 4th order) or 'boris' (Boris method).

        n_out : int
            Number of times along the stage in which the particle distribution
            should be returned (A list with all output bunches is returned
            after tracking).

        **model_params
            Keyword arguments which will be given to the wakefield model. Each
            model requires a different set of parameters which are listed
            below:

        Model 'focusing_blowout'
        ------------------------
        No additional parameters required.

        Model 'simple_blowout'
        ----------------------
        laser : LaserPulse
            Laser driver of the plasma stage.

        Model 'custom_blowout'
        ----------------------
        laser : LaserPulse
            Laser driver of the plasma stage.

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


        Model 'cold_fluid_1d'
        ---------------------
        laser : LaserPulse
            Laser driver of the plasma stage.

        laser_evolution : bool
            If True (default), the laser pulse is evolved
            using a laser envelope model. If False, the pulse envelope stays
            unchanged throughout the computation.

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

        dz_fields : float (optional)
            Determines how often the plasma wakefields should be updated. If
            dz_fields=0 (default value), the wakefields are calculated at every
            step of the Runge-Kutta solver for the beam particle evolution
            (most expensive option). If specified, the wakefields are only
            updated in steps determined by dz_fields. For example, if
            dz_fields=10e-6, the plasma wakefields are only updated every time
            the simulation window advances by 10 micron. By default, if not
            specified, the value of `dz_fields` will be `xi_max-xi_min`, i.e.,
            the length the simulation box.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic'.

        Model 'quasistatic_2d'
        ---------------------
        laser : LaserPulse
            Laser driver of the plasma stage.

        laser_evolution : bool
            If True (default), the laser pulse is evolved
            using a laser envelope model. If False, the pulse envelope stays
            unchanged throughout the computation.

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
            Number of plasma particles per radial cell. By default `ppc=2`.

        dz_fields : float (optional)
            Determines how often the plasma wakefields should be updated. If
            dz_fields=0 (default value), the wakefields are calculated at every
            step of the Runge-Kutta solver for the beam particle evolution
            (most expensive option). If specified, the wakefields are only
            updated in steps determined by dz_fields. For example, if
            dz_fields=10e-6, the plasma wakefields are only updated every time
            the simulation window advances by 10 micron. By default, if not
            specified, the value of `dz_fields` will be `xi_max-xi_min`, i.e.,
            the length the simulation box.

        r_max_plasma : float
            Maximum radial extension of the plasma column. If `None`, the
            plasma extends up to the `r_max` boundary of the simulation box.

        parabolic_coefficient : float or callable
            The coefficient for the transverse parabolic density profile. The
            radial density distribution is calculated as
            `n_r = n_p * (1 + parabolic_coefficient * r**2)`, where n_p is the
            local on-axis plasma density. If a `float` is provided, the same
            value will be used throwout the stage. Alternatively, a function
            which returns the value of the coefficient at the given position
            `z` (e.g. `def func(z)`) might also be provided.

        p_shape : str
            Particle shape to be used for the beam charge deposition. Possible
            values are 'linear' or 'cubic' (default).

        max_gamma : float
            Plasma particles whose `gamma` exceeds `max_gamma` are considered
            to violate the quasistatic condition and are put at rest (i.e.,
            `gamma=1.`, `pr=pz=0.`). By default `max_gamma=10`.

        """

        self.length = length
        self.density = self._get_density_profile(density)
        if isinstance(wakefield_model, wf.Wakefield):
            self.wakefield = wakefield_model
        else:
            self.wakefield = self._get_wakefield(wakefield_model, model_params)
        self.bunch_pusher = bunch_pusher
        self.n_out = n_out

    def track(self, bunch, out_initial=False, opmd_diag=False, diag_dir=None):
        """
        Track bunch through plasma stage.

        Parameters:
        -----------
        bunch : ParticleBunch
            Particle bunch to be tracked.

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
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        bunch_list = self._track_numerically(bunch, out_initial, opmd_diag)
        if opmd_diag is not False:
            opmd_diag.increase_z_pos(self.length)
        return bunch_list

    def _get_density_profile(self, density):
        """ Get density profile function """
        if isinstance(density, float):
            def uniform_density(z):
                return np.ones_like(z) * density
            return uniform_density
        elif callable(density):
            return density
        else:
            raise ValueError(
                'Type {} not supported for density.'.format(type(density)))

    def _get_wakefield(self, model, model_params):
        """ Initialize and return corresponding wakefield model. """
        if model in wakefield_models:
            return wakefield_models[model](self.density, **model_params)
        else:
            raise ValueError(
                'Wakefield model "{}" not recognized.'.format(model))

    def _track_numerically(self, bunch, out_initial, opmd_diag):
        """ Track bunch numerically. """
        start = time.time()

        # Initialize current time.
        t = 0.
        # Final time of the tracking.
        t_final = self.length / ct.c
        # Initialize current time of the output and diagnostics.
        t_output = 0.
        # Time step of the output and diagnostics.
        dt_output = t_final / self.n_out
        # Initialize current time of the particle bunch.
        t_bunch = 0.
        # Time step of the particle bunch.
        dt_bunch = self._get_optimized_dt(bunch)
        # Initialize current time of the fields and determine their time step.
        if isinstance(self.wakefield, NumericalField):
            num_wf = True
            t_fields = 0.
            dt_fields = self.wakefield.dt_update
        else:
            num_wf = False
            t_fields = 0.
            dt_fields = np.inf

        # Update the fields (trigger initial calculation).
        self.wakefield.update(t, [bunch])

        # Initialize list where the output bunches will be stored.
        bunch_list = []

        # Generate output of initial bunch and fields.
        if out_initial:
            bunch_list.append(
                ParticleBunch(
                    deepcopy(bunch.q),
                    deepcopy(bunch.x),
                    deepcopy(bunch.y),
                    deepcopy(bunch.xi),
                    deepcopy(bunch.px),
                    deepcopy(bunch.py),
                    deepcopy(bunch.pz),
                    prop_distance=deepcopy(bunch.prop_distance),
                    name=bunch.name
                )
            )
            if opmd_diag is not False:
                opmd_diag.write_diagnostics(
                    t, dt_bunch, [bunch], self.wakefield)

        # Perform tracking.
        while t < t_final:
            # Determine next time for the output, bunch and fields.
            t_next_output = t_output + dt_output
            t_next_bunch = t_bunch + dt_bunch
            t_next_fields = t_fields + dt_fields

            # If the next closest time is `t_next_output`, generate output and
            # advance to `t_next_output`
            if t_next_output < min(t_next_bunch, t_next_fields):
                t_output += dt_output
                t = t_output
                bunch_list.append(
                    ParticleBunch(
                        deepcopy(bunch.q),
                        deepcopy(bunch.x),
                        deepcopy(bunch.y),
                        deepcopy(bunch.xi),
                        deepcopy(bunch.px),
                        deepcopy(bunch.py),
                        deepcopy(bunch.pz),
                        prop_distance=deepcopy(bunch.prop_distance),
                        name=bunch.name
                    )
                )
                if opmd_diag is not False:
                    opmd_diag.write_diagnostics(
                        t, dt_bunch, [bunch], self.wakefield)
            # If the next closest time is `t_next_bunch`, push bunch and
            # advance to `t_next_bunch`
            elif t_next_bunch < min(t_next_output, t_next_fields):
                if not num_wf:
                    self.wakefield.update(t, [bunch])
                bunch.evolve(self.wakefield, dt_bunch, self.bunch_pusher)
                t_bunch += dt_bunch
                t = t_bunch
            # If the next closest time is `t_next_fields`, update fields and
            # advance to `t_next_fields`
            elif t_next_fields < min(t_next_output, t_next_bunch) and num_wf:
                self.wakefield.update(t, [bunch])
                t_fields += dt_fields
                t = t_fields

        end = time.time()
        print("Done ({:1.3f} seconds).".format(end-start))
        print('-'*80)

        return bunch_list

    def _get_optimized_dt(self, beam):
        """ Get tracking time step. """
        gamma = np.sqrt(1 + beam.px**2 + beam.py**2 + beam.pz**2)
        mean_gamma = np.average(gamma, weights=beam.q)
        # calculate maximum focusing along stage.
        z = np.linspace(0, self.length, 100)
        n_p = self.density(z)
        w_p = np.sqrt(max(n_p)*ct.e**2/(ct.m_e*ct.epsilon_0))
        max_kx = (ct.m_e/(2*ct.e*ct.c))*w_p**2
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * max_kx/mean_gamma)
        period_x = 1/w_x
        dt = 0.1*period_x
        return dt
