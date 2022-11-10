""" This module contains the definition of the PlasmaStage class """

import numpy as np
import scipy.constants as ct

import wake_t.physics_models.plasma_wakefields as wf
from wake_t.diagnostics import OpenPMDDiagnostics
from wake_t.tracking.tracker import Tracker
from wake_t.fields.base import Field


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
                 bunch_pusher='rk4', dt_bunch='auto', n_out=1,
                 name='Plasma stage', external_fields=[], **model_params):
        """
        Initialize plasma stage.

        Parameters
        ----------
        length : float
            Length of the plasma stage in m.

        density : float
            Plasma density in units of m^{-3}.

        wakefield_model : str or Field
            Wakefield model to be used. Possible values are 'blowout',
            'custom_blowout', 'focusing_blowout', 'cold_fluid_1d' and
            'quasistatic_2d'. If `None`, no wakefields will be computed.

        bunch_pusher : str
            The pusher used to evolve the particle bunches in time within
            the specified fields. Possible values are 'rk4' (Runge-Kutta
            method of 4th order) or 'boris' (Boris method).

        dt_bunch : float
            The time step for evolving the particle bunches. If `None`, it will
            be automatically set to `dt = T/(10*2*pi)`, where T is the smallest
            expected betatron period of the bunch along the plasma stage.

        n_out : int
            Number of times along the stage in which the particle distribution
            should be returned (A list with all output bunches is returned
            after tracking).

        name : str
            Name of the plasma stage. This is only used for displaying the
            progress bar during tracking. By default, `'Plasma stage'`.

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
            specified by the parameter above. By default, 0.


        Model 'cold_fluid_1d'
        ---------------------
        laser : LaserPulse
            Laser driver of the plasma stage.

        laser_evolution : bool
            If True (default), the laser pulse is evolved
            using a laser envelope model. If False, the pulse envelope stays
            unchanged throughout the computation.

        laser_envelope_substeps : int
            Number of substeps of the laser envelope solver per `dz_fields`.
            The time step of the envelope solver is therefore
            `dz_fields / c / laser_envelope_substeps`.

        laser_envelope_nxi, laser_envelope_nr : int, optional
            If given, the laser envelope will run in a grid of size
            (`laser_envelope_nxi`, `laser_envelope_nr`) instead
            of (`n_xi`, `n_r`). This allows the laser to run in a finer (or
            coarser) grid than the plasma wake. It is not necessary to specify
            both parameters. If one of them is not given, the resolution of
            the plasma grid with be used for that direction.

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

        laser_envelope_substeps : int
            Number of substeps of the laser envelope solver per `dz_fields`.
            The time step of the envelope solver is therefore
            `dz_fields / c / laser_envelope_substeps`.

        laser_envelope_nxi, laser_envelope_nr : int, optional
            If given, the laser envelope will run in a grid of size
            (`laser_envelope_nxi`, `laser_envelope_nr`) instead
            of (`n_xi`, `n_r`). This allows the laser to run in a finer (or
            coarser) grid than the plasma wake. It is not necessary to specify
            both parameters. If one of them is not given, the resolution of
            the plasma grid with be used for that direction.

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
        self.wakefield = self._get_wakefield(wakefield_model, model_params)
        self.bunch_pusher = bunch_pusher
        self.dt_bunch = dt_bunch
        self.n_out = n_out
        self.name = name
        self.external_fields = external_fields

    def track(self, bunches=[], out_initial=False, opmd_diag=False,
              diag_dir=None):
        """
        Track bunch through plasma stage.

        Parameters:
        -----------
        bunches : ParticleBunch or list of ParticleBunch
            Particle bunches to be tracked.

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
        # Make sure `bunches` is a list.
        if not isinstance(bunches, list):
            bunches = [bunches]

        if not isinstance(self.dt_bunch, list):
            dt_bunch = [self.dt_bunch] * len(bunches)
        else:
            dt_bunch = self.dt_bunches

        # Create diagnostics instance.
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)

        fields = []
        if self.wakefield is not None:
            fields.append(self.wakefield)
        fields.extend(self.external_fields)

        # Create tracker.
        tracker = Tracker(
            t_final=self.length/ct.c,
            bunches=bunches,
            dt_bunches=dt_bunch,
            fields=fields,
            n_diags=self.n_out,
            opmd_diags=opmd_diag,
            bunch_pusher=self.bunch_pusher,
            auto_dt_bunch_f=self._get_optimized_dt,
            section_name=self.name
        )

        # Do tracking.
        bunch_list = tracker.do_tracking()

        # If only tracking one bunch, do not return list of lists.
        if len(bunch_list) == 1:
            bunch_list = bunch_list[0]

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
        if model is None:
            return None
        elif isinstance(model, Field):
            return model
        elif model in wakefield_models:
            return wakefield_models[model](self.density, **model_params)
        else:
            raise ValueError(
                'Wakefield model "{}" not recognized.'.format(model))

    def _get_optimized_dt(self, beam):
        """ Get tracking time step. """
        # Get minimum gamma in the bunch (assumes px,py << pz).
        min_gamma = np.sqrt(np.min(beam.pz)**2 + 1)
        # calculate maximum focusing along stage.
        z = np.linspace(0, self.length, 100)
        n_p = self.density(z)
        w_p = np.sqrt(max(n_p)*ct.e**2/(ct.m_e*ct.epsilon_0))
        max_kx = (ct.m_e/(2*ct.e*ct.c))*w_p**2
        w_x = np.sqrt(ct.e*ct.c/ct.m_e * max_kx/min_gamma)
        period_x = 1/w_x
        dt = 0.1*period_x
        return dt
