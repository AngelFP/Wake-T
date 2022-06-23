from copy import deepcopy
import numpy as np
import scipy.constants as ct
from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.fields.analytic_field import AnalyticField
from wake_t.fields.numerical_field import NumericalField


class Tracker():

    def __init__(
            self, t_final, bunches=[], dt_bunches=[], fields=[],
            n_diags=0, opmd_diags=False, auto_dt_bunch_f=None,
            bunch_pusher='rk4'):
        self.t_final = t_final
        self.bunches = bunches
        self.dt_bunches = dt_bunches
        self.fields = fields if fields is not None else [AnalyticField()]
        self.num_fields = [f for f in fields if isinstance(f, NumericalField)]
        self.dt_fields = [f.dt_update for f in self.num_fields]
        self.opmd_diags = opmd_diags
        self.n_diags = n_diags
        self.auto_dt_bunch_f = auto_dt_bunch_f
        self.bunch_pusher = bunch_pusher

        self.objects_to_track = [*self.bunches, *self.num_fields]
        self.dt_objects = [*self.dt_bunches, *self.dt_fields]

        self.auto_bunch_indices = []
        for i, dt in enumerate(self.dt_bunches):
            if dt == 'auto':
                self.auto_bunch_indices.append(i)

        if self.n_diags > 0:
            self.objects_to_track.append('diags')
            self.dt_diags = self.t_final/self.n_diags
            self.dt_objects.append(self.dt_diags)
            self.bunch_list = [[]] * len(bunches)
        
        self.t_tracking = 0.

    def do_tracking(self):

        # Calculate fields at t=0
        for field in self.num_fields:
            field.update(self.bunches)

        # Write initial diagnostics
        self.write_diagnostics()

        # Allocate arrays containing the time step and current time of all
        # objects during tracking.
        t_objects = np.zeros(len(self.dt_objects))
        dt_objects = np.zeros(len(self.dt_objects))

        # Fill up time steps.
        for i, dt in enumerate(self.dt_objects):
            if dt == 'auto':
                dt_objects[i] = self.auto_dt_bunch_f(self.bunches[i])
            else:
                dt_objects[i] = dt

        # Start tracking loop.
        while True:

            # Calculate next time of all objects.
            t_next_objects = t_objects + dt_objects

            # Get index of the object with smallest next time.
            # float32 is used to avoid precision issues. Since
            # `t_next_objects` is calculated by repeatedly adding dt_objects,
            # it can happen that two objects that are supposed to have exactly
            # the same `t_next_objects` do not have it due to precision issues.
            # Going from float64 to float32 reduces the number of decimals,
            # thus making sure that two "almost identical" numbers at float64
            # are actually identical as float32. 
            i_next = np.argmin(t_next_objects.astype(np.float32))

            # Get next object and its corresponding time and time step.
            obj_next = self.objects_to_track[i_next]
            dt_next = dt_objects[i_next]
            t_next = t_next_objects[i_next]

            # If the next time of the object is beyond `t_final`, the tracking
            # is finished.
            if np.float32(t_next) > np.float32(self.t_final):
                break

            
            if isinstance(obj_next, ParticleBunch):
                obj_next.evolve(
                    self.fields, self.t_tracking, dt_next, self.bunch_pusher)
                # Update the time steps labeled as `'auto'`.
                for i in self.auto_bunch_indices:
                    dt_objects[i] = self.auto_dt_bunch_f(self.bunches[i])
                # Determine if this was the last push.
                final_push = np.float32(t_next) == np.float32(self.t_final)
                # Determine is next push brings the bunch beyond `t_final`.
                next_push_beyond_final_time = (
                    t_next + dt_objects[i_next] > self.t_final)
                # Make sure the last push of the bunch advances it to exactly
                # `t_final`.
                if not final_push and next_push_beyond_final_time:
                        dt_objects[i_next] = self.t_final - t_next

            elif isinstance(obj_next, NumericalField):
                obj_next.update(self.bunches)

            elif obj_next == 'diags':
                self.write_diagnostics()

            t_objects[i_next] += dt_objects[i_next]

            self.t_tracking = t_next

        if self.opmd_diags is not False:
            self.opmd_diag.increase_z_pos(self.t_final * ct.c)

        return self.bunch_list

    def write_diagnostics(self):
        for i, bunch in enumerate(self.bunches):
                self.bunch_list[i].append(
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

        if self.opmd_diags is not False:
            self.opmd_diags.write_diagnostics(
                self.t_tracking, self.dt_diags, self.bunches, self.fields)
