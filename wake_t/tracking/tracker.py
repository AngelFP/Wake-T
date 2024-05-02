""" This module contains the Tracker class. """
from typing import Optional, Callable, List, Literal

import numpy as np
import scipy.constants as ct

from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.fields.base import Field
from wake_t.fields.analytical_field import AnalyticalField
from wake_t.fields.numerical_field import NumericalField
from wake_t.diagnostics.openpmd_diag import OpenPMDDiagnostics
from wake_t.utilities.numba import (
    num_threads, set_num_threads, get_num_threads
)
from .progress_bar import get_progress_bar


class Tracker():
    """Class in charge of evolving in time the particle bunches and fields.

    There are 3 main ingredients in the simulation of an accelerator stage:
    particle bunches, electromagnetic fields, and diagnostics. Each of these
    elements has a certain periodicity: the bunches are evolved with a certain
    time step (which can be adaptive), the numerical fields are also updated
    with a certain time step, and the diagnostics are generated with a regular
    periodicity.

    Thus, in time, a simulation looks something like:

        Bunch: x-x-x--x---x-----x------x--------x---------x----x|
        Field: x-------x-------x-------x-------x-------x-------x|
        Diags: x-----------x-----------x-----------x-----------x|
        Time:  ------------------------------------------------>| End

    Where the `x` denote the moments in time where each quantity is updated.

    The job of the `Tracker` is to orchestrate this flow and update each
    element at the right time.

    Parameters
    ----------
    t_final : float
        Final time of the tracking.
    bunches : list, optional
        List of `ParticleBunch`es to track.
    dt_bunches : list, optional
        List of time steps. There should be one value per bunch. Possible
        values are any float >0 for a constant time step, or 'auto' for
        enabling an adaptive time step.
    fields : list, optional
        List of `Field`s in which to evolve the particles.
    n_diags : int, optional
        Number of diagnostics to output.
    opmd_diags : OpenPMDDiagnostics, optional
        Instace of OpenPMDDiagnostics in charge of generating the openPMD
        output. If not given, no diagnostics will be generated.
    auto_dt_bunch_f : callable, optional
        Function used to determine the adaptive time step for bunches in
        which the time step is set to `'auto'`. The function should take
        solely a `ParticleBunch` as argument.
    bunch_pusher : str, optional
        The particle pusher used to evolve the bunches. Possible values
        are `'boris'` or `'rk4'`.
    push_bunches_before_diags : bool, optional
        Whether to push the bunches before saving them to the diagnostics.
        Since the time step of the diagnostics can be different from that
        of the bunches, it could happen that the bunches appear in the
        diagnostics as they were at the last push, but not at the actual
        time of the diagnostics. Setting this parameter to ``True``
        (default) ensures that an additional push is given to all bunches
        to evolve them to the diagnostics time before saving.
        This additional push will always have a time step smaller than
        the the time step of the bunch, so it has no detrimental impact
        on the accuracy of the simulation. However, it could make
        convergence studies more difficult to interpret,
        since the number of pushes will depend on `n_diags`. Therefore,
        it is exposed as an option so that it can be disabled if needed.
    show_progress_bar : bool, optional
        Whether to show a progress bar of the tracking. By default ``True``.
    section_name : str, optional
        Name of the section to be tracked. This will be appended to the
        beginning of the progress bar.

    """

    def __init__(
        self,
        t_final: float,
        bunches: Optional[List[ParticleBunch]] = [],
        dt_bunches: Optional[List[float]] = [],
        fields: Optional[List[Field]] = [],
        n_diags: Optional[int] = 0,
        opmd_diags: Optional[OpenPMDDiagnostics] = None,
        auto_dt_bunch_f: Optional[Callable[[ParticleBunch], float]] = None,
        bunch_pusher: Optional[Literal['boris', 'rk4']] = 'boris',
        push_bunches_before_diags: Optional[bool] = True,
        show_progress_bar: Optional[bool] = True,
        section_name: Optional[str] = 'Simulation'
    ) -> None:
        self.t_final = t_final
        self.bunches = bunches
        self.dt_bunches = dt_bunches
        self.fields = fields if len(fields) > 0 else [AnalyticalField()]
        self.opmd_diags = opmd_diags
        self.n_diags = n_diags
        self.auto_dt_bunch_f = auto_dt_bunch_f
        self.bunch_pusher = bunch_pusher
        self.push_bunches_before_diags = push_bunches_before_diags
        self.show_progress_bar = show_progress_bar
        self.section_name = section_name

        # Get all numerical fields and their time steps.
        self.num_fields = [f for f in fields if isinstance(f, NumericalField)]
        for field in self.num_fields:
            field.adjust_dt(self.t_final)
        self.dt_fields = [f.dt_update for f in self.num_fields]

        # Make lists with all objects to track and their time steps.
        self.objects_to_track = [*self.bunches, *self.num_fields]
        self.dt_objects = [*self.dt_bunches, *self.dt_fields]

        # Get list of bunches with adaptive time step.
        self.auto_dt_bunches = []
        for i, dt in enumerate(self.dt_bunches):
            if dt == 'auto':
                self.auto_dt_bunches.append(self.bunches[i])

        # If needed, add diagnostics to objects to track.
        if self.n_diags > 0:
            self.objects_to_track.append('diags')
            self.dt_diags = self.t_final/self.n_diags
            self.dt_objects.append(self.dt_diags)
            self.bunch_list = []
            for bunch in bunches:
                self.bunch_list.append([])

        # Initialize tracking time.
        self.t_tracking = 0.

    def do_tracking(self) -> List[List[ParticleBunch]]:
        """Do the tracking.

        Returns
        -------
        list
            A list with `n` items, where `n` is the number of bunches to track.
            Each item is another list with `n_diag` copies of the particle
            bunch along the tracking.
        """
        # Get current number of threads before setting it to the number
        # requested by Wake-T.
        num_threads_outside_waket = get_num_threads()
        set_num_threads(num_threads)

        # Initialize progress bar.
        progress_bar = get_progress_bar(
            description=self.section_name,
            total_length=self.t_final*ct.c,
            disable=not self.show_progress_bar,
        )

        # Calculate fields at t=0.
        for field in self.num_fields:
            field.update(self.bunches)

        # Generate initial diagnostics.
        self.generate_diagnostics()

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
            t_current = t_objects[i_next]

            # If the next time of the object is beyond `t_final`, the tracking
            # is finished.
            if np.float32(t_next) > np.float32(self.t_final):
                break

            # Advance tracking time.
            self.t_tracking = t_next

            # If next object is a ParticleBunch, update it.
            if isinstance(obj_next, ParticleBunch):
                self.evolve_bunch(
                    bunch=obj_next,
                    t_current=t_current,
                    t_next=t_next,
                    dt_next=dt_next,
                    i_next=i_next,
                    dt_objects=dt_objects,
                )

            # If next object is a NumericalField, update it.
            elif isinstance(obj_next, NumericalField):
                obj_next.update(self.bunches)

            # If next object are the diagnostics, generate them.
            elif obj_next == 'diags':
                # Evolve all bunches to the diagnostics time.
                if self.push_bunches_before_diags:
                    for i, obj in enumerate(self.objects_to_track):
                        if isinstance(obj, ParticleBunch):
                            dt_bunch = t_next - t_objects[i]
                            self.evolve_bunch(
                                bunch=obj,
                                t_current=t_objects[i],
                                t_next=t_next,
                                dt_next=dt_bunch,
                                i_next=i,
                                dt_objects=dt_objects,
                            )
                            t_objects[i] += dt_bunch
                self.generate_diagnostics()

            # Advance current time of the update object.
            t_objects[i_next] += dt_next

            # Update progress bar.
            progress_bar.update(self.t_tracking*ct.c - progress_bar.n)

        # Finalize tracking by increasing z position of diagnostics.
        if self.opmd_diags is not None:
            self.opmd_diags.increase_z_pos(self.t_final * ct.c)

        # Close progress bar.
        progress_bar.close()

        # Reset the number of threads to the value outside of Wake-T.
        # This should help avoiding Wake-T to affect the behavior of other
        # applications that require a different number of threads.
        set_num_threads(num_threads_outside_waket)

        return self.bunch_list

    def evolve_bunch(
        self,
        bunch: ParticleBunch,
        t_current: float,
        t_next: float,
        dt_next: float,
        i_next: int,
        dt_objects: List,
    ):
        """Evolve particle bunch to next time step.

        Parameters
        ----------
        bunch : ParticleBunch
            The particle bunch to evolve.
        t_current : float
            The current time of the simulation.
        t_next : float
            The time of the next step.
        dt_next : float
            The time step by which to advance the bunch.
        i_next : int
            The index of the bunch in the list of objects to track.
        dt_objects : List
            The time steps of all objects to track.
        """
        bunch.evolve(self.fields, t_current, dt_next, self.bunch_pusher)
        # Update the time step if set to `'auto'`.
        if bunch in self.auto_dt_bunches:
            dt_objects[i_next] = self.auto_dt_bunch_f(bunch)
        # Determine if this was the last push.
        final_push = np.float32(t_next) == np.float32(self.t_final)
        # Determine if next push brings the bunch beyond `t_final`.
        next_push_beyond_final_time = (
            t_next + dt_objects[i_next] > self.t_final)
        # Make sure the last push of the bunch advances it to exactly
        # `t_final`.
        if not final_push and next_push_beyond_final_time:
            dt_objects[i_next] = self.t_final - t_next

    def generate_diagnostics(self) -> None:
        """Generate tracking diagnostics."""
        # Make copy of current bunches and store in output list.
        for i, bunch in enumerate(self.bunches):
            self.bunch_list[i].append(bunch.copy())

        # If needed, write also the openPMD diagnostics.
        if self.opmd_diags is not None:
            self.opmd_diags.write_diagnostics(
                self.t_tracking, self.dt_diags, self.bunches, self.fields)
