from typing import Optional, Union, List

import scipy.constants as ct

from wake_t.diagnostics import OpenPMDDiagnostics
from wake_t.tracking.tracker import Tracker
from wake_t.fields.base import Field
from wake_t.particles.particle_bunch import ParticleBunch


class FieldElement():
    """
    Generic class for any beamline element based on field tracking.

    Parameters
    ----------
    length : float or str
        Length of the plasma stage in m.
    dt_bunch : float
        The time step for evolving the particle bunches. An adaptive time
        step can be used if this parameter is set to ``'auto'`` and a
        ``auto_dt_bunch`` function is provided.
    bunch_pusher : str
        The pusher used to evolve the particle bunches in time within
        the specified fields. Possible values are ``'rk4'`` (Runge-Kutta
        method of 4th order) or ``'boris'`` (Boris method).
    n_out : int
        Number of times along the stage in which the particle distribution
        should be returned (A list with all output bunches is returned
        after tracking).
    name : str
        Name of the plasma stage. This is only used for displaying the
        progress bar during tracking. By default, ``'field element'``.
    fields : list
        List of Fields that will be applied to the particle bunches.
    auto_dt_bunch : callable, optional
        Function used to determine the adaptive time step for bunches in
        which the time step is set to ``'auto'``. The function should take
        solely a ``ParticleBunch`` as argument.

    """

    def __init__(
        self,
        length: float,
        dt_bunch: Union[float, str],
        bunch_pusher: Optional[str] = 'rk4',
        n_out: Optional[int] = 1,
        name: Optional[str] = 'field element',
        fields: Optional[List[Field]] = [],
        auto_dt_bunch: Optional[str] = None
    ) -> None:
        self.length = length
        self.bunch_pusher = bunch_pusher
        self.dt_bunch = dt_bunch
        self.n_out = n_out
        self.name = name
        self.fields = fields
        self.auto_dt_bunch = auto_dt_bunch

    def track(
        self,
        bunches: Optional[Union[ParticleBunch, List[ParticleBunch]]] = [],
        opmd_diag: Optional[Union[bool, OpenPMDDiagnostics]] = False,
        diag_dir: Optional[str] = None
    ) -> Union[List[ParticleBunch], List[List[ParticleBunch]]]:
        """
        Track bunch through element.

        Parameters
        ----------
        bunches : ParticleBunch or list of ParticleBunch
            Particle bunches to be tracked.
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

        Returns
        -------
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
        elif not opmd_diag:
            opmd_diag = None

        # Create tracker.
        tracker = Tracker(
            t_final=self.length/ct.c,
            bunches=bunches,
            dt_bunches=dt_bunch,
            fields=self.fields,
            n_diags=self.n_out,
            opmd_diags=opmd_diag,
            bunch_pusher=self.bunch_pusher,
            auto_dt_bunch_f=self.auto_dt_bunch,
            section_name=self.name
        )

        # Do tracking.
        bunch_list = tracker.do_tracking()

        # If only tracking one bunch, do not return list of lists.
        if len(bunch_list) == 1:
            bunch_list = bunch_list[0]

        return bunch_list
