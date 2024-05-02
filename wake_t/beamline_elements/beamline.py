from typing import Optional, Union, List

from wake_t.diagnostics import OpenPMDDiagnostics
from wake_t.particles.particle_bunch import ParticleBunch


class Beamline():
    """
    Class for grouping beamline elements and allowing easier tracking.

    """

    def __init__(
        self,
        elements: List
    ) -> None:
        self.elements = elements

    def track(
        self,
        bunches: Optional[Union[ParticleBunch, List[ParticleBunch]]] = [],
        opmd_diag: Optional[bool] = False,
        diag_dir: Optional[str] = None,
        show_progress_bar: Optional[bool] = True,
    ) -> Union[List[ParticleBunch], List[List[ParticleBunch]]]:
        """
        Track bunch through beamline.

        Parameters
        ----------
        bunches : ParticleBunch or list of ParticleBunch
            Particle bunches to be tracked.
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
        show_progress_bar : bool, optional
            Whether to show a progress bar of the tracking through each
            element. By default ``True``.

        Returns
        -------
        A list of size 'n_out' containing the bunch distribution at each step.

        """
        bunch_list = []
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        for element in self.elements:
            bunch_list.extend(
                element.track(
                    bunches,
                    opmd_diag=opmd_diag,
                    show_progress_bar=show_progress_bar,
                )
            )
        return bunch_list
