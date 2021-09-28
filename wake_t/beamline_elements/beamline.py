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
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        for i, element in enumerate(self.elements):
            bunch_list.extend(
                element.track(
                    bunch,
                    out_initial=out_initial and i == 0,
                    opmd_diag=opmd_diag
                    )
                )
        return bunch_list
