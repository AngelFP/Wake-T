""" Contains the base class for all EM fields. """

class Field():
    """
    Base class for all EM fields.
    
    It defines the interface for gathering all field components
    (i.e., Ex, Ey, Ez, Bx, By, Bz), updating the fields, and getting the
    openpmd diagnostics.
    """

    def __init__(self, openpmd_diag_supported=False):
        self.__openpmd_diag_supported = openpmd_diag_supported

    def gather(self, x, y, z, ex, ey, ez, bx, by, bz):
        """Gather all field components at the specified locations.

        Parameters
        ----------
        x : ndarray
            1D array containing the x position where to gather the fields.
        y : ndarray
            1D array containing the x position where to gather the fields.
        z : ndarray
            1D array containing the x position where to gather the fields.
        ex : ndarray
            1D array where the gathered Ex values will be stored.
        ey : ndarray
            1D array where the gathered Ey values will be stored
        ez : ndarray
            1D array where the gathered Ez values will be stored
        bx : ndarray
            1D array where the gathered Bx values will be stored
        by : ndarray
            1D array where the gathered By values will be stored
        bz : ndarray
            1D array where the gathered Bz values will be stored
        """
        self._gather(x, y, z, ex, ey, ez, bx, by, bz)

    def update(self, t, bunches):
        """Update field.

        This method provides the option of updating the field as a function of
        time and a list of particle bunches.

        Parameters
        ----------
        t : float
            A time value (in seconds).
        bunches : list
            List of `ParticleBunch`es.
        """
        self._update(t, bunches)

    def get_openpmd_diagnostics_data(self, global_time):
        """Get the data for including the field in the openPMD diagnostics.

        Parameters
        ----------
        global_time : float
            Current global time of the simulation.

        Returns
        -------
        Dict or None
            If openPMD output is supported by the field, returns a Dict with
            all the required data.
        """
        if self.__openpmd_diag_supported:
            return self._get_openpmd_diagnostics_data(global_time)

    def _gather(self, x, y, z, ex, ey, ez, bx, by, bz):
        """To be implemented by the subclasses."""
        raise NotImplementedError

    def _update(self, t, bunches):
        """(Optional) To be implemented by the subclasses."""
        pass

    def _get_openpmd_diagnostics_data(self):
        """To be implemented by the subclasses."""
        raise NotImplementedError
