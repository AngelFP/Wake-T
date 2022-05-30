class Wakefield():

    """ Base class for all wakefields """

    def __init__(self):
        self.openpmd_diag_supported = False

    def Wx(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError

    def Wy(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError

    def Wz(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError

    def get_openpmd_diagnostics_data(self):
        if self.openpmd_diag_supported:
            return self._get_openpmd_diagnostics_data()

    def _get_openpmd_diagnostics_data(self):
        raise NotImplementedError
