import numpy as np

from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class CombinedWakefield(Wakefield):
    def __init__(self, wakefield_list):
        super().__init__()
        self.wakefield_list = wakefield_list

    def Wx(self, x, y, xi, px, py, pz, q, t):
        wx = np.zeros(x.shape[0])
        for wf in self.wakefield_list:
            wx += wf.Wx(x, y, xi, px, py, pz, q, t)
        return wx

    def Wy(self, x, y, xi, px, py, pz, q, t):
        wy = np.zeros(x.shape[0])
        for wf in self.wakefield_list:
            wy += wf.Wy(x, y, xi, px, py, pz, q, t)
        return wy

    def Wz(self, x, y, xi, px, py, pz, q, t):
        wz = np.zeros(x.shape[0])
        for wf in self.wakefield_list:
            wz += wf.Wz(x, y, xi, px, py, pz, q, t)
        return wz
