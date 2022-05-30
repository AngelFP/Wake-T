import numpy as np
import scipy.constants as ct

from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class PlasmaLensField(Wakefield):
    def __init__(self, dB_r):
        super().__init__()
        self.dB_r = dB_r  # [T/m]

    def Wx(self, x, y, xi, px, py, pz, q, t):
        # By = -x*self.dB_r
        gamma = np.sqrt(1 + px*px + py*py + pz*pz)
        return - pz*ct.c/gamma * (-x*self.dB_r)

    def Wy(self, x, y, xi, px, py, pz, q, t):
        # Bx = y*self.dB_r
        gamma = np.sqrt(1 + px*px + py*py + pz*pz)
        return pz*ct.c/gamma * y*self.dB_r

    def Wz(self, x, y, xi, px, py, pz, q, t):
        gamma = np.sqrt(1 + px*px + py*py + pz*pz)
        return (px*(-x*self.dB_r) - py*y*self.dB_r)*ct.c/gamma


class PlasmaLensFieldRelativistic(Wakefield):
    def __init__(self, k_x):
        super().__init__()
        self.k_x = k_x  # [T/m]

    def Wx(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.k_x*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.k_x*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        return np.zeros(len(x))
