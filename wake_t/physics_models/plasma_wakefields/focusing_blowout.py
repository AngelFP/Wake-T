import numpy as np
import scipy.constants as ct

from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class FocusingBlowoutField(Wakefield):
    def __init__(self, density_function):
        super().__init__()
        self.density_function = density_function

    def Wx(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        return np.zeros(len(xi))

    def calculate_focusing(self, xi, t):
        z = t*ct.c + xi  # z postion of each particle at time t
        n_p = self.density_function(z)
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        return (ct.m_e/(2*ct.e*ct.c))*w_p**2
