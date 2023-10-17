import numpy as np
import scipy.constants as ct

from wake_t.fields.analytical_field import AnalyticalField
from wake_t.utilities.numba import prange


class FocusingBlowoutField(AnalyticalField):
    def __init__(self, density_function):
        self.density = density_function

        def e_x(x, y, xi, t, ex, k):
            for i in prange(x.shape[0]):
                ex[i] = ct.c * k[i] * x[i]

        def e_y(x, y, xi, t, ey, k):
            for i in prange(x.shape[0]):
                ey[i] = ct.c * k[i] * y[i]

        super().__init__(e_x=e_x, e_y=e_y)

    def _pre_gather(self, x, y, xi, t):
        z = t*ct.c + xi
        n_p = self.density(z)
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        k = (ct.m_e/(2*ct.e*ct.c))*w_p**2
        self.constants = k
