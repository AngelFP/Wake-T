import numpy as np
import scipy.constants as ct

from .solver import calculate_wakefields
from wake_t.fields.rz_wakefield import RZWakefield


class Quasistatic2DWakefield(RZWakefield):

    def __init__(self, density_function, laser=None, laser_evolution=True,
                 laser_envelope_substeps=1, r_max=None, xi_min=None,
                 xi_max=None, n_r=100, n_xi=100, ppc=2, dz_fields=None,
                 r_max_plasma=None, parabolic_coefficient=0., p_shape='cubic',
                 max_gamma=10, plasma_pusher='rk4'):
        self.ppc = ppc
        self.r_max_plasma = r_max_plasma
        self.parabolic_coefficient = self._get_parabolic_coefficient_fn(
            parabolic_coefficient)
        self.p_shape = p_shape
        self.max_gamma = max_gamma
        self.plasma_pusher = plasma_pusher
        super().__init__(
            density_function=density_function,
            laser=laser,
            laser_evolution=laser_evolution,
            laser_envelope_substeps=laser_envelope_substeps,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            model_name='quasistatic_2d'
        )

    def _calculate_wakefield(self, bunches):
        parabolic_coefficient = self.parabolic_coefficient(self.t*ct.c)

        # Get laser envelope
        if self.laser is not None:
            a_env = np.abs(self.laser.get_envelope()) ** 2
        else:
            a_env = np.zeros((self.n_xi, self.n_r))

        # Calculate plasma wakefields
        calculate_wakefields(
            a_env, bunches, self.r_max, self.xi_min, self.xi_max,
            self.n_r, self.n_xi, self.ppc, self.n_p,
            r_max_plasma=self.r_max_plasma,
            parabolic_coefficient=parabolic_coefficient,
            p_shape=self.p_shape, max_gamma=self.max_gamma,
            plasma_pusher=self.plasma_pusher,
            fld_arrays=[self.rho, self.chi, self.e_r, self.e_z, self.b_t,
                        self.xi_fld, self.r_fld])

    def _get_parabolic_coefficient_fn(self, parabolic_coefficient):
        """ Get parabolic_coefficient profile function """
        if isinstance(parabolic_coefficient, float):
            def uniform_parabolic_coefficient(z):
                return np.ones_like(z) * parabolic_coefficient
            return uniform_parabolic_coefficient
        elif callable(parabolic_coefficient):
            return parabolic_coefficient
        else:
            raise ValueError(
                'Type {} not supported for parabolic_coefficient.'.format(
                    type(parabolic_coefficient)))
