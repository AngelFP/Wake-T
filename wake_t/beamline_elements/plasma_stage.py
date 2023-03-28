""" This module contains the definition of the PlasmaStage class """

from typing import Optional, Union, Callable, List

import numpy as np
import scipy.constants as ct

import wake_t.physics_models.plasma_wakefields as wf
from wake_t.fields.base import Field
from .field_element import FieldElement


wakefield_models = {
    'simple_blowout': wf.SimpleBlowoutWakefield,
    'custom_blowout': wf.CustomBlowoutWakefield,
    'focusing_blowout': wf.FocusingBlowoutField,
    'cold_fluid_1d': wf.NonLinearColdFluidWakefield,
    'quasistatic_2d': wf.Quasistatic2DWakefield,
    'quasistatic_2d_ion': wf.Quasistatic2DWakefieldIon,
}


class PlasmaStage(FieldElement):
    """
    Main class for defining a plasma acceleration stage.

    Parameters
    ----------
    length : float
        Length of the plasma stage in m.
    density : float
        Plasma density in units of m^{-3}.
    wakefield_model : str or Field
        Wakefield model to be used. Possible values are ``'blowout'``,
        ``'custom_blowout'``, ``'focusing_blowout'``, ``'cold_fluid_1d'``
        and ``'quasistatic_2d'``. If ``None``, no wakefields will be
        computed.
    bunch_pusher : str
        The pusher used to evolve the particle bunches in time within
        the specified fields. Possible values are ``'rk4'`` (Runge-Kutta
        method of 4th order) or ``'boris'`` (Boris method).
    dt_bunch : float
        The time step for evolving the particle bunches. If ``None``, it
        will be automatically set to ``dt = T/(10*2*pi)``, where T is the
        smallest expected betatron period of the bunch along the plasma
        stage.
    n_out : int
        Number of times along the stage in which the particle distribution
        should be returned (A list with all output bunches is returned
        after tracking).
    name : str
        Name of the plasma stage. This is only used for displaying the
        progress bar during tracking. By default, ``'Plasma stage'``.
    **model_params
        Keyword arguments which will be given to the wakefield model. Each
        model requires a different set of parameters. See the documentation
        for each of them for more details.

    See Also
    --------
    wake_t.physics_models.plasma_wakefields.Quasistatic2DWakefield
    wake_t.physics_models.plasma_wakefields.NonLinearColdFluidWakefield

    """

    def __init__(
        self,
        length: float,
        density: Union[float, Callable[[float], float]],
        wakefield_model: Optional[str] = 'simple_blowout',
        bunch_pusher: Optional[str] = 'rk4',
        dt_bunch: Optional[Union[float, int]] = 'auto',
        n_out: Optional[int] = 1,
        name: Optional[str] = 'Plasma stage',
        external_fields: Optional[List[Field]] = [],
        **model_params
    ) -> None:
        self.density = self._get_density_profile(density)
        self.wakefield = self._get_wakefield(wakefield_model, model_params)
        self.external_fields = external_fields
        fields = []
        if self.wakefield is not None:
            fields.append(self.wakefield)
        fields.extend(self.external_fields)
        super().__init__(
            length=length,
            dt_bunch=dt_bunch,
            bunch_pusher=bunch_pusher,
            n_out=n_out,
            name=name,
            fields=fields,
            auto_dt_bunch=self._get_optimized_dt
        )

    def _get_density_profile(self, density):
        """ Get density profile function """
        if isinstance(density, float):
            def uniform_density(z):
                return np.ones_like(z) * density
            return uniform_density
        elif callable(density):
            return density
        else:
            raise ValueError(
                'Type {} not supported for density.'.format(type(density)))

    def _get_wakefield(self, model, model_params):
        """ Initialize and return corresponding wakefield model. """
        if model is None:
            return None
        elif isinstance(model, Field):
            return model
        elif model in wakefield_models:
            return wakefield_models[model](self.density, **model_params)
        else:
            raise ValueError(
                'Wakefield model "{}" not recognized.'.format(model))

    def _get_optimized_dt(self, beam):
        """ Get tracking time step. """
        # Get minimum gamma in the bunch (assumes px,py << pz).
        min_gamma = np.sqrt(np.min(beam.pz)**2 + 1)
        # calculate maximum focusing along stage.
        z = np.linspace(0, self.length, 100)
        n_p = self.density(z)
        q_over_m = beam.q_species / beam.m_species
        w_p = np.sqrt(max(n_p)*ct.e**2/(ct.m_e*ct.epsilon_0))
        max_kx = (ct.m_e/(2*ct.e*ct.c))*w_p**2
        w_x = np.sqrt(np.abs(q_over_m*ct.c * max_kx/min_gamma))
        period_x = 1/w_x
        dt = 0.1*period_x
        return dt
