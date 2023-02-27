""" This module contains the definition of the ActivePlasmaLens class """

from typing import Optional, Union, Callable

import numpy as np
import scipy.constants as ct

from wake_t.beamline_elements import PlasmaStage
from wake_t.physics_models.em_fields.linear_b_theta import LinearBThetaField


class ActivePlasmaLens(PlasmaStage):
    """
    Class defining an active plasma lens.

    This elements is a subclass of :class:`PlasmaStage`, where a linear
    azimuthal magnetic field is added externally. It also includes
    convenient methods to specify the field gradient and whether the plasma
    wakefields should be taken into account.

    Parameters
    ----------
    length : float
        Length of the plasma lens in :math:`m`.
    foc_strength : float
        Focusing strength of the plasma lens in :math:`T/m`. Defined so that
        a positive value is focusing for electrons.
    wakefields : bool
        If ``True``, the beam-induced wakefields in the plasma lens will be
        computed using the model specified in ``'wakefield_model'`` and
        taken into account for the beam evolution.
    wakefield_model : str
        Name of the model which should be used for computing the
        beam-induced wakefields. Recommended models are ``'cold_fluid_1d'`` or
        ``'quasistatic_2d'``. See :class:`PlasmaStage` documentation for other
        possibilities.
    density : float or callable
        Optional. Required only if ``wakefields=True``. Plasma density
        of the APL in units of :math:`m^{-3}`. See :class:`PlasmaStage`
        documentation for more details.
    bunch_pusher : str
        The pusher used to evolve the particle bunches in time within
        the specified fields. Possible values are ``'rk4'`` (Runge-Kutta
        method of 4th order) or ``'boris'`` (Boris method).
    n_out : int
        Number of times along the lens in which the particle distribution
        should be returned (A list with all output bunches is returned
        after tracking).
    name : str
        Name of the plasma lens. This is only used for displaying the
        progress bar during tracking. By default, ``'Active plasma lens'``.
    **model_params
        Optional. Required only if ``wakefields=True``. Keyword arguments
        which will be given to the wakefield model. See :class:`PlasmaStage`
        documentation for more details.

    See Also
    --------
    PlasmaStage

    """

    def __init__(
        self,
        length: float,
        foc_strength: float,
        wakefields: bool = False,
        density: Optional[Union[float, Callable[[float], float]]] = None,
        wakefield_model: Optional[str] = 'quasistatic_2d',
        bunch_pusher: Optional[str] = 'rk4',
        dt_bunch: Optional[Union[float, int]] = 'auto',
        n_out: Optional[int] = 1,
        name: Optional[str] = 'Active plasma lens',
        **model_params
    ) -> None:
        self.foc_strength = foc_strength
        self.wakefields = wakefields
        if not self.wakefields:
            wakefield_model = None
        if density is None:
            if wakefields:
                raise ValueError(
                    'A density value is required to compute plasma wakefields')
            else:
                # Give any value (it won't be used.)
                density = 0.
        self.apl_field = LinearBThetaField(-self.foc_strength)
        super().__init__(
            length=length,
            density=density,
            wakefield_model=wakefield_model,
            bunch_pusher=bunch_pusher,
            dt_bunch=dt_bunch,
            n_out=n_out,
            name=name,
            external_fields=[self.apl_field],
            **model_params
        )

    def _get_optimized_dt(self, beam):
        """ Get tracking time step. """
        # If plasma wakefields are active, use default dt.
        if self.wakefields:
            dt = super()._get_optimized_dt(beam)
        # Otherwise, determine dt from the APL focusing strength.
        else:
            # Get minimum gamma in the bunch (assumes px,py << pz).
            q_over_m = beam.q_species / beam.m_species
            min_gamma = np.sqrt(np.min(beam.pz)**2 + 1)
            w_x = np.sqrt(np.abs(q_over_m*ct.c * self.foc_strength/min_gamma))
            T_x = 1/w_x
            dt = 0.1*T_x
        return dt
