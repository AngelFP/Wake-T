""" This module contains the definition of the ActivePlasmaLens class """

from typing import Optional, Union, Callable, Literal

import numpy as np
import scipy.constants as ct

from .plasma_stage import PlasmaStage, DtBunchType
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
    dt_bunch : float
        The time step for evolving the particle bunches. If ``None``, it will
        be automatically set to :math:`dt = T/(10*2*pi)`, where T is the
        smallest expected betatron period of the bunch along the plasma lens
        (T is calculated from `foc_strength` if `wakefields=False`,
        otherwise the focusing strength of a blowout is used).
        A list of values can also be provided. In this case, the list
        should have the same order as the list of bunches given to the
        ``track`` method.
    push_bunches_before_diags : bool, optional
        Whether to push the bunches before saving them to the diagnostics.
        Since the time step of the diagnostics can be different from that
        of the bunches, it could happen that the bunches appear in the
        diagnostics as they were at the last push, but not at the actual
        time of the diagnostics. Setting this parameter to ``True``
        (default) ensures that an additional push is given to all bunches
        to evolve them to the diagnostics time before saving.
        This additional push will always have a time step smaller than
        the the time step of the bunch, so it has no detrimental impact
        on the accuracy of the simulation. However, it could make
        convergence studies more difficult to interpret,
        since the number of pushes will depend on `n_diags`. Therefore,
        it is exposed as an option so that it can be disabled if needed.
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
        bunch_pusher: Optional[Literal['boris', 'rk4']] = 'boris',
        dt_bunch: Optional[DtBunchType] = 'auto',
        push_bunches_before_diags: Optional[bool] = True,
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
            push_bunches_before_diags=push_bunches_before_diags,
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
