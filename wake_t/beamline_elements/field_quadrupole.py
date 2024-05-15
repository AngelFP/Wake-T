from typing import Literal, Optional

import numpy as np
import scipy.constants as ct

from .plasma_stage import DtBunchType
from .field_element import FieldElement
from wake_t.physics_models.em_fields.quadrupole import QuadrupoleField


class FieldQuadrupole(FieldElement):
    """
    Class defining a quadrupole as a field element.

    Parameters
    ----------
    length : float
        Length of the quadrupole lens in :math:`m`.
    foc_strength : float
        Focusing strength of the quadrupole in :math:`T/m`. Defined so
        that a positive value is focusing for electrons in the :math:`x`
        plane and defocusing in the :math:`y` plane.
    dt_bunch : float, str, or list of float and str
        The time step for evolving the particle bunches. If ``'auto'``, it will
        be automatically set to :math:`dt = T/(10*2*pi)`, where T is the
        betatron period of the particle with the lowest energy in the bunch.
        A list of values can also be provided. In this case, the list
        should have the same order as the list of bunches given to the
        ``track`` method.
    bunch_pusher : str
        The pusher used to evolve the particle bunches in time within
        the specified fields. Possible values are ``'rk4'`` (Runge-Kutta
        method of 4th order) or ``'boris'`` (Boris method).
    n_out : int, optional
        Number of times along the lens in which the particle distribution
        should be returned (A list with all output bunches is returned
        after tracking).
    name : str, optional
        Name of the quadrupole. This is only used for displaying the
        progress bar during tracking. By default, 'quadrupole'
    """

    def __init__(
        self,
        length: float,
        foc_strength: float,
        dt_bunch: Optional[DtBunchType] = 'auto',
        bunch_pusher: Literal['boris', 'rk4'] = 'boris',
        n_out: Optional[int] = 1,
        name: Optional[str] = 'quadrupole',
    ) -> None:
        self.foc_strength = foc_strength
        super().__init__(
            length=length,
            dt_bunch=dt_bunch,
            bunch_pusher=bunch_pusher,
            n_out=n_out,
            name=name,
            fields=[QuadrupoleField(foc_strength)],
            auto_dt_bunch=self._get_optimized_dt,
        )

    def _get_optimized_dt(self, beam):
        """ Get tracking time step. """
        # Get minimum gamma in the bunch (assumes px,py << pz).
        q_over_m = beam.q_species / beam.m_species
        min_gamma = np.sqrt(np.min(beam.pz)**2 + 1)
        w_x = np.sqrt(np.abs(q_over_m*ct.c * self.foc_strength/min_gamma))
        T_x = 1/w_x
        dt = 0.1*T_x
        return dt
