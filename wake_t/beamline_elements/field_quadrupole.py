from typing import List, Literal
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
        that a positive value is focusing for electrons in the :math:`y`
        plane and defocusing in the :math:`x` plane.
    dt_bunch : float | str | List[float  |  str]
        The time step for evolving the particle bunches.
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
        dt_bunch: float | str | List[float | str],
        bunch_pusher: Literal['boris', 'rk4'] = 'boris',
        n_out: int | None = 1,
        name: str | None = 'quadrupole',
    ) -> None:
        self.foc_strength = foc_strength
        super().__init__(
            length=length,
            dt_bunch=dt_bunch,
            bunch_pusher=bunch_pusher,
            n_out=n_out,
            name=name,
            fields=[QuadrupoleField(foc_strength)],
            auto_dt_bunch=None,
        )
