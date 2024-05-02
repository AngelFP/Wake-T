"""
This module contains the definition of the PlasmaRamp class as well as
predefined ramp profiles.

"""

from typing import Optional, Union, Callable, Literal
from functools import partial

import numpy as np

from .plasma_stage import PlasmaStage, DtBunchType


# Define type alias for the ramp profiles.
Profile = Callable[[float, float, float, float, float], float]


def inverse_square_profile(z, decay_length=None, density_top=None,
                           density_down=None, position_down=None):
    if decay_length is None:
        decay_length = position_down / (np.sqrt(density_top/density_down) - 1)
    return density_top / np.square(1 + z/decay_length)


def exponential_profile(z, decay_length=None, density_top=None,
                        density_down=None, position_down=None):
    if decay_length is None:
        decay_length = position_down / np.log(density_top / density_down)
    return density_top * np.exp(-z / decay_length)


def gaussian_profile(z, decay_length=None, density_top=None,
                     density_down=None, position_down=None):
    if decay_length is None:
        decay_length = (position_down /
                        np.sqrt(2*np.log(density_top / density_down)))
    return density_top * np.exp(-z**2/(2*decay_length**2))


ramp_profiles = {
    'inverse_square': inverse_square_profile,
    'exponential': exponential_profile,
    'gaussian': gaussian_profile
}


class PlasmaRamp(PlasmaStage):
    """
    Class defining a plasma density ramp.

    This elements is a subclass of :class:`PlasmaStage` that exposes
    convenient attributes for defining a plasma density ramp, such as
    predefined density profiles.

    Parameters
    ----------
    length : float
        Length of the plasma ramp in m.
    profile : string or callable
        Longitudinal density profile of the ramp. Possible string values
        are ``'gaussian'``, ``'inverse square'`` and ``'exponential'``.
        A callable might also be provided as a function of the form
        ``'def func(z, decay_length, density_top, density_down,
        position_down)'`` that returns the density value at the given
        position ``z``.
    ramp_type : string
        Possible types are ``'upramp'`` and ``'downramp'``.
    wakefield_model : str
        Wakefield model to be used. Possible values are ``'blowout'``,
        ``'custom_blowout'``, ``'focusing_blowout'``, ``'cold_fluid_1d'`` and
        ``'quasistatic_2d'``.
    decay_length : float
        Optional. Characteristic decay length of the ramp. If not provided,
        it will be determined from ``plasma_dens_top``, ``plasma_dens_down``
        and ``position_down``.
    plasma_dens_top : float
        Optional. Needed only if `decay_length=None`. Plasma density at the
        beginning (end) of the downramp (upramp) in units of :math:`m^{-3}`.
    plasma_dens_down : float
        Optional. Needed only if ``decay_length=None``. Plasma density at the
        position ``position_down`` in units of :math:`m^{-3}`.
    position_down : float
        Optional. Needed only if ``decay_length=None``. Position where the
        plasma density will be equal to ``plasma_dens_down`` measured from
        the beginning (end) of the downramp (upramp). If not provided,
        the ``length`` value is assigned.
    bunch_pusher : str
        The pusher used to evolve the particle bunches in time within
        the specified fields. Possible values are ``'rk4'`` (Runge-Kutta
        method of 4th order) or ``'boris'`` (Boris method).
    dt_bunch : float
        The time step for evolving the particle bunches. If ``None``, it will
        be automatically set to :math:`dt = T/(10*2*pi)`, where T is the
        smallest expected betatron period of the bunch along the plasma stage.
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
        Number of times along the stage in which the particle distribution
        should be returned (A list with all output bunches is returned
        after tracking).
    name : str
        Name of the plasma ramp. This is only used for displaying the
        progress bar during tracking. By default, ``'Plasma ramp'``.
    **model_params
        Keyword arguments which will be given to the wakefield model. Each
        model requires a different set of parameters. See :class:`PlasmaStage`
        documentation for more details.

    See Also
    --------
    PlasmaStage

    """

    def __init__(
        self,
        length: float,
        profile: Optional[Union[str, Profile]] = 'inverse_square',
        ramp_type: Optional[str] = 'upramp',
        wakefield_model: Optional[str] = 'focusing_blowout',
        decay_length: Optional[float] = None,
        plasma_dens_top: Optional[float] = None,
        plasma_dens_down: Optional[float] = None,
        position_down: Optional[float] = None,
        bunch_pusher: Optional[Literal['boris', 'rk4']] = 'boris',
        dt_bunch: Optional[DtBunchType] = 'auto',
        push_bunches_before_diags: Optional[bool] = True,
        n_out: Optional[int] = 1,
        name: Optional[str] = 'Plasma ramp',
        **model_params
    ) -> None:
        self.ramp_type = ramp_type
        if position_down is None:
            position_down = length
        # If a function profile is not provided, generate from presets.
        if not callable(profile):
            if profile in ramp_profiles:
                profile = ramp_profiles[profile]
            else:
                raise ValueError(
                    'Ramp profile "{}" not recognized'.format(profile))
        profile = partial(
            profile, decay_length=decay_length,
            density_top=plasma_dens_top, density_down=plasma_dens_down,
            position_down=position_down)
        self.profile = profile
        super().__init__(
            length=length,
            density=self.ramp_profile,
            wakefield_model=wakefield_model,
            bunch_pusher=bunch_pusher,
            dt_bunch=dt_bunch,
            push_bunches_before_diags=push_bunches_before_diags,
            n_out=n_out,
            name=name,
            **model_params
        )

    def ramp_profile(self, z):
        """ Return the density value at a certain z location. """
        # For an upramp, invert z coordinate.
        if self.ramp_type == 'upramp':
            z = self.length - z
        return self.profile(z)
