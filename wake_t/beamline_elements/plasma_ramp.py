"""
This module contains the definition of the PlasmaRamp class as well as
predefined ramp profiles.

"""

from functools import partial
import numpy as np

from wake_t.beamline_elements import PlasmaStage


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

    """ Convenience class to define a plasma density ramp """

    def __init__(self, length, profile='inverse_square', ramp_type='upramp',
                 wakefield_model='focusing_blowout', decay_length=None,
                 plasma_dens_top=None, plasma_dens_down=None,
                 position_down=None, n_out=1, **model_params):
        """
        Initialize plasma ramp.

        Parameters:
        -----------
        length : float
            Length of the plasma ramp in m.

        profile : string or callable
            Longitudinal density profile of the ramp. Possible string values
            are 'gaussian', 'inverse square' and 'exponential'. A callable
            might also be provided as a function of the form 'def func(z)'
            which returns the density value at the given position z.

        ramp_type : string
            Possible types are 'upramp' and 'downramp'.

        wakefield_model : str
            Wakefield model to be used. Possible values are 'blowout',
            'custom_blowout', 'focusing_blowout', 'cold_fluid_1d' and
            'quasistatic_2d'.

        decay_length : float
            Optional. Characteristic decay length of the ramp. If not provided,
            it will be determined from `plasma_dens_top`, `plasma_dens_down`
            and `position_down`.

        plasma_dens_top : float
            Optional. Needed only if `decay_length=None`. Plasma density at the
            beginning (end) of the downramp (upramp) in units of m^{-3}.

        plasma_dens_down : float
            Optional. Needed only if `decay_length=None`. Plasma density at the
            position `position_down` in units of m^{-3}.

        position_down : float
            Optional. Needed only if `decay_length=None`. Position where the
            plasma density will be equal to `plasma_dens_down` measured from
            the beginning (end) of the downramp (upramp). If not provided,
            the `length` value is assigned.

        n_out : int
            Number of times along the stage in which the particle distribution
            should be returned (A list with all output bunches is returned
            after tracking).

        **model_params
            Keyword arguments which will be given to the wakefield model. Each
            model requires a different set of parameters. See `PlasmaStage`
            documentation for more details.

        """

        self.ramp_type = ramp_type
        if position_down is None:
            position_down = length
        # If a function profile is not provided, generate from presets.
        if not callable(profile):
            if profile in ramp_profiles:
                ramp_profile = ramp_profiles[profile]
                profile = partial(
                    ramp_profile, decay_length=decay_length,
                    density_top=plasma_dens_top, density_down=plasma_dens_down,
                    position_down=position_down)
            else:
                raise ValueError(
                    'Ramp profile "{}" not recognized'.format(profile))
        self.profile = profile
        super().__init__(
            length, self.ramp_profile, wakefield_model, n_out, **model_params)

    def ramp_profile(self, z):
        """ Return the density value at a certain z location. """
        # For an upramp, invert z coordinate.
        if self.ramp_type == 'upramp':
            z = self.length - z
        return self.profile(z)
