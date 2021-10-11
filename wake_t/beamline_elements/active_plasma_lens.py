""" This module contains the definition of the ActivePlasmaLens class """

import numpy as np
import scipy.constants as ct

from wake_t.beamline_elements import PlasmaStage
from wake_t.physics_models.plasma_wakefields import (
    PlasmaLensField, CombinedWakefield)


class ActivePlasmaLens(PlasmaStage):

    """ Convenience class to define an active plasma lens. """

    def __init__(self, length, foc_strength, wakefields=False, density=None,
                 wakefield_model='quasistatic_2d', n_out=1, **model_params):
        """
        Initialize plasma lens.

        Parameters:
        -----------
        length : float
            Length of the plasma lens in m.

        foc_strength : float
            Focusing strength of the plasma lens in T/m.

        wakefields : bool
            If True, the beam-induced wakefields in the plasma lens will be
            computed using the model specified in 'wakefield_model' and
            taken into account for the beam evolution.

        wakefield_model : str
            Name of the model which should be used for computing the
            beam-induced wakefields. Recommended models are 'cold_fluid_1d' or
            'quasistatic_2d'. See `PlasmaStage` documentation for other
            possibilities.

        density : float or callable
            Optional. Required only if `wakefields=true`. Plasma density
            of the APL in units of m^{-3}. See `PlasmaStage` documentation
            for more details.

        n_out : int
            Number of times along the lens in which the particle distribution
            should be returned (A list with all output bunches is returned
            after tracking).

        **model_params
            Optional. Required only if `wakefields=true`. Keyword arguments
            which will be given to the wakefield model. See `PlasmaStage`
            documentation for more details.

        """

        self.foc_strength = foc_strength
        self.wakefields = wakefields
        if density is None:
            if wakefields:
                raise ValueError(
                    'A density value is required to compute plasma wakefields')
            else:
                # Give any value (it won't be used.)
                density = 0.
        super().__init__(length, density, wakefield_model, n_out,
                         **model_params)

    def _get_wakefield(self, model, model_params):
        """ Return the APL field, including plasma wakefields if needed. """
        wakefield = PlasmaLensField(self.foc_strength)
        if self.wakefields:
            wf_model = super()._get_wakefield(model, model_params)
            wakefield = CombinedWakefield([wakefield, wf_model])
        return wakefield

    def _get_optimized_dt(self, beam):
        """ Get tracking time step. """
        # If plasma wakefields are active, use default dt.
        if self.wakefields:
            dt = super()._get_optimized_dt(beam)
        # Otherwise, determine dt from the APL focusing strength.
        else:
            gamma = np.sqrt(1 + beam.px**2 + beam.py**2 + beam.pz**2)
            mean_gamma = np.average(gamma, weights=beam.q)
            w_x = np.sqrt(ct.e*ct.c/ct.m_e * self.foc_strength/mean_gamma)
            T_x = 1/w_x
            dt = 0.1*T_x
        return dt
