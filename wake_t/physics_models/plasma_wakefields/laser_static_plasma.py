from typing import List, Optional, Callable

import numpy as np
import scipy.constants as ct

from wake_t.fields.rz_wakefield import RZWakefield
from wake_t.physics_models.laser.laser_pulse import LaserPulse

from wake_t.utilities.other import ProfStart, ProfStop


class LaserStaticPlasma(RZWakefield):
    """
    This model can be used to propagate laser pulses through a vacuum or a
    specially varying static plasma density. Specifically, it can be used
    when the laser is too weak to form a plasma wake, namely when a0 << 1.
    This model does not calculate any electric or magnetic fields and should
    not be used with a particle bunch.

    Parameters
    ----------
    density_function : callable
        Function that returns the density value at the given position z.
        This parameter is given by the `PlasmaStage` and does not need
        to be specified by the user.
    r_max : float
        Maximum radial position up to which laser will be calculated.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        laser will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        laser will be calculated.
    n_r : int
        Number of grid elements along r to calculate the laser.
    n_xi : int
        Number of grid elements along xi to calculate the laser.
    dz_fields : float, optional
        Determines how often the laser and plasma refractive index should
        be updated. If dz_fields=0 (default value), the laser is calculated
        at every step of the Runge-Kutta solver for the beam particle evolution
        (most expensive option). If specified, the laser is only
        updated in steps determined by dz_fields. For example, if
        dz_fields=10e-6, the laser is only updated every time
        the simulation window advances by 10 micron. By default, if not
        specified, the value of `dz_fields` will be `xi_max-xi_min`, i.e.,
        the length the simulation box.
    r_max_plasma : float, optional
        Maximum radial extension of the plasma column. If ``None``, the
        plasma extends up to the ``r_max`` boundary of the simulation box.
    laser : LaserPulse, optional
        Laser driver of the plasma stage.
    laser_evolution : bool, optional
        If True (default), the laser pulse is evolved
        using a laser envelope model. If False, the pulse envelope stays
        unchanged throughout the computation.
    laser_envelope_substeps : int, optional
        Number of substeps of the laser envelope solver per `dz_fields`.
        The time step of the envelope solver is therefore
        `dz_fields / c / laser_envelope_substeps`.
    laser_envelope_nxi, laser_envelope_nr : int, optional
        If given, the laser envelope will run in a grid of size
        (`laser_envelope_nxi`, `laser_envelope_nr`) instead
        of (`n_xi`, `n_r`). This allows the laser to run in a finer (or
        coarser) grid than the plasma refractive index. It is not necessary
        to specify both parameters. If one of them is not given, the resolution
        of the plasma grid will be used for that direction.
    laser_envelope_use_phase : bool, optional
        Determines whether to take into account the terms related to the
        longitudinal derivative of the complex phase in the envelope
        solver.
    field_diags : list, optional
        List of fields to save to openpmd diagnostics.
        By default ['rho', 'chi', 'a_mod', 'a_phase', 'a']. Can also be 'all'.
        Each entry can also be a dict to give more precise control of the
        outputted data. The dict can have the following keys:
        {
            "field" : list of field names to output
            "r_min" , "r_max" , "xi_min" , "xi_max" :
                Cut the diagnostic box in r and xi.
            "r_stride" , "xi_stride" :
                Add a stride in units of dr and dxi to r and xi.
            "do_transpose" :
                Wheater to transpose data for the output (default True).
            "diag_name" :
                Name to append to the field name if there are multiple
                diagnostics with the same field.
        }

    See Also
    --------
    Quasistatic2DWakefieldIon

    """

    def __init__(
        self,
        density_function: Callable[[float], float],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        dz_fields: Optional[float] = None,
        r_max_plasma: Optional[float] = None,
        laser: Optional[LaserPulse] = None,
        laser_evolution: Optional[bool] = True,
        laser_envelope_substeps: Optional[int] = 1,
        laser_envelope_nxi: Optional[int] = None,
        laser_envelope_nr: Optional[int] = None,
        laser_envelope_use_phase: Optional[bool] = True,
        field_diags: Optional[List[str]] = None,
    ) -> None:
        if field_diags is None:
            field_diags = ["rho", "chi", "a_mod", "a_phase", "a"]
        super().__init__(
            density_function=density_function,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            laser=laser,
            laser_evolution=laser_evolution,
            laser_envelope_substeps=laser_envelope_substeps,
            laser_envelope_nxi=laser_envelope_nxi,
            laser_envelope_nr=laser_envelope_nr,
            laser_envelope_use_phase=laser_envelope_use_phase,
            field_diags=field_diags,
            model_name="laser_in_vacuum",
        )
        self.r_max_plasma = r_max_plasma

    def _calculate_wakefield(self, bunches):

        ProfStart("laser_static_plasma")

        # Use a reference plasma density for internal units to be able to simulate vacuum
        self.n_p = 1e23

        # Get laser envelope
        if self.laser is not None:
            a_env = np.abs(self.laser.get_envelope())
            # If linearly polarized, divide by sqrt(2) so that the
            # ponderomotive force on the plasma particles is correct.
            if self.laser.polarization == "linear":
                a_env /= np.sqrt(2)
        else:
            a_env = np.zeros((self.n_xi, self.n_r))

        density_elec = self.density_function(self.t * ct.c, self.r_fld)
        if self.r_max_plasma is not None:
            density_elec = np.where(self.r_fld > self.r_max_plasma, 0, density_elec)
        # Gamma for particles with no momentum but that see a laser
        gamma_elec = 0.5 * (2 + a_env**2)
        # Convert to normalized units
        self.rho[2:-2, 2:-2] = (density_elec / self.n_p) * gamma_elec
        self.chi[2:-2, 2:-2] = density_elec / self.n_p

        ProfStop("laser_static_plasma")
