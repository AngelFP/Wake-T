"""This module contains the base class for plasma wakefields in r-z geometry"""

from typing import Optional, Callable, List

import numpy as np
import scipy.constants as ct

from wake_t.particles.interpolation import gather_main_fields_cyl_linear
from .numerical_field import NumericalField
from wake_t.physics_models.laser.laser_pulse import LaserPulse


class RZWakefield(NumericalField):
    """Base class for plasma wakefields in r-z geometry.

    Parameters
    ----------
    density_function : callable
        Function of that returns the relative value of the plasma density
        at each `z` and `r` position.
    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated.
    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.
    n_r : int
        Number of grid elements along r to calculate the wakefields.
    n_xi : int
        Number of grid elements along xi to calculate the wakefields.
    dz_fields : float, optional
        Determines how often the plasma wakefields should be updated.
        For example, if ``dz_fields=10e-6``, the plasma wakefields are
        only updated every time the simulation window advances by
        10 micron. By default ``dz_fields=xi_max-xi_min``, i.e., the
        length the simulation box.
    species_rho_diags : bool, optional
        Whether the model should save the charge density of each plasma species
        separately.
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
        coarser) grid than the plasma wake. It is not necessary to specify
        both parameters. If one of them is not given, the resolution of
        the plasma grid with be used for that direction.
    laser_envelope_use_phase : bool, optional
        Determines whether to take into account the terms related to the
        longitudinal derivative of the complex phase in the envelope
        solver.
    field_diags : list, optional
        List of fields to save to openpmd diagnostics. By default ['rho', 'E',
        'B', 'a_mod', 'a_phase']. Can also be 'all'.
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
    particle_diags : list, optional
        List of particle quantities to save to openpmd diagnostics. By default
        [].
    model_name : str, optional
        Name of the wakefield model. This will be stored in the openPMD
        diagnostics.

    """

    def __init__(
        self,
        density_function: Callable[[float, float], float],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        dz_fields=None,
        species_rho_diags: Optional[bool] = False,
        laser: Optional[LaserPulse] = None,
        laser_evolution: Optional[bool] = True,
        laser_envelope_substeps: Optional[int] = 1,
        laser_envelope_nxi: Optional[int] = None,
        laser_envelope_nr: Optional[int] = None,
        laser_envelope_use_phase: Optional[bool] = True,
        field_diags: Optional[List[str]] = None,
        particle_diags: Optional[List[str]] = [],
        model_name: Optional[str] = "",
    ) -> None:
        dz_fields = xi_max - xi_min if dz_fields is None else dz_fields
        self.density_function = density_function
        self.laser = laser
        self.laser_evolution = laser_evolution
        self.laser_envelope_substeps = laser_envelope_substeps
        self.laser_envelope_nxi = laser_envelope_nxi
        self.laser_envelope_nr = laser_envelope_nr
        self.laser_envelope_use_phase = laser_envelope_use_phase
        self.species_rho_diags = species_rho_diags
        self.r_max = r_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.n_r = n_r
        self.n_xi = n_xi
        self.dr = r_max / n_r
        self.dxi = (xi_max - xi_min) / (n_xi - 1)
        if field_diags is None:
            self.field_diags = ["rho", "E", "B"]
            if self.laser is not None:
                self.field_diags += ["a_mod", "a_phase", "a"]
        else:
            self.field_diags = field_diags
        self.particle_diags = particle_diags
        self.model_name = model_name
        # If a laser is included, make sure it is evolved for the whole
        # duration of the plasma stage. See `force_even_updates` parameter.
        super().__init__(
            dt_update=dz_fields / ct.c,
            openpmd_diag_supported=True,
            force_even_updates=laser is not None,
        )

    def _initialize_properties(self, bunches):
        # Initialize laser.
        if self.laser is not None:
            self.laser.set_envelope_solver_params(
                self.xi_min,
                self.xi_max,
                self.r_max,
                self.n_xi,
                self.n_r,
                self.dt_update,
                self.laser_envelope_substeps,
                self.laser_envelope_nxi,
                self.laser_envelope_nr,
                self.laser_envelope_use_phase,
            )
            self.laser.initialize_envelope()

        # Initialize field arrays
        self.rho = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.rho_e = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.rho_i = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.chi = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.e_z = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.e_r = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.e_t = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.b_z = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.b_r = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.b_t = np.zeros((self.n_xi + 4, self.n_r + 4))
        self.r_fld = np.linspace(self.dr / 2, self.r_max - self.dr / 2, self.n_r)
        self.xi_fld = np.linspace(self.xi_min, self.xi_max, self.n_xi)

    def _evolve_properties(self, bunches):
        if self.laser is not None:
            # Evolve laser envelope
            if self.laser_evolution:
                self.laser.evolve(self.chi[2:-2, 2:-2], self.n_p)

    def _calculate_field(self, bunches):
        self.n_p = self.density_function(self.t * ct.c, 0.0)
        self.rho[:] = 0.0
        self.chi[:] = 0.0
        self.e_z[:] = 0.0
        self.e_r[:] = 0.0
        self.b_t[:] = 0.0
        if self.species_rho_diags:
            self.rho_e[:] = 0.0
            self.rho_i[:] = 0.0
        self._calculate_wakefield(bunches)

    def _calculate_wakefield(self, bunches):
        """To be implemented by the subclasses."""
        raise NotImplementedError

    def _gather(self, x, y, z, t, ex, ey, ez, bx, by, bz, bunch_name):
        gather_main_fields_cyl_linear(
            self.e_r,
            self.e_z,
            self.b_t,
            self.xi_fld[0],
            self.xi_fld[-1],
            self.r_fld[0],
            self.r_fld[-1],
            self.dxi,
            self.dr,
            x,
            y,
            z,
            ex,
            ey,
            ez,
            bx,
            by,
            bz,
        )

    def _get_openpmd_diagnostics_data(self, global_time):
        # Generate dictionary for openPMD diagnostics.
        diag_data = {}
        diag_data["fields"] = []
        diag_data["field_solver"] = "other"
        diag_data["field_solver_params"] = self.model_name
        diag_data["field_boundary"] = ["other"] * 4
        diag_data["field_boundary_params"] = ["none"] * 4
        diag_data["particle_boundary"] = ["other"] * 4
        diag_data["particle_boundary_params"] = ["none"] * 4
        diag_data["current_smoothing"] = "none"
        diag_data["charge_correction"] = "none"

        # Cell-centered in 'r' and node centered in 'z'.
        dr = np.abs(self.r_fld[1] - self.r_fld[0])
        dz = np.abs(self.xi_fld[1] - self.xi_fld[0])

        def add_diag(
            options,
            name,
            array,
            factor=None,
            comps=None,
            nghost=2,
            pos_xi=0.0,
            pos_r=0.5,
            grid_spacing_xi=dz,
            grid_spacing_r=dr,
            grid_labels_xi="z",
            grid_labels_r="r",
            grid_global_offset_xi=self.xi_min,
            grid_global_offset_r=0.0,
            attrs=None,
        ):

            pos = [pos_xi, pos_r]
            grid_spacing = [grid_spacing_xi, grid_spacing_r]
            grid_labels = [grid_labels_xi, grid_labels_r]
            grid_global_offset = [grid_global_offset_xi, grid_global_offset_r]

            r_begin = None
            r_end = None
            r_stride = None
            xi_begin = None
            xi_end = None
            xi_stride = None

            if "r_min" in options:
                r_begin = round(
                    (options["r_min"] - grid_global_offset[1]) / grid_spacing[1]
                    - pos[1]
                )
                r_begin = max(0, min(r_begin, array[0].shape[1] - 1))
            if "r_max" in options:
                r_end = (
                    round(
                        (options["r_max"] - grid_global_offset[1]) / grid_spacing[1]
                        - pos[1]
                    )
                    + 1
                )
                r_end = max(1, min(r_end, array[0].shape[1]))
            if "xi_min" in options:
                xi_begin = round(
                    (options["xi_min"] - grid_global_offset[0]) / grid_spacing[0]
                    - pos[0]
                )
                xi_begin = max(0, min(xi_begin, array[0].shape[0] - 1))
            if "xi_max" in options:
                xi_end = (
                    round(
                        (options["xi_max"] - grid_global_offset[0]) / grid_spacing[0]
                        - pos[0]
                    )
                    + 1
                )
                xi_end = max(1, min(xi_end, array[0].shape[0]))

            if r_begin is not None and r_begin != 0:
                grid_global_offset[1] += r_begin * grid_spacing[1]

            if xi_begin is not None and xi_begin != 0:
                grid_global_offset[0] += xi_begin * grid_spacing[0]

            if "r_stride" in options:
                r_stride = options["r_stride"]
                grid_spacing[1] *= r_stride

            if "xi_stride" in options:
                xi_stride = options["xi_stride"]
                grid_spacing[0] *= xi_stride

            grid_global_offset[0] += global_time * ct.c

            if "do_transpose" not in options or options["do_transpose"]:
                pos.reverse()
                grid_spacing.reverse()
                grid_labels.reverse()
                grid_global_offset.reverse()

            array2 = []

            for a in array:
                if nghost is not None and nghost != 0:
                    a = a[nghost:-nghost, nghost:-nghost]

                if (
                    r_begin is not None
                    or r_end is not None
                    or r_stride is not None
                    or xi_begin is not None
                    or xi_end is not None
                    or xi_stride is not None
                ):
                    a = a[xi_begin:xi_end:xi_stride, r_begin:r_end:r_stride]

                if "do_transpose" not in options or options["do_transpose"]:
                    a = a.T

                a = np.ascontiguousarray(a)

                if factor is not None:
                    a = np.ascontiguousarray(a * factor)

                array2.append(a)

            if name in diag_data["fields"]:
                name = name + "_" + str(options["diag_name"])

            diag_data["fields"].append(name)
            diag_data[name] = {}

            if comps is not None:
                diag_data[name]["comps"] = {}
                for comp, arr in zip(comps, array2):
                    diag_data[name]["comps"][comp] = {}
                    diag_data[name]["comps"][comp]["array"] = arr
                    diag_data[name]["comps"][comp]["position"] = pos
            else:
                diag_data[name]["array"] = array2[0]
                diag_data[name]["position"] = pos

            diag_data[name]["grid"] = {}
            diag_data[name]["grid"]["spacing"] = grid_spacing
            diag_data[name]["grid"]["labels"] = grid_labels
            diag_data[name]["grid"]["global_offset"] = grid_global_offset
            diag_data[name]["attributes"] = attrs if attrs is not None else {}

        rho_norm = self.n_p * (-ct.e)
        chi_norm = self.n_p * ct.e * ct.e * ct.mu_0 / ct.m_e

        all_field_data_pre = self.field_diags

        if not isinstance(all_field_data_pre, list):
            all_field_data_pre = [all_field_data_pre]

        all_field_data = [d for d in all_field_data_pre if not isinstance(d, str)]

        if any(isinstance(d, str) for d in all_field_data_pre):
            all_field_data.insert(
                0, {"field": [d for d in all_field_data_pre if isinstance(d, str)]}
            )

        for idx, i_field_data in enumerate(all_field_data):
            if isinstance(i_field_data, dict):
                options = i_field_data
                allowed_options = [
                    "field",
                    "r_min",
                    "r_max",
                    "xi_min",
                    "xi_max",
                    "r_stride",
                    "xi_stride",
                    "do_transpose",
                    "diag_name",
                ]
                for o in options.keys():
                    if o not in allowed_options:
                        raise ValueError(
                            f"Unknown field_diags option {o}, must be in {allowed_options}"
                        )
            else:
                raise ValueError(
                    f"field_diags must be list of str or dict, but got {i_field_data}"
                )

            if "diag_name" not in options:
                options["diag_name"] = "diag" + str(idx)

            fields = options["field"] if "field" in options else "all"
            if isinstance(fields, str):
                fields = [fields]

            available_fields = ["all"]

            # Add requested fields to diagnostics.
            available_fields.append("E")
            if "E" in fields or "all" in fields:
                add_diag(
                    options, "E", [self.e_r, self.e_t, self.e_z], comps=["r", "t", "z"]
                )
            available_fields.append("B")
            if "B" in fields or "all" in fields:
                add_diag(
                    options, "B", [self.b_r, self.b_t, self.b_z], comps=["r", "t", "z"]
                )
            available_fields.append("rho")
            if "rho" in fields or "all" in fields:
                add_diag(options, "rho", [self.rho], factor=rho_norm)
            if self.species_rho_diags:
                available_fields.append("rho_e")
                if "rho_e" in fields or "all" in fields:
                    add_diag(options, "rho_e", [self.rho_e], factor=rho_norm)
                available_fields.append("rho_i")
                if "rho_i" in fields or "all" in fields:
                    add_diag(options, "rho_i", [self.rho_i], factor=rho_norm)
            if self.laser is not None:
                if self.laser.polarization == "linear":
                    pol = np.array([1, 0j])
                else:
                    pol = np.array([np.sqrt(1 / 2), np.sqrt(1 / 2) * 1j])
                laser_attrs = {
                    "envelopeField": "normalized_vector_potential",
                    "angularFrequency": 2 * np.pi * ct.c / self.laser.l_0,
                    "polarization": pol,
                }

                available_fields.append("a_mod")
                if "a_mod" in fields or "all" in fields:
                    add_diag(
                        options,
                        "a_mod",
                        [np.abs(self.laser.get_envelope())],
                        nghost=None,
                        attrs={"polarization": self.laser.polarization},
                    )
                available_fields.append("a_phase")
                if "a_phase" in fields or "all" in fields:
                    add_diag(
                        options,
                        "a_phase",
                        [np.angle(self.laser.get_envelope())],
                        nghost=None,
                    )
                available_fields.append("a")
                if "a" in fields or "all" in fields:
                    add_diag(
                        options,
                        "a",
                        [self.laser.get_envelope()],
                        nghost=None,
                        attrs=laser_attrs,
                    )
                if self.laser.use_subgrid:
                    available_fields.append("a_subgrid")
                    if "a_subgrid" in fields or "all" in fields:
                        add_diag(
                            options,
                            "a_subgrid",
                            [self.laser._a_env[:-2]],
                            nghost=None,
                            grid_spacing_xi=self.laser.subgrid_params["subgrid"]["dz"],
                            grid_spacing_r=self.laser.subgrid_params["subgrid"]["dr"],
                            attrs=laser_attrs,
                        )
                available_fields.append("chi")
                if "chi" in fields or "all" in fields:
                    add_diag(options, "chi", [self.chi], factor=chi_norm)
            if "ion_densities" in self.__dict__:
                for i in range(len(self.ion_atomic_number)):
                    for j in range(self.ion_atomic_number[i] + 1):
                        name = f"n_{self.ion_names[i]}_ionlevel_{j}"
                        available_fields.append(name)
                        if name in fields or "all" in fields:
                            add_diag(
                                options,
                                name,
                                [self.ion_densities[self.ion_start_index[i] + j, :, :]],
                                factor=self.n_p,
                            )
            if "elec_density" in self.__dict__:
                available_fields.append("n_electrons")
                if "n_electrons" in fields or "all" in fields:
                    add_diag(
                        options, "n_electrons", [self.elec_density], factor=self.n_p
                    )
            if "q_bunch" in self.__dict__:
                available_fields.append("charge_profile")
                if "charge_profile" in fields or "all" in fields:
                    s_d = ct.c / np.sqrt(ct.e**2 * self.n_p / (ct.m_e * ct.epsilon_0))
                    # Inverse of the normalization applied when depositing the
                    # beam charge onto the grid (q_bunch is stored normalized).
                    charge_norm = 2 * np.pi * ct.e * self.dr * self.dxi * s_d * self.n_p
                    add_diag(
                        options, "charge_profile", [self.q_bunch], factor=charge_norm
                    )
            for f in fields:
                if f not in available_fields:
                    raise ValueError(
                        f"Diagnostic field {f} is not available! "
                        + f"Available fields are {available_fields}"
                    )

        return diag_data
