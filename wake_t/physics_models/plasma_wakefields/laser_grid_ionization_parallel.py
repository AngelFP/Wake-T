from typing import Dict, List, Optional, Callable, Union
import math
import numpy as np
import scipy.constants as ct

from wake_t.fields.rz_wakefield import RZWakefield
from wake_t.physics_models.laser.laser_pulse import LaserPulse
from wake_t.utilities.numba import num_threads
from wake_t.utilities.other import ProfStart, ProfStop

import os
import cppimport

cppmodule = cppimport.imp_from_filepath(
    os.path.join(os.path.dirname(__file__), "parallel_solver.cpp")
)


class LaserGridIonizationParallel(RZWakefield):
    """
    This model can be used to propagate laser pulses through a neutral gas or
    plasma that gets ionized from the laser. Specifically, it can be used
    when the laser is too weak to form a plasma wake, namely when a0 << 1.
    This model does not calculate any electric or magnetic fields and should
    not be used with a particle bunch.

    Parameters
    ----------
    density_function : callable
        Function that returns the initial electron density value at the
        given position z. This parameter is given by the `PlasmaStage`
        and does not need to be specified by the user.
        Usually this should be zero.
    ion_species: list
        List of ion species names. Currently supported are H, He, N and Ar.
    ion_densities: list
        List of initial densities for each ion species. Can be a function of z and r.
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
    species_rho_diags : bool, optional
        Whether the model should save the charge density of each plasma species
        separately.
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
        One can get the per-ion-level density using 'n_<ion_species>_ionlevel_<level>'
        and the electron density using n_electrons.
        Note that E, B and rho are not set by this model.
        By default ['chi', 'a_mod', 'a_phase', 'a']. Can also be 'all'.
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
        density_function: Callable[[float, float], float],
        ion_species: List[Union[str, Dict]],
        ion_densities: List[Union[float, Callable[[float, float], float]]],
        r_max: float,
        xi_min: float,
        xi_max: float,
        n_r: int,
        n_xi: int,
        dz_fields: Optional[float] = None,
        species_rho_diags: Optional[bool] = False,
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
            field_diags = ["chi", "a_mod", "a_phase", "a"]
        super().__init__(
            density_function=density_function,
            r_max=r_max,
            xi_min=xi_min,
            xi_max=xi_max,
            n_r=n_r,
            n_xi=n_xi,
            dz_fields=dz_fields,
            species_rho_diags=species_rho_diags,
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

        ion_species_lookup = {
            "H": {"mass_u": 1.007975, "ionization_energy_eV": [13.59843449]},
            "H2": {
                "mass_u": 2.015950,
                "ionization_energy_eV": [15.42593000, 13.59843449],
            },
            "He": {
                "mass_u": 4.002602,
                "ionization_energy_eV": [24.58738880, 54.4177650],
            },
            "N": {
                "mass_u": 14.006855,
                "ionization_energy_eV": [
                    14.53413,
                    29.60125,
                    47.4453,
                    77.4735,
                    97.8901,
                    552.06732,
                    667.046116,
                ],
            },
            "Ar": {
                "mass_u": 39.948,
                "ionization_energy_eV": [
                    15.7596117,
                    27.62967,
                    40.735,
                    59.58,
                    74.84,
                    91.290,
                    124.41,
                    143.4567,
                    422.60,
                    479.76,
                    540.4,
                    619.0,
                    685.5,
                    755.13,
                    855.5,
                    918.375,
                    4120.6656,
                    4426.2228,
                ],
            },
        }

        if not isinstance(ion_species, list):
            ion_species = [ion_species]

        if not isinstance(ion_densities, list):
            ion_densities = [ion_densities]

        self.ion_species = []
        self.ion_names = []

        for species in ion_species:
            if isinstance(species, str):
                if species not in ion_species_lookup:
                    raise ValueError(
                        f"ion_species {species} not found in {list(ion_species_lookup.keys())}"
                    )
                self.ion_species.append(ion_species_lookup[species])
                name = species
            else:
                if (
                    not isinstance(species, dict)
                    or species.keys() != ion_species_lookup["He"].keys()
                ):
                    raise ValueError(
                        f"ion_species {species} must be str or dict like {ion_species_lookup['He']}"
                    )
                self.ion_species.append(species)
                name = "Ion"
            prev_name_cout = self.ion_names.count(name)
            self.ion_names.append(
                name if prev_name_cout == 0 else name + str(prev_name_cout)
            )

        self.initial_ion_densities = []

        for density in ion_densities:
            if isinstance(density, float):

                def uniform_density(z, r, density=density):
                    return np.ones_like(z) * np.ones_like(r) * density

                self.initial_ion_densities.append(uniform_density)
            elif callable(density):
                self.initial_ion_densities.append(density)
            else:
                raise ValueError(
                    f"Type {type(density)} of {density} not supported for ion density."
                )

        self.ion_mass = np.array(
            [species["mass_u"] * ct.u for species in self.ion_species]
        )

        self.ion_atomic_number = np.array(
            [len(species["ionization_energy_eV"]) for species in self.ion_species]
        )

        self.adk_prefactors = np.zeros(
            (len(self.ion_atomic_number), np.max(self.ion_atomic_number), 4)
        )

        for i, species in enumerate(self.ion_species):
            wa = (
                ct.alpha**3
                * ct.c
                / ct.physical_constants["classical electron radius"][0]
            )
            Ea = (
                ct.m_e
                * ct.c
                * ct.c
                / ct.e
                * ct.alpha**4
                / ct.physical_constants["classical electron radius"][0]
            )
            UH = ion_species_lookup["H"]["ionization_energy_eV"][0]
            l_eff = np.sqrt(UH / species["ionization_energy_eV"][0]) - 1.0
            dt = (xi_max - xi_min) / (n_xi * ct.c)

            for j in range(self.ion_atomic_number[i]):
                Uion = species["ionization_energy_eV"][j]
                n_eff = (j + 1) * np.sqrt(UH / Uion)
                C2 = np.pow(2, 2 * n_eff) / (
                    n_eff * math.gamma(n_eff + l_eff + 1.0) * math.gamma(n_eff - l_eff)
                )
                self.adk_prefactors[i, j, 0] = -(2.0 * n_eff - 1.0)
                self.adk_prefactors[i, j, 1] = (
                    dt
                    * wa
                    * C2
                    * (Uion / (2.0 * UH))
                    * np.pow(2 * np.pow(Uion / UH, 3.0 / 2.0) * Ea, 2 * n_eff - 1)
                )
                self.adk_prefactors[i, j, 2] = (
                    -2.0 / 3.0 * np.pow(Uion / UH, 3.0 / 2.0) * Ea
                )
                self.adk_prefactors[i, j, 3] = (
                    (3.0 / ct.pi) * np.pow(Uion / UH, -3.0 / 2.0) / Ea
                )

        self.ion_start_index = (
            np.cumsum(self.ion_atomic_number + 1) - self.ion_atomic_number - 1
        )
        self.ion_densities = np.zeros(
            (np.sum(self.ion_atomic_number + 1), self.n_xi + 4, self.n_r + 4)
        )
        self.elec_density = np.zeros((self.n_xi + 4, self.n_r + 4))

    def _calculate_wakefield(self, bunches):

        ProfStart("LaserGridIonization")

        # Use a reference plasma density for internal units to be able to simulate vacuum
        self.n_p = 1e23

        if self.laser is None:
            raise ValueError("Must use a laser with LaserGridIonization.")

        # Get laser envelope
        a_env = self.laser.get_envelope()
        is_linear_pol = self.laser.polarization == "linear"

        density_elec = self.density_function(self.t * ct.c, self.r_fld)
        if self.r_max_plasma is not None:
            density_elec = np.where(self.r_fld > self.r_max_plasma, 0, density_elec)

        # Convert to normalized units and set initial electron density
        self.elec_density[:, :] = 0
        self.elec_density[-2, 2:-2] = density_elec / self.n_p

        self.ion_densities[:, :, :] = 0
        for i, density in enumerate(self.initial_ion_densities):
            density_ion = density(self.t * ct.c, self.r_fld)
            if self.r_max_plasma is not None:
                density_ion = np.where(self.r_fld > self.r_max_plasma, 0, density_ion)
            self.ion_densities[self.ion_start_index[i], -2, 2:-2] = (
                density_ion / self.n_p
            )

        omega0 = 2 * ct.pi * ct.c / self.laser.l_0

        cppmodule.calculate_grid_ionization(
            a_env,
            self.chi[2:-2, 2:-2],
            self.n_xi,
            self.n_r,
            len(self.ion_species),
            self.ion_densities[:, 2:-1, 2:-2],
            self.elec_density[2:-1, 2:-2],
            self.ion_start_index.astype(np.int32),
            self.ion_atomic_number.astype(np.int32),
            self.ion_mass,
            omega0,
            self.adk_prefactors,
            self.laser.polarization == "linear",
            1 / np.abs(self.xi_fld[1] - self.xi_fld[0]),
            num_threads,
        )

        if self.species_rho_diags:
            # Gamma for particles with no momentum but that see a laser
            # If linearly polarized, divide by 2 so that the
            # ponderomotive force on the plasma particles is correct.
            gamma_elec = 0.5 * (2 + np.abs(a_env) ** 2 * (0.5 if is_linear_pol else 1))
            self.rho_e[2:-2, 2:-2] = self.elec_density[2:-2, 2:-2] * gamma_elec

            for i in range(len(self.ion_atomic_number)):
                for j in range(self.ion_atomic_number[i] + 1):
                    self.rho_i[2:-2, 2:-2] -= (
                        j * self.ion_densities[self.ion_start_index[i] + j, 2:-2, 2:-2]
                    )

        ProfStop("LaserGridIonization")

    def _evolve_properties(self, bunches):
        ProfStart("EvolveLaserParallel")
        if self.laser is not None:
            if self.laser_evolution:
                k_0 = 2 * np.pi / self.laser.l_0
                k_p = np.sqrt(ct.e**2 * self.n_p / (ct.m_e * ct.epsilon_0)) / ct.c

                if self.laser.use_subgrid:
                    raise ValueError(
                        "Cannot use laser subgrid with parallel laser solver"
                    )

                omega0 = 2 * ct.pi * ct.c / self.laser.l_0

                substep_pos = ct.c * (
                    self.t
                    - self.dt_update
                    + np.arange(self.laser.solver_params["nt"])
                    * self.laser.solver_params["dt"]
                )
                r, z = np.meshgrid(self.r_fld, substep_pos)

                initial_elec_density = self.density_function(z, r) / self.n_p
                if self.r_max_plasma is not None:
                    initial_elec_density = np.where(
                        self.r_fld > self.r_max_plasma, 0, initial_elec_density
                    )

                initial_ion_densities = np.zeros(
                    (
                        np.sum(self.ion_atomic_number + 1),
                        self.laser.solver_params["nt"],
                        self.n_r,
                    )
                )
                for i, density in enumerate(self.initial_ion_densities):
                    density_ion = density(z, r) / self.n_p
                    if self.r_max_plasma is not None:
                        density_ion = np.where(
                            self.r_fld > self.r_max_plasma, 0, density_ion
                        )
                    initial_ion_densities[self.ion_start_index[i], :, :] = density_ion

                cppmodule.parallel_solver(
                    self.laser._a_env,
                    self.laser._a_env_old,
                    self.chi[2:-2, 2:-2],
                    k_0,
                    k_p,
                    self.laser.solver_params["zmin"],
                    self.laser.solver_params["zmax"],
                    self.laser.solver_params["nz"],
                    self.laser.solver_params["rmax"],
                    self.laser.solver_params["nr"],
                    self.laser.solver_params["dt"],
                    self.laser.solver_params["nt"],
                    self.laser.solver_params["use_phase"],
                    self.laser.n_steps == 0,
                    len(self.ion_species),
                    self.ion_densities[:, 2:-1, 2:-2],
                    initial_ion_densities,
                    self.elec_density[2:-1, 2:-2],
                    initial_elec_density,
                    self.ion_start_index.astype(np.int32),
                    self.ion_atomic_number.astype(np.int32),
                    self.ion_mass,
                    omega0,
                    self.adk_prefactors,
                    self.laser.polarization == "linear",
                    1 / np.abs(self.xi_fld[1] - self.xi_fld[0]),
                    num_threads,
                )

                # Update arrays and step count.
                self.laser._update_output_envelope()
                self.laser.n_steps += 1

        ProfStop("EvolveLaserParallel")
