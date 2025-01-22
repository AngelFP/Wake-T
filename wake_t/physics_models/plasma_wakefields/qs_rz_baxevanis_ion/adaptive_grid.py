"""Contains the definition of the `AdaptiveGrid` class."""

from typing import Optional

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.utilities.numba import njit_serial
from wake_t.particles.interpolation import gather_main_fields_cyl_linear
from .psi_and_derivatives import calculate_psi_with_interpolation
from .b_theta import calculate_b_theta_with_interpolation
from .b_theta_bunch import calculate_bunch_source, deposit_bunch_charge
from .utils import longitudinal_gradient, radial_gradient


class AdaptiveGrid:
    """Grid whose size dynamically adapts to the extent of a particle bunch.

    The number of radial cells is fixed, but its transverse extent is
    continually adapted to fit the whole particle distribution (unless
    a fixed value for `r_max` is given).

    The longitudinal grid spacing is always constant, and set to the
    longitudinal step of the plasma particles. The longitudinal
    extent and number of grid points are always adjusted so that the whole
    particle bunch fits within the grid.

    Parameters
    ----------
    x, y, xi : ndarray
        The transverse and longitudinal coordinates of the bunch particles.
    bunch_name : str
        The name of the bunch that is being covered by the grid.
    nr, nxi : int
        Radial and longitudinal resolution of the grid.
    xi_plasma : ndarray
        Array containing the possible longitudinal locations of the plasma
        particles.
    r_max : float, optional
        Radial extent of the grid. If not given, the radial extent is
        dynamically updated with the beam size. If given, the radial extent
        is always fixed to the specified value.
    r_lim : float, optional
        Limit to the radial extent of the grid. If not given, the radial extent
        is dynamically updated with the beam size to fit the whole beam.
        If given, the radial extent will never be larger than the specified
        value. Bunch particles that escape the grid transversely with
        deposit to and gather from the base grid (if they haven't escaped
        from it too).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xi: np.ndarray,
        bunch_name: str,
        nr: int,
        nxi: int,
        xi_plasma: np.ndarray,
        r_max: Optional[float] = None,
        r_lim: Optional[float] = None,
    ):
        self.bunch_name = bunch_name
        self.xi_plasma = xi_plasma
        self.dxi = (xi_plasma[-1] - xi_plasma[0]) / (nxi - 1)
        self.nr_border = 2
        self.nxi_border = 2
        self.nr = nr + self.nr_border
        self._r_max = r_max
        self._r_lim = r_lim
        self._r_max_hist = []

        self._update(x, y, xi)

    @property
    def r_min_cell(self):
        """Radial position of first cell (ignoring guard cells)."""
        return self.r_grid[0]

    @property
    def r_max_cell(self):
        """Radial position of last cell (ignoring guard and border cells)."""
        return self.r_grid[-1 - self.nr_border]

    @property
    def r_max(self):
        """Radial extent of the grid, ignoring guard and border cells."""
        return self.r_max_cell + 0.5 * self.dr

    @property
    def r_max_cell_guard(self):
        """Radial position of last guard grid cell."""
        return self.r_grid[-1] + 2 * self.dr

    def update_if_needed(self, x, y, xi, n_p, pp_hist):
        """
        Update grid size and fields if bunch extent has changed sufficiently.

        Parameters
        ----------
        x, y, xi : ndarray
            The transverse and longitudinal coordinates of the bunch particles.
        n_p : float
            The plasma density.
        pp_hist : dict
            Dictionary containing arrays with the history of the plasma
            particles.
        """
        update_r = False
        # Only trigger radial update if the radial size is not fixed.
        if self._r_max is None:
            r_max_beam = np.max(np.sqrt(x**2 + y**2))
            update_r = (r_max_beam > self.r_max_cell) or (
                r_max_beam < self.r_max_cell * 0.9
            )
            # It a radial limit is set, update only if limit has not been
            # reached.
            if update_r and self._r_lim is not None:
                if r_max_beam < self._r_lim:
                    update_r = True
                elif self.r_max_cell != self._r_lim:
                    update_r = True
                else:
                    update_r = False

        xi_min_beam = np.min(xi)
        xi_max_beam = np.max(xi)
        update_xi = (xi_min_beam < self.xi_grid[0 + self.nxi_border]) or (
            xi_max_beam > self.xi_grid[-1 - self.nxi_border]
        )
        if update_r or update_xi:
            self._update(x, y, xi)
            self.calculate_fields(n_p, pp_hist, reset_fields=False)

    def calculate_fields(self, n_p, pp_hist, reset_fields=True):
        """Calculate the E and B fields from the plasma at the grid.

        Parameters
        ----------
        n_p : float
            The plasma density.
        pp_hist : dict
            Dictionary containing arrays with the history of the plasma
            particles.
        reset_fields : bool
            Whether the fields should be reset to zero before calculating.
        """
        if reset_fields:
            self._reset_fields()
        s_d = ge.plasma_skin_depth(n_p * 1e-6)
        calculate_fields_on_grid(
            self.i_grid,
            self.r_grid,
            s_d,
            self.psi_grid,
            self.b_t,
            pp_hist["r_hist"],
            pp_hist["log_r_hist"],
            pp_hist["sum_1_hist"],
            pp_hist["sum_2_hist"],
            pp_hist["a_0_hist"],
            pp_hist["a_i_hist"],
            pp_hist["b_i_hist"],
        )

        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)
        longitudinal_gradient(
            self.psi_grid[2:-2, 2:-2], self.dxi / s_d, self.e_z[2:-2, 2:-2]
        )
        radial_gradient(
            self.psi_grid[2:-2, 2:-2], self.dr / s_d, self.e_r[2:-2, 2:-2]
        )
        self.e_r -= self.b_t
        self.e_z *= -E_0
        self.e_r *= -E_0
        self.b_t *= E_0 / ct.c

    def calculate_bunch_source(self, bunch, n_p, p_shape):
        """Calculate the source term (B_theta) of the bunch within the grid.

        Parameters
        ----------
        bunch : ParticleBunch
            The particle bunch.
        n_p : float
            The plasma density.
        p_shape : str
            The particle shape.
        """
        self.b_t_bunch[:] = 0.0
        self.q_bunch[:] = 0.0
        all_deposited = deposit_bunch_charge(
            bunch.x,
            bunch.y,
            bunch.xi,
            bunch.q,
            n_p,
            self.nr - self.nr_border,
            self.nxi,
            self.r_grid,
            self.xi_grid,
            self.dr,
            self.dxi,
            p_shape,
            self.q_bunch,
        )
        calculate_bunch_source(self.q_bunch, self.nr, self.nxi, self.b_t_bunch)
        return all_deposited

    def gather_fields(self, x, y, z, ex, ey, ez, bx, by, bz):
        """Gather the plasma fields at the location of the bunch particles.

        Parameters
        ----------
        x, y, z : ndarray
            The transverse and longitudinal coordinates of the bunch particles.
        ex, ey, ez, bx, by, bz : ndarray
            The arrays where the gathered field components will be stored.
        """
        return gather_main_fields_cyl_linear(
            self.e_r,
            self.e_z,
            self.b_t,
            self.xi_min,
            self.xi_max,
            self.r_min_cell,
            self.r_max_cell,
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

    def get_openpmd_data(self, global_time, diags):
        """Get the field data at the grid to store in the openpmd diagnostics.

        Parameters
        ----------
        global_time : float
            The global time of the simulation.
        diags : list
            A list of strings with the names of the fields to include in the
            diagnostics.

        Returns
        -------
        dict
        """

        # Grid parameters.
        grid_spacing = [self.dr, self.dxi]
        grid_labels = ["r", "z"]
        grid_global_offset = [0.0, global_time * ct.c + self.xi_min]

        # Initialize field diags lists.
        names = []
        comps = []
        attrs = []
        arrays = []

        # Add requested fields to lists.
        if "E" in diags:
            names += ["E"]
            comps += [["r", "z"]]
            attrs += [{}]
            arrays += [
                [
                    np.ascontiguousarray(self.e_r.T[2:-2, 2:-2]),
                    np.ascontiguousarray(self.e_z.T[2:-2, 2:-2]),
                ]
            ]
        if "B" in diags:
            names += ["B"]
            comps += [["t"]]
            attrs += [{}]
            arrays += [[np.ascontiguousarray(self.b_t.T[2:-2, 2:-2])]]

        # Create dictionary with all diagnostics data.
        comp_pos = [[0.5, 0.0]] * len(names)
        fld_zip = zip(names, comps, attrs, arrays, comp_pos)
        diag_data = {}
        diag_data["fields"] = []
        for fld, comps, attrs, arrays, pos in fld_zip:
            fld += "_" + self.bunch_name
            diag_data["fields"].append(fld)
            diag_data[fld] = {}
            if comps is not None:
                diag_data[fld]["comps"] = {}
                for comp, arr in zip(comps, arrays):
                    diag_data[fld]["comps"][comp] = {}
                    diag_data[fld]["comps"][comp]["array"] = arr
                    diag_data[fld]["comps"][comp]["position"] = pos
            else:
                diag_data[fld]["array"] = arrays[0]
                diag_data[fld]["position"] = pos
            diag_data[fld]["grid"] = {}
            diag_data[fld]["grid"]["spacing"] = grid_spacing
            diag_data[fld]["grid"]["labels"] = grid_labels
            diag_data[fld]["grid"]["global_offset"] = grid_global_offset
            diag_data[fld]["attributes"] = attrs

        return diag_data

    def _update(self, x, y, xi):
        """Update the grid size."""
        # Create grid in r
        if self._r_max is None:
            r_max = np.max(np.sqrt(x**2 + y**2))
        else:
            r_max = self._r_max
        if self._r_lim is not None:
            r_max = r_max if r_max <= self._r_lim else self._r_lim
        self._r_max_hist.append(r_max)
        self.dr = r_max / (self.nr - self.nr_border)
        r_max += self.nr_border * self.dr
        self.r_grid = np.linspace(self.dr / 2, r_max - self.dr / 2, self.nr)

        # Create grid in xi
        xi_min_beam = np.min(xi)
        xi_max_beam = np.max(xi)
        self.i_grid = np.where(
            (self.xi_plasma > xi_min_beam - self.dxi * (1 + self.nxi_border))
            & (self.xi_plasma < xi_max_beam + self.dxi * (1 + self.nxi_border))
        )[0]
        self.xi_grid = self.xi_plasma[self.i_grid]
        self.xi_max = self.xi_grid[-1]
        self.xi_min = self.xi_grid[0]
        self.nxi = self.xi_grid.shape[0]

        # Create field arrays.
        self.psi_grid = np.zeros((self.nxi + 4, self.nr + 4))
        self.b_t = np.zeros((self.nxi + 4, self.nr + 4))
        self.e_r = np.zeros((self.nxi + 4, self.nr + 4))
        self.e_z = np.zeros((self.nxi + 4, self.nr + 4))
        self.b_t_bunch = np.zeros((self.nxi + 4, self.nr + 4))
        self.q_bunch = np.zeros((self.nxi + 4, self.nr + 4))

    def _reset_fields(self):
        """Reset value of the fields at the grid."""
        self.psi_grid[:] = 0.0
        self.b_t[:] = 0.0
        self.e_r[:] = 0.0
        self.e_z[:] = 0.0


@njit_serial()
def calculate_fields_on_grid(
    i_grid,
    r_grid,
    s_d,
    psi_grid,
    bt_grid,
    r_hist,
    log_r_hist,
    sum_1_hist,
    sum_2_hist,
    a_0_hist,
    a_i_hist,
    b_i_hist,
):
    """Compute the plasma fields on the grid.

    Compiling this method in numba avoids significant overhead.
    """
    n_points = i_grid.shape[0]
    n_elec = int(r_hist.shape[-1] / 2)
    for i in range(n_points):
        j = i_grid[i]
        psi = psi_grid[i + 2, 2:-2]
        b_theta = bt_grid[i + 2, 2:-2]
        calculate_psi_with_interpolation(
            r_eval=r_grid / s_d,
            r=r_hist[j, :n_elec],
            log_r=log_r_hist[j, :n_elec],
            sum_1_arr=sum_1_hist[j, :n_elec + 1],
            sum_2_arr=sum_2_hist[j, :n_elec + 1],
            psi=psi,
        )
        calculate_psi_with_interpolation(
            r_eval=r_grid / s_d,
            r=r_hist[j, n_elec:],
            log_r=log_r_hist[j, n_elec:],
            sum_1_arr=sum_1_hist[j, n_elec + 1:],
            sum_2_arr=sum_2_hist[j, n_elec + 1:],
            psi=psi,
            add=True,
        )
        calculate_b_theta_with_interpolation(
            r_fld=r_grid / s_d,
            a_0=a_0_hist[j],
            a=a_i_hist[j],
            b=b_i_hist[j],
            r=r_hist[j, :n_elec],
            b_theta=b_theta,
        )
