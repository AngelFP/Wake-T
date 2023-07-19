"""Contains the definition of the `AdaptiveGrid` class."""

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.utilities.numba import njit_serial
from wake_t.particles.interpolation import gather_main_fields_cyl_linear
from .psi_and_derivatives import calculate_psi
from .b_theta import calculate_b_theta
from .b_theta_bunch import calculate_bunch_source


class AdaptiveGrid():
    """Grid whose size dynamically adapts to the extent of a particle bunch.

    The number of radial cells is fixed, but it its transverse extent is
    continually adapted to fit the whole particle distribution.

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
    nr : int
        Radial resolution of the grid.
    xi_plasma : ndarray
        Array containing the possible longitudinal locations of the plasma
        particles.
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xi: np.ndarray,
        bunch_name: str,
        nr: int,
        xi_plasma: np.ndarray
    ):
        self.bunch_name = bunch_name
        self.nr = nr
        self.xi_plasma = xi_plasma
        self.dxi = xi_plasma[1] - xi_plasma[0]

        self._update(x, y, xi)

    def update_if_needed(self, x, y, xi):
        """
        Update the grid size if bunch extent has changed sufficiently.

        Parameters
        ----------
        x, y, xi : ndarray
            The transverse and longitudinal coordinates of the bunch particles.
        """
        r_max_beam = np.max(np.sqrt(x**2 + y**2))
        xi_min_beam = np.min(xi)
        xi_max_beam = np.max(xi)
        if (
            (r_max_beam > self.r_grid[-1]) or
            (xi_min_beam < self.xi_grid[0]) or
            (xi_max_beam > self.xi_grid[-1]) or
            (r_max_beam < self.r_grid[-1] * 0.9)
        ):
            self._update(x, y, xi)
        else:
            self._reset_fields()

    def calculate_fields(self, n_p, pp_hist):
        """Calculate the E and B fields from the plasma at the grid.

        Parameters
        ----------
        n_p : float
            The plasma density.
        pp_hist : dict
            Dictionary containing arrays with the history of the plasma
            particles.
        """
        s_d = ge.plasma_skin_depth(n_p * 1e-6)
        calculate_fields_on_grid(
            self.i_grid, self.r_grid, s_d,
            self.psi_grid, self.b_t, self.log_r_grid, pp_hist['r_hist'],
            pp_hist['sum_1_hist'], pp_hist['sum_2_hist'],
            pp_hist['i_sort_hist'], pp_hist['psi_max_hist'],
            pp_hist['a_0_hist'], pp_hist['a_i_hist'], pp_hist['b_i_hist'])

        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p * 1e-6)
        dxi_psi, dr_psi = np.gradient(self.psi_grid[2:-2, 2:-2], self.dxi/s_d,
                                      self.dr/s_d, edge_order=2)
        self.e_z[2:-2, 2:-2] = -dxi_psi * E_0
        self.b_t *= E_0 / ct.c
        self.e_r[2:-2, 2:-2] = -dr_psi * E_0 + self.b_t[2:-2, 2:-2] * ct.c

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
        self.b_t_bunch[:] = 0.
        calculate_bunch_source(bunch, n_p, self.nr, self.nxi, self.r_grid[0],
                               self.xi_grid[0], self.dr, self.dxi, p_shape,
                               self.b_t_bunch)

    def gather_fields(self, x, y, z, ex, ey, ez, bx, by, bz):
        """Gather the plasma fields at the location of the bunch particles.

        Parameters
        ----------
        x, y, z : ndarray
            The transverse and longitudinal coordinates of the bunch particles.
        ex, ey, ez, bx, by, bz : ndarray
            The arrays where the gathered field components will be stored.
        """
        gather_main_fields_cyl_linear(
            self.e_r, self.e_z, self.b_t, self.xi_min, self.xi_max,
            self.r_grid[0], self.r_grid[-1], self.dxi, self.dr, x, y, z,
            ex, ey, ez, bx, by, bz)

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
        grid_labels = ['r', 'z']
        grid_global_offset = [0., global_time*ct.c + self.xi_min]

        # Initialize field diags lists.
        names = []
        comps = []
        attrs = []
        arrays = []

        # Add requested fields to lists.
        if 'E' in diags:
            names += ['E']
            comps += [['r', 'z']]
            attrs += [{}]
            arrays += [
                [np.ascontiguousarray(self.e_r.T[2:-2, 2:-2]),
                 np.ascontiguousarray(self.e_z.T[2:-2, 2:-2])]
            ]
        if 'B' in diags:
            names += ['B']
            comps += [['t']]
            attrs += [{}]
            arrays += [
                [np.ascontiguousarray(self.b_t.T[2:-2, 2:-2])]
            ]

        # Create dictionary with all diagnostics data.
        comp_pos = [[0.5, 0.]] * len(names)
        fld_zip = zip(names, comps, attrs, arrays, comp_pos)
        diag_data = {}
        diag_data['fields'] = []
        for fld, comps, attrs, arrays, pos in fld_zip:
            fld += '_' + self.bunch_name
            diag_data['fields'].append(fld)
            diag_data[fld] = {}
            if comps is not None:
                diag_data[fld]['comps'] = {}
                for comp, arr in zip(comps, arrays):
                    diag_data[fld]['comps'][comp] = {}
                    diag_data[fld]['comps'][comp]['array'] = arr
                    diag_data[fld]['comps'][comp]['position'] = pos
            else:
                diag_data[fld]['array'] = arrays[0]
                diag_data[fld]['position'] = pos
            diag_data[fld]['grid'] = {}
            diag_data[fld]['grid']['spacing'] = grid_spacing
            diag_data[fld]['grid']['labels'] = grid_labels
            diag_data[fld]['grid']['global_offset'] = grid_global_offset
            diag_data[fld]['attributes'] = attrs

        return diag_data

    def _update(self, x, y, xi):
        """Update the grid size."""
        # Create grid in r
        r_max_beam = np.max(np.sqrt(x**2 + y**2))
        self.r_max = r_max_beam * 1.1
        self.dr = self.r_max / self.nr
        self.r_grid = np.linspace(self.dr/2, self.r_max - self.dr/2, self.nr)
        self.log_r_grid = np.log(self.r_grid)

        # Create grid in xi
        xi_min_beam = np.min(xi)
        xi_max_beam = np.max(xi)
        self.i_grid = np.where(
            (self.xi_plasma > xi_min_beam - self.dxi) &
            (self.xi_plasma < xi_max_beam + self.dxi)
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

    def _reset_fields(self):
        """Reset value of the fields at the grid."""
        self.psi_grid[:] = 0.
        self.b_t[:] = 0.
        self.e_r[:] = 0.
        self.e_z[:] = 0.


@njit_serial()
def calculate_fields_on_grid(
        i_grid, r_grid, s_d,
        psi_grid, bt_grid, log_r_grid, r_hist, sum_1_hist, sum_2_hist,
        i_sort_hist, psi_max_hist, a_0_hist, a_i_hist, b_i_hist):
    """Compute the plasma fields on the grid.

    Compiling this method in numba avoids significant overhead.
    """
    n_points = i_grid.shape[0]
    n_elec = int(r_hist.shape[-1] / 2)
    for i in range(n_points):
        j = i_grid[i]
        psi = psi_grid[i + 2, 2:-2]
        b_theta = bt_grid[i + 2, 2:-2]
        calculate_psi(
            r_eval=r_grid / s_d,
            log_r_eval=log_r_grid - np.log(s_d),
            r=r_hist[j, :n_elec],
            sum_1=sum_1_hist[j, :n_elec],
            sum_2=sum_2_hist[j, :n_elec],
            idx=i_sort_hist[j, :n_elec],
            psi=psi
        )
        calculate_psi(
            r_eval=r_grid / s_d,
            log_r_eval=log_r_grid - np.log(s_d),
            r=r_hist[j, n_elec:],
            sum_1=sum_1_hist[j, n_elec:],
            sum_2=sum_2_hist[j, n_elec:],
            idx=i_sort_hist[j, n_elec:],
            psi=psi
        )
        psi -= psi_max_hist[j]

        calculate_b_theta(
            r_fld=r_grid / s_d,
            a_0=a_0_hist[j],
            a_i=a_i_hist[j],
            b_i=b_i_hist[j],
            r=r_hist[j, :n_elec],
            idx=i_sort_hist[j, :n_elec],
            b_theta=b_theta
        )
