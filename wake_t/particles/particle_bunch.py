"""
This module contains the class defining a particle bunch.
"""
# TODO: clean methods to set and get bunch matrix
from __future__ import annotations
from copy import deepcopy
from typing import Optional

import numpy as np
import scipy.constants as ct
from aptools.plotting.quick_diagnostics import full_phase_space

from .push.runge_kutta_4 import apply_rk4_pusher
from .push.boris_pusher import apply_boris_pusher


class ParticleBunch():
    """ Defines a particle bunch.

    Parameters
    ----------
    w : ndarray
        Weight of the macroparticles, i.e., the number of real particles
        represented by each macroparticle. In practice, :math:`w = q_m / q`,
        where :math:`q_m` and :math:`q` are, respectively the charge of the
        macroparticle and of the real particle (e.g., an electron).
    x, y, xi : ndarray
        Position of the macropparticles in the x, y, and xi directions in
        units of m.
    px, py, pz : ndarray
        Momentum of the macropparticles in the x, y, and z directions in
        non-dimensional units (beta*gamma).
    bunch_matrix : ndarray
        6 x N matrix, where N is the number of particles, containing the
        phase-space information of the bunch. If provided, the arguments x
        to pz are not considered. The matrix contains (x, px, y, py, z, pz)
        if matrix_type='standard' or (x, x', y, y', xi, dp) if
        matrix_type='alternative'.
    matrix_type : string
        Indicates the type of bunch_matrix. Possible values are 'standard'
        or 'alternative' (see above).
    gamma_ref : float
        Reference energy with respect to which the particle momentum dp is
        calculated. Only needed if bunch_matrix is used and
        matrix_type='alternative'.
    tags : ndarray
        Individual tags assigned to each particle.
    prop_distance : float
        Propagation distance of the bunch along the beamline.
    t_flight : float
        Time of flight of the bunch along the beamline.
    z_injection: float
        Particles have a ballistic motion for z<z_injection (in meters).
    name : str
        Name of the particle bunch. Used for species identification
        in openPMD diagnostics.
    q_species, m_species : float
        Charge and mass of a single particle of the species represented
        by the macroparticles. For an electron bunch (default),
        ``q_species=-e`` and ``m_species=m_e``

    """

    _n_unnamed = 0  # Number of unnamed ParticleBunch instances

    def __init__(
        self,
        w: np.ndarray,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        xi: Optional[np.ndarray] = None,
        px: Optional[np.ndarray] = None,
        py: Optional[np.ndarray] = None,
        pz: Optional[np.ndarray] = None,
        bunch_matrix: Optional[np.ndarray] = None,
        matrix_type: Optional[str] = 'standard',
        gamma_ref: Optional[float] = None,
        tags: Optional[np.ndarray] = None,
        prop_distance: Optional[float] = 0,
        t_flight: Optional[float] = 0,
        z_injection: Optional[float] = None,
        name: Optional[str] = None,
        q_species: Optional[float] = -ct.e,
        m_species: Optional[float] = ct.m_e
    ) -> None:
        if bunch_matrix is not None:
            if matrix_type == 'standard':
                self.set_phase_space_from_matrix(bunch_matrix)
            elif matrix_type == 'alternative':
                self.set_phase_space_from_alternative_matrix(bunch_matrix,
                                                             gamma_ref)
        else:
            self.x = x
            self.y = y
            self.xi = xi
            self.px = px
            self.py = py
            self.pz = pz
        self.w = w
        # self.mu = 0
        self.tags = tags
        self.prop_distance = prop_distance
        self.t_flight = t_flight
        self.z_injection = z_injection
        self.x_ref = 0
        self.theta_ref = 0
        self.set_name(name)
        self.q_species = q_species
        self.m_species = m_species
        self.__field_arrays_allocated = False
        self.__rk4_arrays_allocated = False

    @property
    def q(self) -> np.ndarray:
        """Get an array with the charge of each macroparticle.

        This property is implemented for convenience and for backward
        compatibility.
        """
        return self.w * self.q_species

    @q.setter
    def q(self, q_new):
        """Set the total charge of each macroparticle.

        This property is implemented for convenience and for backward
        compatibility.
        """
        self.w = np.abs(q_new / self.q_species)

    def set_name(self, name):
        """ Set the particle bunch name """
        if name is None:
            name = 'elec_bunch_{}'.format(ParticleBunch._n_unnamed)
            ParticleBunch._n_unnamed += 1
        self.name = name

    def set_phase_space(self, x, y, xi, px, py, pz):
        """Sets the phase space coordinates"""
        self.x = x
        self.y = y
        self.xi = xi
        self.px = px
        self.py = py
        self.pz = pz

    def set_phase_space_from_matrix(self, beam_matrix):
        """
        Sets the phase space coordinates from a matrix with the values of
        (x, px, y, py, xi, pz).

        """
        self.x = beam_matrix[0]
        self.y = beam_matrix[2]
        self.xi = beam_matrix[4]
        self.px = beam_matrix[1]
        self.py = beam_matrix[3]
        self.pz = beam_matrix[5]

    def set_phase_space_from_alternative_matrix(self, beam_matrix, gamma_ref):
        """
        Sets the phase space coordinates from a matrix with the values of
        (x, x', y, y', xi, dp).

        Parameters
        ----------
        bunch_matrix : array
            6 x N matrix, where N is the number of particles, containing the
            phase-space information of the bunch as (x, x', y, y', xi, dp) in
            units of (m, rad, m, rad, m, -). dp is defined as
            dp = (g-g_ref)/g_ref, while x' = px/p_kin and y' = py/p_kin, where
            p_kin is the kinetic momentum of each particle.

        gamma_ref : float
            Reference energy with respect to which the particle momentum dp is
            calculated.

        """
        dp = beam_matrix[5]
        gamma = (dp + 1)*gamma_ref
        p_kin = np.sqrt(gamma**2 - 1)
        self.x = beam_matrix[0]
        self.px = beam_matrix[1] * p_kin
        self.y = beam_matrix[2]
        self.py = beam_matrix[3] * p_kin
        self.xi = beam_matrix[4]
        self.pz = np.sqrt(gamma**2 - self.px**2 - self.py**2 - 1)

    def set_bunch_matrix(self, beam_matrix):
        """Sets the 6D phase space and charge of the bunch"""
        self.x = beam_matrix[0]
        self.y = beam_matrix[1]
        self.xi = beam_matrix[2]
        self.px = beam_matrix[3]
        self.py = beam_matrix[4]
        self.pz = beam_matrix[5]
        self.w = beam_matrix[6] / self.q_species

    def get_bunch_matrix(self):
        """Returns a matrix with the 6D phase space and charge of the bunch"""
        return np.array([self.x, self.y, self.xi, self.px, self.py, self.pz,
                         self.w * self.q_species])

    def get_6D_matrix(self):
        """
        Returns the 6D phase space matrix of the bunch containing
        (x, px, y, py, xi, pz)
        """
        return np.array([self.x, self.px, self.y, self.py, self.xi, self.pz])

    def get_6D_matrix_with_charge(self):
        """
        Returns the 6D phase space matrix of the bunch containing
        (x, px, y, py, xi, pz)
        """
        return np.array([self.x, self.px, self.y, self.py, self.xi, self.pz,
                         self.w * self.q_species])

    def get_alternative_6D_matrix(self):
        """
        Returns the 6D matrix of the bunch containing
        (x, x', y, y', xi, dp)
        """
        g = np.sqrt(1 + self.px**2 + self.py**2 + self.pz**2)
        g_avg = np.average(g, weights=self.w)
        b_avg = np.sqrt(1 - g_avg**(-2))
        dp = (g-g_avg)/(g_avg*b_avg)
        p_kin = np.sqrt(g**2 - 1)
        return np.array([self.x, self.px/p_kin, self.y, self.py/p_kin,
                         self.xi, dp]), g_avg

    def increase_prop_distance(self, dist):
        """Increases the propagation distance"""
        self.prop_distance += dist

    def reposition_xi(self, xi_c):
        """Recenter bunch along xi around the specified xi_c"""
        current_xi_c = np.average(self.xi, weights=self.w)
        dxi = xi_c - current_xi_c
        self.xi += dxi

    def get_openpmd_diagnostics_data(self, global_time):
        """
        Returns a dictionary with the necessary data to write the openPMD
        diagnostics of the particle bunch.

        """
        diag_dict = {
            'x': self.x,
            'y': self.y,
            'z': self.xi,
            'px': self.px * self.m_species * ct.c,
            'py': self.py * self.m_species * ct.c,
            'pz': self.pz * self.m_species * ct.c,
            'w': self.w,
            'q': self.q_species,
            'm': self.m_species,
            'name': self.name,
            'z_off': global_time * ct.c
        }
        return diag_dict

    def show(self, **kwargs):
        """ Show the phase space of the bunch in all dimensions. """
        full_phase_space(
            self.x, self.y, self.xi, self.px, self.py, self.pz,
            self.w * self.q_species, show=True, **kwargs)

    def evolve(self, fields, t, dt, pusher='rk4'):
        """Evolve particle bunch to the next time step.

        Parameters
        ----------
        fields : list
            List of fields in which to evolve the particle bunch.
        t : float
            The current time.
        dt : float
            Time step by which to evolve the bunch.
        pusher : str, optional
            The particle pusher to use. Either 'rk4' or 'boris'. By
            default 'rk4'.
        """

        if self.z_injection is not None:
            if (np.amax(self.xi) + self.prop_distance) < self.z_injection:
                fields = []

        if pusher == 'rk4':
            apply_rk4_pusher(self, fields, t, dt)
        elif pusher == 'boris':
            apply_boris_pusher(self, fields, t, dt)
        else:
            raise ValueError(
                f"Bunch pusher '{pusher}' not recognized. "
                "Possible values are 'boris' and 'rk4'"
            )
        self.prop_distance += dt * ct.c

    def copy(self) -> ParticleBunch:
        """Return a copy of the bunch.

        To improve performance, this copy won't contain copies of auxiliary
        arrays, only of the particle coordinates and properties.
        """
        bunch_copy = ParticleBunch(
            w=deepcopy(self.w),
            x=deepcopy(self.x),
            y=deepcopy(self.y),
            xi=deepcopy(self.xi),
            px=deepcopy(self.px),
            py=deepcopy(self.py),
            pz=deepcopy(self.pz),
            prop_distance=deepcopy(self.prop_distance),
            name=deepcopy(self.name),
            q_species=deepcopy(self.q_species),
            m_species=deepcopy(self.m_species)
        )
        bunch_copy.x_ref = self.x_ref
        bunch_copy.theta_ref = self.theta_ref
        return bunch_copy

    def get_field_arrays(self):
        """Get the arrays where the gathered fields will be stored."""
        if not self.__field_arrays_allocated:
            self.__preallocate_field_arrays()
        return (
            self.__e_x, self.__e_y, self.__e_z,
            self.__b_x, self.__b_y, self.__b_z
        )

    def get_rk4_arrays(self):
        """Get the arrays needed by the RK4 pusher."""
        if not self.__rk4_arrays_allocated:
            self.__preallocate_rk4_arrays()
        return (
            self.__x_rk4, self.__y_rk4, self.__xi_rk4,
            self.__px_rk4, self.__py_rk4, self.__pz_rk4,
            self.__dx_rk4, self.__dy_rk4, self.__dxi_rk4,
            self.__dpx_rk4, self.__dpy_rk4, self.__dpz_rk4,
            self.__k_x, self.__k_y, self.__k_xi,
            self.__k_px, self.__k_py, self.__k_pz
        )

    def __preallocate_field_arrays(self):
        """Preallocate the arrays where the gathered fields will be stored."""
        n_part = len(self.x)
        self.__e_x = np.zeros(n_part)
        self.__e_y = np.zeros(n_part)
        self.__e_z = np.zeros(n_part)
        self.__b_x = np.zeros(n_part)
        self.__b_y = np.zeros(n_part)
        self.__b_z = np.zeros(n_part)
        self.__field_arrays_allocated = True

    def __preallocate_rk4_arrays(self):
        """Preallocate the arrays needed by the RK4 pusher."""
        n_part = len(self.x)

        self.__k_x = np.zeros(n_part)
        self.__k_y = np.zeros(n_part)
        self.__k_xi = np.zeros(n_part)
        self.__k_px = np.zeros(n_part)
        self.__k_py = np.zeros(n_part)
        self.__k_pz = np.zeros(n_part)

        self.__x_rk4 = np.zeros(n_part)
        self.__y_rk4 = np.zeros(n_part)
        self.__xi_rk4 = np.zeros(n_part)
        self.__px_rk4 = np.zeros(n_part)
        self.__py_rk4 = np.zeros(n_part)
        self.__pz_rk4 = np.zeros(n_part)

        self.__dx_rk4 = np.zeros(n_part)
        self.__dy_rk4 = np.zeros(n_part)
        self.__dxi_rk4 = np.zeros(n_part)
        self.__dpx_rk4 = np.zeros(n_part)
        self.__dpy_rk4 = np.zeros(n_part)
        self.__dpz_rk4 = np.zeros(n_part)

        self.__rk4_arrays_allocated = True
