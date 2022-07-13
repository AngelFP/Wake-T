"""
This module contains the class defining a particle bunch.
"""
# TODO: clean methods to set and get bunch matrix
import numpy as np
import scipy.constants as ct
from aptools.plotting.quick_diagnostics import full_phase_space

from .push.runge_kutta_4 import apply_rk4_pusher
from .push.boris_pusher import apply_boris_pusher


class ParticleBunch():

    """ Defines a particle bunch. """

    _n_unnamed = 0  # Number of unnamed ParticleBunch instances

    def __init__(self, q, x=None, y=None, xi=None, px=None, py=None, pz=None,
                 bunch_matrix=None, matrix_type='standard', gamma_ref=None,
                 tags=None, prop_distance=0, t_flight=0, z_injection=None,
                 name=None):
        """
        Initialize particle bunch.

        Parameters:
        -----------
        q : array
            Charge of each particle in units of C.

        x : array
            Position of each particle in the x-plane in units of m.

        y : array
            Position of each particle in the y-plane in units of m.

        xi : array
            Position of each particle in the xi-plane in units of m.

        px : array
            Momentum of each particle in the x-plane in non-dimensional units
            (beta*gamma).

        py : array
            Momentum of each particle in the y-plane in non-dimensional units
            (beta*gamma).

        pz : array
            Momentum of each particle in the z-plane in non-dimensional units
            (beta*gamma).

        bunch_matrix : array
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

        tags : array
            Individual tags assigned to each particle.

        prop_distance : float
            Propagation distance of the bunch along the beamline.

        t_flight : float
            Time of flight of the bunch along the beamline.

        z_injection: float (in meters)
            Particles have a ballistic motion for z<z_injection.

        name : str
            Name of the particle bunch. Used for species identification
            in openPMD diagnostics.

        """
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
        self.q = q
        # self.mu = 0
        self.tags = tags
        self.prop_distance = prop_distance
        self.t_flight = t_flight
        self.z_injection = z_injection
        self.x_ref = 0
        self.theta_ref = 0
        self.set_name(name)
        self.__field_arrays_allocated = False
        self.__rk4_arrays_allocated = False

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

    def get_field_arrays(self):
        """Get the arrays where the gathered fields will be stored."""
        if not self.__field_arrays_allocated:
            self.__preallocate_field_arrays()
        return (
            self.__e_x, self.__e_y, self.__e_z,
            self.__b_x, self.__b_y, self.__b_z
        )

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

        Parameters:
        -----------
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
        self.q = beam_matrix[6]

    def get_bunch_matrix(self):
        """Returns a matrix with the 6D phase space and charge of the bunch"""
        return np.array([self.x, self.y, self.xi, self.px, self.py, self.pz,
                         self.q])

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
        return np.array(
            [self.x, self.px, self.y, self.py, self.xi, self.pz, self.q])

    def get_alternative_6D_matrix(self):
        """
        Returns the 6D matrix of the bunch containing
        (x, x', y, y', xi, dp)
        """
        g = np.sqrt(1 + self.px**2 + self.py**2 + self.pz**2)
        g_avg = np.average(g, weights=self.q)
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
        current_xi_c = np.average(self.xi, weights=self.q)
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
            'px': self.px * ct.m_e * ct.c,
            'py': self.py * ct.m_e * ct.c,
            'pz': self.pz * ct.m_e * ct.c,
            'w': self.q / ct.e,
            'q': -ct.e,
            'm': ct.m_e,
            'name': self.name,
            'z_off': global_time * ct.c
        }
        return diag_dict

    def show(self, **kwargs):
        """ Show the phase space of the bunch in all dimensions. """
        full_phase_space(
            self.x, self.y, self.xi, self.px, self.py, self.pz, self.q,
            show=True, **kwargs)

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

        if self.prop_distance < self.z_injection:
            fields = []

        if pusher == 'rk4':
            apply_rk4_pusher(self, fields, t, dt)
        elif pusher == 'boris':
            apply_boris_pusher(self, fields, t, dt)
        self.prop_distance += dt * ct.c
