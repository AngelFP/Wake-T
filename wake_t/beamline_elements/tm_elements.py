""" Contains the classes of all elements tracked using transfer matrices. """
from typing import Optional, Union, List
import time

import numpy as np
import scipy.constants as ct

from wake_t.particles.push.transfer_matrix import track_with_transfer_map
from wake_t.particles.particle_bunch import ParticleBunch
from wake_t.utilities.other import print_progress_bar
from wake_t.utilities.bunch_manipulation import (
    convert_to_ocelot_matrix, convert_from_ocelot_matrix, rotation_matrix_xz)
from wake_t.physics_models.collective_effects.csr import get_csr_calculator
from wake_t.diagnostics import OpenPMDDiagnostics


class TMElement():
    # TODO: fix backtracking issues.
    """
    Base class for all elements where the tracking is performed using
    transfer maps. It can also be used to define an arbitrary element
    that can be modeled with transfer maps of up to second order.

    Parameters
    ----------
    length : float
        Length of the drift space in meters.
    theta : float
        Bending angle that a particle with an energy given by `gamma_ref`
        would experience.
    k1 : float
        Normalized quadrupole strength in units of 1/m**2.
    k2 : float
        Normalized sextupole strength in units of 1/m**3.
    gamma_ref : float, optional
        Reference gamma (relativistic Lorentz factor) to be used for the
        tracking. If not specified, the average gamma of the particle
        bunch is used.
    csr_on : bool
        Whether to include CSR forces.
    n_out : int
        Number of times along the element in which the particle
        distribution should be returned or written to file (if openpmd
        diagnostics are activated).
    order : int
        Highest order of the transfer matrices. Possible values are `1` and
        `2`.

    """

    def __init__(
        self,
        length: Optional[float] = 0.,
        theta: Optional[float] = 0.,
        k1: Optional[float] = 0.,
        k2: Optional[float] = 0.,
        gamma_ref: Optional[float] = None,
        csr_on: Optional[bool] = False,
        n_out: Optional[int] = None,
        order: Optional[int] = 2
    ) -> None:
        self.length = length
        self.theta = theta
        self.k1 = k1
        self.k2 = k2
        self.gamma_ref = gamma_ref
        self.csr_on = csr_on
        self.n_out = n_out
        self.order = order
        self.csr_calculator = get_csr_calculator()
        self.element_name = ''

    def track(
        self,
        bunch: ParticleBunch,
        backtrack: Optional[bool] = False,
        out_initial: Optional[bool] = False,
        opmd_diag: Optional[Union[bool, OpenPMDDiagnostics]] = False,
        diag_dir: Optional[str] = None,
        show_progress_bar: Optional[bool] = True,
    ) -> List[ParticleBunch]:
        """
        Track bunch through element.

        Parameters
        ----------
        bunch : ParticleBunch
            Particle bunch to be tracked.
        backtrack : bool
            Whether to perform the tracking backwards.
        out_initial : bool
            Whether to include the initial bunch in the output.
        opmd_diag : bool or OpenPMDDiagnostics
            Determines whether to write simulation diagnostics to disk (i.e.
            particle distributions and fields). The output is written to
            HDF5 files following the openPMD standard. The number of outputs
            the `n_out` value. It is also possible to provide an already
            existing OpenPMDDiagnostics instance instead of a boolean value.
        diag_dir : str
            Directory into which the openPMD output will be written. By default
            this is a 'diags' folder in the current directory. Only needed if
            `opmd_diag=True`.
        show_progress_bar : bool, optional
            Whether to show a progress bar of the tracking. By default
            ``True``.

        Returns
        -------
        A list of size 'n_out' containing the bunch distribution at each step.

        """
        # Convert bunch to ocelot units and reference frame
        bunch_mat, g_avg = self._get_beam_matrix_for_tracking(bunch)
        if self.gamma_ref is None:
            self.gamma_ref = g_avg

        # Add element to CSR calculator
        if self.csr_on:
            self.csr_calculator.add_lattice_element(self)
        else:
            self.csr_calculator.clear()
        # Determine track and output steps
        l_step, track_steps, output_steps = self._determine_steps()

        # Create diagnostics if needed
        if type(opmd_diag) is not OpenPMDDiagnostics and opmd_diag:
            opmd_diag = OpenPMDDiagnostics(write_dir=diag_dir)
        elif not opmd_diag:
            opmd_diag = None

        # Print output header
        if show_progress_bar:
            print('')
            print(self.element_name.capitalize())
            print('-'*len(self.element_name))
            self._print_element_properties()
            csr_string = 'on' if self.csr_on else 'off'
            print('CSR {}.'.format(csr_string))
            print('')
            n_steps = len(track_steps)
            st_0 = 'Tracking in {} step(s)... '.format(n_steps)

        # Start tracking
        start_time = time.time()
        output_bunch_list = list()
        if out_initial:
            output_bunch_list.append(bunch.copy())
            if opmd_diag is not None:
                opmd_diag.write_diagnostics(
                    0., l_step/ct.c, [output_bunch_list[-1]])
        for i in track_steps:
            if show_progress_bar:
                print_progress_bar(st_0, i+1, n_steps)
            l_curr = (i+1) * l_step * (1-2*backtrack)
            # Track with transfer matrix
            bunch_mat = track_with_transfer_map(
                bunch_mat, l_step, self.length, -self.theta, self.k1, self.k2,
                self.gamma_ref, order=self.order)
            # Apply CSR
            if self.csr_on:
                self.csr_calculator.apply_csr(bunch_mat, bunch.q,
                                              self.gamma_ref, self, l_curr)
            # Add bunch to output list
            if i in output_steps:
                new_bunch_mat = convert_from_ocelot_matrix(
                    bunch_mat, self.gamma_ref)
                new_bunch = self._create_new_bunch(
                    bunch, new_bunch_mat, l_curr)
                output_bunch_list.append(new_bunch)
                if opmd_diag is not None:
                    opmd_diag.write_diagnostics(
                        l_curr/ct.c, l_step/ct.c, [output_bunch_list[-1]])

        # Update bunch data
        self._update_input_bunch(bunch, bunch_mat, output_bunch_list)

        # Add element length to diagnostics position
        if opmd_diag is not None:
            opmd_diag.increase_z_pos(self.length)

        # Finalize
        if show_progress_bar:
            tracking_time = time.time() - start_time
            print('Done ({} s).'.format(tracking_time))
            print('-'*80)
        return output_bunch_list

    def _get_beam_matrix_for_tracking(self, bunch):
        bunch_mat = bunch.get_6D_matrix()
        # obtain with respect to reference displacement
        bunch_mat[0] -= bunch.x_ref
        # rotate by the reference angle so that it enters normal to the element
        if bunch.theta_ref != 0:
            rot = rotation_matrix_xz(-bunch.theta_ref)
            bunch_mat = np.dot(rot, bunch_mat)
        return convert_to_ocelot_matrix(bunch_mat, bunch.w, self.gamma_ref)

    def _determine_steps(self):
        if self.n_out is not None:
            l_step = self.length/self.n_out
            n_track = self.n_out
            frac_out = 1
        else:
            l_step = self.length
            n_track = 1
            frac_out = 0
        if self.csr_on:
            csr_track_step = self.csr_calculator.get_csr_step(self)
            n_csr = int(self.length / csr_track_step)
            if self.n_out is not None:
                frac_out = int(n_csr / self.n_out)
            l_step = csr_track_step
            n_track = n_csr
        track_steps = np.arange(0, n_track)
        if frac_out != 0:
            output_steps = track_steps[::-frac_out]
        else:
            output_steps = []
        return l_step, track_steps, output_steps

    def _update_input_bunch(self, bunch, bunch_mat, output_bunch_list):
        if len(output_bunch_list) == 0:
            new_bunch_mat = convert_from_ocelot_matrix(
                bunch_mat, self.gamma_ref)
            last_bunch = self._create_new_bunch(bunch, new_bunch_mat,
                                                self.length)
        else:
            last_bunch = output_bunch_list[-1].copy()
        bunch.set_phase_space(last_bunch.x, last_bunch.y, last_bunch.xi,
                              last_bunch.px, last_bunch.py, last_bunch.pz)
        bunch.prop_distance = last_bunch.prop_distance
        bunch.theta_ref = last_bunch.theta_ref
        bunch.x_ref = last_bunch.x_ref

    def _create_new_bunch(self, old_bunch, new_bunch_mat, prop_dist):
        if self.theta != 0:
            # angle rotated for prop_dist
            theta_step = self.theta*prop_dist/self.length
            # magnet bending radius
            rho = abs(self.length/self.theta)
            # new reference angle and transverse displacement
            new_theta_ref = old_bunch.theta_ref + theta_step
            sign = -theta_step/abs(theta_step)
            new_x_ref = (
                old_bunch.x_ref +
                sign*rho*(np.cos(new_theta_ref)-np.cos(old_bunch.theta_ref)))
        else:
            # new reference angle and transverse displacement
            new_theta_ref = old_bunch.theta_ref
            new_x_ref = (old_bunch.x_ref + prop_dist *
                         np.sin(old_bunch.theta_ref))
        # new prop. distance
        new_prop_dist = old_bunch.prop_distance + prop_dist
        # rotate distribution if reference angle != 0
        if new_theta_ref != 0:
            rot = rotation_matrix_xz(new_theta_ref)
            new_bunch_mat = np.dot(rot, new_bunch_mat)
        new_bunch_mat[0] += new_x_ref
        # create new bunch
        new_bunch = ParticleBunch(old_bunch.w, bunch_matrix=new_bunch_mat,
                                  prop_distance=new_prop_dist,
                                  name=old_bunch.name,
                                  q_species=old_bunch.q_species,
                                  m_species=old_bunch.m_species)
        new_bunch.theta_ref = new_theta_ref
        new_bunch.x_ref = new_x_ref
        return new_bunch

    def _print_element_properties(self):
        "To be implemented by each element. Prints the element properties"
        raise NotImplementedError


class Drift(TMElement):
    """
    Defines a drift space where the tracking is performed by transfer matrices
    up to second order.

    Parameters
    ----------
    length : float
        Length of the drift space in meters.
    gamma_ref : float, optional
        Reference gamma (relativistic Lorentz factor) to be used for the
        tracking. If not specified, the average gamma of the particle
        bunch is used.
    csr_on : bool
        Whether to include CSR forces.
    n_out : int
        Number of times along the element in which the particle
        distribution should be returned or written to file (if openpmd
        diagnostics are activated).
    order : int
        Highest order of the transfer matrices. Possible values are `1` and
        `2`.

    """
    def __init__(
        self,
        length: Optional[float] = 0.,
        gamma_ref: Optional[float] = None,
        csr_on: Optional[bool] = False,
        n_out: Optional[int] = None,
        order: Optional[int] = 2
    ) -> None:
        super().__init__(length, 0, 0, 0, gamma_ref, csr_on, n_out, order)
        self.element_name = 'drift'

    def _print_element_properties(self):
        print('Length = {:1.4f} m'.format(self.length))


class Dipole(TMElement):
    """
    Defines a dipole where the tracking is performed by transfer matrices
    up to second order.

    Parameters
    ----------
    length : float
        Length of the drift space in meters.
    theta : float
        Bending angle that a particle with an energy given by `gamma_ref`
        would experience.
    gamma_ref : float, optional
        Reference gamma (relativistic Lorentz factor) to be used for the
        tracking. If not specified, the average gamma of the particle
        bunch is used.
    csr_on : bool
        Whether to include CSR forces.
    n_out : int
        Number of times along the element in which the particle
        distribution should be returned or written to file (if openpmd
        diagnostics are activated).
    order : int
        Highest order of the transfer matrices. Possible values are `1` and
        `2`.

    """
    def __init__(
        self,
        length: Optional[float] = 0.,
        theta: Optional[float] = 0.,
        gamma_ref: Optional[float] = None,
        csr_on: Optional[bool] = False,
        n_out: Optional[int] = None,
        order: Optional[int] = 2
    ) -> None:
        super().__init__(length, theta, 0, 0, gamma_ref, csr_on, n_out, order)
        self.element_name = 'dipole'

    def _print_element_properties(self):
        ang_deg = self.theta * 180/ct.pi
        b_field = (ct.m_e*ct.c/ct.e) * self.theta*self.gamma_ref/self.length
        print('Bending angle = {:1.4f} rad ({:1.4f} deg)'.format(
            self.theta, ang_deg))
        print('Dipole field = {:1.4f} T'.format(b_field))


class Quadrupole(TMElement):
    """
    Defines a quadrupole where the tracking is performed by transfer matrices
    up to second order.

    Parameters
    ----------
    length : float
        Length of the drift space in meters.
    k1 : float
        Normalized quadrupole strength in units of 1/m**2.
    gamma_ref : float, optional
        Reference gamma (relativistic Lorentz factor) to be used for the
        tracking. If not specified, the average gamma of the particle
        bunch is used.
    csr_on : bool
        Whether to include CSR forces.
    n_out : int
        Number of times along the element in which the particle
        distribution should be returned or written to file (if openpmd
        diagnostics are activated).
    order : int
        Highest order of the transfer matrices. Possible values are `1` and
        `2`.

    """
    def __init__(
        self,
        length: Optional[float] = 0.,
        k1: Optional[float] = 0.,
        gamma_ref: Optional[float] = None,
        csr_on: Optional[bool] = False,
        n_out: Optional[int] = None,
        order: Optional[int] = 2
    ) -> None:
        super().__init__(length, 0, k1, 0, gamma_ref, csr_on, n_out, order)
        self.element_name = 'quadrupole'

    def _print_element_properties(self):
        g = self.k1 * self.gamma_ref*(ct.m_e*ct.c/ct.e)
        print('Quadrupole gradient = {:1.4f} T/m'.format(g))


class Sextupole(TMElement):
    """
    Defines a sextupole where the tracking is performed by transfer matrices
    up to second order.

    Parameters
    ----------
    length : float
        Length of the drift space in meters.
    k2 : float
        Normalized sextupole strength in units of 1/m^3.
    gamma_ref : float, optional
        Reference gamma (relativistic Lorentz factor) to be used for the
        tracking. If not specified, the average gamma of the particle
        bunch is used.
    csr_on : bool
        Whether to include CSR forces.
    n_out : int
        Number of times along the element in which the particle
        distribution should be returned or written to file (if openpmd
        diagnostics are activated).
    order : int
        Highest order of the transfer matrices. Possible values are `1` and
        `2`.

    """
    def __init__(
        self,
        length: Optional[float] = 0.,
        k2: Optional[float] = 0.,
        gamma_ref: Optional[float] = None,
        csr_on: Optional[bool] = False,
        n_out: Optional[int] = None,
        order: Optional[int] = 2
    ) -> None:
        super().__init__(length, 0, 0, k2, gamma_ref, csr_on, n_out, order)
        self.element_name = 'sextupole'

    def _print_element_properties(self):
        g = self.k2 * self.gamma_ref*(ct.m_e*ct.c/ct.e)
        print('Sextupole gradient = {:1.4f} T/m^2'.format(g))
