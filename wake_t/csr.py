"""
This module contains the definition of the CSRCalculator class.

As indicated below, the methods for computing the reference trajectory and
the CSR effects on the bunch are strongly based on the 1D CSR model from OCELOT
(https://github.com/ocelot-collab/ocelot) written by S. Tomin and M. Dohlus.

"""


import numpy as np
from scipy import stats
import scipy.constants as ct


class CSRCalculator():

    """Class taking care of calculating and applying CSR effects."""

    def __init__(self):
        """Initialize CSR calculator."""
        self._ref_traj = None
        self._lattice_elements = []
        self._lattice_element_steps = []
        self._lattice_element_traj_steps = []
        self.set_settings()

    def set_settings(self, csr_step=0.1, csr_traj_step=0.0005, n_bins=2000):
        """Set the setting for CSR calculation."""
        self._csr_step = csr_step
        self._csr_traj_step = csr_traj_step
        self._n_bins = n_bins

    def add_lattice_element(self, element):
        """Add lattice element to CSR calculator"""
        self._lattice_elements.append(element)
        self._set_element_steps(element)
        self._calculate_trajectory(element)

    def get_csr_step(self, element):
        """Return the CSR step size used in the specified lattice element."""
        el_idx = self._lattice_elements.index(element)
        ds_csr, ds_traj = self._lattice_element_traj_steps[el_idx]
        return ds_csr

    def clear(self):
        """
        Clear the CSR trajectory and list of lattice elements.

        Calling this method allows for a new CSR calculation to be started
        without being affected by the previous beamline.

        """
        self._ref_traj = None
        self._lattice_elements = []
        self._lattice_element_steps = []

    def apply_csr(self, bunch_matrix, bunch_q, gamma, element, element_pos):
        """
        Apply the CSR kick to the specified particle bunch at a certain
        lattice location.

        The implementation of this method is an adaptation of code from OCELOT
        (https://github.com/ocelot-collab/ocelot) written by S. Tomin and
        M. Dohlus.

        Parameters
        ----------

        bunch_matrix : ndarray
            Matrix containing the (x, xp, y, yp, tau, dp) components of each
            bunch particle

        bunch_q : ndarray
            Array containing the charge of each particle.

        gamma : float
            Reference energy used for the CSR calculation.

        element : TMElement
            Lattice element through which the bunch is currently being tracked.

        element_pos : float
            Current position (longitudinal) of the bunch in the lattice
            element.

        """
        el_idx = self._lattice_elements.index(element)
        ds_csr, ds_traj = self._lattice_element_traj_steps[el_idx]

        # Get current position along trajectory.
        s_current = 0
        for el in self._lattice_elements:
            if el is element:
                break
            s_current += el.length
        s_current += element_pos

        # Get line charge profile of bunch as a histogram.
        z = -bunch_matrix[4]
        bunch_hist, bin_edges = np.histogram(z, self._n_bins, weights=bunch_q)

        # Make profile smooth (important to prevent instabilities) using
        # a Gaussian KDE. This method might be changed/optimized.
        bin_size = bin_edges[1] - bin_edges[0]
        n_part = len(bunch_q)
        bin_centers = bin_edges[1:] - bin_size/2
        bunch_pdf = stats.gaussian_kde(
            bin_centers, bw_method=n_part**(-1/5), weights=bunch_hist)
        bunch_hist = bunch_pdf(bin_centers) * bin_size * bunch_hist.sum()
        bin_center_0 = bin_centers[0]

        # Determine iteration range.
        s_array = self._ref_traj[0, :]
        idx = (np.abs(s_array - s_current)).argmin()
        idx_prev = (np.abs(s_array - (s_current - ds_csr))).argmin()
        it_range = np.arange(idx_prev, idx) + 1
        n_iter = len(it_range)

        # Calculate CSR kernel.
        K1 = 0
        for it in it_range:
            K1 += self._calculate_kernel(it, self._ref_traj, self._n_bins,
                                         bin_size, gamma)
        K1 /= n_iter

        # Convolve kernel with line charge
        lam_K1 = np.convolve(bunch_hist, K1[::-1]) / bin_size * ds_csr

        # Calculate and apply energy kick
        z_norm = z * (1./bin_size) - bin_center_0/bin_size
        dE = np.interp(z_norm, np.arange(len(lam_K1)), lam_K1)
        pc_ref = np.sqrt(gamma**2 - 1) * 0.511e-3
        delta_p = dE * 1e-9 / pc_ref
        bunch_matrix[5] += delta_p

    def _set_element_steps(self, element):
        """
        Calculate and store the csr and trajectory step size for the
        given element.

        """
        n_out = element.n_out
        ds_csr = self._csr_step
        ds_traj = self._csr_traj_step
        if n_out is not None:
            ds_out = element.length / n_out
            ds_csr = ds_out / np.ceil(ds_out/ds_csr)
        elif element.length < ds_csr:
            ds_csr = element.length
        elif not element.length % ds_csr == 0:
            ds_csr = element.length / np.ceil(element.length/ds_csr)
        ds_traj = ds_csr / np.ceil(ds_csr/ds_traj)
        self._lattice_element_traj_steps.append([ds_csr, ds_traj])

    def _calculate_trajectory(self, element):
        """
        Calculate the reference trajectory along the given element.

        The implementation of this method is an adaptation of code from OCELOT
        (https://github.com/ocelot-collab/ocelot) written by S. Tomin and
        M. Dohlus.

        """
        el_idx = self._lattice_elements.index(element)
        ds_csr, ds_traj = self._lattice_element_traj_steps[el_idx]
        n_steps = int(np.ceil(element.length / ds_traj))
        ds_traj = element.length / n_steps
        traj = np.zeros((7, n_steps))
        if self._ref_traj is None:
            self._ref_traj = np.transpose([[0, 0, 0, 0, 0, 0, 1.]])
        traj_start = self._ref_traj[:, [-1]]
        e1 = traj_start[4:]
        l_traj = np.linspace(ds_traj, element.length, n_steps)
        traj[0, :] = traj_start[0] + l_traj
        if element.theta != 0:
            rho = element.length / element.theta
            # to support element tilt in the future
            tilt = 0
            rho_x = rho * np.cos(tilt)
            rho_y = rho * np.sin(tilt)
            rho_vect = np.array([-rho_y, rho_x, 0])

            n_vect = rho_vect/rho
            e2 = np.cross(n_vect, e1.T).T
            si = np.sin(l_traj / rho)
            co = np.cos(l_traj / rho)
            omco = 2 * np.sin(l_traj / (2*rho))**2

            traj[1:4, :] = traj_start[1:4] + rho * (e1*si + e2*omco)
            traj[4:, :] = e1*co + e2*si

        else:
            traj[1:4, :] = traj_start[1:4] + e1 * l_traj
            traj[4:, :] = e1

        self._ref_traj = np.append(self._ref_traj, traj, axis=1)
        self._lattice_element_steps.append(n_steps)

    def _calculate_kernel(self, i, traj, n_bins, bin_size, gamma):
        """
        Calculate CSR kernel.

        The implementation of this method is an adaptation of code from OCELOT
        (https://github.com/ocelot-collab/ocelot) written by S. Tomin and
        M. Dohlus.

        Parameters:
        -----------

        i : int
            Iteration index.

        traj : ndarray
            Reference trajectory along which CSR forces are calculated

        n_bins : int
            Number of bins of the longitudinal bunch histogram.

        bin_size : int
            Size of the histogram bins.

        gamma: float
            reference gamma to calculate CSR kernel.

        """

        w_range = np.arange(-n_bins, 0) * bin_size

        # Calculate long-range interactions.
        w, KS = self._calculate_kernel_long_range(i, traj, w_range[0], gamma)

        # Calculate short-range interactions.
        w_min = np.min(w)
        if w_range[0] < w_min:
            m = np.where(w_range < w_min)[0][-1]
            KS2 = self._calculate_kernel_short_range(
                i, traj, np.append(w_range[0:m+1], w_min), gamma)
            KS1 = KS[0]
            KS2 = (KS2[-1] - KS2) + KS1
            KS = np.append(KS2[0:-1], np.interp(w_range[m+1:], w, KS))
        else:
            KS = np.interp(w_range, w, KS)

        four_pi_eps0 = 1./(1e-7*ct.c**2)
        K1 = np.diff(np.diff(KS, append=0), append=0) / bin_size / four_pi_eps0
        return K1

    def _calculate_kernel_long_range(self, i, traj, wmin, gamma):
        """
        The implementation of this method is an adaptation of code from OCELOT
        (https://github.com/ocelot-collab/ocelot) written by S. Tomin and
        M. Dohlus.

        For details about the CSR model, see section 2.6 of:
        https://www.desy.de/~dohlus/UWake/
        Two%20Methods%20for%20the%20Calculation%20of%20CSR%20Fields.pdf

        Parameters:
        -----------

        i : int
            Iteration index.

        traj: ndarray
            Reference trajectory along which CSR forces are calculated.

        wmin: float
            Leftmost edge of the longitudinal bunch histogram.

        gamma: float
            reference gamma to calculate CSR kernel.

        """

        # Relativistic parameters
        gamma_sq_inv = 1. / gamma ** 2
        beta_sq = 1. - gamma_sq_inv
        beta = np.sqrt(beta_sq)

        i_0 = self._estimate_start_index(i, traj, wmin, beta)
        # i_0 = 0

        # Reference trajectory positions with respect to the current one.
        s = traj[0, i_0:i] - traj[0, i]

        # Distance (in x, y, z) from the current to previous trajectory points.
        n0 = traj[1, i] - traj[1, i_0:i]
        n1 = traj[2, i] - traj[2, i_0:i]
        n2 = traj[3, i] - traj[3, i_0:i]

        # Distance from the current to previous trajectory points.
        R = np.sqrt(n0*n0 + n1*n1 + n2*n2)

        w = s + beta * R

        # Indices of trajectory locations where w<wmin
        j = np.where(w <= wmin)[0]
        if len(j) > 0:
            j = j[-1]
            w = w[j:i]
            s = s[j:i]
            n0 = n0[j:]
            n1 = n1[j:]
            n2 = n2[j:]
            R = R[j:]
            j += i_0
        else:
            j = i_0

        # Tangential unit vectors along trajectory until current position.
        t4 = traj[4, j:i]
        t5 = traj[5, j:i]
        t6 = traj[6, j:i]

        t4_i = traj[4, i]
        t5_i = traj[5, i]
        t6_i = traj[6, i]

        # Calculate kernel
        R_inv = 1 / R
        n0 *= R_inv
        n1 *= R_inv
        n2 *= R_inv

        x = n0 * t4 + n1 * t5 + n2 * t6
        K = ((beta * (x - n0 * t4_i - n1 * t5_i - n2 * t6_i) -
              beta_sq * (1. - t4 * t4_i - t5 * t5_i - t6 * t6_i) -
              gamma_sq_inv) * R_inv -
             (1. - beta * x) / w * gamma_sq_inv)

        # Integrate kernel
        K[:-1] += K[1:]
        K *= 0.5 * np.diff(s, append=0)
        K = np.cumsum(K[::-1])[::-1]

        return w, K

    def _calculate_kernel_short_range(self, i, traj, w_range, gamma):
        """
        Calculate short-range contributions to CSR kernel.

        The implementation of this method is an adaptation of code from OCELOT
        (https://github.com/ocelot-collab/ocelot) written by S. Tomin and
        M. Dohlus.

        Parameters:
        -----------

        i : int
            Iteration index.

        traj: ndarray
            Reference trajectory along which CSR forces are calculated.

        wrange: ndarray
            Region of beam < w_min.

        gamma: float
            reference gamma to calculate CSR kernel.

        """
        # Relativistic parameters
        gamma_sq = gamma**2
        gamma_sq_inv = 1./gamma_sq
        beta_sq = 1. - gamma_sq_inv
        beta = np.sqrt(beta_sq)

        # winf
        Rv1 = traj[1:4, i] - traj[1:4, 0]
        s1 = traj[0, 0] - traj[0, i]
        ev1 = traj[4:, 0]
        evo = traj[4:, i]
        winfms1 = np.dot(Rv1, ev1)

        aup = -Rv1 + winfms1*ev1
        a2 = np.dot(aup, aup)
        a = np.sqrt(a2)

        uup = aup/a if a != 0 else None

        winf = s1 + winfms1
        s = winf + gamma * (gamma * (w_range - winf) -
                            beta * np.sqrt(gamma_sq*(w_range-winf)**2 + a2))
        R = (w_range-s) / beta

        KS = (beta * (1. - np.dot(ev1, evo)) * np.log(R[0]/R) -
              (beta_sq * np.dot(ev1, evo) - 1.) * np.log(
                  (winf - s + R) / (winf - s[0] + R[0])) +
              gamma_sq_inv * np.log(w_range[0]/w_range))
        if a2/R[1]**2 > 1e-7:
            KS -= (beta * np.dot(uup, evo) *
                   (np.arctan((s[0] - winf)/a) - np.arctan((s-winf)/a)))
        return KS

    def _estimate_start_index(
            self, i, traj, w_min, beta, i_min=1000, n_test=10):
        """
        This method estimates the index of the first trajectory point from
        which CSR effects should be computed.

        This method can significantly reduce the computing time of CSR effects
        by pre-discaring regions of the reference trajectory which do not
        influence the CSR calculation (i.e. the points where w <= wmin). This
        is performed by testing the w <= wmin condition for a subset of n_test
        equally-spaced points along the reference trajectory. The index of the
        last tested trajectory point in which w <= wmin is returned.

        Parameters
        ----------

        i : int
            Iteration index

        traj : ndarray
            Reference trajectory along which CSR forces are calculated

        w_min : float
            Leftmost edge of the longitudinal bunch binning.

        beta : float
            Relativistic factor.

        i_min : int
            Minimum iteration index. When i<i_min, no estimation of the
            starting index is performed (0 is returned).

        n_test : int
            Number of points along the trajectory in which to test whether they
            should be taken into account for the CSR calculation.

        Returns
        -------
        The estimated start index, which is always <= than the real one.

        """
        i_0 = 0
        if i > i_min:
            idx = np.linspace(0, i, n_test, dtype=np.int32)
            s = traj[0, idx] - traj[0, i]
            n0 = traj[1, i] - traj[1, idx]
            n1 = traj[2, i] - traj[2, idx]
            n2 = traj[3, i] - traj[3, idx]
            R = np.sqrt(n0*n0 + n1*n1 + n2*n2)
            w = s + beta * R
            j = np.where(w <= w_min)[0]
            if len(j) > 0:
                i_0 = idx[j[-1]]
        return i_0


_csr_calculator = CSRCalculator()


def get_csr_calculator():
    """Return the single instance of CSRCalculator"""
    return _csr_calculator


def set_csr_settings(csr_step=0.1, csr_traj_step=0.0005, n_bins=2000):
    """
    Set the setting for CSR calculation.

    Parameters:
    -----------

    csr_step : float
        Iteration index.

    csr_traj_step : float
        Reference trajectory along which CSR forces are calculated.

    n_bins : int
        Number of bins used for determining the longitudinal charge profile
        of the bunch.

    """
    _csr_calculator.set_settings(csr_step, csr_traj_step, n_bins)


def reset_csr_calculator():
    """
    Reset CSR calculator by clearing the stored reference trajectory and
    lattice elements. Needed to start a new, independent calculation after
    a previous one.

    """
    _csr_calculator.clear()
