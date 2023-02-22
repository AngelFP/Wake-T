"""Contains the definition of the `PlasmaParticles` class."""

import numpy as np


class PlasmaParticles():
    """
    Class containing the 1D slice of plasma particles used in the quasi-static
    Baxevanis wakefield model.

    Parameters
    ----------
    r_max : float
        Maximum radial extension of the simulation box in normalized units.
    r_max_plasma : float
        Maximum radial extension of the plasma column in normalized units.
    parabolic_coefficient : float
        The coefficient for the transverse parabolic density profile.
    dr : float
        Radial step size of the discretized simulation box.
    ppc : int
        Number of particles per cell.
    pusher : str
        Particle pusher used to evolve the plasma particles. Possible
        values are `'rk4'` and `'ab5'`.

    """

    def __init__(self, r_max, r_max_plasma, parabolic_coefficient, dr, ppc,
                 pusher):
        # Calculate total number of plasma particles.
        n_part = int(np.round(r_max_plasma / dr * ppc))

        # Readjust plasma extent to match number of particles.
        dr_p = dr / ppc
        r_max_plasma = n_part * dr_p

        # Store parameters.
        self.r_max = r_max
        self.r_max_plasma = r_max_plasma
        self.parabolic_coefficient = parabolic_coefficient
        self.dr = dr
        self.ppc = ppc
        self.dr_p = dr / ppc
        self.pusher = pusher
        self.n_part = n_part

    def initialize(self):
        """Initialize column of plasma particles."""

        # Initialize particle arrays.
        self.r = np.linspace(
            self.dr_p / 2, self.r_max_plasma - self.dr_p / 2, self.n_part)
        self.pr = np.zeros(self.n_part)
        self.pz = np.zeros(self.n_part)
        self.gamma = np.ones(self.n_part)
        self.q = (self.dr_p * self.r
                  + self.dr_p * self.parabolic_coefficient * self.r**3)

        # Allocate arrays that will contain the fields experienced by the
        # particles.
        self.allocate_field_arrays()

        # Allocate arrays needed for the particle pusher.
        if self.pusher == 'ab5':
            self.allocate_ab5_arrays()
        elif self.pusher == 'rk4':
            self.allocate_rk4_arrays()
            self.allocate_rk4_field_arrays()

    def allocate_field_arrays(self):
        """Allocate arrays for the fields experienced by the particles.

        In order to evolve the particles to the next longitudinal position,
        it is necessary to know the fields that they are experiencing. These
        arrays are used for storing the value of these fields at the location
        of each particle.
        """
        self.__a2 = np.zeros(self.n_part)
        self.__nabla_a2 = np.zeros(self.n_part)
        self.__b_t_0 = np.zeros(self.n_part)
        self.__b_t = np.zeros(self.n_part)
        self.__psi = np.zeros(self.n_part)
        self.__dr_psi = np.zeros(self.n_part)
        self.__dxi_psi = np.zeros(self.n_part)
        self.__field_arrays = [
            self.__a2, self.__nabla_a2, self.__b_t_0, self.__b_t,
            self.__psi, self.__dr_psi, self.__dxi_psi
        ]

    def get_field_arrays(self):
        """Get arrays containing the fields experienced by the particles."""
        return self.__field_arrays

    def allocate_ab5_arrays(self):
        """Allocate the arrays needed for the 5th order Adams-Bashforth pusher.

        The AB5 pusher needs the derivatives of r and pr for each particle
        at the last 5 plasma slices. This method allocates the arrays that will
        store these derivatives.
        """
        self.__dr_1 = np.zeros(self.n_part)
        self.__dr_2 = np.zeros(self.n_part)
        self.__dr_3 = np.zeros(self.n_part)
        self.__dr_4 = np.zeros(self.n_part)
        self.__dr_5 = np.zeros(self.n_part)
        self.__dpr_1 = np.zeros(self.n_part)
        self.__dpr_2 = np.zeros(self.n_part)
        self.__dpr_3 = np.zeros(self.n_part)
        self.__dpr_4 = np.zeros(self.n_part)
        self.__dpr_5 = np.zeros(self.n_part)
        self.__dr_arrays = [
            self.__dr_1, self.__dr_2, self.__dr_3, self.__dr_4, self.__dr_5]
        self.__dpr_arrays = [
            self.__dpr_1, self.__dpr_2, self.__dpr_3, self.__dpr_4,
            self.__dpr_5]

    def get_ab5_arrays(self):
        """Get the arrays needed by the 5th order Adams-Bashforth pusher."""
        return self.__dr_arrays, self.__dpr_arrays

    def allocate_rk4_arrays(self):
        """Allocate the arrays needed for the 4th order Runge-Kutta pusher.

        The RK4 pusher needs the derivatives of r and pr for each particle at
        the current slice and at 3 intermediate substeps. This method allocates
        the arrays that will store these derivatives.
        """
        self.__dr_1 = np.zeros(self.n_part)
        self.__dr_2 = np.zeros(self.n_part)
        self.__dr_3 = np.zeros(self.n_part)
        self.__dr_4 = np.zeros(self.n_part)
        self.__dpr_1 = np.zeros(self.n_part)
        self.__dpr_2 = np.zeros(self.n_part)
        self.__dpr_3 = np.zeros(self.n_part)
        self.__dpr_4 = np.zeros(self.n_part)
        self.__dr_arrays = [self.__dr_1, self.__dr_2, self.__dr_3, self.__dr_4]
        self.__dpr_arrays = [
            self.__dpr_1, self.__dpr_2, self.__dpr_3, self.__dpr_4]

    def get_rk4_arrays(self):
        """Get the arrays needed by the 4th order Runge-Kutta pusher."""
        return self.__dr_arrays, self.__dpr_arrays

    def allocate_rk4_field_arrays(self):
        """Allocate field arrays needed by the 4th order Runge-Kutta pusher.

        In order to compute the derivatives of r and pr at the 3 subteps
        of the RK4 pusher, the field values at the location of the particles
        in these substeps are needed. This method allocates the arrays
        that will store these field values.
        """
        self.__a2_2 = np.zeros(self.n_part)
        self.__nabla_a2_2 = np.zeros(self.n_part)
        self.__b_t_0_2 = np.zeros(self.n_part)
        self.__b_t_2 = np.zeros(self.n_part)
        self.__psi_2 = np.zeros(self.n_part)
        self.__dr_psi_2 = np.zeros(self.n_part)
        self.__dxi_psi_2 = np.zeros(self.n_part)
        self.__a2_3 = np.zeros(self.n_part)
        self.__nabla_a2_3 = np.zeros(self.n_part)
        self.__b_t_0_3 = np.zeros(self.n_part)
        self.__b_t_3 = np.zeros(self.n_part)
        self.__psi_3 = np.zeros(self.n_part)
        self.__dr_psi_3 = np.zeros(self.n_part)
        self.__dxi_psi_3 = np.zeros(self.n_part)
        self.__a2_4 = np.zeros(self.n_part)
        self.__nabla_a2_4 = np.zeros(self.n_part)
        self.__b_t_0_4 = np.zeros(self.n_part)
        self.__b_t_4 = np.zeros(self.n_part)
        self.__psi_4 = np.zeros(self.n_part)
        self.__dr_psi_4 = np.zeros(self.n_part)
        self.__dxi_psi_4 = np.zeros(self.n_part)
        self.__rk4_flds = [
            [self.__a2, self.__nabla_a2, self.__b_t_0, self.__b_t,
             self.__psi, self.__dr_psi, self.__dxi_psi],
            [self.__a2_2, self.__nabla_a2_2, self.__b_t_0_2, self.__b_t_2,
             self.__psi_2, self.__dr_psi_2, self.__dxi_psi_2],
            [self.__a2_3, self.__nabla_a2_3, self.__b_t_0_3, self.__b_t_3,
             self.__psi_3, self.__dr_psi_3, self.__dxi_psi_3],
            [self.__a2_4, self.__nabla_a2_4, self.__b_t_0_4, self.__b_t_4,
             self.__psi_4, self.__dr_psi_4, self.__dxi_psi_4]
        ]

    def get_rk4_field_arrays(self, i):
        """Get field arrays for the four substeps of the RK4 pusher."""
        return self.__rk4_flds[i]
