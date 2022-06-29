import numpy as np


class PlasmaParticles():
    """
    Class containing the 1D slice of plasma particles used in the quasi-static
    Baxevanis wakefield model.
    """

    def __init__(self, r_max, r_max_plasma, parabolic_coefficient, dr, ppc,
                 pusher):
        """Create particle collection.

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
        
        n_part = int(np.round(r_max_plasma / dr * ppc))
        # Readjust plasma extent to match number of particles.
        dr_p = dr / ppc
        r_max_plasma = n_part * dr_p

        self.r_max = r_max
        self.r_max_plasma = r_max_plasma
        self.parabolic_coefficient = parabolic_coefficient
        self.dr = dr
        self.ppc = ppc
        self.dr_p = dr / ppc
        self.pusher = pusher
        self.n_part = n_part

    def initialize(self):        
        self.r = np.linspace(self.dr_p / 2, self.r_max_plasma - self.dr_p / 2, self.n_part)
        self.pr = np.zeros(self.n_part)
        self.pz = np.zeros(self.n_part)
        self.gamma = np.ones(self.n_part)
        self.q = self.dr_p * self.r + self.dr_p * self.parabolic_coefficient * self.r**3
        self.allocate_field_arrays()
        if self.pusher == 'ab5':
            self.allocate_ab5_arrays()

    def allocate_ab5_arrays(self):
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
        self.__dr_arrays = [self.__dr_1, self.__dr_2, self.__dr_3, self.__dr_4, self.__dr_5]
        self.__dpr_arrays = [self.__dpr_1, self.__dpr_2, self.__dpr_3, self.__dpr_4, self.__dpr_5]

    def get_ab5_arrays(self):
        return self.__dr_arrays, self.__dpr_arrays

    def allocate_field_arrays(self):
        self.__a2 = np.zeros(self.n_part)
        self.__nabla_a2 = np.zeros(self.n_part)
        self.__b_theta_0 = np.zeros(self.n_part)
        self.__b_theta = np.zeros(self.n_part)
        self.__psi = np.zeros(self.n_part)
        self.__dr_psi = np.zeros(self.n_part)
        self.__dxi_psi = np.zeros(self.n_part)

    def get_field_arrays(self):
        return self.__a2, self.__nabla_a2, self.__b_theta_0, self.__b_theta

    def get_psi_arrays(self):
        return self.__psi, self.__dr_psi, self.__dxi_psi
