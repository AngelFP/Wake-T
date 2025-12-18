"""Contains the definition of the `PlasmaParticleContainerPy` class."""

import numba


class PlasmaParticleContainerPy:
    # Particle properties
    r: numba.float64[::1]  # radial position
    log_r: numba.float64[::1]  # logarithm of radial position
    pr: numba.float64[::1]  # momentum in r
    pz: numba.float64[::1]  # momentum in z
    gamma: numba.float64[::1]  # lorentz factor
    w: numba.float64[::1]  # weight
    w_center: numba.float64[::1]  # weight until the particle center.
    r_to_x: numba.int32[::1]  # -1 if the particle crossed r=0, else +1
    id: numba.int32[::1]  # unique (within a species) id
    mass: float  # single particle mass
    charge: float  # single particle charge

    # Fields at particles
    psi: numba.float64[::1]  # plasma wakefield potential
    dr_psi: numba.float64[::1]  # focussing field
    dxi_psi: numba.float64[::1]  # accelerating field
    b_t: numba.float64[::1]  # theta component of magnetic field
    b_t_0: numba.float64[::1]  # beam contribution to b_t
    nabla_a2: numba.float64[::1]  # gradient of laser envelope
    a2: numba.float64[::1]  # absolute value squared of laser envelope
    rho: numba.float64[::1]  # charge density
    chi: numba.float64[::1]  # refractive index

    # Temp arrays for field solvers
    a_i: numba.float64[::1]  # b_t
    b_i: numba.float64[::1]  # b_t
    sum_1: numba.float64[::1]  # psi
    sum_2: numba.float64[::1]  # psi
    sum_3: numba.float64[::1]  # dxi_psi
    a_0: numba.float64[::1]  # b_t
    A: numba.float64[::1]  # b_t
    B: numba.float64[::1]  # b_t
    C: numba.float64[::1]  # b_t
    K: numba.float64[::1]  # b_t
    U: numba.float64[::1]  # b_t

    # AB2 pusher
    dr: numba.float64[:, ::1]  # last two values of dzeta_r for AB2 pusher
    dpr: numba.float64[:, ::1]  # last two values of dzeta_pr for AB2 pusher

    # History arrays, usee the same name as above
    r_hist: numba.float64[:, ::1]
    log_r_hist: numba.float64[:, ::1]
    xi_hist: numba.float64[:, ::1]
    pr_hist: numba.float64[:, ::1]
    pz_hist: numba.float64[:, ::1]
    w_hist: numba.float64[:, ::1]
    r_to_x_hist: numba.int32[:, ::1]
    id_hist: numba.int32[:, ::1]
    sum_1_hist: numba.float64[:, ::1]
    sum_2_hist: numba.float64[:, ::1]
    a_i_hist: numba.float64[:, ::1]
    b_i_hist: numba.float64[:, ::1]
    a_0_hist: numba.float64[::1]
    i_push: int  # number of slices pushed
    xi_current: float  # current zeta position

    # Flags
    num_particles: int  # number of particles in this species
    is_ion: bool  # is this an ion or electron. Ions don't contribute to b_t and chi
    do_push: bool  # if this particle should be pushed or if it is static
    store_history: bool  # if the full trajectory of the particles should be stored

    def __init__(self, serialized_list=None):
        if serialized_list is not None:
            (
                self.r,
                self.log_r,
                self.pr,
                self.pz,
                self.gamma,
                self.w,
                self.w_center,
                self.r_to_x,
                self.id,
                self.mass,
                self.charge,
                self.psi,
                self.dr_psi,
                self.dxi_psi,
                self.b_t,
                self.b_t_0,
                self.nabla_a2,
                self.a2,
                self.rho,
                self.chi,
                self.a_i,
                self.b_i,
                self.sum_1,
                self.sum_2,
                self.sum_3,
                self.a_0,
                self.A,
                self.B,
                self.C,
                self.K,
                self.U,
                self.dr,
                self.dpr,
                self.r_hist,
                self.log_r_hist,
                self.xi_hist,
                self.pr_hist,
                self.pz_hist,
                self.w_hist,
                self.r_to_x_hist,
                self.id_hist,
                self.sum_1_hist,
                self.sum_2_hist,
                self.a_i_hist,
                self.b_i_hist,
                self.a_0_hist,
                self.i_push,
                self.xi_current,
                self.num_particles,
                self.is_ion,
                self.do_push,
                self.store_history,
            ) = serialized_list

    def serialize(self):
        return (
            self.r,
            self.log_r,
            self.pr,
            self.pz,
            self.gamma,
            self.w,
            self.w_center,
            self.r_to_x,
            self.id,
            self.mass,
            self.charge,
            self.psi,
            self.dr_psi,
            self.dxi_psi,
            self.b_t,
            self.b_t_0,
            self.nabla_a2,
            self.a2,
            self.rho,
            self.chi,
            self.a_i,
            self.b_i,
            self.sum_1,
            self.sum_2,
            self.sum_3,
            self.a_0,
            self.A,
            self.B,
            self.C,
            self.K,
            self.U,
            self.dr,
            self.dpr,
            self.r_hist,
            self.log_r_hist,
            self.xi_hist,
            self.pr_hist,
            self.pz_hist,
            self.w_hist,
            self.r_to_x_hist,
            self.id_hist,
            self.sum_1_hist,
            self.sum_2_hist,
            self.a_i_hist,
            self.b_i_hist,
            self.a_0_hist,
            self.i_push,
            self.xi_current,
            self.num_particles,
            self.is_ion,
            self.do_push,
            self.store_history,
        )


# PlasmaParticleContainer can be used inside numba jit functions
# PlasmaParticleContainerPy can be used outside numba jit functions.
# Both can not be used as parameters that pass from outside to inside a numba jit function
# because that would break function caching
PlasmaParticleContainer = numba.experimental.jitclass(PlasmaParticleContainerPy)
