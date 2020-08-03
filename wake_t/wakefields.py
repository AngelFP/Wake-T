""" This module contains the possible plasma wakefields """

import numpy as np
import scipy.constants as ct
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, interp2d
import aptools.plasma_accel.general_equations as ge
import matplotlib.pyplot as plt
try:
    from VisualPIC.DataHandling.dataContainer import DataContainer
    vpic_installed = True
except ImportError:
    vpic_installed = False

from wake_t.quasistatic_2d import calculate_wakefield


class Wakefield():

    """ Base class for all wakefields """

    def __init__(self):
        pass

    def Wx(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError

    def Wy(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError

    def Wz(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError

    def Kx(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError

    def Ez_p(self, x, y, xi, px, py, pz, q, t):
        raise NotImplementedError


class SimpleBlowoutWakefield(Wakefield):
    def __init__(self, n_p, driver, field_offset=0):
        """
        [n_p] = m^-3
        """
        self.n_p = n_p
        self.field_off = field_offset
        self.driver = driver
        self._calculate_base_quantities()

    def _calculate_base_quantities(self):
        w_p = ge.plasma_frequency(self.n_p*1e-6)
        self.l_p = 2*np.pi*ct.c / w_p
        self.g_x = w_p**2/2 * ct.m_e / (ct.e * ct.c)
        self.E_z_p = w_p**2/2 * ct.m_e / ct.e
        self.l_c = self.driver.xi_c
        self.b_w = self.driver.get_group_velocity(self.n_p)

    def Wx(self, x, y, xi, px, py, pz, q, t):
        return ct.c * self.g_x*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        return ct.c * self.g_x*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        return self.E_z_p * (self.l_p/2 + xi - self.l_c - self.field_off +
                             (1-self.b_w)*ct.c*t)

    def Kx(self, x, y, xi, px, py, pz, q, t):
        return self.g_x*np.ones_like(x)

    def Ez_p(self, x, y, xi, px, py, pz, q, t):
        return self.E_z_p*np.ones_like(x)


class CustomBlowoutWakefield(Wakefield):
    def __init__(self, n_p, driver, lon_field=None, lon_field_slope=None,
                 foc_strength=None, xi_fields=None):
        """
        [n_p] = m^-3
        """
        self.n_p = n_p
        self.xi_fields = xi_fields
        self.driver = driver
        self._calculate_base_quantities(lon_field, lon_field_slope,
                                        foc_strength)

    def _calculate_base_quantities(self, lon_field, lon_field_slope,
                                   foc_strength):
        self.g_x = foc_strength
        self.E_z_0 = lon_field
        self.E_z_p = lon_field_slope
        self.l_c = self.driver.xi_c
        self.b_w = self.driver.get_group_velocity(self.n_p)

    def Wx(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.g_x*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.g_x*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        if self.xi_fields is None:
            self.xi_fields = np.average(xi, weights=q)
        return self.E_z_0 + self.E_z_p*(xi - self.xi_fields
                                        + (1-self.b_w)*ct.c*t)

    def Kx(self, x, y, xi, px, py, pz, q, t):
        return self.g_x*np.ones_like(x)

    def Kz(self, x, y, xi, px, py, pz, q, t):
        return self.E_z_p*np.ones_like(x)


class WakefieldFromPICSimulation(Wakefield):
    def __init__(self, simulation_code, simulation_path, driver, timestep=0,
                 n_p=None, filter_fields=False, sigma_filter=20,
                 reverse_tracking=False):
        """
        [n_p] = m^-3
        """
        self.driver = driver
        self.b_w = driver.get_group_velocity(n_p)
        self.filter_fields = filter_fields
        self.sigma_filter = sigma_filter
        self.reverse_tracking = reverse_tracking
        self._load_fields(simulation_code, simulation_path, driver, timestep,
                          n_p)

    def _load_fields(self, simulation_code, simulation_path, driver, timestep,
                     n_p):
        # Load data
        self.dc = DataContainer()
        self.dc.SetDataFolderLocation(simulation_path)
        simulation_parameters = {"isLaser": False,
                                 "SimulationCode": simulation_code}
        if n_p is not None:
            simulation_parameters["n_p"] = n_p/1e24
        self.dc.SetSimulationParameters(simulation_parameters)
        self.dc.LoadData()

        # time
        self.current_ts = timestep
        self.timesteps = []

        self.create_fields()
        print("Done.")
        # todo: implement separate components for transverse fields

    def check_if_update_fields(self, time):
        possible_ts = np.where(self.timesteps_in_sec < time)[0]
        if len(possible_ts) > 1:
            if not self.reverse_tracking:
                requested_ts_index = possible_ts[-1]
                current_ts_index = np.where(
                    self.timesteps == self.current_ts)[0][0]
                if current_ts_index < requested_ts_index:
                    # update current time step
                    self.current_ts = self.timesteps[requested_ts_index]
                    print("Updating fields using timestep {} ...".format(
                        self.current_ts))
                    self.create_fields()
                    print("Done.")
            else:
                requested_ts_index = possible_ts[0]
                current_ts_index = np.where(
                    self.timesteps == self.current_ts)[0][0]
                if current_ts_index > requested_ts_index:
                    self.current_ts = self.timesteps[requested_ts_index]
                    print("Updating fields using timestep {} ...".format(
                        self.current_ts))
                    self.create_fields()
                    print("Done.")

    def create_fields(self):
        # Simulation geometry
        geom = self.dc.GetSimulationDimension()

        # Read fields
        E_z_field = self.dc.GetDomainField("Ez")
        W_x_field = self.dc.GetDomainField("Wx")
        K_x_field = self.dc.GetDomainField("dx Wx")

        # read data
        E_z_data = E_z_field.GetAllFieldDataISUnits(self.current_ts)
        W_x_data = W_x_field.GetAllFieldData(self.current_ts, "V/m")
        K_x_data = K_x_field.GetAllFieldData(self.current_ts, "T/m")

        # filter noise in data
        if self.filter_fields:
            s = self.sigma_filter
            E_z_data = ndimage.gaussian_filter1d(E_z_data, s, 1)
            W_x_data = ndimage.gaussian_filter1d(W_x_data, s, 1)
            K_x_data = ndimage.gaussian_filter1d(K_x_data, s, 1)

        # axes
        if geom == "3D":
            z_axis = E_z_field.GetAxisInISUnits("z", self.current_ts)
            xi_axis = z_axis - np.min(z_axis)
            x_axis = E_z_field.GetAxisInISUnits("x", self.current_ts)
            y_axis = E_z_field.GetAxisInISUnits("y", self.current_ts)
        elif geom == "thetaMode":
            z_axis = E_z_field.GetAxisInISUnits("z", self.current_ts)
            xi_axis = z_axis - np.min(z_axis)
            r_axis = E_z_field.GetAxisInISUnits("r", self.current_ts)

        # create interpolation functions
        if geom == "3D":
            self.E_z = RegularGridInterpolator((x_axis, y_axis, xi_axis),
                                               E_z_data, fill_value=0,
                                               bounds_error=False)
            self.W_x = RegularGridInterpolator((x_axis, y_axis, xi_axis),
                                               W_x_data, fill_value=0,
                                               bounds_error=False)
            self.K_x = RegularGridInterpolator((x_axis, y_axis, xi_axis),
                                               K_x_data, fill_value=0,
                                               bounds_error=False)
        elif geom == "thetaMode":
            self.E_z = RegularGridInterpolator((r_axis, xi_axis), E_z_data,
                                               fill_value=0,
                                               bounds_error=False)
            self.W_x = RegularGridInterpolator((r_axis, xi_axis), W_x_data,
                                               fill_value=0,
                                               bounds_error=False)
            self.K_x = RegularGridInterpolator((r_axis, xi_axis), K_x_data,
                                               fill_value=0,
                                               bounds_error=False)

        if len(self.timesteps) == 0:
            self.timesteps = E_z_field.GetTimeSteps()
            self.timesteps_in_sec = np.zeros(len(self.timesteps))
            for i, ts in enumerate(self.timesteps):
                self.timesteps_in_sec[i] = E_z_field.GetTimeInUnits("s", ts)
            if self.reverse_tracking:
                current_ts_index = np.where(
                    self.timesteps == self.current_ts)[0][0]
                self.timesteps = self.timesteps[:current_ts_index+1]
                self.timesteps_in_sec = self.timesteps_in_sec[
                    :current_ts_index+1]
                self.timesteps_in_sec = self.timesteps_in_sec[::-1]

    def Wx(self, x, y, xi, px, py, pz, q, t):
        geom = self.dc.GetSimulationDimension()
        # matrix of coordinate points
        if geom == "3D":
            R = np.array([x, y, xi + (1-self.b_w)*ct.c*t]).T
        elif geom == "thetaMode":
            R = np.array([np.sqrt(np.square(x)+np.square(y)),
                          xi + (1-self.b_w)*ct.c*t]).T
            theta = np.arctan2(x, y)
        return self.W_x(R)*np.sin(theta)

    def Wy(self, x, y, xi, px, py, pz, q, t):
        geom = self.dc.GetSimulationDimension()
        if geom == "3D":
            R = np.array([y, x, xi + (1-self.b_w)*ct.c*t]).T
        elif geom == "thetaMode":
            R = np.array([np.sqrt(np.square(x)+np.square(y)),
                          xi + (1-self.b_w)*ct.c*t]).T
            theta = np.arctan2(x, y)
        return self.W_x(R)*np.cos(theta)

    def Wz(self, x, y, xi, px, py, pz, q, t):
        geom = self.dc.GetSimulationDimension()
        if geom == "3D":
            R = np.array([x, y, xi + (1-self.b_w)*ct.c*t]).T
        elif geom == "thetaMode":
            R = np.array([np.sqrt(np.square(x)+np.square(y)),
                          xi + (1-self.b_w)*ct.c*t]).T
        return self.E_z(R)

    def Kx(self, x, y, xi, px, py, pz, q, t):
        geom = self.dc.GetSimulationDimension()
        # matrix of coordinate points
        if geom == "3D":
            R = np.array([x, y, xi + (1-self.b_w)*ct.c*t]).T
        elif geom == "thetaMode":
            R = np.array([np.sqrt(np.square(x)+np.square(y)),
                          xi + (1-self.b_w)*ct.c*t]).T
        return self.K_x(R)


class NonLinearColdFluidWakefield(Wakefield):
    def __init__(self, density_function, driver=None, laser_evolution=False,
                 laser_z_foc=0, r_max=None, xi_min=None, xi_max=None, n_r=100,
                 n_xi=100, beam_wakefields=False):
        self.density_function = density_function
        self.driver = driver
        self.laser_evolution = laser_evolution
        self.laser_z_foc = laser_z_foc
        self.r_max = r_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.n_r = n_r
        self.n_xi = n_xi
        self.beam_wakefields = beam_wakefields
        self.current_t = -1
        self.current_n_p = None

    def __wakefield_ode_system(self, u_1, u_2, r, z, laser_a0, n_beam):
        if self.beam_wakefields:
            return np.array([u_2, (1+laser_a0**2)/(2*(1+u_1)**2) + n_beam - 1/2])
        # return np.array([u_2, laser_a0**2/2 - u_1]) # linear regime
        else:
            return np.array([u_2, (1+laser_a0**2)/(2*(1+u_1)**2) - 1/2])

    def __calculate_wakefields(self, x, y, xi, px, py, pz, q, t):
        if self.current_t != t:
            self.current_t = t
        else:
            return
        z_beam = t*ct.c + np.average(xi)  # z postion of beam center
        n_p = self.density_function(z_beam)

        if n_p == self.current_n_p and not self.laser_evolution:
            # If density has not changed and driver does not evolve, it is
            # not necessary to recompute fields.
            return
        self.current_n_p = n_p

        s_d = ge.plasma_skin_depth(n_p*1e-6)
        r = np.linspace(0, self.r_max, self.n_r)
        dz = (self.xi_max - self.xi_min) / self.n_xi / s_d
        dr = self.r_max / self.n_r / s_d
        r_part = np.sqrt(x**2 + y**2)
        beam_hist, *_ = np.histogram2d(xi, r_part,
                                       bins=[self.n_xi, self.n_r],
                                       range=[[self.xi_min, self.xi_max],
                                              [0, self.r_max]],
                                       weights=q/ct.e)
        # ,
        # weights=1/(ct.pi*dr*2*dz))
        n = np.arange(self.n_r)
        disc_area = np.pi * dr**2*(1+2*n)
        beam_hist *= 1/(disc_area*dz*n_p)/s_d**3
        n_iter = self.n_xi - 1
        u_1 = np.zeros((n_iter+1, len(r)))
        u_2 = np.zeros((n_iter+1, len(r)))
        z_arr = np.zeros(n_iter+1)
        z_arr[-1] = self.xi_max / s_d
        # calculate distance to laser focus
        if self.laser_evolution:
            dist_z_foc = self.laser_z_foc - ct.c*t
        else:
            dist_z_foc = 0
        for i in np.arange(n_iter):
            z = z_arr[-1] - i*dz
            # get laser a0 at z, z+dz/2 and z+dz
            if self.driver is not None:
                a0_0 = self.driver.get_a0_profile(r, z*s_d, dist_z_foc)
                a0_1 = self.driver.get_a0_profile(r, (z - dz/2)*s_d, dist_z_foc)
                a0_2 = self.driver.get_a0_profile(r, (z - dz)*s_d, dist_z_foc)
            else:
                a0_0 = np.zeros(r.shape[0])
                a0_1 = np.zeros(r.shape[0])
                a0_2 = np.zeros(r.shape[0])
            # perform runge-kutta
            A = dz*self.__wakefield_ode_system(
                u_1[-1-i], u_2[-1-i], r, z*s_d, a0_0, beam_hist[-(i+1)])
            B = dz*self.__wakefield_ode_system(
                u_1[-1-i] + A[0]/2, u_2[-1-i] + A[1]/2, r, (z - dz/2)*s_d,
                a0_1, beam_hist[-(i+1)])
            C = dz*self.__wakefield_ode_system(
                u_1[-1-i] + B[0]/2, u_2[-1-i] + B[1]/2, r, (z - dz/2)*s_d,
                a0_1, beam_hist[-(i+1)])
            D = dz*self.__wakefield_ode_system(
                u_1[-1-i] + C[0], u_2[-1-i] + C[1], r, (z - dz)*s_d, a0_2,
                beam_hist[-(i+1)])
            u_1[-2-i] = u_1[-1-i] + 1/6*(A[0] + 2*B[0] + 2*C[0] + D[0])
            u_2[-2-i] = u_2[-1-i] + 1/6*(A[1] + 2*B[1] + 2*C[1] + D[1])
            z_arr[-2-i] = z - dz
        E_z = -np.gradient(u_1, dz, axis=0, edge_order=2)
        E_z_p = np.gradient(E_z, dz, axis=0, edge_order=2)
        W_r = -np.gradient(u_1, dr, axis=1, edge_order=2)
        K_r = np.gradient(W_r, dr, axis=1, edge_order=2)
        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p*1e-6)

        # For debugging
        # plt.plot(E_z[:,0])
        # plt.plot(K_r[:,0])
        # plt.plot(a0_0[:,0])
        # plt.show()

        # plt.subplot(411)
        # plt.imshow(E_z.T*E_0, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.plot(E_z[:,0]*E_0)
        # plt.subplot(412)
        # plt.imshow(W_r.T*E_0, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.subplot(413)
        # plt.imshow(E_z_p.T*E_0/s_d, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.subplot(414)
        # plt.imshow(beam_hist.T, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.show()

        self.E_z = RectBivariateSpline(z_arr*s_d, r, E_z*E_0, kx=2, ky=2)
        self.W_x = RectBivariateSpline(z_arr*s_d, r, W_r*E_0, kx=2, ky=2)
        self.K_x = RectBivariateSpline(z_arr*s_d, r, K_r*E_0/s_d/ct.c, kx=2, ky=2)
        self.E_z_p = RectBivariateSpline(z_arr*s_d, r, E_z_p*E_0/s_d, kx=2, ky=2)

    def Wx(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        sin = x / r
        return self.W_x(xi, r, grid=False) * sin

    def Wy(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        cos = y / r
        return self.W_x(xi, r, grid=False) * cos

    def Wz(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        return self.E_z(xi, r, grid=False)

    def Kx(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        return self.K_x(xi, r, grid=False)

    def Ez_p(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        return self.E_z_p(xi, r, grid=False)


class Quasistatic2DWakefield(Wakefield):

    def __init__(self, density_function, laser=None, laser_evolution=False,
                 laser_z_foc=0, r_max=None, xi_min=None, xi_max=None, n_r=100,
                 n_xi=100, n_part=1000, dz_fields=0):
        self.density_function = density_function
        self.laser = laser
        self.laser_evolution = laser_evolution
        self.laser_z_foc = laser_z_foc
        self.r_max = r_max
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.n_r = n_r
        self.n_xi = n_xi
        self.n_part = n_part
        self.dz_fields = np.inf if dz_fields is None else dz_fields
        self.current_t = None

    def Wx(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        sin = x / r
        return self.W_x(xi, r, grid=False) * sin

    def Wy(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        cos = y / r
        return self.W_x(xi, r, grid=False) * cos

    def Wz(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        return self.E_z(xi, r, grid=False)

    def Kx(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        return self.K_x(xi, r, grid=False)

    def Ez_p(self, x, y, xi, px, py, pz, q, t):
        self.__calculate_wakefields(x, y, xi, px, py, pz, q, t)
        r = np.sqrt(x*x + y*y)
        return self.E_z_p(xi, r, grid=False)

    def __calculate_wakefields(self, x, y, xi, px, py, pz, q, t):
        if self.current_t is None:
            self.current_t = t
        elif self.current_t != t and t >= self.current_t + self.dz_fields/ct.c:
            self.current_t = t
        else:
            return
        z_beam = t*ct.c + np.average(xi)  # z postion of beam center
        n_p = self.density_function(z_beam)

        # calculate distance to laser focus
        if self.laser_evolution:
            dist_z_foc = self.laser_z_foc - ct.c*t
        else:
            dist_z_foc = 0
            
        flds = calculate_wakefield(self.laser, [x, y, xi, q], self.r_max, self.xi_min, self.xi_max, self.n_r, self.n_xi, self.n_part, n_p, dist_z_foc)
        n_p_mesh, W_r, E_z, E_z_p, K_r, psi_mesh, xi_arr, r_arr = flds

        # For debugging
        # plt.plot(E_z[:,0])
        # plt.plot(K_r[:,0])
        # plt.plot(a0_0[:,0])
        # plt.show()

        # plt.subplot(411)
        # plt.imshow(E_z.T*E_0, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.plot(E_z[:,0]*E_0)
        # plt.subplot(412)
        # plt.imshow(K_r.T*E_0/s_d, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.subplot(413)
        # plt.imshow(E_z_p.T*E_0/s_d, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.subplot(414)
        # plt.imshow(beam_hist.T, aspect='auto',
        #           extent=(self.xi_min, self.xi_max, 0, self.r_max))
        # plt.show()
        E_0 = ge.plasma_cold_non_relativisct_wave_breaking_field(n_p*1e-6)
        s_d = ge.plasma_skin_depth(n_p*1e-6)
        # self.E_z = RegularGridInterpolator(
        #     (xi_arr*s_d, r_arr*s_d), E_z.T*E_0, fill_value=0, bounds_error=False)
        # self.W_x = RegularGridInterpolator(
        #     (xi_arr*s_d, r_arr*s_d), W_r.T*E_0, fill_value=0, bounds_error=False)
        # self.K_x = RegularGridInterpolator(
        #     (xi_arr*s_d, r_arr*s_d), K_r.T*E_0/s_d, fill_value=0, bounds_error=False)
        # self.E_z_p = RegularGridInterpolator(
        #     (xi_arr*s_d, r_arr*s_d), E_z_p.T*E_0/s_d, fill_value=0, bounds_error=False)

        self.E_z = RectBivariateSpline(xi_arr*s_d, r_arr*s_d, E_z.T*E_0, kx=2, ky=2)
        self.W_x = RectBivariateSpline(xi_arr*s_d, r_arr*s_d, W_r.T*E_0, kx=2, ky=2)
        self.K_x = RectBivariateSpline(xi_arr*s_d, r_arr*s_d, K_r.T*E_0/s_d/ct.c, kx=2, ky=2)
        self.E_z_p = RectBivariateSpline(xi_arr*s_d, r_arr*s_d, E_z_p.T*E_0/s_d, kx=2, ky=2)


class PlasmaRampBlowoutField(Wakefield):
    def __init__(self, density_function):
        self.density_function = density_function

    def Wx(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return ct.c*kx*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        return np.zeros(len(xi))

    def Kx(self, x, y, xi, px, py, pz, q, t):
        kx = self.calculate_focusing(xi, t)
        return np.ones(len(xi))*kx

    def calculate_focusing(self, xi, t):
        z = t*ct.c + xi  # z postion of each particle at time t
        n_p = self.density_function(z)
        w_p = np.sqrt(n_p*ct.e**2/(ct.m_e*ct.epsilon_0))
        return (ct.m_e/(2*ct.e*ct.c))*w_p**2


class PlasmaLensField(Wakefield):
    def __init__(self, dB_r):
        self.dB_r = dB_r  # [T/m]

    def Wx(self, x, y, xi, px, py, pz, q, t):
        # By = -x*self.dB_r
        gamma = np.sqrt(1 + px*px + py*py + pz*pz)
        return - pz*ct.c/gamma * (-x*self.dB_r)

    def Wy(self, x, y, xi, px, py, pz, q, t):
        # Bx = y*self.dB_r
        gamma = np.sqrt(1 + px*px + py*py + pz*pz)
        return pz*ct.c/gamma * y*self.dB_r

    def Wz(self, x, y, xi, px, py, pz, q, t):
        gamma = np.sqrt(1 + px*px + py*py + pz*pz)
        return (px*(-x*self.dB_r) - py*y*self.dB_r)*ct.c/gamma

    def Kx(self, x, y, xi, px, py, pz, q, t):
        # not really important
        return np.ones(len(x))*self.dB_r


class PlasmaLensFieldRelativistic(Wakefield):
    def __init__(self, k_x):
        self.k_x = k_x  # [T/m]

    def Wx(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.k_x*x

    def Wy(self, x, y, xi, px, py, pz, q, t):
        return ct.c*self.k_x*y

    def Wz(self, x, y, xi, px, py, pz, q, t):
        return np.zeros(len(x))

    def Kx(self, x, y, xi, px, py, pz, q, t):
        return np.ones(len(x))*self.k_x


class CombinedWakefield(Wakefield):
    def __init__(self, wakefield_list):
        self.wakefield_list = wakefield_list

    def Wx(self, x, y, xi, px, py, pz, q, t):
        wx = np.zeros(x.shape[0])
        for wf in self.wakefield_list:
            wx += wf.Wx(x, y, xi, px, py, pz, q, t)
        return wx

    def Wy(self, x, y, xi, px, py, pz, q, t):
        wy = np.zeros(x.shape[0])
        for wf in self.wakefield_list:
            wy += wf.Wy(x, y, xi, px, py, pz, q, t)
        return wy

    def Wz(self, x, y, xi, px, py, pz, q, t):
        wz = np.zeros(x.shape[0])
        for wf in self.wakefield_list:
            wz += wf.Wz(x, y, xi, px, py, pz, q, t)
        return wz

    def Kx(self, x, y, xi, px, py, pz, q, t):
        kx = np.zeros(x.shape[0])
        for wf in self.wakefield_list:
            kx += wf.Kx(x, y, xi, px, py, pz, q, t)
        return kx

