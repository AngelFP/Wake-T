import numpy as np
import scipy.constants as ct
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
try:
    from VisualPIC.DataHandling.dataContainer import DataContainer
    vpic_installed = True
except ImportError:
    vpic_installed = False

from wake_t.physics_models.plasma_wakefields.base_wakefield import Wakefield


class WakefieldFromPICSimulation(Wakefield):
    def __init__(self, simulation_code, simulation_path, driver, timestep=0,
                 n_p=None, filter_fields=False, sigma_filter=20,
                 reverse_tracking=False):
        """
        [n_p] = m^-3
        """
        super().__init__()
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
