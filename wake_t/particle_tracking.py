""" This module contains the numerical trackers and equations of motion """

import numpy as np
import scipy.constants as ct


def runge_kutta_4(beam_matrix, WF, t0, dt, iterations):
    for i in np.arange(iterations):
        t = t0 + i*dt
        A = dt*equations_of_motion_relativistic(beam_matrix, t, WF)
        B = dt*equations_of_motion_relativistic(beam_matrix + A/2, t+dt/2, WF)
        C = dt*equations_of_motion_relativistic(beam_matrix + B/2, t+dt/2, WF)
        D = dt*equations_of_motion_relativistic(beam_matrix + C, t+dt, WF)
        beam_matrix += 1/6*(A + 2*B + 2*C + D)
    return beam_matrix
    
def equations_of_motion_relativistic(beam_matrix, t, WF):
    K = -ct.e/(ct.m_e*ct.c)
    x, px, y, py, xi, pz = beam_matrix
    gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    return np.array([px*ct.c/gamma, K*WF.Wx(x, y, xi, t),
                     py*ct.c/gamma, K*WF.Wy(x, y, xi, t),
                     (pz/gamma-1)*ct.c, K*WF.Ez(x, y, xi, t)])
