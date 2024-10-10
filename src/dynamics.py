

import casadi as ca
from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from l4casadi import L4CasADi
from l4casadi.naive import NaiveL4CasADiModule
from l4casadi.realtime import RealTimeL4CasADi
from liecasadi import Quaternion
from scipy.spatial.transform import Rotation as R
import torch
import json
import os
import sys
import pandas as pd
from tqdm import tqdm
# from numba import jit
import h5py

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)

from src.models import ScaledModel
from src.utils import load_model, TrajectoryConfiguration
from src.plotting import TrajectoryPlotter
from dataclasses import dataclass

print(DEVICE)

class Aircraft:
    """ 
    NN based flight dynamics for aircraft. 

    Coordinate Systems:

    Body Frame (B)(NED):
    Right handed coordinate system with:
        - x-axis : in the plane of symmetry. Runs from tail to tip (forward 
            positive) and is denoted north (N)
        - z-axis : in the plane of symmetry, orthogonal to x-axis (down 
            positive) and is denoted down (D)
        - y-axis : perpendicular to x and z (right positive) denoted east (E)

    Inertial Frame (ECEF)(NED):
    We approximate the earth as non-rotating and locally flat. The control 
    parameter omega_e_i_ecf governs the rotation of the earth and affects the 
    coriolis force used in the dynamics. The inertial frame is also taken as a 
    NED right-handed coordinate system with axes aligned with the cardinal 
    directions and gravity.

    State:

    The state of the system is characterised by the aircraft's:
        - Orientation_{B -> ECEF} (Quaternion q): such that q*v*q_inv transforms 
            a vector v from the body frame B to ECEF.
        - Position_{ECEF}    (Vector p = (x, y, z))
        - Velocity_{ECEF}    (Vector v = (u, v, w))
        - Angular Velocity{B} (vector omega)

    Control:

    Parameters used to modify the dynamics. State variables that don't evolve 
    under the dynamics are placed in control so that they can be modified 
    throughout the simulation.

        - delta_a : aileron deflection (-5 to 5)
        - delta_e : elevator deflection (-5 to 5)
        - delta_r : rudder deflection (neglected with current model)
        - throttle : vector giving force in x, y, z. Usually disabled.
        - CoM : center of mass offset from the body frame origin
        - v_wind_ecf_e : wind velocity in the ECF frame
        - omega_e_i_ecf : angular velocity of the earth in the inertial frame
        
    Aerodynamics:

    The model predicts aerodynamic coefficients based on the parameters:
        - alpha (angle of attack)
        - beta (angle of sideslip)
        - delta_a (aileron deflection)
        - delta_e (elevator deflection)
        - q_bar (dynamic pressure)

    And outputs the aerodynamic coefficients (CX, CY, CZ, Cl, Cm, Cn) which 
    relate to the forces and moments in the body frame:

    F_X = q_bar * S * C_X
    M_l = q_bar * S * b * C_X
    M_m = q_bar * S * c * C_X
    M_b = q_bar * S * b * C_X

    Where S is the reference area for the aircraft, b is span and c is chord

    """
    def __init__(
            self, 
            params:dict, 
            model:torch.nn, 
            EPSILON:float = 1e-6, 
            STEPS:int = 100,
            LINEAR:bool = False
            ):
        
        """ 
        EPSILON: smoothing parameter for non-smoothly-differentiable functions
        STEPS: number of integration steps in each state update
        """
        Ixx = params['Ixx']
        Iyy = params['Iyy']
        Ixz = params['Ixz']
        Izz = params['Izz']
        self._inertia_tensor = ca.vertcat(
            ca.horzcat(Ixx, 0  , Ixz),
            ca.horzcat(0  , Iyy, 0  ),
            ca.horzcat(Ixz, 0  , Izz)
        )
        self.LINEAR = LINEAR

        
        
        self.EPSILON = EPSILON
        self.STEPS = STEPS

        self.q_frd_ned = ca.MX.sym('q_frd_ned', 4)
        self.p_ned = ca.MX.sym('p_ned', 3)
        self.v_ned = ca.MX.sym('v_ned', 3)
        self.omega_frd_ned = ca.MX.sym('omega_frd_ned', 3)

        self.linear_coeffs = ca.DM(np.array(pd.read_csv('data/glider/linearised.csv')))

        self.state = ca.vertcat(
            self.q_frd_ned, 
            self.p_ned, 
            self.v_ned, 
            self.omega_frd_ned
            )
        
        self.num_states = self.state.size()[0]

        self.v_wind_ned = ca.MX.sym('v_wind_ned', 3)
        self.aileron = ca.MX.sym('aileron')
        self.elevator = ca.MX.sym('elevator')
        self.rudder = ca.MX.sym('rudder')
        self.throttle = ca.MX.sym('throttle', 3)

        self.com = ca.MX.sym('com', 3)



        self.control = ca.vertcat(
            self.aileron, 
            self.elevator, 
            self.rudder, 
            self.throttle, 
            self.com,
            self.v_wind_ned
            )
        
        self.num_controls = self.control.size()[0]

        # self.STEP = STEP # timestep to be used for integration

        self.grav = ca.vertcat(0, 0, 9.81)
        self.S = params['reference_area']
        self.b = params['span']
        self.c = params['chord']

        self.mass = params['mass']
        Realtime = False
        if Realtime:
            self.model = RealTimeL4CasADi(model, approximation_order=1)
        else:
            self.model = L4CasADi(model, name = 'AeroModel', 
                                  generate_jac_jac=True
                                  )
        self._inv_inertia_tensor = ca.inv(self.inertia_tensor)
        self.qbar
        self.beta
        self.alpha
        self.phi
        
        self.x_dot
        self.p_ned
        

    @property
    def inertia_tensor(self):
        """
        Inertia Tensor around the Centre of Mass
        """
        com = self.com
        # inertia_tensor = (self._inertia_tensor 
        #                   + self.mass * (ca.dot(com, com) 
        #                   * ca.diag([1, 1, 1]) - ca.dot(ca.MX(com).T, com)))
        
        com_term = - ca.vertcat(
            ca.horzcat(0, com[0]*com[1], com[0]*com[2]),
            ca.horzcat(com[1]*com[0], 0, com[1]*com[2]),
            ca.horzcat(com[2]*com[1], com[2]*com[1], 0)
        )

        inertia_tensor = self._inertia_tensor - self.mass * com_term
        return inertia_tensor
    
    @property
    def inverse_inertia_tensor(self):
        """
        Inverted Inertia Tensor around Centre of Mass
        """
        pass


    
    @property
    def v_frd_rel(self):
        q_frd_ned = Quaternion(self.q_frd_ned)
        v_ned = Quaternion(ca.vertcat(self.v_ned, 0))
        v_wind_ned = Quaternion(ca.vertcat(self.v_wind_ned, 0))
        result = (q_frd_ned.inverse() * (v_ned - v_wind_ned) 
                  * q_frd_ned).coeffs()[:3]
        
        result = result + self.EPSILON

        self._v_frd_rel = ca.Function(
            'v_frd_rel', 
            [self.state, self.control], 
            [result]
            ).expand()
        
        return result
    
    @property
    def airspeed(self):
        airspeed = ca.sqrt(ca.sumsqr(self.v_frd_rel) + self.EPSILON)
        self._airspeed = ca.Function(
            'airspeed', 
            [self.state, self.control], 
            [airspeed]
            ).expand()
        return airspeed
    
    @property
    def alpha(self):
        v_frd_rel = self.v_frd_rel
        alpha = ca.atan2(v_frd_rel[2], v_frd_rel[0] + self.EPSILON)
        self._alpha = ca.Function('alpha', [self.state, self.control], [alpha]).expand()
        return alpha
    
    # @property
    # def phi(self):
    #     """
    #     Roll angle
    #     """

    #     aircraft_up = Quaternion(self.q_frd_ned)

    #     sinr_cosp = 2 * (aircraft_up.w * aircraft_up.x + aircraft_up.y * aircraft_up.z);
    #     cosr_cosp = 1 - 2 * (aircraft_up.x * aircraft_up.x + aircraft_up.y * aircraft_up.y);
    #     roll = ca.atan2(sinr_cosp, cosr_cosp);

    #     self._phi = ca.Function('roll', [self.state, self.control], [roll]).expand()
    #     return roll

    @property
    def phi(self):
        q_frd_ned = self.q_frd_ned  # Quaternions [x, y, z, w]
        x, y, z, w = q_frd_ned[0], q_frd_ned[1], q_frd_ned[2], q_frd_ned[3]
        phi = ca.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        self._phi = ca.Function('phi', [self.state], [phi]).expand()
        return phi

    @property
    def theta(self):
        q_frd_ned = self.q_frd_ned
        x, y, z, w = q_frd_ned[0], q_frd_ned[1], q_frd_ned[2], q_frd_ned[3]
        theta = ca.asin(2 * (w * y - z * x))
        self._theta = ca.Function('theta', [self.state], [theta]).expand()
        return theta

    @property
    def psi(self):
        q_frd_ned = self.q_frd_ned
        x, y, z, w = q_frd_ned[0], q_frd_ned[1], q_frd_ned[2], q_frd_ned[3]
        psi = ca.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        self._psi = ca.Function('psi', [self.state], [psi]).expand()
        return psi

    @property
    def p(self):
        omega_frd_frd = self.omega_frd_ned  # Angular rates [p, q, r]
        p = omega_frd_frd[0]
        self._p = ca.Function('p', [self.state], [p]).expand()
        return p

    @property
    def q(self):
        omega_frd_frd = self.omega_frd_ned
        q = omega_frd_frd[1]
        self._q = ca.Function('q', [self.state], [q]).expand()
        return q

    @property
    def r(self):
        omega_frd_frd = self.omega_frd_ned
        r = omega_frd_frd[2]
        self._r = ca.Function('r', [self.state], [r]).expand()
        return r

    @property
    def phi_dot(self):
        phi = self.phi
        theta = self.theta
        p = self.p
        q = self.q
        r = self.r
        phi_dot = p + ca.sin(phi) * ca.tan(theta) * q + ca.cos(phi) * ca.tan(theta) * r
        self._phi_dot = ca.Function('phi_dot', [self.state], [phi_dot]).expand()
        return phi_dot

    @property
    def theta_dot(self):
        phi = self.phi
        q = self.q
        r = self.r
        theta_dot = ca.cos(phi) * q - ca.sin(phi) * r
        self._theta_dot = ca.Function('theta_dot', [self.state], [theta_dot]).expand()
        return theta_dot

    @property
    def psi_dot(self):
        phi = self.phi
        theta = self.theta
        q = self.q
        r = self.r
        psi_dot = (ca.sin(phi) / ca.cos(theta)) * q + (ca.cos(phi) / ca.cos(theta)) * r
        self._psi_dot = ca.Function('psi_dot', [self.state], [psi_dot]).expand()
        return psi_dot

    
    @property
    def beta(self):
        v_frd_rel = self.v_frd_rel
        beta = ca.asin(v_frd_rel[1] / self.airspeed)
        self._beta = ca.Function('beta', [self.state, self.control], [beta]).expand()
        return beta
    
    @property
    def qbar(self):
        qbar = 0.5 * 1.225 * ca.dot(self.v_frd_rel, self.v_frd_rel)
        self._qbar = ca.Function('qbar', [self.state, self.control], [qbar]).expand()
        return qbar
    
    @property
    def coefficients(self):
        """
        Forward pass of the ml model to retrieve aerodynamic coefficients.
        """
        inputs = ca.vertcat(
            self.qbar, 
            self.alpha, 
            self.beta, 
            self.aileron, 
            self.elevator
            )
        
        if self.LINEAR:
            outputs = ca.mtimes(self.linear_coeffs, ca.vertcat(inputs, 1))
        else:
            outputs = self.model(ca.reshape(inputs, 1, -1))
            outputs = ca.vertcat(outputs.T)
            print("Outputs", outputs.shape)

        stall_angle_alpha = np.deg2rad(10)
        stall_angle_beta = np.deg2rad(10)

        steepness = 2

        alpha_scaling = 1 / (1 + ca.exp(steepness * (self.alpha - stall_angle_alpha)))
        beta_scaling = 1 / (1 + ca.exp(steepness * (self.beta - stall_angle_beta)))
        
        
        outputs[2] *= alpha_scaling
        outputs[2] *= beta_scaling

        outputs[4] *= alpha_scaling
        outputs[4] *= beta_scaling


        # self._coefficients = ca.Function(
        #     'coefficients', 
        #     [self.state, self.control], 
        #     [outputs]
        #     )
        
        


        # angular rate contributions
        # outputs[0] += 0.05 * self.omega_b_i_frd[0]
        # outputs[1] += -0.05 * self.omega_frd_ned[2]
        # outputs[2] += -0.1 * self.omega_frd_ned[0]
        outputs[1] += -0.05 * self.omega_frd_ned[2]
        outputs[2] += -0.1 * self.omega_frd_ned[0]

        """
        The relative amplitudes of the damping factors are taken from the cessna
        model which has:

        Roll:
        Clp ~ -0.5
        Clr ~ 0.1

        Pitch:
        Cmq ~ -12

        Yaw:
        Cnp ~ -0.03
        Cnr ~ -0.1

        We scale this down by a factor of 100

        NOTE:

        We flip the yaw moment, TODO: investigate
        We scale the roll moment down by a factor of 4 meaning we use the 
        windtunnel scale rather than the sim scale.
        """
        scale = 1
        # roll moment rates
        outputs[3] /= 4
        outputs[3] += -0.005 * self.omega_frd_ned[0] * scale
        outputs[3] += 0.001 * self.omega_frd_ned[2] * scale

        # pitching moment rates
        outputs[4] += -0.03 * self.omega_frd_ned[1] * scale

        # yaw moment rates
        outputs[5] *= -1
        outputs[5] += -0.0003 * self.omega_frd_ned[0] * scale
        outputs[5] += -0.001 * self.omega_frd_ned[2] * scale

        self._coefficients = ca.Function(
            'coefficients', 
            [self.state, self.control], 
            [outputs]
            )

        return outputs

    @property
    def forces_frd(self):
        forces = self.coefficients[:3] * self.qbar * self.S
        forces += self.throttle

        self._forces_frd = ca.Function('forces_frd', 
            [self.state, self.control], [forces])
        return forces
    
    @property
    def moments_aero_frd(self):
        moments_aero = (self.coefficients[3:] * self.qbar * self.S 
                        * ca.vertcat(self.b, self.c, self.b))

        self._moments_aero_frd = ca.Function('moments_aero_frd', 
            [self.state, self.control], [moments_aero])
        return moments_aero
    
    @property
    def moments_from_forces_frd(self):
        com = self.com
        moments_from_forces = ca.cross(com, self.forces_frd)
        self._moments_from_forces_frd = ca.Function('moments_from_forces_frd', 
            [self.state, self.control], [moments_from_forces])
        
        return moments_from_forces
    
    @property
    def moments_frd(self):
        moments = self.moments_aero_frd + self.moments_from_forces_frd
        self._moments_frd = ca.Function('moments_frd', 
            [self.state, self.control], [moments])
        
        return moments
    
    @property
    def forces_ned(self):
        forces_frd = Quaternion(ca.vertcat(self.forces_frd, 0))
        q_frd_ned = Quaternion(self.q_frd_ned)
        forces_ned = q_frd_ned * forces_frd * q_frd_ned.inverse()
        self._forces_ned = ca.Function('forces_ned', 
            [self.state, self.control], [forces_ned.coeffs()[:3]])
        
        return forces_ned.coeffs()[:3]

    @property
    def q_frd_ned_dot(self):
        q_frd_ecf = Quaternion(self.q_frd_ned)
        omega_frd_ned = Quaternion(ca.vertcat(self.omega_frd_ned, 0))

        q_frd_ned_dot = 0.5 * q_frd_ecf * omega_frd_ned
        self._q_frd_ned_dot = ca.Function('q_frd_ecf_dot', 
            [self.state, self.control], [q_frd_ned_dot.coeffs()]).expand()

        return q_frd_ned_dot.coeffs()
    
    @property
    def p_ned_dot(self):
        self._p_ned_dot = ca.Function('p_ecf_cm_O_dot', 
            [self.state, self.control], [self.v_ned]).expand()
        
        return self.v_ned
    
    @property
    def v_ned_dot(self):

        forces = self.forces_ned
        grav = self.grav
        mass = self.mass

        v_ned_dot =  forces / mass + grav
        
        self._v_ned_dot = ca.Function(
            'v_ecf_cm_e_dot', 
            [self.state, self.control], 
            [v_ned_dot]
            )

        return v_ned_dot
    
    def compute_euler_and_body_rates(self, q_frd_ned, omega_frd_frd):
        """
        Computes the Euler angles and angular rates given the orientation quaternion and angular rates in the FRD frame.
        
        Parameters:
        q_frd_ned: A CasADi array representing the quaternions (w, x, y, z) that map from FRD to NED. Shape: (4, n)
        omega_frd_frd: A CasADi array representing the angular rates in the FRD frame [p, q, r]. Shape: (3, n)
        
        Returns:
        A dictionary containing vectorized Euler angles, angular rates, and their derivatives.
        """

        # Extract the components of the quaternions
        x, y, z, w = q_frd_ned[0, :], q_frd_ned[1, :], q_frd_ned[2, :], q_frd_ned[3, :]
        n = q_frd_ned.shape[1]  # number of sets
        phi = np.zeros(n)
        theta = np.zeros(n)
        psi = np.zeros(n)
        T_list = []
        for i in range(n):
            x, y, z, w = q_frd_ned[0, i], q_frd_ned[1, i], q_frd_ned[2, i], q_frd_ned[3, i]
            C = ca.vertcat(
                    ca.horzcat((w**2 + x**2 - y**2 - z**2), 2*(x*y + w*z), (2*(x*z - w*y))),
                    ca.horzcat((2*(x*y - w*z)), (w**2 - x**2 + y**2 - z**2), (2*(y*z + w*x))),
                    ca.horzcat((2*(x*z + w*y)), (2*(y*z - w*x)), (w**2 - x**2 - y**2 + z**2))
                )
            # Calculate the Euler angles from the DCM
            phi[i] = ca.atan2(C[1,2], C[2,2])  # Roll angle
            theta[i] = -ca.asin(C[0,2])  # Pitch angle
            psi[i] = ca.atan2(C[0,1], C[0,0])  # Yaw angle

            T_i = ca.vertcat(
                ca.horzcat(1, 0, -ca.sin(theta[i])),
                ca.horzcat(0, ca.cos(phi[i]), ca.sin(phi[i]) * ca.cos(theta[i])),
                ca.horzcat(0, -ca.sin(phi[i]), ca.cos(phi[i]) * ca.cos(theta[i]))
            )
            T_list.append(T_i)
        
        
        
        
        # Find the Euler angle derivatives from the body rates
        euler_rates = np.zeros((3, n))
        for i in range(n):
            euler_rates[:, i] = ca.solve(T_list[i], omega_frd_frd[:, i]).full().flatten()
        
        # Extract the individual angular rates p, q, r from omega_frd_frd
        p, q, r = omega_frd_frd[0, :], omega_frd_frd[1, :], omega_frd_frd[2, :]
        
        return {
            'phi': phi, 'theta': theta, 'psi': psi,
            'p': p, 'q': q, 'r': r,
            'phi_dot': euler_rates[0, :],
            'theta_dot': euler_rates[1, :],
            'psi_dot': euler_rates[2, :],
            'T_list': T_list  # Include the list of transformation matrices in the output if needed
        }

    @property
    def omega_frd_ned_dot(self):
        J_frd = self.inertia_tensor
        J_frd_inv = self._inv_inertia_tensor
        
        omega_frd_ned = self.omega_frd_ned
        moments = self.moments_frd


        omega_frd_ned_dot = ca.mtimes(J_frd_inv, (moments 
             - ca.cross(omega_frd_ned, ca.mtimes(J_frd, omega_frd_ned))))
        
        self._omega_frd_ned_dot = ca.Function(
            'omega_frd_ned_dot', 
            [self.state, self.control], 
            [omega_frd_ned_dot]
            )

        return omega_frd_ned_dot
    
    @property
    def x_dot(self):
        x_dot = ca.vertcat(
            self.q_frd_ned_dot, 
            self.p_ned_dot, 
            self.v_ned_dot, 
            self.omega_frd_ned_dot
            )
        self.dynamics = ca.Function(
            'dynamics', 
            [self.state, self.control], 
            [x_dot]
            )
        return x_dot
    
    # @jit
    def state_step(self, state, control, dt_scaled):
        """ 
        Runge kutta step for state update. Due to the multiplicative nature
        of the quaternion integration we cannot rely on conventional 
        integerators. 
        """

        k1 = self.dynamics(state, control)
        state_k1 = state + dt_scaled / 2 * k1

        k2 = self.dynamics(state_k1, control)
        state_k2 = state + dt_scaled / 2 * k2


        k3 = self.dynamics(state_k2, control)
        state_k3 = state + dt_scaled * k3

        k4 = self.dynamics(state_k3, control)

        state = state + dt_scaled / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Normalize the quaternion
        quaternion = Quaternion(state[:4])
        state[:4] = quaternion.normalize().coeffs()

        # self._state_step = ca.Function('step', [state, control, dt_scaled], [state])

        return state


    @property
    def state_update(self, normalisation_interval: int = 10):
        """
        Runge Kutta integration with quaternion update, for loop over self.STEPS
        """
        dt = ca.MX.sym('dt')
        state = self.state
        control_sym = self.control
        num_steps = self.STEPS

        dt_scaled = dt / num_steps
        input_to_fold = ca.vertcat(self.state, self.control, dt)
        fold_output = ca.vertcat(self.state_step(state, control_sym, dt_scaled), control_sym, dt)
        folded_update = ca.Function('folder', [input_to_fold], [fold_output])
        
        F = folded_update.fold(num_steps)
        state = F(input_to_fold)[:self.num_states]
        # for i in range(num_steps):
        #     state = self.state_step(state, control_sym, dt_scaled)
        
        #     if i % normalisation_interval == 0:
        #         state[:4] = Quaternion(state[:4]).normalize().coeffs()
        
        # state[:4] = Quaternion(state[:4]).normalize().coeffs()
        # if self.LINEAR:
        #     return ca.Function(
        #     'state_update', 
        #     [self.state, self.control, dt], 
        #     [state]).expand()
        return ca.Function(
            'state_update', 
            [self.state, self.control, dt], 
            [state]
            ) #, {'jit':True}
    

   

if __name__ == '__main__':
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))
    trajectory_config = TrajectoryConfiguration(traj_dict)


    aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=True)

    perturbation = False
    
    trim_state_and_control = [0.823338, -0.538343, -7.4267e-06, 0.179728, 0, 0, 0, 33.3716, 10.3374, -61.5815, -0.00673001, -0.000351288, -0.00355378, -0.139053, 48.0152, 0, -1.6163e-37, -4.70829e-38, -8.30193e-38, 0.0554644, 0.00336277, -0.00549335, 7.05297e-38, -3.29138e-37, -1.17549e-38]
    if trim_state_and_control is not None:
        state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
        control = np.array(trim_state_and_control[aircraft.num_states:])
    else:

        x0 = np.zeros(3)
        v0 = ca.vertcat([80, 0, 0])
        q0 = Quaternion(ca.vertcat(0, 0, 0, 1))
        # q0 = ca.vertcat([0.215566, -0.568766, 0.255647, 0.751452])#q0.inverse()
        omega0 = np.array([0, 0, 0])

        state = ca.vertcat(q0, x0, v0, omega0)
        control = np.zeros(aircraft.num_controls)
        control[0] = 0.05
        control[1] = 4
        control[6:9] = traj_dict['aircraft']['aero_centre_offset']

    dyn = aircraft.state_update
    dt = .1
    tf = 100.0
    state_list = np.zeros((aircraft.num_states, int(tf / dt)))

    # dt_sym = ca.MX.sym('dt', 1)
    t = 0
    target_roll = np.deg2rad(60)
    ele_pos = True
    ail_pos = True
    control_list = np.zeros((aircraft.num_controls, int(tf / dt)))
    for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
        if np.isnan(state[0]):
            print('Aircraft crashed')
            break
        else:
            state_list[:, i] = state.full().flatten()
            control_list[:, i] = control
            state = dyn(state, control, dt)
            # print("Roll Angle:", np.rad2deg(aircraft._phi(state, control).full().flatten()))
            # print("Switch condition: ", np.rad2deg((aircraft._phi(state, control).full().flatten() - np.pi)))
            # if np.rad2deg(aircraft._phi(state, control).full().flatten()) > 60:
            #     if ail_pos:
            #         control[0] += 0.5
            #     else:
            #         control[0] = 0
            #         ail_pos = True
            # elif np.rad2deg(aircraft._phi(state, control).full().flatten()) < -60:
            #     if not ail_pos:
            #         control[0] -= 0.5
            #     else:
            #         control[0] = 0
            #         ail_pos = False

            # if np.rad2deg(aircraft._alpha(state, control).full().flatten()) > 5:
            #     if ele_pos:
            #         control[1] += 0.5
            #     else:
            #         control[1] = 0
            #         ele_pos = True
            # elif np.rad2deg(aircraft._alpha(state, control).full().flatten()) < -5:
            #     if not ele_pos:
            #         control[1] -= 0.5
            #     else:
            #         control[1] = 0
            #         ele_pos = False
            # control_list.append(control[0])
                    
            t += 1
    print(state)
    # state_list = state_list[:, :t-10]
    # print(state_list[0, :])
    first = None
    # t -=5
    def save(filepath):
        with h5py.File(filepath, "a") as h5file:
            grp = h5file.create_group(f'iteration_0')
            grp.create_dataset('state', data=state_list[:, :t])
            grp.create_dataset('control', data=control_list[:, :t])
    
    
    filepath = os.path.join("data", "trajectories", "simulation.h5")
    if os.path.exists(filepath):
        os.remove(filepath)
    save(filepath)

    plotter = TrajectoryPlotter(filepath, aircraft)
    plotter.plot(0)
    plt.show(block = True)