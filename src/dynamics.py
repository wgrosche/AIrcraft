

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
from numba import jit

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)

from src.models import ScaledModel
from src.utils import load_model
from src.plotting import plot_state

print(DEVICE)

def load_model(
        filepath:str = os.path.join(NETWORKPATH,'model-dynamics.pth'), 
        device = DEVICE
        ) -> ScaledModel:
    checkpoint = torch.load(filepath, map_location=device)

    scaler = (checkpoint['input_mean'], 
              checkpoint['input_std'], 
              checkpoint['output_mean'], 
              checkpoint['output_std'])
    
    model = ScaledModel(5, 6, scaler=scaler)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

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
        # Determinant of the inertia tensor
        # det_I = Ixx * Iyy * Izz - Ixz**2 * Iyy

        # Inverse inertia tensor
        self.inv_inertia_tensor = ca.inv(self._inertia_tensor)
        
        # (1 / det_I) * ca.vertcat(
        #     ca.horzcat(Iyy * Izz, 0, -Ixz * Iyy),
        #     ca.horzcat(0, Ixx * Izz - Ixz**2, 0),
        #     ca.horzcat(-Ixz * Iyy, 0, Ixx * Iyy)
        # )
        self.EPSILON = EPSILON
        self.STEPS = STEPS

        self.q_frd_ecf = ca.MX.sym('q_frd_ecf', 4)
        self.p_ecf_cm_O = ca.MX.sym('p_ecf_cm_O', 3)
        self.v_ecf_cm_e = ca.MX.sym('v_ecf_cm_e', 3)
        self.omega_b_i_frd = ca.MX.sym('omega_b_i', 3)

        self.linear_coeffs = ca.DM(np.array(pd.read_csv('data/glider/linearised.csv')))

        self.state = ca.vertcat(
            self.q_frd_ecf, 
            self.p_ecf_cm_O, 
            self.v_ecf_cm_e, 
            self.omega_b_i_frd
            )
        
        self.num_states = self.state.size()[0]

        self.v_wind_ecf_e = ca.MX.sym('v_wind_ecf_e', 3)
        self.aileron = ca.MX.sym('aileron')
        self.elevator = ca.MX.sym('elevator')
        self.rudder = ca.MX.sym('rudder')
        self.throttle = ca.MX.sym('throttle', 3)
        self.omega_e_i_ecf = ca.MX.sym('omega_e_i', 3)
        self.com = ca.MX.sym('com', 3)

        self.control = ca.vertcat(
            self.aileron, 
            self.elevator, 
            self.rudder, 
            self.throttle, 
            self.com,
            self.v_wind_ecf_e, 
            self.omega_e_i_ecf
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
            self.model = L4CasADi(model, name = 'AeroModel'
                              , generate_jac_jac=True
                              )
            # inputs = ca.vertcat(
            # self.qbar, 
            # self.alpha, 
            # self.beta, 
            # self.aileron, 
            # self.elevator
            # )
            # self.model.build(inputs)
            # self.
        self.qbar
        self.beta
        self.alpha
        
        self.x_dot
        

    @property
    def inertia_tensor(self):
        inertia_tensor = (self._inertia_tensor 
                          + self.mass * (ca.dot(self.com, self.com) 
                          * ca.diag([1, 1, 1]) - ca.cross(self.com, self.com)))
        return inertia_tensor

    @property
    def v_frd_rel(self):
        q_frd_ecf = Quaternion(self.q_frd_ecf)
        v_ecf_cm_e = Quaternion(ca.vertcat(self.v_ecf_cm_e, 0))
        v_wind_ecf_e = Quaternion(ca.vertcat(self.v_wind_ecf_e, 0))
        result = (q_frd_ecf.inverse() * (v_ecf_cm_e - v_wind_ecf_e) 
                  * q_frd_ecf).coeffs()[:3]
        
        result = result + self.EPSILON

        self._v_frd_rel = ca.Function(
            'v_frd_rel', 
            [self.state, self.control], 
            [result]
            )
        
        return result
    
    @property
    def airspeed(self):
        airspeed = ca.sqrt(ca.sumsqr(self.v_frd_rel) + self.EPSILON)
        self._airspeed = ca.Function(
            'airspeed', 
            [self.state, self.control], 
            [airspeed]
            )
        return airspeed
    
    @property
    def alpha(self):
        v_frd_rel = self.v_frd_rel
        alpha = ca.atan2(v_frd_rel[2], v_frd_rel[0] + self.EPSILON)#v_frd_rel[2] /(v_frd_rel[0] + self.EPSILON)#
        self._alpha = ca.Function('alpha', [self.state, self.control], [alpha])
        return alpha
    
    @property
    def beta(self):
        v_frd_rel = self.v_frd_rel
        beta = ca.asin(v_frd_rel[1] / self.airspeed)#v_frd_rel[1] / self.airspeed#ca.asin(v_frd_rel[1] / self.airspeed)
        self._beta = ca.Function('beta', [self.state, self.control], [beta])
        return beta
    
    @property
    def qbar(self):
        qbar = 0.5 * 1.225 * ca.dot(self.v_frd_rel, self.v_frd_rel)
        self._qbar = ca.Function('qbar', [self.state, self.control], [qbar])
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


        self._coefficients = ca.Function(
            'coefficients', 
            [self.state, self.control], 
            [outputs]
            )
        
        


        # angular rate contributions
        # outputs[0] += 0.05 * self.omega_b_i_frd[0]
        outputs[1] += -0.05 * self.omega_b_i_frd[2]
        outputs[2] += -0.1 * self.omega_b_i_frd[0]

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
        outputs[3] += -0.005 * self.omega_b_i_frd[0] * scale
        outputs[3] += 0.001 * self.omega_b_i_frd[2] * scale

        # pitching moment rates
        outputs[4] += -0.1 * self.omega_b_i_frd[1] * scale

        # yaw moment rates
        outputs[5] *= -1
        outputs[5] += -0.0003 * self.omega_b_i_frd[0] * scale
        outputs[5] += -0.001 * self.omega_b_i_frd[2] * scale

        # self._coefficients = ca.Function(
        #     'coefficients', 
        #     [self.state, self.control], 
        #     [outputs]
        #     )

        return outputs

    @property
    def forces_frd(self):
        forces = self.coefficients[:3] * self.qbar * self.S
        # forces *= ca.vertcat(1, 1, 1)
        forces += self.throttle
        self._forces_frd = ca.Function(
            'forces_frd', 
            [self.state, self.control], 
            [forces]
            )
        return forces
    
    @property
    def moments_aero(self):
        moments_aero = (self.coefficients[3:] 
                        * self.qbar 
                        * self.S 
                        * ca.vertcat(self.b, self.c, self.b))
        # moments_aero *= ca.vertcat(1, 1, 1)

        self._moments_aero = ca.Function(
            'moments_aero', 
            [self.state, self.control], 
            [moments_aero]
            )
        return moments_aero
    
    @property
    def moments_from_forces(self):
        moments_from_forces = ca.cross(self.com, self.forces_frd)

        self._moments_from_forces = ca.Function(
            'moments_from_forces', 
            [self.state, self.control], 
            [moments_from_forces]
            )
        return moments_from_forces
    
    @property
    def moments_frd(self):

        moments = self.moments_aero + self.moments_from_forces

        self._moments_frd = ca.Function(
            'moments_frd', 
            [self.state, self.control], 
            [moments]
            )
        
        return moments

    @property
    def forces_ecf(self):
        forces_frd = Quaternion(ca.vertcat(self.forces_frd, 0))
        q_frd_ecf = Quaternion(self.q_frd_ecf)
        forces_ecf = q_frd_ecf * forces_frd * q_frd_ecf.inverse()
        self._forces_ecf = ca.Function(
            'forces_ecf', 
            [self.state, self.control], 
            [forces_ecf.coeffs()[:3]]
            )
        return forces_ecf.coeffs()[:3]

    @property
    def omega_e_i_frd(self):
        q_frd_ecf = Quaternion(self.q_frd_ecf)
        omega_e_i_ecf = Quaternion(ca.vertcat(self.omega_e_i_ecf, 0))

        result = q_frd_ecf.inverse() * omega_e_i_ecf * q_frd_ecf
        self._omega_e_i_frd = ca.Function(
            'omega_e_i_frd', 
            [self.state, self.control], 
            [result.coeffs()[:3]]
            )
        return result.coeffs()[:3]

    @property
    def q_frd_ecf_dot(self):
        q_frd_ecf = Quaternion(self.q_frd_ecf)
        omega_b_i_frd = Quaternion(ca.vertcat(self.omega_b_i_frd, 0))
        omega_e_i_frd = Quaternion(ca.vertcat(self.omega_e_i_frd, 0))

        q_frd_ecf_dot = 0.5 * q_frd_ecf * (omega_b_i_frd - omega_e_i_frd)

        self._q_frd_ecf_dot = ca.Function(
            'q_frd_ecf_dot', 
            [self.state, self.control], 
            [q_frd_ecf_dot.coeffs()]
            )

        return q_frd_ecf_dot.coeffs()
    
    @property
    def p_ecf_cm_O_dot(self):
        self._p_ecf_cm_O_dot = ca.Function(
            'p_ecf_cm_O_dot', 
            [self.state, self.control], 
            [self.v_ecf_cm_e]
            )
        return self.v_ecf_cm_e
    
    @property
    def v_ecf_cm_e_dot(self):

        forces = self.forces_ecf
        grav = self.grav
        mass = self.mass
        omega_e_i_frd = self.omega_e_i_frd
        v_ecf_cm_e = self.v_ecf_cm_e

        v_ecf_cm_e_dot =  1 / mass * forces \
            + grav \
            - 2 * ca.cross(omega_e_i_frd, v_ecf_cm_e) # Coriolis term
        
        self._v_ecf_cm_e_dot = ca.Function(
            'v_ecf_cm_e_dot', 
            [self.state, self.control], 
            [v_ecf_cm_e_dot]
            )

        return v_ecf_cm_e_dot
    

    @property
    def omega_b_i_frd_dot(self):
        J_frd = self.inertia_tensor
        J_frd_inv = self.inv_inertia_tensor
        
        omega_b_i_frd = self.omega_b_i_frd
        moments = self.moments_frd


        omega_b_i_frd_dot = ca.mtimes(
            J_frd_inv, 
            (moments 
             - ca.cross(omega_b_i_frd, ca.mtimes(J_frd, omega_b_i_frd))))
        self._omega_b_i_frd_dot = ca.Function(
            'omega_b_i_frd_dot', 
            [self.state, self.control], 
            [omega_b_i_frd_dot]
            )

        return omega_b_i_frd_dot
    
    @property
    def x_dot(self):
        x_dot = ca.vertcat(
            self.q_frd_ecf_dot, 
            self.p_ecf_cm_O_dot, 
            self.v_ecf_cm_e_dot, 
            self.omega_b_i_frd_dot
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
        
        state[:4] = Quaternion(state[:4]).normalize().coeffs()
        if self.LINEAR:
            return ca.Function(
            'state_update', 
            [self.state, self.control, dt], 
            [state]).expand()
        return ca.Function(
            'state_update', 
            [self.state, self.control, dt], 
            [state]
            ) #, {'jit':True}

   

if __name__ == '__main__':
    aircraft_params = json.load(open(os.path.join(BASEPATH, 'data', 'glider', 'glider_fs.json')))
    perturbation = False

    model = load_model()
    
    aircraft = Aircraft(aircraft_params, model, STEPS=100, LINEAR=True)
    
    trim_state_and_control = None
    if trim_state_and_control is not None:
        state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
        control = ca.vertcat(trim_state_and_control[aircraft.num_states:])
    else:

        x0 = np.zeros(3)
        v0 = np.array([50, 0, 0])
        q0 = np.array([1, 0, 0, 0])
        omega0 = np.array([0, 0, 0])

        state = ca.vertcat(q0, x0, v0, omega0)
        control = np.zeros(aircraft.num_controls)
        control[0] = 0
        control[6:9] = aircraft_params['aero_centre_offset']


    dyn = aircraft.state_update
    dt = 1
    tf = 100.0
    state_list = np.zeros((aircraft.num_states, int(tf / dt)))

    dt_sym = ca.MX.sym('dt', 1)
    t = 0

    for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
        if np.isnan(state[0]):
            print('Aircraft crashed')
            break
        else:
            state_list[:, i] = state.full().flatten()
            state = dyn(state, control, dt)
            t += 1

    # state_list = state_list[:, :t-10]
    # print(state_list[0, :])
    first = None
    t -= 5

    # print(f"Final State: {state_list[:, t-10:t]}")
    fig = plt.figure(figsize=(18, 9))
    fig = plot_state(fig, state_list, control, aircraft, t, dt, first=0)
    fig.savefig('test.png')
    plt.show(block=True)