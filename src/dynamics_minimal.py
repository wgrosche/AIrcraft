

import casadi as ca
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

from liecasadi import Quaternion
from scipy.spatial.transform import Rotation as R
import torch
import json
import os
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
# from numba import jit
import h5py

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)

from src.models import ScaledModel
from src.utils import load_model, TrajectoryConfiguration
from src.plotting_minimal import TrajectoryPlotter
from dataclasses import dataclass

print(DEVICE)

@dataclass
class AircraftOpts:
    epsilon:float = 1e-6
    physical_integration_substeps:int = 1
    linear_path:Path = None
    poly_path:Path = None
    nn_model_path:Path = None
    aircraft_config:TrajectoryConfiguration.AircraftConfiguration = None
    realtime:bool = False # Work in progress implementation of faster nn eval

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
    M_l = q_bar * S * b * C_l
    M_m = q_bar * S * c * C_m
    M_b = q_bar * S * b * C_n

    Where S is the reference area for the aircraft, b is span and c is chord

    """


    def __init__(
            self,
            opts:AircraftOpts
            ):
        
        """ 
        EPSILON: smoothing parameter for non-smoothly-differentiable functions
        STEPS: number of integration steps in each state update
        """
        self.LINEAR = False

        if opts.linear_path:
            self.LINEAR = True
            self.linear_coeffs = ca.DM(np.array(pd.read_csv(opts.linear_path)))
            self.fitted_models = None
        elif opts.poly_path:
            import pickle
            with open(opts.poly_path, 'rb') as file:
                self.fitted_models = pickle.load(file)

        elif opts.nn_model_path:
            try:
                from l4casadi import L4CasADi
                from l4casadi.naive import NaiveL4CasADiModule
                from l4casadi.realtime import RealTimeL4CasADi
            except:
                print("L4CasADi not installed")
            self.LINEAR = False
            self.fitted_models = None
            model = load_model(filepath=opts.nn_model_path)
            if opts.realtime:
                self.model = RealTimeL4CasADi(model, approximation_order=1)
            else:
                self.model = L4CasADi(model, name = 'AeroModel', 
                                    generate_jac_jac=True
                                    )
        
        self.EPSILON = opts.epsilon
        self.STEPS = opts.physical_integration_substeps

        self.grav = ca.vertcat(0, 0, 9.81)
        self.S = opts.aircraft_config.reference_area
        self.b = opts.aircraft_config.span
        self.c = opts.aircraft_config.chord
        self.mass = opts.aircraft_config.mass
        self.com = opts.aircraft_config.aero_centre_offset # position of aerodynamic centre relative to the centre of mass
        # self.com[0] = 0.028
        self.rudder_moment_arm = 0.5 # distance between centre of mass and the tail of the plane (used for damping calculations)
        # self.com[0] *= 1 #* 0.6
        # self.com[2] *= 1
        # self.com[2] = 0
        # self.com = np.array([0.0134613, -7.8085e-09, 0.00436365])
        self.length = opts.aircraft_config.length
        self.opts = opts

        self.dt_sym = ca.MX.sym('dt')

        self.state
        self.control

    
    @property
    def state(self):
        if not hasattr(self, '_state_initialized') or not self._state_initialized:
            # Define _symbolic state variables once
            self._q_frd_ned = ca.MX.sym('q_frd_ned', 4)
            self._p_ned = ca.MX.sym('p_ned', 3)
            self._v_ned = ca.MX.sym('v_ned', 3)
            self._omega_frd_ned = ca.MX.sym('omega_frd_ned', 3)

            self._state = ca.vertcat(
            self._p_ned, 
            self._v_ned, 
            self._q_frd_ned, 
            self._omega_frd_ned
            )

            self.num_states = self._state.size()[0]
            
            # Set the flag to indicate initialization
            self._state_initialized = True

        # Bundle state variables together
        return self._state
    
    @property
    def control(self):
        if not hasattr(self, '_control_initialized') or not self._control_initialized:
            self._aileron = ca.MX.sym('aileron')
            self._elevator = ca.MX.sym('elevator')
            self._thrust = ca.MX.sym('thrust')

            self._control = ca.vertcat(
            self._aileron, 
            self._elevator,
            self._thrust
            )
            self.num_controls = self._control.size()[0]

            self._control_initialized = True

        return self._control

    @property
    def inertia_tensor(self):
        """
        Inertia Tensor around the Centre of Mass
        """

        Ixx = self.opts.aircraft_config.Ixx
        Iyy = self.opts.aircraft_config.Iyy
        Ixz = self.opts.aircraft_config.Ixz
        Izz = self.opts.aircraft_config.Izz

        aero_inertia_tensor = ca.vertcat(
            ca.horzcat(Ixx, 0  , Ixz),
            ca.horzcat(0  , Iyy, 0  ),
            ca.horzcat(Ixz, 0  , Izz)
        )

        mass = self.mass

        com = self.com
        
        x, y, z = com[0], com[1], com[2]

        com_term = ca.vertcat(
            ca.horzcat(y**2 + z**2, -x*y, -x*z),
            ca.horzcat(-y*x, x**2 + z**2, -y*z),
            ca.horzcat(-z*x, -z*y, x**2 + y**2)
        )

        inertia_tensor = aero_inertia_tensor + mass * com_term

        return inertia_tensor
    
    @property
    def inverse_inertia_tensor(self):
        """
        Inverted Inertia Tensor around Centre of Mass
        """
        return ca.inv(self.inertia_tensor)

    @property
    def v_frd_rel(self):
        q_frd_ned = Quaternion(self._q_frd_ned)
        v_ned = Quaternion(ca.vertcat(self._v_ned, 0))

        self._v_frd_rel = (q_frd_ned.inverse() * (v_ned) 
                            * q_frd_ned).coeffs()[:3] + self.EPSILON
        
        return ca.Function('v_frd_rel', [self.state, self.control], 
            [self._v_frd_rel]).expand()
    
    @property
    def airspeed(self):
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        self._airspeed = ca.sqrt(ca.sumsqr(self._v_frd_rel) + self.EPSILON)

        return ca.Function('airspeed', [self.state, self.control], 
            [self._airspeed]).expand()
    
    @property
    def alpha(self):
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        v_frd_rel = self._v_frd_rel
        self._alpha = ca.atan2(v_frd_rel[2], v_frd_rel[0] + self.EPSILON)
        return ca.Function('alpha', [self.state, self.control], [self._alpha]).expand()
    

    
    # @property
    # def elevator_alpha(self):
    #     if not hasattr(self, '_v_frd_rel'):
    #         self.v_frd_rel
    #     v_frd_rel = self._v_frd_rel
    #     self._elevator_alpha = ca.atan2(v_frd_rel[2] + self.length * self._omega_frd_ned[1], v_frd_rel[0] + self.EPSILON)
        # return ca.Function('elevator_alpha', [self.state, self.control], [self._elevator_alpha]).expand()
    

    @property
    def elevator_alpha(self):
        """
        Includes changed angle of attack due to pitch rate
        """
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        v_frd_rel = self._v_frd_rel
        self._elevator_alpha = ca.atan2(v_frd_rel[2] + self.rudder_moment_arm * self._omega_frd_ned[1], v_frd_rel[0] + self.EPSILON)
        return ca.Function('elevator_alpha', [self.state, self.control], [self._elevator_alpha]).expand()
    

    @property
    def left_wing_alpha(self):
        """
        Includes changed angle of attack due to roll rate
        """
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        v_frd_rel = self._v_frd_rel
        self._left_wing_alpha = ca.atan2(v_frd_rel[2] - self.b * self._omega_frd_ned[0] / 4, v_frd_rel[0] + self.EPSILON)

        self._left_wing_alpha = ca.fmax(ca.fmin(self._left_wing_alpha, np.deg2rad(30)), np.deg2rad(-30))
        return ca.Function('left_wing_alpha', [self.state, self.control], [self._left_wing_alpha]).expand()
    
    @property
    def right_wing_alpha(self):
        """
        Includes changed angle of attack due to roll rate
        """
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        v_frd_rel = self._v_frd_rel
        self._right_wing_alpha = ca.atan2(v_frd_rel[2] + self.b * self._omega_frd_ned[0] / 4, v_frd_rel[0] + self.EPSILON)

        self._right_wing_alpha = ca.fmax(ca.fmin(self._right_wing_alpha, np.deg2rad(30)), np.deg2rad(-30))
        return ca.Function('right_wing_alpha', [self.state, self.control], [self._right_wing_alpha]).expand()


    @property
    def phi(self):
        x, y, z, w = self._q_frd_ned[0], self._q_frd_ned[1], self._q_frd_ned[2], self._q_frd_ned[3]
        self._phi = ca.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        return ca.Function('phi', [self.state], [self._phi]).expand()

    @property
    def theta(self):
        x, y, z, w = self._q_frd_ned[0], self._q_frd_ned[1], self._q_frd_ned[2], self._q_frd_ned[3]
        self._theta = ca.asin(2 * (w * y - z * x))
        return ca.Function('theta', [self.state], [self._theta]).expand()

    @property
    def psi(self):
        x, y, z, w = self._q_frd_ned[0], self._q_frd_ned[1], self._q_frd_ned[2], self._q_frd_ned[3]
        self._psi = ca.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return ca.Function('psi', [self.state], [self._psi]).expand()

    @property
    def phi_dot(self):
        if not hasattr(self, '_phi'):
            self.phi

        if not hasattr(self, '_theta'):
            self.theta

        phi = self._phi
        theta = self._theta
        p, q, r = self._omega_frd_ned[0], self._omega_frd_ned[1], self._omega_frd_ned[2]

        self._phi_dot = p + ca.sin(phi) * ca.tan(theta) * q + ca.cos(phi) * ca.tan(theta) * r

        return ca.Function('phi_dot', [self.state], [self._phi_dot]).expand()

    @property
    def theta_dot(self):
        if not hasattr(self, '_phi'):
            self.phi
        phi = self._phi
        q, r = self._omega_frd_ned[1], self._omega_frd_ned[2]

        self._theta_dot = ca.cos(phi) * q - ca.sin(phi) * r
        return ca.Function('theta_dot', [self.state], [self._theta_dot]).expand()

    @property
    def psi_dot(self):
        if not hasattr(self, '_phi'):
            self.phi
        if not hasattr(self, '_theta'):
            self.theta
        phi = self._phi
        theta = self._theta
        
        q, r = self._omega_frd_ned[1], self._omega_frd_ned[2]

        self._psi_dot = (ca.sin(phi) / ca.cos(theta)) * q + (ca.cos(phi) / ca.cos(theta)) * r
        return ca.Function('psi_dot', [self.state], [self._psi_dot]).expand()

    @property
    def beta(self):
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        if not hasattr(self, '_airspeed'):
            self.airspeed

        v_frd_rel = self._v_frd_rel
        self._beta = ca.asin(v_frd_rel[1] / self._airspeed)
        return ca.Function('beta', [self.state, self.control], [self._beta]).expand()
    
    @property
    def rudder_beta(self):
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        if not hasattr(self, '_airspeed'):
            self.airspeed

        
        v_frd_rel = self._v_frd_rel
        v_frd_rel[1] = v_frd_rel[1] - self.rudder_moment_arm * self._omega_frd_ned[2]

        airspeed = ca.sqrt(ca.sumsqr(v_frd_rel) + self.EPSILON)
        self._rudder_beta = ca.asin(v_frd_rel[1] / airspeed)
        self._rudder_beta = ca.fmax(ca.fmin(self._rudder_beta, np.deg2rad(20)), np.deg2rad(-20))
        return ca.Function('rudder_beta', [self.state, self.control], [self._rudder_beta]).expand()
         
    @property
    def qbar(self):
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        self._qbar = 0.5 * 1.225 * ca.dot(self._v_frd_rel, self._v_frd_rel)
        return ca.Function('qbar', [self.state, self.control], [self._qbar]).expand()
    
    @property
    def left_wing_qbar(self):
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel
        
        r = self._omega_frd_ned[2]
        new_vel = self._v_frd_rel[0] 
        new_vel[0] += self.b * r / 4
        new_vel = ca.fmin(100, new_vel)
        self._left_wing_qbar = 0.5 * 1.225 * ca.dot(new_vel, new_vel)
        return ca.Function('left_wing_qbar', [self.state, self.control], [self._left_wing_qbar]).expand()
    
    @property
    def right_wing_qbar(self):
        if not hasattr(self, '_v_frd_rel'):
            self.v_frd_rel

        r = self._omega_frd_ned[2]
        new_vel = self._v_frd_rel[0] 
        new_vel[0] += self.b * r / 4

        new_vel = ca.fmin(100, new_vel)
        self._right_wing_qbar = 0.5 * 1.225 * ca.dot(new_vel, new_vel)
        return ca.Function('right_wing_qbar', [self.state, self.control], [self._right_wing_qbar]).expand()
        
    @property
    def coefficients(self):
        """
        Forward pass of the ml model to retrieve aerodynamic coefficients.

        To calculate damping factors the effective velocities (under the angular rotation) of the relevant lifting surfaces are calculated and passed as inputs to the model.
        """
        if not hasattr(self, '_qbar'):
            self.qbar
        if not hasattr(self, '_alpha'):
            self.alpha
        if not hasattr(self, '_beta'):
            self.beta
        if not hasattr(self, '_elevator_alpha'):
            self.elevator_alpha
        if not hasattr(self, '_right_wing_alpha'):
            self.right_wing_alpha
        if not hasattr(self, '_left_wing_alpha'):
            self.left_wing_alpha
        if not hasattr(self, '_right_wing_qbar'):
            self.right_wing_qbar
        if not hasattr(self, '_left_wing_qbar'):
            self.left_wing_qbar
        if not hasattr(self, '_rudder_beta'):
            self.rudder_beta

        rudder_beta = self._rudder_beta
        left_wing_qbar = self._left_wing_qbar
        right_wing_qbar = self._right_wing_qbar
        left_wing_alpha = self._left_wing_alpha
        right_wing_alpha = self._right_wing_alpha
        elevator_alpha = self._elevator_alpha
        
        qbar = self._qbar
        alpha = self._alpha
        beta = self._beta
        aileron = self._aileron
        elevator = self._elevator

        inputs = ca.vertcat(
            qbar, 
            alpha, 
            beta, 
            aileron, 
            elevator
            )

        # TODO: adapt reference areas for the relevant lifting surfaces. currently they are contributing too much (probably, especially the wings with the roll moment)
        
        if self.LINEAR:
            outputs = ca.mtimes(self.linear_coeffs, ca.vertcat(inputs, 1))

        elif self.fitted_models is not None:
            outputs = ca.vertcat(*[self.fitted_models['casadi_functions'][i](inputs) for i in self.fitted_models['casadi_functions'].keys()])



            # # roll rate contribution with terms due to changed angle of attack and airspeed
            left_wing_inputs = ca.vertcat(
                qbar,
                left_wing_alpha,
                0,
                0,
                0
            )
            left_wing_lift_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](left_wing_inputs) for i in self.fitted_models['casadi_functions'].keys()])
            right_wing_inputs = ca.vertcat(
                qbar,
                right_wing_alpha,
                0,
                0,
                0
            )
            right_wing_lift_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](right_wing_inputs) for i in self.fitted_models['casadi_functions'].keys()])

            outputs[3] += self.b / 4 * (right_wing_lift_coeff[2] / 2 - left_wing_lift_coeff[2] / 2) # span over 4 assumes wing lift attacks at centre of wing, coeffs/2 to adjust reference area as the lift of each wing should be halved relative to the whole plane
            # pitch coefficient contribution due to pitch rate
            elevator_inputs = ca.vertcat(
                qbar,
                elevator_alpha,
                beta,
                aileron,
                elevator
            )
            elevator_damped_pitch_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](elevator_inputs) for i in self.fitted_models['casadi_functions'].keys()])
            outputs[4] = elevator_damped_pitch_coeff[4]

            # yaw coefficient contribution due to yaw rate
            rudder_inputs = ca.vertcat(
                qbar,
                alpha,
                rudder_beta,
                aileron,
                elevator
            )  
            rudder_damped_yaw_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](rudder_inputs) for i in self.fitted_models['casadi_functions'].keys()])
            outputs[5] = rudder_damped_yaw_coeff[5]

        else:
            outputs = self.model(ca.reshape(inputs, 1, -1))
            outputs = ca.vertcat(outputs.T)


        # # outputs[0] = 0
        # outputs[1] = 0
        # # outputs[2] = 0
        # outputs[3] = 0
        # # outputs[4] = 0
        # outputs[5] = 0


        # stall scaling
        stall_angle_alpha = np.deg2rad(10)
        stall_angle_beta = np.deg2rad(10)

        steepness = 10

        alpha_scaling = 1 / (1 + ca.exp(steepness * (ca.fabs(alpha) - stall_angle_alpha)))
        beta_scaling = 1 / (1 + ca.exp(steepness * (ca.fabs(beta) - stall_angle_beta)))
        
        # outputs[2] *= -1
        outputs[2] *= alpha_scaling
        outputs[2] *= beta_scaling

        outputs[4] *= alpha_scaling
        outputs[4] *= beta_scaling

        # p, q, r = self._omega_frd_ned[0], self._omega_frd_ned[1], self._omega_frd_ned[2]

        # # angular rate contributions
        # # outputs[0] += 0.05 * self.omega_b_i_frd[0]
        # # outputs[1] += -0.05 * self.omega_frd_ned[2]
        # # outputs[2] += -0.1 * self.omega_frd_ned[0]
        # outputs[1] += -0.05 * r
        # outputs[2] += -0.1 * p

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
        # scale = 1
        # # roll moment rates
        
        # outputs[3] /= 16
        # outputs[3] += -0.005 * p * scale
        # outputs[3] += 0.001 * r * scale

        # # pitching moment rates
        # outputs[4] += -0.2 * q * scale

        # # yaw moment rates
        # outputs[5] *= -1
        # # outputs[5] += -0.0003 * p * scale
        # outputs[5] += -0.001 * r * scale

        self._coefficients = outputs

        return ca.Function(
            'coefficients', 
            [self.state, self.control], 
            [self._coefficients]
            )

    @property
    def forces_frd(self):
        if not hasattr(self, '_coefficients'):
            self.coefficients
        if not hasattr(self, '_qbar'):
            self.qbar

        forces = self._coefficients[:3] * self._qbar * self.S# + self._thrust
        # antialign drag and velocity
        forces[0] = ca.sign(self._v_frd_rel[0])*forces[0]
        # forces += self._throttle
        speed_threshold = 80.0  # m/s
        penalty_factor = 10.0  # Scale factor for additional drag

        # Smooth penalty function for increased drag
        excess_speed = self._airspeed - speed_threshold
        additional_drag = penalty_factor * ca.fmax(0, excess_speed)**2  # Quadratic penalty for smoothness

        # Apply the additional drag along the x-axis (forward direction)
        forces[0] -= additional_drag

        self._forces_frd = forces
        return ca.Function('forces_frd', 
            [self.state, self.control], [forces]).expand()
    
    @property
    def moments_aero_frd(self):
        if not hasattr(self, '_coefficients'):
            self.coefficients
        if not hasattr(self, '_qbar'):
            self.qbar
        moments_aero = (self._coefficients[3:] * self._qbar * self.S 
                        * ca.vertcat(self.b, self.c, self.b))

        self._moments_aero_frd = moments_aero
        return ca.Function('moments_aero_frd', 
            [self.state, self.control], [moments_aero]).expand()
    
    @property
    def moments_from_forces_frd(self):
        if not hasattr(self, '_forces_frd'):
            self.forces_frd
        self._moments_from_forces_frd = ca.cross(self.com, self._forces_frd)
        
        return ca.Function('moments_from_forces_frd', 
            [self.state, self.control], [self._moments_from_forces_frd]).expand()
    
    @property
    def moments_frd(self):
        if not hasattr(self, '_moments_aero_frd'):
            self.moments_aero_frd
        if not hasattr(self, '_moments_from_forces_frd'):
            self.moments_from_forces_frd
        self._moments_frd = self._moments_aero_frd + self._moments_from_forces_frd
        
        return ca.Function('moments_frd', 
            [self.state, self.control], [self._moments_frd]).expand()
    
    @property
    def forces_ned(self):
        if not hasattr(self, '_forces_frd'):
            self.forces_frd
        forces_frd = Quaternion(ca.vertcat(self._forces_frd, 0))
        q_frd_ned = Quaternion(self._q_frd_ned)
        self._forces_ned = (q_frd_ned * forces_frd * q_frd_ned.inverse()).coeffs()[:3]
        return ca.Function('forces_ned', 
            [self.state, self.control], [self._forces_ned]).expand()
        
    @property
    def q_frd_ned_dot(self):
        q_frd_ecf = Quaternion(self._q_frd_ned)
        omega_frd_ned = Quaternion(ca.vertcat(self._omega_frd_ned, 0))

        self._q_frd_ned_dot = (0.5 * q_frd_ecf * omega_frd_ned).coeffs()
        return ca.Function('q_frd_ecf_dot', 
            [self.state, self.control], [self._q_frd_ned_dot]).expand()
    @property
    def q_frd_ned_update(self):
        dt = self.dt_sym
        q_frd_ned = Quaternion(self._q_frd_ned)
        omega_frd_ned = self._omega_frd_ned  # Assume this is a 3D vector
        # dt = self.dt  # Time step

        theta = ca.norm_2(omega_frd_ned)  # Angular velocity magnitude
        half_theta = 0.5 * dt * theta

        # Compute exponential map terms
        exp_q = ca.vertcat(ca.if_else(theta > 1e-6,  # Avoid division by zero
                                    ca.sin(half_theta) * omega_frd_ned / theta,
                                    ca.MX.zeros(3, 1)),# Vector part, 
                                    ca.cos(half_theta))   # Scalar part

        # Compute the updated quaternion
        q_next = Quaternion.product(exp_q, q_frd_ned.coeffs())

        return ca.Function('q_frd_ned_update',
                        [self.state, self.control, dt], [q_next]).expand()


    @property
    def p_ned_dot(self):
        self._p_ned_dot = self._v_ned
        
        return ca.Function('p_ecf_cm_O_dot', 
            [self.state, self.control], [self._v_ned]).expand()
    
    @property
    def v_ned_dot(self):
        if not hasattr(self, '_forces_ned'):
            self.forces_ned
        forces = self._forces_ned
        grav = self.grav
        mass = self.mass

        self._v_ned_dot =  forces / mass + grav
        
        return ca.Function(
            'v_ecf_cm_e_dot', 
            [self.state, self.control], 
            [self._v_ned_dot]
            ).expand()

    @property
    def omega_frd_ned_dot(self):
        if not hasattr(self, '_moments_frd'):
            self.moments_frd
        J_frd = self.inertia_tensor
        J_frd_inv = self.inverse_inertia_tensor
        
        omega_frd_ned = self._omega_frd_ned
        moments = self._moments_frd


        self._omega_frd_ned_dot = ca.mtimes(J_frd_inv, (moments 
             - ca.cross(omega_frd_ned, ca.mtimes(J_frd, omega_frd_ned))))
        
        return ca.Function(
            'omega_frd_ned_dot', 
            [self.state, self.control], 
            [self._omega_frd_ned_dot]
            ).expand()
    
    @property
    def state_derivative(self):
        if not hasattr(self, '_q_frd_ned_dot'):
            self.q_frd_ned_dot
        if not hasattr(self, '_p_ned_dot'):
            self.p_ned_dot
        if not hasattr(self, '_v_ned_dot'):
            self.v_ned_dot
        if not hasattr(self, '_omega_frd_ned_dot'):
            self.omega_frd_ned_dot
        self._state_derivative = ca.vertcat(
            self._p_ned_dot, 
            self._v_ned_dot, 
            self._q_frd_ned_dot, 
            self._omega_frd_ned_dot
            )
        
        return ca.Function(
            'dynamics', 
            [self.state, self.control], 
            [self._state_derivative], ['x', 'u'], ['x_dot']
            ).expand()
    
    # @jit
    def state_step(self, state, control, dt_scaled):
        """ 
        Runge kutta step for state update. Due to the multiplicative nature
        of the quaternion integration we cannot rely on conventional 
        integerators. 
        """
        state_derivative = self.state_derivative
        k1 = state_derivative(state, control)
        state_k1 = state + dt_scaled / 2 * k1

        k2 = state_derivative(state_k1, control)
        state_k2 = state + dt_scaled / 2 * k2


        k3 = state_derivative(state_k2, control)
        state_k3 = state + dt_scaled * k3

        k4 = state_derivative(state_k3, control)

        state = state + dt_scaled / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Normalize the quaternion
        # quaternion = Quaternion(state[6:10])

        state[6:10] = self.q_frd_ned_update(state, control, dt_scaled)
        # quaternion.normalize().coeffs()

        # self._state_step = ca.Function('step', [state, control, dt_scaled], [state])

        return state


    @property
    def state_update(self, normalisation_interval: int = 10):
        """
        Runge Kutta integration with quaternion update, for loop over self.STEPS
        """
        dt = self.dt_sym
        state = self.state
        control_sym = self.control
        num_steps = self.STEPS

        if num_steps == 1:
            state = self.state_step(state, control_sym, dt)
        else:

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
            ).expand() #, {'jit':True}


def perturb_quaternion(q, delta_theta=0.01):
    """ Perturbs a quaternion by a small rotation. """
    # Generate a small random rotation axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize to unit vector
    
    # Create small rotation quaternion
    delta_q = R.from_rotvec(delta_theta * axis).as_quat()  # [x, y, z, w]
    
    # Apply rotation (Hamilton product)
    q_perturbed = R.from_quat(q) * R.from_quat(delta_q)
    
    return q_perturbed.as_quat()  # Return perturbed quaternion

if __name__ == '__main__':
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
    model_path = Path(NETWORKPATH) / 'model-dynamics.pth'
    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'

    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)
    # opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)

    perturbation = False
    
    trim_state_and_control = [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]# [0, 0, 0, 60, 2.29589e-41, 0, 0, 9.40395e-38, -2.93874e-39, 1, 0, 1.46937e-39, 0, -5.73657e-45, 0, 0, 0.0134613, -7.8085e-09, 0.00436365]
    if trim_state_and_control is not None:
        state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
        control = np.array(trim_state_and_control[aircraft.num_states:-3])
        control[0] = 0
        control[1] = 0
        aircraft.com = np.array(trim_state_and_control[-3:])
    else:

        x0 = np.zeros(3)
        v0 = ca.vertcat([60, 0, 0])
        # would be helpful to have a conversion here between actual pitch, roll and yaw angles and the Quaternion q0, so we can enter the angles in a sensible way.
        q0 = Quaternion(ca.vertcat(0, 0, 0, 1))
        # q0 = Quaternion(ca.vertcat(.259, 0, 0, 0.966))
        # q0 = ca.vertcat([0.215566, -0.568766, 0.255647, 0.751452])#q0.inverse()
        omega0 = np.array([0, 0, 0])

        state = ca.vertcat(x0, v0, q0, omega0)
        control = np.zeros(aircraft.num_controls)
        control[0] = +0
        control[1] = 5
        # control[6:9] = traj_dict['aircraft']['aero_centre_offset']

    dyn = aircraft.state_update
    # dt_sym = ca.MX.sym('dt')
    # dyn = ca.Function('step', [aircraft.state, aircraft.control, dt_sym], [aircraft.state_step(aircraft.state, aircraft.control, dt_sym)]).expand()
    dt = .01
    tf = 500
    state_list = np.zeros((aircraft.num_states, int(tf / dt)))
    # investigate stiffness:

    # Define f(state, control) (e.g., the dynamics function)
    f = aircraft.state_update(aircraft.state, aircraft.control, aircraft.dt_sym)

    # Compute the Jacobian of f w.r.t state
    J = ca.jacobian(f, aircraft.state)

    # Create a CasADi function for numerical evaluation
    J_func = ca.Function('J', [aircraft.state, aircraft.control, aircraft.dt_sym], [J])

    # Evaluate J numerically for a specific state and control
    J_val = J_func(state, control, .01)

    # Compute eigenvalues using numpy
    eigvals = np.linalg.eigvals(np.array(J_val))

    print(eigvals)

    import numpy as np

    # Define perturbations (adjust as needed)
    state_perturbations = np.linspace(-0.1, 0.1, num=5)  # Small deviations
    control_perturbations = np.linspace(-0.0, 0.0, num=5)

    # Storage for eigenvalues
    eigenvalues_list = []

    for dx in state_perturbations:
        for du in control_perturbations:
            perturbed_state = state + dx
            perturbed_quaternion = perturb_quaternion(state[6:10].toarray().flatten())
            perturbed_state[6:10] = perturbed_quaternion
            perturbed_control = control + du
            
            # Evaluate the discrete-time Jacobian at perturbed states
            J_val = J_func(perturbed_state, perturbed_control, 0.01)
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(J_val)
            eigenvalues_list.append(eigenvalues)

    # Convert to NumPy array for easier analysis
    eigenvalues_array = np.array(eigenvalues_list)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,6))
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
    plt.gca().add_patch(unit_circle)

    for eigvals in eigenvalues_array:
        plt.scatter(eigvals.real, eigvals.imag, color='blue', alpha=0.5)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Eigenvalues Under Perturbed States & Controls")
    plt.grid()
    plt.show(block = True)

    # Define timestep range (log scale for better resolution)
    timesteps = np.logspace(-4, 0, num=20)  # From very small to larger dt values
    max_eigenvalues = []

    for dt in timesteps:
        # Compute discrete-time Jacobian at this timestep
        J_val = J_func(state, control, dt)  # Get continuous Jacobian
        # J_d = np.eye(J_val.shape[0]) + dt * J_val  # First-order discretization (Euler)

        # Compute eigenvalues and store the largest norm
        eigvals = np.linalg.eigvals(J_val)
        max_eigenvalues.append(max(np.abs(eigvals)))

    # Plot results
    plt.figure(figsize=(7, 5))
    plt.plot(timesteps, max_eigenvalues, marker='o', linestyle='-')
    plt.xscale("log")  # Log scale for better visualization
    plt.yscale("log")
    plt.axhline(1, color='r', linestyle='--', label="Unit Circle Bound")
    plt.xlabel("Timestep (Δt)")
    plt.ylabel("Max Eigenvalue Norm")
    plt.title("Max Eigenvalue Norm vs. Timestep")
    plt.legend()
    plt.grid()
    plt.show(block = True)


    



    # # dt_sym = ca.MX.sym('dt', 1)
    # t = 0
    # ele_pos = True
    # ail_pos = True
    # control_list = np.zeros((aircraft.num_controls, int(tf / dt)))
    # for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
    #     if np.isnan(state[0]):
    #         print('Aircraft crashed')
    #         break
    #     else:
    #         state_list[:, i] = state.full().flatten()
    #         control_list[:, i] = control
    #         state = dyn(state, control, dt)
                    
    #         t += 1
    # # print(state)
    # # J_val = J_func(state, control)
    # # eigvals = np.linalg.eigvals(np.array(J_val))

    # # print(eigvals)
    # first = None
    # # t -=10
    # def save(filepath):
    #     with h5py.File(filepath, "a") as h5file:
    #         grp = h5file.create_group(f'iteration_0')
    #         grp.create_dataset('state', data=state_list[:, :t])
    #         grp.create_dataset('control', data=control_list[:, :t])
    
    
    # filepath = os.path.join("data", "trajectories", "simulation.h5")
    # if os.path.exists(filepath):
    #     os.remove(filepath)
    # save(filepath)

    # plotter = TrajectoryPlotter(aircraft)
    # plotter.plot(filepath=filepath)
    # plt.show(block = True)