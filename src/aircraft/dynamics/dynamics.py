

import casadi as ca
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

from liecasadi import Quaternion
from scipy.spatial.transform import Rotation as R
import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import h5py

from aircraft.config import BASEPATH, NETWORKPATH, DATAPATH, DEVICE
from aircraft.surrogates.models import ScaledModel
from aircraft.utils.utils import load_model, TrajectoryConfiguration, AircraftConfiguration, perturb_quaternion

from dataclasses import dataclass


print(DEVICE)

@dataclass
class SixDOFOpts:
    epsilon:float = 1e-6
    physical_integration_substeps:int = 10
    gravity:ca.MX = ca.vertcat(0, 0, 9.81)
    mass:float = 1.0

@dataclass
class AircraftOpts(SixDOFOpts):
    linear_path:Optional[Path] = None
    poly_path:Optional[Path] = None
    nn_model_path:Optional[Path] = None
    aircraft_config:Optional[AircraftConfiguration] = None
    realtime:bool = False # Work in progress implementation of faster nn 
    stall_angle_alpha:Tuple[float] = (np.deg2rad(-10), np.deg2rad(10))
    stall_angle_beta:Tuple[float] = (np.deg2rad(-10), np.deg2rad(10))
    
    def __post_init__(self):
        if self.aircraft_config is not None:
            self.mass = self.aircraft_config.mass
    

class SixDOF(ABC):
    """
    Baseclass for 6DOF Dynamics Simulation in a NED system
    """
    def __init__(self, opts:Optional[SixDOFOpts] = SixDOFOpts()):
        self.gravity = opts.gravity
        self.dt_sym = ca.MX.sym('dt')
        self.epsilon = opts.epsilon
        self.mass = opts.mass
        self.com = None

    def _ensure_initialized(self, *properties):
        """Ensure properties are initialized by calling them"""
        for prop in properties:
            if not hasattr(self, f"_{prop}"):
                getattr(self, prop)

    @property
    def state(self) -> ca.MX:
        if not hasattr(self, '_state_initialized') or not self._state_initialized:
            # Define _symbolic state variables once
            self._q_frd_ned = ca.MX.sym('q_frd_ned', 4)
            self._p_ned = ca.MX.sym('p_ned', 3)
            self._v_ned = ca.MX.sym('v_ned', 3)
            self._omega_frd_ned = ca.MX.sym('omega_frd_ned', 3)
            
            # Bundle state variables together
            self._state = ca.vertcat(
            self._p_ned, 
            self._v_ned, 
            self._q_frd_ned, 
            self._omega_frd_ned
            )

            self.num_states = self._state.size()[0]
            
            # Set the flag to indicate initialization
            self._state_initialized = True

        return self._state
    
    
    @property
    @abstractmethod
    def control(self) -> ca.MX:
        pass
    
    
    @property
    @abstractmethod
    def inertia_tensor(self) -> ca.MX:
        """
        Inertia Tensor around the Centre of Mass
        """
        pass
    
    @property
    def inverse_inertia_tensor(self) -> ca.MX:
        """
        Inverted Inertia Tensor around Centre of Mass
        """
        return ca.inv(self.inertia_tensor)
    

    @property
    def v_frd_rel(self):
        q_frd_ned = Quaternion(self._q_frd_ned)
        
        if hasattr(self, 'v_wind'):
            vel = self._v_ned + self.v_wind
        else:
            vel = self._v_ned

        v_ned = Quaternion(ca.vertcat(vel, 0))

        self._v_frd_rel = (q_frd_ned.inverse() * (v_ned) 
                            * q_frd_ned).coeffs()[:3] + self.epsilon
        
        return ca.Function('v_frd_rel', [self.state, self.control], 
            [self._v_frd_rel])
    
    @property
    def airspeed(self):
        self._ensure_initialized('v_frd_rel')
        self._airspeed = ca.sqrt(ca.sumsqr(self._v_frd_rel) + self.epsilon)

        return ca.Function('airspeed', [self.state, self.control], 
            [self._airspeed])
    
    @property
    def alpha(self):
        self._ensure_initialized('v_frd_rel')
        v_frd_rel = self._v_frd_rel
        self._alpha = ca.atan2(v_frd_rel[2], v_frd_rel[0] + self.epsilon)
        return ca.Function('alpha', [self.state, self.control], [self._alpha])    
    
    @property
    def phi(self):
        x, y, z, w = self._q_frd_ned[0], self._q_frd_ned[1], self._q_frd_ned[2], self._q_frd_ned[3]
        self._phi = ca.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        return ca.Function('phi', [self.state], [self._phi])

    @property
    def theta(self):
        x, y, z, w = self._q_frd_ned[0], self._q_frd_ned[1], self._q_frd_ned[2], self._q_frd_ned[3]
        self._theta = ca.asin(2 * (w * y - z * x))
        return ca.Function('theta', [self.state], [self._theta])

    @property
    def psi(self):
        x, y, z, w = self._q_frd_ned[0], self._q_frd_ned[1], self._q_frd_ned[2], self._q_frd_ned[3]
        self._psi = ca.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return ca.Function('psi', [self.state], [self._psi])

    @property
    def phi_dot(self):
        self._ensure_initialized('phi', 'theta')

        phi = self._phi
        theta = self._theta
        p, q, r = self._omega_frd_ned[0], self._omega_frd_ned[1], self._omega_frd_ned[2]

        self._phi_dot = p + ca.sin(phi) * ca.tan(theta) * q + ca.cos(phi) * ca.tan(theta) * r

        return ca.Function('phi_dot', [self.state], [self._phi_dot])

    @property
    def theta_dot(self):
        self._ensure_initialized('phi')
        phi = self._phi
        q, r = self._omega_frd_ned[1], self._omega_frd_ned[2]

        self._theta_dot = ca.cos(phi) * q - ca.sin(phi) * r
        return ca.Function('theta_dot', [self.state], [self._theta_dot])

    @property
    def psi_dot(self):

        self._ensure_initialized('phi', 'theta')
        phi = self._phi
        theta = self._theta
        
        q, r = self._omega_frd_ned[1], self._omega_frd_ned[2]

        self._psi_dot = (ca.sin(phi) / ca.cos(theta)) * q + (ca.cos(phi) / ca.cos(theta)) * r
        return ca.Function('psi_dot', [self.state], [self._psi_dot])

    @property
    def beta(self):
        self._ensure_initialized('v_frd_rel', 'airspeed')

        v_frd_rel = self._v_frd_rel
        self._beta = ca.asin(v_frd_rel[1] / self._airspeed)
        return ca.Function('beta', [self.state, self.control], [self._beta])

    @property
    def qbar(self):
        self._ensure_initialized('v_frd_rel')
        self._qbar = 0.5 * 1.225 * ca.dot(self._v_frd_rel, self._v_frd_rel)
        return ca.Function('qbar', [self.state, self.control], [self._qbar])

    @property
    @abstractmethod
    def _forces_frd(self) -> ca.MX:
        pass

    @property
    def forces_frd(self) -> ca.Function:
        return ca.Function('forces_frd',
            [self.state, self.control], [self._forces_frd])

    @property
    def _moments_from_forces_frd(self) -> ca.MX:
        return ca.cross(self.com, self._forces_frd)

    @property
    def moments_from_forces_frd(self) -> ca.Function:
        return ca.Function('moments_from_forces_frd',
            [self.state, self.control], [self._moments_from_forces_frd])

    @property
    @abstractmethod
    def _moments_aero_frd(self) -> ca.MX:
        pass

    @property
    def moments_aero_frd(self) -> ca.Function:
        return ca.Function('moments_aero_frd',
            [self.state, self.control], [self._moments_aero_frd])

    @property
    def moments_frd(self):
        self._ensure_initialized('moments_aero_frd', 'moments_from_forces_frd')
        self._moments_frd = self._moments_aero_frd + self._moments_from_forces_frd
        
        return ca.Function('moments_frd', 
            [self.state, self.control], [self._moments_frd])
    
    @property
    def forces_ned(self):
        if not hasattr(self, '_forces_frd'):
            self.forces_frd
        forces_frd = Quaternion(ca.vertcat(self._forces_frd, 0))
        q_frd_ned = Quaternion(self._q_frd_ned)
        self._forces_ned = (q_frd_ned * forces_frd * q_frd_ned.inverse()).coeffs()[:3]
        return ca.Function('forces_ned', 
            [self.state, self.control], [self._forces_ned])
        
    @property
    def q_frd_ned_dot(self):
        q_frd_ecf = Quaternion(self._q_frd_ned)
        omega_frd_ned = Quaternion(ca.vertcat(self._omega_frd_ned, 0))

        self._q_frd_ned_dot = (0.5 * q_frd_ecf * omega_frd_ned).coeffs()
        return ca.Function('q_frd_ecf_dot', 
            [self.state, self.control], [self._q_frd_ned_dot])
    
    @property
    def q_frd_ned_update(self):
        dt = self.dt_sym
        q_frd_ned = Quaternion(self._q_frd_ned)
        omega_frd_ned = self._omega_frd_ned  # Assume this is a 3D vector

        # Angular velocity magnitude
        theta = ca.norm_2(omega_frd_ned)  
        half_theta = 0.5 * dt * theta

        # Use Taylor series expansion for small angles to improve stability
        sin_half_theta = ca.if_else(theta > 1e-6, 
                                    ca.sin(half_theta), 
                                    half_theta)  # Approximate sin(x) ≈ x for small x
        cos_half_theta = ca.if_else(theta > 1e-6, 
                                    ca.cos(half_theta), 
                                    1 - (half_theta ** 2) / 2)  # Approximate cos(x) ≈ 1 - x^2/2

        # Normalized axis of rotation (handle division by zero safely)
        axis = ca.if_else(theta > 1e-6, 
                        omega_frd_ned / theta, 
                        ca.MX.zeros(3, 1))

        # Compute exponential map terms (quaternion)
        exp_q = ca.vertcat(sin_half_theta * axis, cos_half_theta)

        # Compute the updated quaternion and normalize it
        q_next = Quaternion.product(exp_q, q_frd_ned.coeffs())
        q_next = q_next / ca.norm_2(q_next)  # Ensure unit norm

        return ca.Function('q_frd_ned_update', [self.state, self.control, dt], [q_next])



    @property
    def p_ned_dot(self):
        self._p_ned_dot = self._v_ned
        
        return ca.Function('p_ecf_cm_O_dot', 
            [self.state, self.control], [self._v_ned])
    
    @property
    def v_ned_dot(self):
        if not hasattr(self, '_forces_ned'):
            self.forces_ned
        forces = self._forces_ned
        grav = self.gravity
        mass = self.mass

        self._v_ned_dot =  forces / mass + grav
        
        return ca.Function(
            'v_ecf_cm_e_dot', 
            [self.state, self.control], 
            [self._v_ned_dot]
            )

    @property
    def omega_frd_ned_dot(self):
        if not hasattr(self, '_moments_frd'):
            self.moments_frd
        
        omega_frd_ned = self._omega_frd_ned
        moments = self._moments_frd


        self._omega_frd_ned_dot = ca.mtimes(self.inverse_inertia_tensor, (moments 
             - ca.cross(omega_frd_ned, ca.mtimes(self.inertia_tensor, omega_frd_ned))))
        
        return ca.Function(
            'omega_frd_ned_dot', 
            [self.state, self.control], 
            [self._omega_frd_ned_dot]
            )

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
            )
    
    def state_step(self, state, control, dt_scaled):
        """ 
        Runge-Kutta step for state update.
        Due to the multiplicative nature of the quaternion integration,
        we cannot rely on conventional integrators.
        """
        state_derivative = self.state_derivative

        # Precompute scaled timestep constants
        half_dt = dt_scaled / 2
        sixth_dt = dt_scaled / 6

        # Runge-Kutta intermediate steps
        k1 = state_derivative(state, control)
        state_k1 = state + half_dt * k1

        k2 = state_derivative(state_k1, control)
        state_k2 = state + half_dt * k2

        k3 = state_derivative(state_k2, control)
        state_k3 = state + dt_scaled * k3

        k4 = state_derivative(state_k3, control)

        # Aggregate RK4 step
        state = state + sixth_dt * (k1 + 2 * k2 + 2 * k3 + k4)

        # Quaternion update to maintain unit norm
        state[6:10] = self.q_frd_ned_update(state, control, dt_scaled)
        state[6:10] = Quaternion(state[6:10]).normalize()

        return state



    @property
    def state_update(self, normalisation_interval: int = 10):
        """
        Runge Kutta integration with quaternion update, for loop over self.STEPS
        """
        dt = self.dt_sym
        state = self.state
        control_sym = self.control
        num_steps = 100# self.STEPS

        # if num_steps == 1:
        #     state = self.state_step(state, control_sym, dt)
        # else:

        dt_scaled = dt / num_steps
        input_to_fold = ca.vertcat(self.state, self.control, dt)
        fold_output = ca.vertcat(self.state_step(state, control_sym, dt_scaled), control_sym, dt)
        folded_update = ca.Function('folder', [input_to_fold], [fold_output])
        
        F = folded_update.fold(num_steps)
        state = F(input_to_fold)[:self.num_states]

        return ca.Function(
            'state_update', 
            [self.state, self.control, dt], 
            [state]
            ) #, {'jit':True}

class Aircraft(SixDOF):
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
        super().__init__()
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
            except ImportError:
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
        self.rudder_moment_arm = 0.5 # distance between centre of mass and the tail of the plane (used for damping calculations)
        self.length = opts.aircraft_config.length
        self.opts = opts

        self.dt_sym = ca.MX.sym('dt')

        self.state
        self.control

    @property
    def control(self):
        if not hasattr(self, '_control_initialized') or not self._control_initialized:
            self._aileron = ca.MX.sym('aileron')
            self._elevator = ca.MX.sym('elevator')
            self._thrust = ca.MX.sym('thrust', 3)

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

        i_xx = self.opts.aircraft_config.Ixx
        i_yy = self.opts.aircraft_config.Iyy
        i_xz = self.opts.aircraft_config.Ixz
        i_zz = self.opts.aircraft_config.Izz

        aero_inertia_tensor = ca.vertcat(
            ca.horzcat(i_xx, 0  , i_xz),
            ca.horzcat(0  , i_yy, 0  ),
            ca.horzcat(i_xz, 0  , i_zz)
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
    def elevator_alpha(self):
        """
        Includes changed angle of attack due to pitch rate
        """
        self._ensure_initialized('v_frd_rel')
        v_frd_rel = self._v_frd_rel
        self._elevator_alpha = ca.atan2(v_frd_rel[2] + self.rudder_moment_arm * self._omega_frd_ned[1], v_frd_rel[0] + self.EPSILON)
        return ca.Function('elevator_alpha', [self.state, self.control], [self._elevator_alpha])
    

    @property
    def left_wing_alpha(self):
        """
        Includes changed angle of attack due to roll rate
        """
        self._ensure_initialized('_v_frd_rel')
        v_frd_rel = self._v_frd_rel
        self._left_wing_alpha = ca.atan2(v_frd_rel[2] - self.b * self._omega_frd_ned[0] / 4, v_frd_rel[0] + self.EPSILON)

        self._left_wing_alpha = ca.fmax(ca.fmin(self._left_wing_alpha, np.deg2rad(30)), np.deg2rad(-30))
        return ca.Function('left_wing_alpha', [self.state, self.control], [self._left_wing_alpha])
    
    @property
    def right_wing_alpha(self):
        """
        Includes changed angle of attack due to roll rate
        """
        self._ensure_initialized('v_frd_rel')
        v_frd_rel = self._v_frd_rel
        self._right_wing_alpha = ca.atan2(v_frd_rel[2] + self.b * self._omega_frd_ned[0] / 4, v_frd_rel[0] + self.EPSILON)

        self._right_wing_alpha = ca.fmax(ca.fmin(self._right_wing_alpha, np.deg2rad(30)), np.deg2rad(-30))
        return ca.Function('right_wing_alpha', [self.state, self.control], [self._right_wing_alpha])
    
    @property
    def rudder_beta(self):
        self._ensure_initialized('v_frd_rel', 'airspeed')

        
        v_frd_rel = self._v_frd_rel
        v_frd_rel[1] = v_frd_rel[1] - self.rudder_moment_arm * self._omega_frd_ned[2]

        airspeed = ca.sqrt(ca.sumsqr(v_frd_rel) + self.EPSILON)
        self._rudder_beta = ca.asin(v_frd_rel[1] / airspeed)
        self._rudder_beta = ca.fmax(ca.fmin(self._rudder_beta, np.deg2rad(20)), np.deg2rad(-20))
        return ca.Function('rudder_beta', [self.state, self.control], [self._rudder_beta])
    
    @property
    def left_wing_qbar(self):
        self._ensure_initialized('v_frd_rel')
        
        r = self._omega_frd_ned[2]
        new_vel = self._v_frd_rel[0] 
        new_vel[0] += self.b * r / 4
        new_vel = ca.fmin(100, new_vel)
        self._left_wing_qbar = 0.5 * 1.225 * ca.dot(new_vel, new_vel)
        return ca.Function('left_wing_qbar', [self.state, self.control], [self._left_wing_qbar])
    
    @property
    def right_wing_qbar(self):
        self._ensure_initialized('v_frd_rel')

        r = self._omega_frd_ned[2]
        new_vel = self._v_frd_rel[0] 
        new_vel[0] += self.b * r / 4

        new_vel = ca.fmin(100, new_vel)
        self._right_wing_qbar = 0.5 * 1.225 * ca.dot(new_vel, new_vel)
        return ca.Function('right_wing_qbar', [self.state, self.control], [self._right_wing_qbar])
        
    @property
    def coefficients(self):
        """
        Forward pass of the ml model to retrieve aerodynamic coefficients.

        To calculate damping factors the effective velocities (under the angular rotation) of the relevant lifting surfaces are calculated and passed as inputs to the model.
        """

        self._ensure_initialized(
            'qbar', 'alpha', 'beta', 'elevator_alpha', 
            'right_wing_alpha', 'left_wing_alpha', 
            'right_wing_qbar', 'left_wing_qbar', 'rudder_beta'
        )

        inputs = ca.vertcat(
            self._qbar, 
            self._alpha, 
            self._beta, 
            self._aileron, 
            self._elevator
            )
        
        if self.LINEAR:
            outputs = ca.mtimes(self.linear_coeffs, ca.vertcat(inputs, 1))

        elif self.fitted_models is not None:
            outputs = ca.vertcat(*[self.fitted_models['casadi_functions'][i](inputs) for i in self.fitted_models['casadi_functions'].keys()])



            # roll rate contribution with terms due to changed angle of attack and airspeed
            left_wing_inputs = ca.vertcat(
                self._left_wing_qbar, # may want to use qbar here instead
                self._left_wing_alpha,
                0,
                0,
                0
            )
            left_wing_lift_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](left_wing_inputs) for i in self.fitted_models['casadi_functions'].keys()])
            right_wing_inputs = ca.vertcat(
                self._right_wing_qbar, # may want to use qbar here instead
                self._right_wing_alpha,
                0,
                0,
                0
            )
            right_wing_lift_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](right_wing_inputs) for i in self.fitted_models['casadi_functions'].keys()])

            outputs[3] += self.b / 4 * (right_wing_lift_coeff[2] / 2 - left_wing_lift_coeff[2] / 2) # span over 4 assumes wing lift attacks at centre of wing, coeffs/2 to adjust reference area as the lift of each wing should be halved relative to the whole plane
            # pitch coefficient contribution due to pitch rate
            elevator_inputs = ca.vertcat(
                self._qbar,
                self._elevator_alpha,
                self._beta,
                self._aileron,
                self._elevator
            )
            elevator_damped_pitch_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](elevator_inputs) for i in self.fitted_models['casadi_functions'].keys()])
            outputs[4] = elevator_damped_pitch_coeff[4]

            # yaw coefficient contribution due to yaw rate
            rudder_inputs = ca.vertcat(
                self._qbar,
                self._alpha,
                self._rudder_beta,
                self._aileron,
                self._elevator
            )  
            rudder_damped_yaw_coeff = ca.vertcat(*[self.fitted_models['casadi_functions'][i](rudder_inputs) for i in self.fitted_models['casadi_functions'].keys()])
            outputs[5] = rudder_damped_yaw_coeff[5]

        else:
            outputs = self.model(ca.reshape(inputs, 1, -1))
            outputs = ca.vertcat(outputs.T)


        # stall scaling
        stall_angle_alpha = np.deg2rad(10)
        stall_angle_beta = np.deg2rad(10)

        steepness = 10

        alpha_scaling = 1 / (1 + ca.exp(steepness * (ca.fabs(self._alpha) - stall_angle_alpha)))
        beta_scaling = 1 / (1 + ca.exp(steepness * (ca.fabs(self._beta) - stall_angle_beta)))
        
        outputs[2] *= alpha_scaling
        outputs[2] *= beta_scaling

        outputs[4] *= alpha_scaling
        outputs[4] *= beta_scaling


        self._coefficients = outputs

        return ca.Function(
            'coefficients', 
            [self.state, self.control], 
            [self._coefficients]
            )
    
    @property
    def _forces_frd(self):
        self._ensure_initialized('coefficients', 'qbar')

        forces = self._coefficients[:3] * self._qbar * self.S

        # antialign drag and velocity
        forces[0] = ca.sign(self._v_frd_rel[0])*forces[0]

        # forces += self._thrust

        speed_threshold = 80.0  # m/s
        penalty_factor = 10.0  # Scale factor for additional drag

        # Smooth penalty function for increased drag
        excess_speed = self._airspeed - speed_threshold
        additional_drag = penalty_factor * ca.fmax(0, excess_speed)**2  # Quadratic penalty for smoothness

        # Apply the additional drag along the x-axis (forward direction)
        forces[0] -= additional_drag
        return forces
    
    @property
    def _moments_aero_frd(self):
        self._ensure_initialized('coefficients', 'qbar')
        moments_aero = (self._coefficients[3:] * self._qbar * self.S 
                        * ca.vertcat(self.b, self.c, self.b))

        return moments_aero

class AircraftLinear(SixDOF):

    def __init__(self):
        super().__init__()

    def coefficients(self) -> ca.Function:
        """
        
        """

        self._coefficients = None
        return ca.Function('coefficients', [self.state, self.control], [self._coefficients])
    
class Quadrotor(SixDOF):
    super().__init__()

    @property
    def control(self):
        if not hasattr(self, '_control_initialized') or not self._control_initialized:
            self._thrust = ca.MX.sym('thrust', 4)

            self._control = ca.vertcat(
            self._thrust
            )
            self.num_controls = self._control.size()[0]

            self._control_initialized = True

        return self._control
    
    @property
    def _forces_frd(self):
        self._ensure_initialized('coefficients', 'qbar')

        forces = self._coefficients[:3] * self._qbar * self.S

        # antialign drag and velocity
        forces[0] = ca.sign(self._v_frd_rel[0])*forces[0]

        # forces += self._thrust

        speed_threshold = 80.0  # m/s
        penalty_factor = 10.0  # Scale factor for additional drag

        # Smooth penalty function for increased drag
        excess_speed = self._airspeed - speed_threshold
        additional_drag = penalty_factor * ca.fmax(0, excess_speed)**2  # Quadratic penalty for smoothness

        # Apply the additional drag along the x-axis (forward direction)
        forces[0] -= additional_drag
        return forces
    
    @property
    def _moments_aero_frd(self):
        self._ensure_initialized('coefficients', 'qbar')
        moments_aero = (self._coefficients[3:] * self._qbar * self.S 
                        * ca.vertcat(self.b, self.c, self.b))

        return moments_aero
    




class AircraftTrim(Aircraft):
    def __init__(
            self,
            opts:AircraftOpts
            ):
        
        super().__init__(opts)
    
    @property
    def control(self):
        if not hasattr(self, '_control_initialized') or not self._control_initialized:
            self._aileron = ca.MX.sym('aileron')
            self._elevator = ca.MX.sym('elevator')
            self._thrust = ca.MX.sym('thrust')
            self._com = ca.MX.sym('com', 3)

            self._control = ca.vertcat(
            self._aileron, 
            self._elevator,
            self._thrust,
            self._com
            )
            self.num_controls = self._control.size()[0]

            self._control_initialized = True

        return self._control

    @property
    def inertia_tensor(self):
        """
        Inertia Tensor around the Centre of Mass
        """

        i_xx = self.opts.aircraft_config.Ixx
        i_yy = self.opts.aircraft_config.Iyy
        i_xz = self.opts.aircraft_config.Ixz
        i_zz = self.opts.aircraft_config.Izz

        aero_inertia_tensor = ca.vertcat(
            ca.horzcat(i_xx, 0  , i_xz),
            ca.horzcat(0  , i_yy, 0  ),
            ca.horzcat(i_xz, 0  , i_zz)
        )

        mass = self.mass

        com = self._com
        
        x, y, z = com[0], com[1], com[2]

        com_term = ca.vertcat(
            ca.horzcat(y**2 + z**2, -x*y, -x*z),
            ca.horzcat(-y*x, x**2 + z**2, -y*z),
            ca.horzcat(-z*x, -z*y, x**2 + y**2)
        )

        inertia_tensor = aero_inertia_tensor + mass * com_term

        return inertia_tensor

if __name__ == '__main__':
    from aircraft.plotting.plotting import TrajectoryPlotter

    mode = 1
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    if mode == 0:
        model_path = Path(NETWORKPATH) / 'model-dynamics.pth'
        opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    elif mode == 1:
        poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
        opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)
    elif mode == 2:
        linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
        opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)

    perturbation = False
    
    trim_state_and_control = [0, 0, 0, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    # trim_state_and_control = [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]

    if trim_state_and_control is not None:
        state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
        control = np.zeros(aircraft.num_controls)
        control[:3] = trim_state_and_control[aircraft.num_states:-3]
        control[0] = 0
        control[1] = 0
        aircraft.com = np.array(trim_state_and_control[-3:])
    else:
        x0 = np.zeros(3)
        v0 = ca.vertcat([60, 0, 0])
        # would be helpful to have a conversion here between actual pitch, roll and yaw angles and the Quaternion q0, so we can enter the angles in a sensible way.
        q0 = Quaternion(ca.vertcat(0, 0, 0, 1))
        omega0 = np.array([0, 0, 0])
        state = ca.vertcat(x0, v0, q0, omega0)
        control = np.zeros(aircraft.num_controls)
        control[0] = +0
        control[1] = 5

    dyn = aircraft.state_update
    jacobian_elevator = ca.jacobian(aircraft.state_derivative(aircraft.state, aircraft.control), aircraft.control[1])
    jacobian_func = ca.Function('jacobian_func', [aircraft.state, aircraft.control], [jacobian_elevator])
    jacobian_elevator_val = jacobian_func(state, control)

    print("Jacobian of state derivatives w.r.t. elevator:")
    print(jacobian_elevator_val)
    dt = .1
    tf = 5
    state_list = np.zeros((aircraft.num_states, int(tf / dt)))
    t = 0
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
                    
            t += 1

    first = None
    t -=10
    def save(filepath):
        with h5py.File(filepath, "a") as h5file:
            grp = h5file.create_group('iteration_0')
            grp.create_dataset('state', data=state_list[:, :t])
            grp.create_dataset('control', data=control_list[:, :t])
    
    
    filepath = os.path.join("data", "trajectories", "simulation.h5")
    if os.path.exists(filepath):
        os.remove(filepath)
    save(filepath)

    plotter = TrajectoryPlotter(aircraft)
    plotter.plot(filepath=filepath)
    plt.show(block = True)