

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
from aircraft.dynamics.base import SixDOFOpts, SixDOF


print(DEVICE)



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
    def __init__(self):
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
    


