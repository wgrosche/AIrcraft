

import casadi as ca
from abc import abstractmethod
from typing import Union, Tuple, Optional
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
from aircraft.utils.utils import load_model, AircraftConfiguration
from dataclasses import dataclass
from aircraft.dynamics.base import SixDOFOpts, SixDOF

@dataclass
class AircraftOpts(SixDOFOpts):
    linear_path:Optional[Path] = None
    poly_path:Optional[Path] = None
    nn_model_path:Optional[Path] = None
    aircraft_config:AircraftConfiguration = AircraftConfiguration({})
    realtime:bool = False # Work in progress implementation of faster nn 
    stall_angle_alpha:Tuple[float, float] = (float(np.deg2rad(-10)), float(np.deg2rad(10)))
    stall_angle_beta:Tuple[float, float] = (float(np.deg2rad(-10)), float(np.deg2rad(10)))
    stall_scaling:bool = True
    
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
            opts:AircraftOpts,
            **kwargs
            ):
        
        """ 
        """
        super().__init__(opts = opts, **kwargs)

        self.opts = opts
        self.grav = self.opts.gravity
        self.stall_scaling = self.opts.stall_scaling
        self.initialise_aircraft(self.opts.aircraft_config)

        self.state
        self.control

    def initialise_aircraft(self, config:AircraftConfiguration) -> None:
        self.S = config.reference_area
        self.b = config.span
        self.c = config.chord
        self.mass = config.mass
        self.com = config.aero_centre_offset
        self.rudder_moment_arm = config.rudder_moment_arm
        self.length = config.length

        i_xx = config.Ixx
        i_yy = config.Iyy
        i_xz = config.Ixz
        i_zz = config.Izz

        self.static_inertia_tensor = ca.vertcat(
                                        ca.horzcat(i_xx, 0  , i_xz),
                                        ca.horzcat(0  , i_yy, 0  ),
                                        ca.horzcat(i_xz, 0  , i_zz)
                                    )
        
    @property
    def control(self) -> ca.MX:
        if not hasattr(self, '_control_initialized') or not self._control_initialized:
            self._aileron = ca.MX.sym('aileron')  # type: ignore[arg-type]
            self._elevator = ca.MX.sym('elevator')  # type: ignore[arg-type]
            self._rudder = ca.MX.sym('rudder')  # type: ignore[arg-type]
            self._thrust = ca.MX.sym('thrust', 3)  # type: ignore[arg-type]

            self._control = ca.vertcat(
            self._aileron, 
            self._elevator,
            self._rudder,
            self._thrust
            )
            

            self._control_initialized = True
            self.num_controls:int = self._control.size()[0]
        assert isinstance(self.num_controls, int)
        assert isinstance(self._control, ca.MX)

        return self._control

    @property
    def inertia_tensor(self) -> ca.MX:
        """
        Inertia Tensor around the Centre of Mass
        """
        mass = self.mass

        com = self.com
        
        x, y, z = com[0], com[1], com[2]

        com_term = ca.vertcat(
            ca.horzcat(y**2 + z**2, -x*y, -x*z),
            ca.horzcat(-y*x, x**2 + z**2, -y*z),
            ca.horzcat(-z*x, -z*y, x**2 + y**2)
        )

        inertia_tensor = self.static_inertia_tensor + mass * com_term

        return inertia_tensor

    @property
    def elevator_alpha(self) -> ca.Function:
        """
        Includes changed angle of attack due to pitch rate
        """
        self._ensure_initialized('v_frd_rel')
        u, _, w = ca.vertsplit(self._v_frd_rel)
        _, q, _ = ca.vertsplit(self._omega_frd_ned)

        self._elevator_alpha = ca.atan2(w + self.rudder_moment_arm * q, u + self.epsilon)
        return ca.Function('elevator_alpha', [self.state, self.control], [self._elevator_alpha])
    

    @property
    def left_wing_alpha(self) -> ca.Function:
        """
        Includes changed angle of attack due to roll rate
        """
        self._ensure_initialized('_v_frd_rel')
        u, _, w = ca.vertsplit(self._v_frd_rel)
        p, _, _ = ca.vertsplit(self._omega_frd_ned)
        self._left_wing_alpha = ca.atan2(w - self.b * p / 4, u + self.epsilon)
        return ca.Function('left_wing_alpha', [self.state, self.control], [self._left_wing_alpha])
    
    @property
    def right_wing_alpha(self) -> ca.Function:
        """
        Includes changed angle of attack due to roll rate
        """
        self._ensure_initialized('v_frd_rel')
        u, _, w = ca.vertsplit(self._v_frd_rel)
        p, _, _ = ca.vertsplit(self._omega_frd_ned)
        self._right_wing_alpha = ca.atan2(w + self.b * p / 4, u + self.epsilon)
        return ca.Function('right_wing_alpha', [self.state, self.control], [self._right_wing_alpha])
    
    @property
    def rudder_beta(self) -> ca.Function:
        self._ensure_initialized('v_frd_rel', 'airspeed')

        v_frd_rel = self._v_frd_rel
        v_frd_rel[1] = v_frd_rel[1] - self.rudder_moment_arm * self._omega_frd_ned[2]

        airspeed = ca.sqrt(ca.sumsqr(v_frd_rel) + self.epsilon)
        self._rudder_beta = ca.asin(v_frd_rel[1] / airspeed)
        return ca.Function('rudder_beta', [self.state, self.control], [self._rudder_beta])
    
    @property
    def left_wing_qbar(self) -> ca.Function:
        self._ensure_initialized('v_frd_rel')
        
        r = self._omega_frd_ned[2]
        new_vel = self._v_frd_rel[0] 
        new_vel[0] += self.b * r / 4
        self._left_wing_qbar = 0.5 * 1.225 * ca.dot(new_vel, new_vel)
        return ca.Function('left_wing_qbar', [self.state, self.control], [self._left_wing_qbar])
    
    @property
    def right_wing_qbar(self) -> ca.Function:
        self._ensure_initialized('v_frd_rel')

        r = self._omega_frd_ned[2]
        new_vel = self._v_frd_rel[0] 
        new_vel[0] += self.b * r / 4
        self._right_wing_qbar = 0.5 * 1.225 * ca.dot(new_vel, new_vel)
        return ca.Function('right_wing_qbar', [self.state, self.control], [self._right_wing_qbar])
    

        

    
    @abstractmethod
    def model_outputs(self, inputs:ca.DM) -> ca.MX:
        """
        Generates the forward pass of the aerodynamics surrogate ready for post-processing by coefficients
        """
        ...
    

    
    @property
    def coefficients(self) -> ca.Function:
        """
        Forward pass of the model to retrieve aerodynamic coefficients.

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
        

        outputs = self.model_outputs(inputs)
        
        # Simplified rudder model
        Cn_rudder = -0.1
        outputs[5] += Cn_rudder * 6 * self._rudder * np.pi / 180

        if self.stall_scaling:
            # stall scaling
            stall_angle_alpha = np.deg2rad(10)
            stall_angle_beta = np.deg2rad(10)

            steepness = 10

            alpha_scaling = 1 / (1 + ca.exp(steepness * (ca.fabs(self._alpha) - stall_angle_alpha)))
            beta_scaling = 1 / (1 + ca.exp(steepness * (ca.fabs(self._beta) - stall_angle_beta)))
            
            outputs[2] *= alpha_scaling
            outputs[2] *= beta_scaling

            outputs[4] *= alpha_scaling
            # outputs[4] *= beta_scaling


        self._coefficients = outputs

        return ca.Function(
            'coefficients', 
            [self.state, self.control], 
            [self._coefficients]
            )
    

    
    @property
    def _forces_frd(self) -> ca.MX:
        self._ensure_initialized('coefficients', 'qbar')

        forces = self._coefficients[:3] * self._qbar * self.S

        # antialign drag and velocity with smoothed sign flip
        epsilon = 1e-2
        smoothed_sign = ca.tanh(self._v_frd_rel[0] / epsilon)
        forces[0] = smoothed_sign * forces[0]

        forces += self._thrust

        return forces
    
    @property
    def _moments_aero_frd(self) -> ca.MX:
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
    def control(self) -> ca.MX:
        if not hasattr(self, '_com_initialized') or not self._com_initialized:
            self._control = super().control
            self._com = ca.MX.sym('com', 3) # type: ignore[arg-type]
            self._control = ca.vertcat(self._control, self._com)
            self._com_initialized = True
            self.num_controls:int = self._control.size()[0]

        assert isinstance(self.num_controls, int)
        assert isinstance(self._control, ca.MX)

        return self._control

    @property
    def inertia_tensor(self):
        """
        Inertia Tensor around the Centre of Mass
        """

        mass = self.mass

        com = self._com
        
        x, y, z = com[0], com[1], com[2]

        com_term = ca.vertcat(
            ca.horzcat(y**2 + z**2, -x*y, -x*z),
            ca.horzcat(-y*x, x**2 + z**2, -y*z),
            ca.horzcat(-z*x, -z*y, x**2 + y**2)
        )

        inertia_tensor = self.static_inertia_tensor + mass * com_term

        return inertia_tensor
    


class LinearAircraft(Aircraft):
    def __init__(
            self,
            *,
            linear_path:Optional[Union[str, Path]] = None,
            **kwargs
            ):
        
        super().__init__(**kwargs)
        self.fitted = False
        if linear_path is not None:
            self.linear_coeffs = ca.DM(np.array(pd.read_csv(linear_path)))
            self.fitted = True
        self.fitted_models = None

    def model_outputs(self, inputs: ca.DM) -> ca.MX:
        if self.fitted:
            return ca.MX(ca.mtimes(self.linear_coeffs, ca.vertcat(inputs, 1)))

        _, alpha, beta, aileron, elevator = ca.vertsplit(inputs)
        
        p, q, r = ca.vertsplit(self._omega_frd_ned)
        
        # Constants for control surface effectiveness
        CD0 = 0.02
        CD_alpha = 0.3

        CL0 = 0.0
        CL_alpha = 5.0  # lift per rad

        CY_beta = -0.98

        Cl_aileron = 0.08
        Cl_p = -0.05  # roll damping

        Cm_elevator = -1.2
        Cm_q = -.5  # pitch damping

        Cn_rudder = -0.1
        Cn_r = -0.05  # yaw damping

        # Core coefficient calculations
        CD = CD0 + CD_alpha * alpha**2
        CL = CL0 + CL_alpha * alpha
        CY = CY_beta * beta

        Cl = Cl_aileron * 4 * aileron  * np.pi / 180+ Cl_p * p
        Cm = Cm_elevator * 5 * elevator  * np.pi / 180 + Cm_q * q
        Cn = Cn_rudder * 6 * self._rudder * np.pi / 180 + Cn_r * r

        return ca.MX(ca.vertcat(-CD, CY, -CL, Cl, Cm, Cn))

    @property
    def coefficients(self) -> ca.Function:
        """
        Returns simplified, hardcoded aerodynamic coefficients for testing.
        Includes basic damping from angular rates (p, q, r).

        Outputs: [CD, CY, CL, Cl, Cm, Cn] (drag, side force, lift, roll, pitch, yaw)
        """
        self._ensure_initialized(
            'qbar', 'alpha', 'beta'
        )

        inputs = ca.vertcat(
            self._qbar, 
            self._alpha, 
            self._beta, 
            self._aileron, 
            self._elevator
            )
        
        self._coefficients = self.model_outputs(inputs)

        return ca.Function(
            'coefficients', 
            [self.state, self.control], 
            [self._coefficients]
            )
    
class NeuralAircraft(Aircraft):
    def __init__(
            self,
            *,
            neural_path:Optional[Union[str, Path]] = None,
            realtime:bool = False,
            **kwargs
            ):
        
        super().__init__(**kwargs)

        try:
            from l4casadi import L4CasADi
            from l4casadi.realtime import RealTimeL4CasADi
        except ImportError as e:
            raise ImportError("Mode 'neural' requires `l4casadi`. Please install it to proceed.") from e

        assert isinstance(neural_path, (str, Path)), "Must supply a valid model path for mode 'neural'"

        model = load_model(filepath=neural_path)
        if realtime:
            self.model = RealTimeL4CasADi(model, approximation_order=1)
        else:
            self.model = L4CasADi(model, name='AeroModel', generate_jac_jac=True)

    def model_outputs(self, inputs: ca.DM) -> ca.MX:
        outputs = self.model(ca.reshape(inputs, 1, -1))
        assert isinstance(outputs, ca.MX), "model output not a ca.MX object"
        outputs = ca.vertcat(outputs.T)
        return ca.MX(outputs)


class PolynomialAircraft(Aircraft):
    def __init__(
            self,
            *,
            poly_path:Optional[Union[str, Path]] = None,
            realtime:bool = False,
            **kwargs
            ):
        assert isinstance(poly_path, Union[str, Path]), "must supply a valid linear path for mode 'polynomial'"
        with open(poly_path, 'rb') as file:
            self.fitted_models = pickle.load(file)
        assert isinstance(self.fitted_models, dict)
        
        super().__init__(**kwargs)


    def model_outputs(self, inputs: ca.DM) -> ca.MX:
        """
        Forward pass of the polynomial model to retrieve aerodynamic coefficients.

        To calculate damping factors the effective velocities (under the angular rotation) of the relevant lifting surfaces are calculated and passed as inputs to the model.
        """
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
        return ca.MX(outputs)