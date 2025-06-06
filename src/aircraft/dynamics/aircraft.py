from __future__ import annotations


import casadi as ca
from abc import abstractmethod
from typing import Tuple, TYPE_CHECKING
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
from aircraft.utils.utils import load_model, AircraftConfiguration
from dataclasses import dataclass
from aircraft.dynamics.base import SixDOFOpts, SixDOF
from aircraft.dynamics.coefficient_models import DefaultModel

__all__ = ['AircraftTrim', 'AircraftOpts']

if TYPE_CHECKING:
    from aircraft.dynamics.aircraft import Aircraft

    
@dataclass
class AircraftOpts(SixDOFOpts):
    coeff_model_type: str = "default"         # "linear", "poly", "nn", or "default"
    coeff_model_path: Path|str = ''   # path to .csv, .pkl, .onnx, etc.
    realtime: bool = False                    # optional kwargs
    aircraft_config: AircraftConfiguration = AircraftConfiguration({})
    stall_angle_alpha: Tuple[float, float] = (float(np.deg2rad(-10)), float(np.deg2rad(10)))
    stall_angle_beta: Tuple[float, float] = (float(np.deg2rad(-10)), float(np.deg2rad(10)))
    stall_scaling: bool = False

    def __post_init__(self):
        from aircraft.dynamics.coefficient_models import COEFF_MODEL_REGISTRY
        self.mass: float = self.aircraft_config.mass

        # Prepare model factory for later
        factory = COEFF_MODEL_REGISTRY.get(self.coeff_model_type, COEFF_MODEL_REGISTRY["default"])
        self.coefficient_model = lambda aircraft: factory(self.coeff_model_path, aircraft, realtime=self.realtime)

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
        self.coefficient_model = self.opts.coefficient_model(self) if self.opts.coefficient_model else DefaultModel(self)
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
        
        assert isinstance(inputs, ca.DM | ca.MX)
        outputs = self.coefficient_model(inputs)
        
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
        # epsilon = 1e-2
        # smoothed_sign = ca.tanh(self._v_frd_rel[0] / epsilon)
        # forces[0] = smoothed_sign * forces[0]
        forces[0] *= ca.sign(self._v_frd_rel[0])
        # forces += self._thrust

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