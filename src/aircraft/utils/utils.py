import casadi as ca
import torch
import os
import sys
import numpy as np
from collections import namedtuple
import json
from aircraft.config import NETWORKPATH, DEVICE, rng
from aircraft.surrogates.models import ScaledModel
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from typing import Union, Optional
from liecasadi import Quaternion

def load_model(
        filepath:Union[str, Path] = os.path.join(NETWORKPATH,'model-dynamics.pth'), 
        device:Optional[torch.device] = None
        ) -> ScaledModel:
    
    if device == None:
        device = DEVICE
            
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)

    scaler = (checkpoint['input_mean'], 
              checkpoint['input_std'], 
              checkpoint['output_mean'], 
              checkpoint['output_std'])
    
    model = ScaledModel(5, 6, scaler=scaler)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def u_from_aero(q_bar, alpha, beta):
    u_squared =((1 - ca.sin(beta)**2) * 2 * q_bar) / ((1 + ca.tan(alpha)**2) * 1.225)

    return ca.sqrt(u_squared)

def v_from_aero(q_bar, alpha, beta):
    v = ca.sin(beta) * ca.sqrt(q_bar * 2 / 1.225)

    return v

def w_from_aero(q_bar, alpha, beta):

    return u_from_aero(q_bar, alpha, beta) * ca.tan(alpha)

def aero_to_state(q_bar, alpha, beta):
    default_state = ca.vertcat(
        ca.DM([0., 0., 0., 1.]),  # Quaternion (orientation)
        ca.DM([0., 0., 0.]),      # Position
        ca.vertcat(               # Velocity
            u_from_aero(q_bar, alpha, beta),
            v_from_aero(q_bar, alpha, beta),
            w_from_aero(q_bar, alpha, beta)
        ),
        ca.DM([0., 0., 0.])       # Angular velocity
    )

    return default_state


class State:
    def __init__(self,
                 position=np.array([0, 0, 0]),
                 velocity=np.array([50, 0, 0]),
                 orientation=np.array([0, 0, 0, 1]),
                 angular_velocity=np.array([0, 0, 0])):
        
        types = {type(position), type(velocity), type(orientation), type(angular_velocity)}
        assert len(types) == 1, "All components must be of the same type"
        self.backend = types.pop()

        self.position = self.backend(position)
        self.velocity = self.backend(velocity)
        self.orientation = self.backend(orientation)
        self.angular_velocity = self.backend(angular_velocity)

    def __call__(self):
        return self.backend.concatenate([
            self.orientation,
            self.position,
            self.velocity,
            self.angular_velocity
        ])

    def to_numpy(self):
        if self.backend == np.ndarray:
            return self
        else:
            return State(
                position=np.array(self.position),
                velocity=np.array(self.velocity),
                orientation=np.array(self.orientation),
                angular_velocity=np.array(self.angular_velocity)
            )

    def to_casadi(self, symbolic=False):
        target_type = ca.MX if symbolic else ca.DM
        if self.backend in [ca.MX, ca.DM] and isinstance(self.position, target_type):
            return self
        else:
            return State(
                position=target_type(self.position),
                velocity=target_type(self.velocity),
                orientation=target_type(self.orientation),
                angular_velocity=target_type(self.angular_velocity)
            )

    def as_vector(self):
        return self.__call__()

    def __repr__(self):
        return f"State(pos={self.position}, vel={self.velocity}, ori={self.orientation}, omega={self.angular_velocity})"
    

class Control:
    def __init__(
            self,
            aileron:np.ndarray = np.array([0]),
            elevator:np.ndarray = np.array([0]),
            rudder:np.ndarray = np.array([0]),
            throttle:np.ndarray = np.array([0, 0, 0]),
            v_wind_ecf_e:np.ndarray = np.array([0, 0, 0]),
            omega_e_i_ecf:np.ndarray = np.array([0, 0, 0]),
            centre_of_mass:np.ndarray = np.array([0, 0, 0])):
        
        self.aileron = np.array(aileron)
        self.elevator = np.array(elevator)
        self.rudder = np.array(rudder)
        self.throttle = np.array(throttle)
        self.v_wind_ecf_e = np.array(v_wind_ecf_e)
        self.omega_e_i_ecf = np.array(omega_e_i_ecf)
        self.centre_of_mass = np.array(centre_of_mass)

    def __call__(self):
        return np.concatenate([self.aileron, 
                               self.elevator, 
                               self.rudder, 
                               self.throttle,
                               self.v_wind_ecf_e,
                               self.omega_e_i_ecf,
                               self.centre_of_mass])
    


Bound = namedtuple('Bound', ['lb', 'ub'])


class StateEnvelopeConfiguration:
    def __init__(self, state_dict:dict):
        self.alpha = Bound(np.array(state_dict.get("alpha", 
                            np.array([-15, 15])))[0],
                            np.array(state_dict.get("alpha", 
                            np.array([-15, 15])))[1])
        self.beta = Bound(np.array(state_dict.get("beta", 
                            np.array([-15, 15])))[0],
                            np.array(state_dict.get("beta", 
                            np.array([-15, 15])))[1])
        
        self.airspeed = Bound(np.array(state_dict.get("airspeed", 
                                        np.array([30, 100])))[0],
                            np.array(state_dict.get("airspeed", 
                                        np.array([30, 100])))[1])
    
class ControlEnvelopeConfiguration:
    def __init__(self, control_dict:dict):
        self.aileron = np.array(control_dict.get("aileron_limit", 
                            np.array([-5, 5])))
        self.elevator = np.array(control_dict.get("elevator_limit", 
                                np.array([-5, 5])))
        self.rudder = np.array(control_dict.get("rudder_limit", 
                                np.array([-5, 5])))
        self.throttle = np.array(control_dict.get("throttle_limit", 
                                np.array([[0,0,0], [0,0,0]])))
        self.centre_of_mass = np.array(control_dict.get("centre_of_mass_limit", 
                                np.array([[-0.5, -0.1, -0.1], [0.5, 0.1, 0.1]])))
        self.lb = np.array([
            self.aileron[0],
            self.elevator[0],
            self.rudder[0],
            *self.throttle[0],
            *self.centre_of_mass[0]])
        
        self.ub = np.array([
            self.aileron[1],
            self.elevator[1],
            self.rudder[1],
            *self.throttle[1],
            *self.centre_of_mass[1]])

class AircraftConfiguration:
    def __init__(self, aircraft_dict:dict):
        self.mass = aircraft_dict.get('mass', 1.0)
        self.span = aircraft_dict.get('span', 1.0)
        self.length = aircraft_dict.get('length', 1.2)
        self.chord = aircraft_dict.get('chord', 1.0)
        self.reference_area = aircraft_dict.get('reference_area', 0.238)
        self.aero_centre_offset = aircraft_dict.get('aero_centre_offset', [0.133, 0, 0.003]) # position of aerodynamic centre relative to the centre of mass
        self.Ixx = aircraft_dict.get('Ixx', 0.155)
        self.Iyy = aircraft_dict.get('Iyy', 0.114)
        self.Izz = aircraft_dict.get('Izz', 0.262)
        self.Ixz = aircraft_dict.get('Ixz', 0.01) 
        self.r_min = aircraft_dict.get('r_min', 10.0)
        self.glide_ratio = aircraft_dict.get('glide_ratio', 10.0)
        self.rudder_moment_arm = aircraft_dict.get('rudder_moment_arm', 0.5) # distance between centre of mass and the tail of the plane (used for damping calculations)
        

        


class WaypointsConfiguration:
    def __init__(self, waypoint_dict:dict):
        self.waypoints = np.array(waypoint_dict.get("waypoints", 
                                np.array([[0,0,0], [0,0,0], [0,0,0]])))
        
        self.waypoint_indices = (waypoint_dict.get("waypoint_indices", 
                                [0,1,2]))
        if waypoint_dict.get('initial_state') is not None:
            self.initial_state = np.array(waypoint_dict.get('initial_state'))
            self.initial_position = np.array(self.initial_state[:3])
            self.waypoints = np.insert(self.waypoints, 0, self.initial_position, axis=0)
        else:
            self.initial_position = self.waypoints[0, :]

        self.final_position = self.waypoints[-1, :]
        self.default_velocity = waypoint_dict.get("default_velocity", 50)
        self.objective_dimension = self.final_position.shape[0]
        self.tolerance = waypoint_dict.get("waypoint_tolerance", 
                                1e-2)

    def __call__(self):
        return self.waypoints

class TrajectoryConfiguration:
    """ 
    """
    def __init__(self, trajectory_dict:dict|str|Path):
        """ Example Trajectory Dict:
         {
         "waypoints": {
            "waypoints" : [[10, 0, 0], [50, 0 , 0], [100, 0 , 0]],
            "initial_state" : [0, 0, 0, 50, 0, 0, 0, 0, 0, 1, 0, 0, 0]
         },
         "aircraft" : {
            "mass"  : 2.0,
            "span"  : 2.0,
            "chord" : 0.124605,
            "reference_area": 0.238,
            "aero_centre_offset":[0.133, 0, 0.003],
            "Ixx": 0.155,
            "Iyy": 0.114,
            "Izz": 0.262,
            "Ixz": 0.01

         },
         "state"    : {
            "alpha" : [-15, 15],
            "beta"  : [-15, 15],
            "airspeed: [30, 100]
         },
         "control"  : {
            "aileron_limit" : [-5, 5],
            "elevator_limit" : [-5, 5],
            "rudder_limit" : [-5, 5],
            "throttle_limit" : [[0, 0, 0], [0, 0, 0]],
            "centre_of_mass_limit" : [[-0.5, -0.1, -0.1], [0.5, 0.1, 0.1],
         }
         } 
         """
        if isinstance(trajectory_dict, str) or isinstance(trajectory_dict, Path):
            trajectory_dict = json.load(open(trajectory_dict, 'r'))
        assert isinstance(trajectory_dict, dict)
        waypoint_dict = trajectory_dict['waypoints']
        aircraft_dict = trajectory_dict['aircraft']
        state_dict = trajectory_dict['state']
        control_dict = trajectory_dict['control']

        self._state = StateEnvelopeConfiguration(state_dict)
        self._control = ControlEnvelopeConfiguration(control_dict)
        self._waypoints = WaypointsConfiguration(waypoint_dict)
        self._aircraft = AircraftConfiguration(aircraft_dict)
        self.trajectory_dict = trajectory_dict
        
    @property
    def control(self):
        return self._control
    
    @property
    def state(self):
        return self._state

    @property
    def waypoints(self):
        return self._waypoints
    
    @property
    def aircraft(self):
        return self._aircraft
    
    def __repr__(self):
        return str(self.trajectory_dict)
    
def perturb_quaternion(q:Quaternion, delta_theta=0.01):
    """ Perturbs a quaternion by a small rotation. """
    # Generate a small random rotation axis
    quat_coeffs = q.coeffs()
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)  # Normalize to unit vector
    
    # Create small rotation quaternion
    delta_q = R.from_rotvec(delta_theta * axis).as_quat(canonical=True)  # [x, y, z, w]
    
    # Apply rotation (Hamilton product)
    q_perturbed = R.from_quat(quat_coeffs) * R.from_quat(delta_q)
    
    return q_perturbed.as_quat(canonical=True)  # Return perturbed quaternion

def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'),)
    config = TrajectoryConfiguration(traj_dict)

    print(config)

if __name__ == "__main__":
    main()


