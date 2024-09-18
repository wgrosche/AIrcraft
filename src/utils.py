import casadi as ca
import torch
import os
import sys
import numpy as np
from collections import namedtuple
import json

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
sys.path.append(BASEPATH)

from src.models import ScaledModel
# from src.dynamics import Aircraft

NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')

def load_model(
        filepath:str = os.path.join(NETWORKPATH,'model-dynamics.pth'), 
        device:torch.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                           else ("mps" if torch.backends.mps.is_available() 
                                                 else "cpu"))
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
    def __init__(
            self, 
            orientation:np.ndarray = np.array([0, 0, 0, 1]), 
            position:np.ndarray = np.array([0, 0, 0]),
            velocity:np.ndarray = np.array([50, 0, 0]), 
            angular_velocity:np.ndarray = np.array([0, 0, 0])):
        
        self.orientation = np.array(orientation)
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.angular_velocity = np.array(angular_velocity)

    def __call__(self):
        return np.concatenate([self.orientation, 
                               self.position, 
                               self.velocity, 
                               self.angular_velocity])
    

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

class TrajectoryConfiguration:
    """ 
    """
    class StateEnvelope:
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
        
    class ControlEnvelope:
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

    class Aircraft:
        def __init__(self, aircraft_dict:dict):
            self.mass = aircraft_dict.get('mass', 1.0)
            self.span = aircraft_dict.get('span', 1.0)
            self.chord = aircraft_dict.get('chord', 1.0)
            self.reference_area = aircraft_dict.get('reference_area', 0.238)
            self.aero_centre_offset = aircraft_dict.get('aero_centre_offset', [0.133, 0, 0.003])
            self.Ixx = aircraft_dict.get('Ixx', 0.155)
            self.Iyy = aircraft_dict.get('Iyy', 0.114)
            self.Izz = aircraft_dict.get('Izz', 0.262)
            self.Ixz = aircraft_dict.get('Ixz', 0.01) 


    class Waypoints:
        def __init__(self, waypoint_dict:dict):
            self.waypoints = np.array(waypoint_dict.get("waypoints", 
                                    np.array([[0,0,0], [0,0,0], [0,0,0]]))).T
            if waypoint_dict.get('initial_state') is not None:
                self.initial_state = np.array(waypoint_dict.get('initial_state'))
                self.initial_position = np.array(self.initial_state[4:7])
                self.waypoints = np.insert(self.waypoints, 0, self.initial_position, axis=1)
            else:
                self.initial_position = self.waypoints[0, :]

            self.final_position = self.waypoints[-1, :]
            self.default_velocity = waypoint_dict.get("default_velocity", 50)
            self.objective_dimension = self.final_position.shape[0]

        def __call__(self):
            return self.waypoints

    
    def __init__(self, trajectory_dict:dict):
        """ Example Trajectory Dict:
         {
         "waypoints": {
            "waypoints" : [[10, 0, 0], [50, 0 , 0], [100, 0 , 0]],
            "initial_state" : [1, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0]
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
        waypoint_dict = trajectory_dict['waypoints']
        aircraft_dict = trajectory_dict['aircraft']
        state_dict = trajectory_dict['state']
        control_dict = trajectory_dict['control']

        self._state = self.StateEnvelope(state_dict)
        self._control = self.ControlEnvelope(control_dict)
        self._waypoints = self.Waypoints(waypoint_dict)
        self._aircraft = self.Aircraft(aircraft_dict)
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

def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'),)
    config = TrajectoryConfiguration(traj_dict)

    print(config)

if __name__ == "__main__":
    main()