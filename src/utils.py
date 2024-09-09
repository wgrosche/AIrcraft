import casadi as ca
import torch
import os
import sys
import numpy as np

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
    

from collections import namedtuple

Bound = namedtuple('Bound', ['lb', 'ub'])

class TrajectoryConfiguration:
    class State:
        def __init__(
                self,
                params
                ):
            self.aileron = Bound(np.array(params.get("aileron_limit", 
                                np.array([-5, 5])))[0],
                                np.array(params.get("aileron_limit", 
                                np.array([-5, 5])))[1])
            self.ub = ub

        @property
        def aileron(self):
            , lb, ub
            return Bound(lb, ub)
        
        @property
        def elevator(self):

            , lb, ub
            return Bound(lb, ub)

        @property
        def airspeed(self):
            , lb, ub
            return Bound(lb, ub)
        
    class Control:
        def __init__(self, lb, ub):
            self.lb = lb
            self.ub = ub

    class Aircraft:
        def __init__(self):
            pass

    
    def __init__(self, params:dict = {}):
        
        self.aileron = np.array(params.get("aileron_limit", 
                                np.array([-5, 5])))
        self.elevator = np.array(params.get("elevator_limit", 
                                np.array([-5, 5])))
        self.rudder = np.array(params.get("rudder_limit", 
                                np.array([-5, 5])))
        self.throttle = np.array(params.get("throttle_limit", 
                                np.array([[0,0,0], [0,0,0]])))
        self.centre_of_mass = np.array(params.get("centre_of_mass_limit", 
                                np.array([[-0.5, -0.1, -0.1], [0.5, 0.1, 0.1]])))

        self.alpha = np.array(params.get("aileron", 
                                         np.array([-np.deg2rad(15), 
                                                   np.deg2rad(15)])))
        self.beta = np.array(params.get("aileron", 
                                        np.array([-np.deg2rad(15), 
                                                  np.deg2rad(15)])))
        self.airspeed = np.array(params.get("aileron", 
                                            np.array([30, 100])))
        
    
        
    @property
    def control(self):
        lb = np.concatenate([
                self.aileron[0],
                self.elevator[0],
                self.rudder[0],
                self.throttle[0],
                self.centre_of_mass[0]])
        
        ub = np.concatenate([
                self.aileron[1],
                self.elevator[1],
                self.rudder[1],
                self.throttle[1],
                self.centre_of_mass[1]])

        return Control(lb, ub)
    
    @property
    def state(self):


    def lb(self):
        return np.concatenate([
            self.aileron[0],
            self.elevator[0],
            self.rudder[0],
            self.throttle[0],
            self.centre_of_mass[0],
            self.alpha[0],
            self.beta[0],
            self.airspeed[0]
        ])
    
    def ub(self):
        return np.concatenate([
            self.aileron[1],
            self.elevator[1],
            self.rudder[1],
            self.throttle[1],
            self.centre_of_mass[1],
            self.alpha[1],
            self.beta[1],
            self.airspeed[1]
        ])
    
    def constrain(self, opti:ca.Opti, state:ca.MX, control:ca.MX):
        pass
    def characteristics(self):
        return {}

    def __repr__(self):
        pass