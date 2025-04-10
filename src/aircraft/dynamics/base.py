
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

        # Compute the exponential map using the Rodrigues' rotation formula
        half_theta = 0.5 * dt * ca.norm_fro(omega_frd_ned)
        sin_half_theta = ca.sin(half_theta)
        cos_half_theta = ca.cos(half_theta)

        # Compute the exponential map terms (quaternion)
        exp_q = ca.vertcat(sin_half_theta * omega_frd_ned, cos_half_theta)

        # Compute the updated quaternion
        q_next = Quaternion.product(exp_q, q_frd_ned.coeffs())

        return ca.Function('q_frd_ned_update', [self.state, self.control, dt], [q_next])
    
    # @property
    # def q_frd_ned_update(self):
    #     dt = self.dt_sym
    #     q_frd_ned = Quaternion(self._q_frd_ned)
    #     omega_frd_ned = self._omega_frd_ned  # Assume this is a 3D vector

    #     # Angular velocity magnitude
    #     theta = ca.norm_2(omega_frd_ned)  
    #     half_theta = 0.5 * dt * theta

    #     # Use Taylor series expansion for small angles to improve stability
    #     sin_half_theta = ca.if_else(theta > 1e-6, 
    #                                 ca.sin(half_theta), 
    #                                 half_theta)  # Approximate sin(x) ≈ x for small x
    #     cos_half_theta = ca.if_else(theta > 1e-6, 
    #                                 ca.cos(half_theta), 
    #                                 1 - (half_theta ** 2) / 2)  # Approximate cos(x) ≈ 1 - x^2/2

    #     # Normalized axis of rotation (handle division by zero safely)
    #     axis = ca.if_else(theta > 1e-6, 
    #                     omega_frd_ned / theta, 
    #                     ca.MX.zeros(3, 1))

    #     # Compute exponential map terms (quaternion)
    #     exp_q = ca.vertcat(sin_half_theta * axis, cos_half_theta)

    #     # Compute the updated quaternion and normalize it
    #     q_next = Quaternion.product(exp_q, q_frd_ned.coeffs())
    #     q_next = q_next / ca.norm_2(q_next)  # Ensure unit norm

    #     return ca.Function('q_frd_ned_update', [self.state, self.control, dt], [q_next])



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
        # state[6:10] = self.q_frd_ned_update(state, control, dt_scaled)
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

        return ca.Function(
            'state_update', 
            [self.state, self.control, dt], 
            [state]
            ) #, {'jit':True}
