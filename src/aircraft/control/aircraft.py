"""
Waypoint constraint formulation for moving horizon mpc:

While a waypoint is not reached all waypoints other than the next and the one after that aren't considered.

Minimising time to next waypoint while maintaining waypoint constraint on current waypoint.

Once waypoint constraint is satisfied soften constraint on new current waypoint and introduce hard constraint on new next waypoint.

Switching will be non-differentiable if naively implemented.

How to handle case where all or one of the next 2 waypoints are out of horizon?

Minimise distances instead of imposing final state constraint.

Formulating the moving horizon mpc:

Num control nodes = 10
Max dt (don't know if this should be flexible) = 0.25s (human reaction time for realistic control input)
"""

import casadi as ca
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass


from aircraft.dynamics.aircraft import Aircraft
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from aircraft.utils.utils import TrajectoryConfiguration, load_model
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
from scipy.interpolate import CubicSpline
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData
from tqdm import tqdm
from typing import Type, List
import logging
import os
from datetime import datetime
from pathlib import Path
from aircraft.dynamics.aircraft import AircraftOpts
from aircraft.control.initialisation import cumulative_distances
from abc import abstractmethod
import time
from aircraft.config import default_solver_options, BASEPATH, NETWORKPATH, DATAPATH, DEVICE, rng
from aircraft.control.base import ControlNode, ControlProblem

plt.ion()

@dataclass
class WaypointControl(ControlProblem):
    def __init__(self, *, trajectory: TrajectoryConfiguration, **kwargs):
        super().__init__(**kwargs)
        self.current_waypoint_param = self.opti.parameter(3)  # x, y, z
        self.next_waypoint_param = self.opti.parameter(3)     # x, y, z
        # Initialize waypoints
        self.waypoints = trajectory.waypoints or []


        

    def _setup_objective(self, nodes):
        """
        Set up the objective function using waypoint parameters
        """
        super()._setup_objective(nodes)
        
        # Primary objective: minimize distance to next waypoint
        next_waypoint_diff = nodes[-1].state[:3] - self.next_waypoint_param
        next_waypoint_dist_sq = ca.dot(next_waypoint_diff, next_waypoint_diff)
        
        # Secondary objective: stay close to current waypoint
        current_waypoint_diff = nodes[0].state[:3] - self.current_waypoint_param
        current_waypoint_dist_sq = ca.dot(current_waypoint_diff, current_waypoint_diff)
        
        # Combine objectives with appropriate weights
        self.opti.minimize(1000 * next_waypoint_dist_sq + 100 * current_waypoint_dist_sq)
        
        # Add control smoothness objectives
        lambda_rate = 100.0
        lambda_smooth = 50.0
        rate_penalty = ca.sumsqr(self.control[:2, 1:] - self.control[:2, :-1])
        smoothness_penalty = ca.sumsqr(self.control[:2, 2:] - 2 * self.control[:2, 1:-1] + self.control[:2, :-2])
        self.opti.minimize(lambda_rate * rate_penalty + lambda_smooth * smoothness_penalty)

    def setup(self, guess, initial_state=None, current_waypoint_idx=0):
        """Set up the waypoints"""
        super().setup(guess)
        
        # Set waypoint parameters
        if self.waypoints and current_waypoint_idx < len(self.waypoints):
            self.current_waypoint_idx = current_waypoint_idx
            self.opti.set_value(self.current_waypoint_param, self.waypoints[current_waypoint_idx].position)
            
            next_idx = min(current_waypoint_idx + 1, len(self.waypoints) - 1)
            self.opti.set_value(self.next_waypoint_param, self.waypoints[next_idx].position)

    def update_parameters(self, initial_state, current_waypoint_idx=None):
        """Update parameters between MPC iterations"""
        self.opti.set_value(self.initial_state_param, initial_state)
        
        if current_waypoint_idx is not None:
            self.current_waypoint_idx = current_waypoint_idx
            
        # Update waypoint parameters
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            self.opti.set_value(self.current_waypoint_param, self.waypoints[self.current_waypoint_idx].position)
            
            next_idx = min(self.current_waypoint_idx + 1, len(self.waypoints) - 1)
            self.opti.set_value(self.next_waypoint_param, self.waypoints[next_idx].position)

    def check_waypoint_reached(self, state_list):
        """
        Check if any state in the list has reached the current waypoint
        
        Args:
            state_list: List of states to check
            
        Returns:
            bool: True if waypoint is reached, False otherwise
        """
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            return False
            
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        for state in state_list:
            position = state[:3]
            if current_waypoint.is_reached(position):
                return True
                
        return False

    def advance_waypoint(self):
        """
        Advance to the next waypoint if available
        
        Returns:
            bool: True if advanced to a new waypoint, False if at the last waypoint
        """
        if self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1
            return True
        return False

class AircraftControl(ControlProblem):
    """
    Class that implements constraints upon state and control for and aircraft
    """

    def __init__(self, *, aircraft: Aircraft, **kwargs):
        dynamics = aircraft.state_update
        self.aircraft = aircraft
        self.plotter = TrajectoryPlotter(aircraft)

        self.current_waypoint_idx = 0
        self.control_limits = kwargs.get('control_limits', {"aileron": [-3, 3], "elevator": [-3, 3], "rudder": [-3, 3]})
        super().__init__(dynamics=dynamics, **kwargs)
        

        
    def control_constraint(self, node: ControlNode):
        super().control_constraint(node)
        self.constraint(
            self.opti.bounded(self.control_limits["aileron"][0], node.control[0], self.control_limits["aileron"][1]), 
            description="Aileron Constraint")
        self.constraint(
            self.opti.bounded(self.control_limits["elevator"][0], node.control[1], self.control_limits["elevator"][1]), 
            description="Elevator Constraint")
        self.constraint(
            self.opti.bounded(self.control_limits["rudder"][0], node.control[2], self.control_limits["rudder"][1]), 
            description="Rudder Constraint")
        

    def state_constraint(self, node: ControlNode, next: ControlNode, dt: ca.MX):
        super().state_constraint(node, next, dt)
        v_rel = self.aircraft.v_frd_rel(node.state, node.control)
        self.constraint(
            self.opti.bounded(20**2, ca.dot(v_rel, v_rel), 80**2), 
            description="Speed constraint")
        self.constraint(
            self.opti.bounded(-np.deg2rad(90), self.aircraft.phi(node.state), np.deg2rad(90)), 
            description="Roll constraint")
        self.constraint(
            self.opti.bounded(-np.deg2rad(10), self.aircraft.beta(node.state, node.control), np.deg2rad(10)), 
            description="Sideslip constraint")
        self.constraint(
            self.opti.bounded(-np.deg2rad(20), self.aircraft.alpha(node.state, node.control), np.deg2rad(20)), 
            description="Attack constraint")
        self.constraint(node.state[2] < 0, description="Height constraint")




    def log(self, iteration):
        super().log(iteration)
        aircraft = self.aircraft
        f = aircraft.state_update(aircraft.state, aircraft.control, aircraft.dt_sym)
        
        # Compute the Jacobian of f w.r.t state
        jacobian = ca.jacobian(f, aircraft.state)
        
        # Create a CasADi function for numerical evaluation
        jacobian_func = ca.Function('J', [aircraft.state, aircraft.control, aircraft.dt_sym], [jacobian])
        
        condition_numbers = np.linalg.cond(jacobian_func(self.opti.debug.value(self.state)[:, 1:],
                                           self.opti.debug.value(self.control),
                                           self.opti.debug.value(self.time)/self.num_nodes))
        
        self.logger.info(f"Condition numbers: {condition_numbers}")
        # Get constraint values and dual variables
        g = self.opti.debug.value(self.opti.g)
        lam_g = self.opti.debug.value(self.opti.lam_g)
        
        # Check which constraints are active (close to bounds)
        tolerance = 1e-6
        active_constraints = np.nonzero(abs(g) > tolerance)[0]
        # Check dynamics violations
        dynamics_violations = []
        for i in range(self.num_nodes):
            current_state = self.opti.debug.value(self.state)[:, i]
            current_control = self.opti.debug.value(self.control)[:, i]
            dt = self.opti.debug.value(self.time)/self.num_nodes
            
            predicted_next = self.dynamics(current_state, current_control, dt)
            actual_next = self.opti.debug.value(self.state)[:, i+1]
            
            dynamics_violation = np.linalg.norm(predicted_next - actual_next)
            if dynamics_violation > 1e-3:
                dynamics_violations.append((i, dynamics_violation))
                self.logger.warning(f"Large dynamics violation at node {i}: {dynamics_violation}")
        
        if not dynamics_violations:
            self.logger.info("No significant dynamics violations detected")
        
        # Log active constraints
        if len(active_constraints) > 0:
            self.logger.info(f"Active constraints: {len(active_constraints)} constraints")
            self.logger.info(f"Constraint values: {g[active_constraints]}")
            self.logger.info(f"Dual variables: {lam_g[active_constraints]}")
        else:
            self.logger.info("No active constraints")
        
        # Log control limits
        control_values = self.opti.debug.value(self.control)
        aileron_limit = abs(control_values[0]) >= 5.0
        elevator_limit = abs(control_values[1]) >= 5.0
        self.logger.info(f"Control limits active - Aileron: {aileron_limit}, Elevator: {elevator_limit}")
        
        state = self.opti.debug.value(self.state)[:, :-1]
        control = self.opti.debug.value(self.control)
        
        airspeed_values = self.aircraft.airspeed(state, control)
        alpha_values = self.aircraft.alpha(state, control)
        beta_values = self.aircraft.beta(state, control)
        
        self.logger.info("State constraints:")
        self.logger.info(f"Airspeed values: {airspeed_values}")
        self.logger.info(f"Alpha values: {alpha_values}")
        self.logger.info(f"Beta values: {beta_values}")
    
    # def callback(self, iteration: int):
    #     super().callback(iteration)
    #     if self.plotter and iteration % 10 == 5:
    #         trajectory_data = TrajectoryData(
    #             state=np.array(self.opti.debug.value(self.state))[:, 1:],
    #             control=np.array(self.opti.debug.value(self.control)),
    #             time=np.array(self.opti.debug.value(self.time))
    #         )
    #         self.plotter.plot(trajectory_data=trajectory_data)
    #         plt.draw()
    #         self.plotter.figure.canvas.start_event_loop(0.0002)

    def solve(self, warm_start: Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start=warm_start)
        trajectory_data = TrajectoryData(
                state=np.array(self.opti.debug.value(self.state))[:, 1:],
                control=np.array(self.opti.debug.value(self.control)),
                time=np.array(self.opti.debug.value(self.time))
            )
        self.plotter.plot(trajectory_data=trajectory_data)
        plt.draw()
        self.plotter.figure.canvas.start_event_loop(0.0002)
        return sol