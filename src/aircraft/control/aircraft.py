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

waypoints = [...]
current_waypoint = waypoints[0]
next_waypoint = waypoints[1]

state = state_0

def check_waypoint_reached(state_list):
    check whether the waypoint condition is met for any state in the state list


while not final_waypoint_reached:
    if check_waypoint_reached(state_list):
        waypoint_index = i+1
        current_waypoint = next_waypoint
        next_waypoint = waypoints[i]
    
    opti problem with warm start?

    


TODO:

    Waypoint class
    AircraftConstraint class
    MHE class



"""


import casadi as ca
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
import os
import sys

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASEPATH)
sys.path.append(BASEPATH)

from aircraft.dynamics.aircraft import Aircraft, Quadrotor
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from aircraft.utils.utils import TrajectoryConfiguration, load_model
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
import h5py
from scipy.interpolate import CubicSpline
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData

import threading
import torch
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

class AircraftControl(ControlProblem):
    def __init__(self, aircraft:Aircraft, num_nodes:int, opts:Optional[dict] = {}, time_guess:float = None):
        dynamics = aircraft.state_update
        self.aircraft = aircraft
        self.plotter = TrajectoryPlotter(aircraft)
        self.scale_time = time_guess
        plt.show()
        super().__init__(dynamics, num_nodes, opts)


    def control_constraint(self, node:ControlNode):
        self.opti.subject_to(self.opti.bounded(-1, node.control[0], 1))
        self.opti.subject_to(self.opti.bounded(-3, node.control[1], 3))
        # self.opti.subject_to(self.opti.bounded(0, node.control[2:], 0))

    def state_constraint(self, node:ControlNode, next:ControlNode, dt:ca.MX):
        super().state_constraint(node, next, dt)
        opti = self.opti
        aircraft = self.aircraft
        beta = aircraft.beta
        alpha = aircraft.alpha
        airspeed = aircraft.airspeed
        roll = aircraft.phi
        opti.subject_to(opti.bounded(20, airspeed(node.state, node.control), 80))
        opti.subject_to(opti.bounded(-np.deg2rad(50), roll(node.state),  np.deg2rad(50)))
        opti.subject_to(opti.bounded(-np.deg2rad(10), beta(node.state, node.control),  np.deg2rad(10)))
        opti.subject_to(opti.bounded(-np.deg2rad(20), alpha(node.state, node.control), np.deg2rad(20)))
        opti.subject_to(next.state[2] < 0)

    def _setup_objective(self, nodes):
        """
        TODO: When enforcing hard constraints on final position make sure that 
        the waypoint tolerance is sufficient to guarantee a node passes within.

        If there are control nodes every 15m and the tolerance is 5m then its unlikely for there to be a node within the tolerance
        """
        super()._setup_objective(nodes)
        
        final_waypoint = [0, 20, -190]

        # tolerance = 2 * np.linalg.norm(np.array([0,0, -200]) - np.array(final_waypoint)) / (self.num_nodes - 1)# TODO: Change from hardcoding to deriving from the initial position and the goal
        final_waypoint_diff = nodes[-1].state[:3] - final_waypoint
        final_waypoint_dist_sq = ca.dot(final_waypoint_diff, final_waypoint_diff)
        # self.opti.subject_to(final_waypoint_dist_sq < tolerance ** 2)
        self.opti.minimize(1000*final_waypoint_dist_sq)

        # lambda_rate = 100.0
        # lambda_smooth = 50.0
        # rate_penalty = ca.sumsqr(self.control[:2, 1:] - self.control[:2, :-1])
        # smoothness_penalty = ca.sumsqr(self.control[:2, 2:] - 2 * self.control[:2, 1:-1] + self.control[:2, :-2])
        # self.opti.minimize(lambda_rate * rate_penalty + lambda_smooth * smoothness_penalty)

    def log(self, iteration):
        super().log()
        aircraft = self.aircraft
        f = aircraft.state_update(aircraft.state, aircraft.control, aircraft.dt_sym)
        
        # Compute the Jacobian of f w.r.t state
        J = ca.jacobian(f, aircraft.state)
        
        # Create a CasADi function for numerical evaluation
        J_func = ca.Function('J', [aircraft.state, aircraft.control, aircraft.dt_sym], [J])
        
        condition_numbers = np.linalg.cond(J_func(self.opti.debug.value(self.state)[:, 1:], 
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
        
    def callback(self, iteration: int):
        # Call the parent class callback to handle saving progress
        super().callback(iteration)

        # Plotting
        if self.plotter and iteration % 10 == 5:
            trajectory_data = TrajectoryData(
                state=np.array(self.opti.debug.value(self.state))[:, 1:],
                control=np.array(self.opti.debug.value(self.control)),
                time=np.array(self.opti.debug.value(self.time))
            )
            self.plotter.plot(trajectory_data=trajectory_data)
            plt.draw()
            self.plotter.figure.canvas.start_event_loop(0.0002)

    def solve(self, warm_start:Optional[ca.OptiSol] = None):
        super().solve(warm_start=warm_start)

        trajectory_data = TrajectoryData(
                state=np.array(self.opti.debug.value(self.state))[:, 1:],
                control=np.array(self.opti.debug.value(self.control)),
                time=np.array(self.opti.debug.value(self.time))
            )
        self.plotter.plot(trajectory_data=trajectory_data)
        plt.draw()
        self.plotter.figure.canvas.start_event_loop(0.0002)

        plt.show(block = True)

    def _initialise_state(self):
        """
        TODO: Implement this method
        """
        pass

    def _initialise_control(self):
        """
        TODO: Implement this method
        """
        pass

    

def test_aircraft_trajectory_opt():
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    
    num_nodes = 30
    time_guess = 10
    dt = time_guess / (num_nodes)

    guess =  np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))

    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
    control = np.zeros(aircraft.num_controls)
    control[:len(trim_state_and_control) - aircraft.num_states - 3] = trim_state_and_control[aircraft.num_states:-3]
    aircraft.com = np.array(trim_state_and_control[-3:])
    
    dyn = aircraft.state_update

    # Initialize trajectory with debugging prints
    for i in tqdm(range(num_nodes + 1), desc='Initialising Trajectory:'):
        guess[:aircraft.num_states, i] = state.full().flatten()
        # control = control + 1 * (rng.random(len(control)) - 0.5)
        guess[aircraft.num_states:, i] = control
        next_state = dyn(state, control, dt)
        # print(f"Node {i}: State = {state}, Control = {control}, Next State = {next_state}")
        state = next_state


    # Second loop: Validate initial guess
    guess2 = np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))
    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])

    for i in tqdm(range(num_nodes + 1), desc='Validating Trajectory:'):
        guess2[:aircraft.num_states, i] = state.full().flatten()
        control = guess[aircraft.num_states:, i]  # Use controls from guess1
        next_state = dyn(state, control, dt)
        guess2[aircraft.num_states:, i] = control
        state = next_state


    
    for i in range(num_nodes + 1):
        print(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i])
        assert np.allclose(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i]), f"Problem in node {i}"




    control_problem = AircraftControl(aircraft, num_nodes, time_guess = time_guess)    
    control_problem.setup(guess)
    control_problem.solve()
    


def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    
    num_nodes = 100
    time_guess = 10
    dt = time_guess / (num_nodes)

    guess =  np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))

    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
    control = np.zeros(aircraft.num_controls)
    control[:len(trim_state_and_control) - aircraft.num_states - 3] = trim_state_and_control[aircraft.num_states:-3]
    aircraft.com = np.array(trim_state_and_control[-3:])
    
    dyn = aircraft.state_update

    # Initialize trajectory with debugging prints
    for i in tqdm(range(num_nodes + 1), desc='Initialising Trajectory:'):
        guess[:aircraft.num_states, i] = state.full().flatten()
        # control = control + 1 * (rng.random(len(control)) - 0.5)
        guess[aircraft.num_states:, i] = control
        next_state = dyn(state, control, dt)
        # print(f"Node {i}: State = {state}, Control = {control}, Next State = {next_state}")
        state = next_state


    # Second loop: Validate initial guess
    guess2 = np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))
    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])

    for i in tqdm(range(num_nodes + 1), desc='Validating Trajectory:'):
        guess2[:aircraft.num_states, i] = state.full().flatten()
        control = guess[aircraft.num_states:, i]  # Use controls from guess1
        next_state = dyn(state, control, dt)
        guess2[aircraft.num_states:, i] = control
        state = next_state


    
    for i in range(num_nodes + 1):
        print(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i])
        assert np.allclose(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i]), f"Problem in node {i}"




    control_problem = AircraftControl(aircraft, num_nodes, time_guess = time_guess)    
    control_problem.setup(guess)
    control_problem.solve()


if __name__ == "__main__":
    main()
    # minimal_quad_test()