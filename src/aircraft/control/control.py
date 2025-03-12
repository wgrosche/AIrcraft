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

from aircraft.dynamics.dynamics import Aircraft
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


from pathlib import Path
from aircraft.dynamics.dynamics import AircraftOpts
from aircraft.control.initialisation import cumulative_distances
from abc import abstractmethod
import time
from aircraft.config import default_solver_options, BASEPATH, NETWORKPATH, DATAPATH, DEVICE, rng
plt.ion()
# from aircraft.config import default_solver_options

@dataclass
class ControlNode:
    index:Optional[int] = None
    state:Optional[ca.MX] = None
    control:Optional[ca.MX] = None

@dataclass
class ControlNodeWaypoints(ControlNode):
    lam:Optional[ca.MX] = None
    mu:Optional[ca.MX] = None
    nu:Optional[ca.MX] = None

    @classmethod
    def from_control_node(cls, node:ControlNode, lam:Optional[ca.MX] = None,
                            mu:Optional[ca.MX] = None,
                            nu:Optional[ca.MX] = None):
        return cls(
            index = node.index,
            state = node.state,
            control = node.control
        )
    
def plot_convergence(self, ax:plt.axes, sol:ca.OptiSol):
    ax.semilogy(sol.stats()['iterations']['inf_du'], label="Dual infeasibility")
    ax.semilogy(sol.stats()['iterations']['inf_pr'], label="Primal infeasibility")

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Infeasibility (log scale)')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show(block = True)

class ControlProblem:
    """
    Control Problem parent class
    """

    def __init__(self, dynamics:ca.Function, num_nodes:int, opts:Optional[dict] = {}):

        self.opti = ca.Opti()
        self.state_dim = dynamics.size1_in(0)
        self.control_dim = dynamics.size1_in(1)
        self.num_nodes = num_nodes
        self.verbose = opts.get('verbose', False)
        self.dynamics = dynamics

        self.scale_state = opts.get('scale_state', None)
        self.scale_control = opts.get('scale_control', None)
        self.scale_time = None
        self.initial_state = opts.get('initial_state', None)
        self.solver_opts = opts.get('solver_options', default_solver_options)

        self.filepath = opts.get('savefile', None)
        if self.filepath:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            self.h5file = h5py.File(self.filepath, "a")

        self.sol_state_list = []
        self.sol_control_list = []
        self.final_times = []

    def control_constraint(self, node:ControlNode):
        """
        Does nothing in base class
        """
        pass

    def state_constraint(self, node:ControlNode, next:ControlNode, dt:ca.MX):
        opti = self.opti
        dynamics = self.dynamics

        # do a couple of physical steps to get to the next node
        num_steps = 10
        h = dt/num_steps
        x = node.state
        
        for _ in range(num_steps):
            x = self.dynamics(x, node.control, h)

        # quaternion constraint
        opti.subject_to(ca.norm_2(x[6:10]) == 1)
        # opti.subject_to(next.state == dynamics(node.state, node.control, dt))
        opti.subject_to(next.state == x)

    def loss(self, time:Optional[ca.MX] = None):
        return time

    def _setup_step(self, index:int, current_node:ControlNode, guess:np.ndarray):
        opti = self.opti

        if self.scale_state and self.scale_control:
            next_node = ControlNode(
                index=0,
                state=ca.DM(self.scale_state) * opti.variable(self.state_dim),
                control=ca.DM(self.scale_control) * opti.variable(self.control_dim),
            )
        else:
            next_node = ControlNode(
                index=0,
                state=opti.variable(self.state_dim),
                control=opti.variable(self.control_dim),
            )
            
        self.state_constraint(current_node, next_node, self.dt)
        self.control_constraint(current_node)

        opti.set_initial(next_node.state, guess[:self.state_dim, index])
        opti.set_initial(next_node.control, guess[self.state_dim:, index])
        return next_node
    
    def _setup_time(self):
        opti = self.opti
        if not self.scale_time:
            self.scale_time = 1
        self.time = self.scale_time * opti.variable()

        opti.subject_to(self.time > 0)
        opti.set_initial(self.time, self.scale_time)

        self.dt = self.time / self.num_nodes

    def _setup_initial_node(self, guess:np.ndarray):
        opti = self.opti
        if self.scale_state and self.scale_control:
            current_node = ControlNode(
                index=0,
                state=ca.DM(self.scale_state) * opti.variable(self.state_dim),
                control=ca.DM(self.scale_control) * opti.variable(self.control_dim),
            )
        else:
            current_node = ControlNode(
                index=0,
                state=opti.variable(self.state_dim),
                control=opti.variable(self.control_dim),
            )

        opti.subject_to(current_node.state == guess[:self.state_dim, 0])
        opti.set_initial(current_node.state, guess[:self.state_dim, 0])
        opti.set_initial(current_node.control, guess[self.state_dim:, 0])

        return current_node
    
    def _setup_variables(self, nodes:List[ControlNode]):
        self.state = ca.hcat([nodes[i].state for i in range(self.num_nodes + 1)])
        self.control = ca.hcat([nodes[i].control for i in range(self.num_nodes)])

        if self.verbose:
            print("State Shape: ", self.state.shape)
            print("Control Shape: ", self.control.shape)

    def _setup_objective(self, nodes):
        self.opti.minimize(self.loss(time = self.time))

    def setup(self, guess:np.ndarray):
        self._setup_time()
        current_node = self._setup_initial_node(guess)
        nodes = [current_node]
        
        for index in range(1, self.num_nodes + 1):
            current_node = self._setup_step(index, current_node, guess)
            nodes.append(current_node)

        self._setup_variables(nodes)
        self._setup_objective(nodes)

    def save_progress(self, iteration, states, controls, time_vals):
        if self.h5file is not None:
            try:
                # Combine and iterate through the last 10 entries
                for i, (state, control, time_val) in enumerate(zip(states[-10:], controls[-10:], time_vals[-10:])):
                    grp_name = f'iteration_{iteration - 10 + i}'
                    grp = self.h5file.require_group(grp_name)  # Creates or accesses the group
                    grp.attrs['timestamp'] = time.time()

                    # Create or overwrite datasets efficiently
                    for name, data in zip(['state', 'control', 'time'], [state, control, time_val]):
                        if name in grp:
                            del grp[name]  # Overwrite if dataset already exists
                        grp.create_dataset(name, data=data, compression='gzip')
            except Exception as e:
                print(f"Error saving progress: {e}")

    def callback(self, iteration: int):
        # Save the progress every 10 iterations
        if self.filepath is not None:
            self.sol_state_list.append(self.opti.debug.value(self.state))
            self.sol_control_list.append(self.opti.debug.value(self.control))
            self.final_times.append(self.opti.debug.value(self.time))
            if iteration % 10 == 0:
                self.save_progress(iteration, self.sol_state_list, self.sol_control_list, self.final_times)

    def solve(self, warm_start:Optional[ca.OptiSol] = None):
        self.opti.solver('ipopt', self.solver_opts)
        self.opti.callback(lambda iteration: self.callback(iteration))
        if warm_start:
            self.opti.set_initial(warm_start.value_variables())
        sol = self.opti.solve()
        return sol
    

class AircraftControl(ControlProblem):
    def __init__(self, aircraft:Aircraft, num_nodes:int, opts:Optional[dict] = {}):
        dynamics = aircraft.state_update
        self.aircraft = aircraft
        self.plotter = TrajectoryPlotter(aircraft)
        self.scale_time = 10
        plt.show()
        super().__init__(dynamics, num_nodes, opts)


    def control_constraint(self, node:ControlNode):
        self.opti.subject_to(self.opti.bounded(-5, node.control[0], 5))
        self.opti.subject_to(self.opti.bounded(-5, node.control[1], 5))
        # self.opti.subject_to(self.opti.bounded(0, node.control[2:], 0))

    def state_constraint(self, node:ControlNode, next:ControlNode, dt:ca.MX):
        super().state_constraint(node, next, dt)
        opti = self.opti
        aircraft = self.aircraft
        beta = aircraft.beta
        alpha = aircraft.alpha
        airspeed = aircraft.airspeed

        # opti.subject_to(opti.bounded(20, airspeed(node.state, node.control), 80))
        # opti.subject_to(opti.bounded(-np.deg2rad(10), beta(node.state, node.control),  np.deg2rad(10)))
        # opti.subject_to(opti.bounded(-np.deg2rad(20), alpha(node.state, node.control), np.deg2rad(20)))
        # opti.subject_to(next.state[2] < 0)

    def _setup_objective(self, nodes):
        """
        TODO: When enforcing hard constraints on final position make sure that 
        the waypoint tolerance is sufficient to guarantee a node passes within.

        If there are control nodes every 15m and the tolerance is 5m then its unlikely for there to be a node within the tolerance
        """
        super()._setup_objective(nodes)
        
        final_waypoint = [0,100, -150]

        tolerance = 2 * np.linalg.norm(np.array([0,0, -200]) - np.array(final_waypoint)) / (self.num_nodes - 1)# TODO: Change from hardcoding to deriving from the initial position and the goal
        final_waypoint_diff = nodes[-1].state[:3] - final_waypoint
        final_waypoint_dist_sq = ca.dot(final_waypoint_diff, final_waypoint_diff)
        self.opti.subject_to(final_waypoint_dist_sq < tolerance ** 2)
        # self.opti.minimize(1000*final_waypoint_dist_sq)

        # lambda_rate = 100.0
        # lambda_smooth = 50.0
        # rate_penalty = ca.sumsqr(self.control[:2, 1:] - self.control[:2, :-1])
        # smoothness_penalty = ca.sumsqr(self.control[:2, 2:] - 2 * self.control[:2, 1:-1] + self.control[:2, :-2])
        # self.opti.minimize(lambda_rate * rate_penalty + lambda_smooth * smoothness_penalty)

    def callback(self, iteration: int):
        aircraft = self.aircraft
        f = aircraft.state_update(aircraft.state, aircraft.control, aircraft.dt_sym)

        # Compute the Jacobian of f w.r.t state
        J = ca.jacobian(f, aircraft.state)

        # Create a CasADi function for numerical evaluation
        J_func = ca.Function('J', [aircraft.state, aircraft.control, aircraft.dt_sym], [J])

        print("Condition Numbers:" , np.linalg.cond(J_func(self.opti.debug.value(self.state)[:, 1:], self.opti.debug.value(self.control), self.opti.debug.value(self.time)/self.num_nodes)))

        # Get constraint values and dual variables
        g = self.opti.debug.value(self.opti.g)
        lam_g = self.opti.debug.value(self.opti.lam_g)
        
        # Check which constraints are active (close to bounds)
        tolerance = 1e-6
        active_constraints = np.nonzero(abs(g) > tolerance)[0]
        
        if iteration % 10 == 0:

            for i in range(self.num_nodes):
                current_state = self.opti.debug.value(self.state)[:, i]
                current_control = self.opti.debug.value(self.control)[:, i]
                dt = self.opti.debug.value(self.time)/self.num_nodes
                num_steps = 10
                h = dt/num_steps
                x = self.opti.debug.value(self.state)[:, i]
                
                for _ in range(num_steps):
                    x = self.dynamics(x, self.opti.debug.value(self.control)[:, i], h)
                # Predicted next state using dynamics

                predicted_next = x#self.dynamics(current_state, current_control, dt)
                actual_next = self.opti.debug.value(self.state)[:, i+1]
                
                dynamics_violation = np.linalg.norm(predicted_next - actual_next)
                if dynamics_violation > 1e-3:
                    print(f"Large dynamics violation at node {i}: {dynamics_violation}")
            print("\nActive constraints at iteration", iteration)
            print("Constraint values:", g[active_constraints])
            print("Dual variables:", lam_g[active_constraints])
            
            # Print specific constraint types
            print("\nControl limits active:")
            print("Aileron:", abs(self.opti.debug.value(self.control[0])) >= 5.0)
            print("Elevator:", abs(self.opti.debug.value(self.control[1])) >= 5.0)
            
            # Print state constraints
            if hasattr(self, 'aircraft'):
                state = self.opti.debug.value(self.state)[:, :-1]
                control = self.opti.debug.value(self.control)
                print("\nState constraints:")
                print(f"Airspeed bounds violated: {self.aircraft.airspeed(state, control)}")
                print(f"Alpha bounds violated: {self.aircraft.alpha(state, control)}")
                print(f"Beta bounds violated: {self.aircraft.beta(state, control)}")


        super().callback(iteration)
        # print("distance to final: ", np.linalg.norm(np.array(self.opti.debug.value(self.state))[:2, -1] - [100,0]))
        # print("Control Deflections: ", self.opti.debug.value(self.control))
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

class WaypointControl(AircraftControl):
    """
    Implements waypoint traversal via complementarity constraint.
    """
    def __init__(self, aircraft:Aircraft, trajectory_config:TrajectoryConfiguration, opts:Optional[dict] = {}):
        """
        To be implemented
        """
        max_control_nodes = opts.get('max_control_nodes', 100)
        # to calculate the num of nodes needed we use the dubins path from initialisation.py and add a tolerance
        num_nodes = 100
        self.waypoint_tolerance = 100
        self.trajectory = trajectory_config
        self.num_waypoints = len(trajectory_config.waypoints.waypoints)

        super().__init__(aircraft, num_nodes, opts)


    def waypoint_constraint(self, node:ControlNodeWaypoints, next:ControlNodeWaypoints):
        """
        Waypoint constraint implementation from:
        https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf
        """
        # tolerance = self.trajectory.waypoints.tolerance
        tolerance = self.waypoint_tolerance
        waypoint_indices = np.array(self.trajectory.waypoints.waypoint_indices)
        num_waypoints = self.num_waypoints
        waypoints = self.waypoints[1:, waypoint_indices]
        opti = self.opti
        
        for j in range(num_waypoints):
            opti.subject_to(next.lam[j] - node.lam[j] + node.mu[j] == 0)
            opti.subject_to(node.mu[j] >= 0)
            if j < num_waypoints - 1:
                opti.subject_to(node.lam[j] - node.lam[j + 1] <= 0)

            diff = node.state[waypoint_indices] - waypoints[j, waypoint_indices]
            opti.subject_to(opti.bounded(0, node.nu[j], tolerance**2))
            opti.subject_to(node.mu[j] * (ca.dot(diff, diff) - node.nu[j]) == 0)

        return None
    
    def _setup_step(self, index:int, current_node:ControlNode, guess:np.ndarray):
        opti = self.opti

        next_node = ControlNodeWaypoints.from_control_node(
            super()._setup_step(index, current_node, guess), 
            lam = opti.variable(self.num_waypoints),
            mu=opti.variable(self.num_waypoints),
            nu=opti.variable(self.num_waypoints))
            
        self.waypoint_constraint(current_node, next_node)
        lam_start = self.state_dim + self.control_dim
        lam_end = lam_start + self.num_waypoints
        opti.set_initial(next_node.lam, guess[lam_start:lam_end, index])

        mu_end = lam_end + self.num_waypoints
        opti.set_initial(next_node.mu, guess[lam_end:mu_end, index])

        nu_end = mu_end + self.num_waypoints
        opti.set_initial(next_node.nu, guess[mu_end:nu_end, index])
        return next_node
    
    def _setup_initial_node(self, guess:np.ndarray):
        opti = self.opti
        current_node = ControlNodeWaypoints.from_control_node(
            super()._setup_initial_node(guess), 
            lam = opti.variable(self.num_waypoints),
            mu=opti.variable(self.num_waypoints),
            nu=opti.variable(self.num_waypoints))

        lam_start = self.state_dim + self.control_dim
        lam_end = lam_start + self.num_waypoints
        opti.set_initial(current_node.lam, guess[lam_start:lam_end, 0])

        mu_end = lam_end + self.num_waypoints
        opti.set_initial(current_node.mu, guess[lam_end:mu_end, 0])

        nu_end = mu_end + self.num_waypoints
        opti.set_initial(current_node.nu, guess[mu_end:nu_end, 0])
        return current_node

def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    
    num_nodes = 500
    dt = 0.01

    guess =  np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))

    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
    control = np.zeros(aircraft.num_controls)
    control[:len(trim_state_and_control) - aircraft.num_states - 3] = trim_state_and_control[aircraft.num_states:-3]
    aircraft.com = np.array(trim_state_and_control[-3:])
    
    dyn = aircraft.state_update

    for i in tqdm(range(num_nodes + 1), desc = 'Initialising Trajectory:'):
            guess[:aircraft.num_states, i] = state.full().flatten()
            control = control + 1 * (rng.random(len(control)) - 0.5)
            guess[aircraft.num_states:, i] = control
            state = dyn(state, control, dt)


    control_problem = AircraftControl(aircraft, num_nodes)    
    control_problem.setup(guess)
    sol = control_problem.solve()


if __name__ == "__main__":
    main()