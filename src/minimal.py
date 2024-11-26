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

from src.dynamics import Aircraft
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from src.utils import TrajectoryConfiguration, load_model
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
import h5py
from scipy.interpolate import CubicSpline
from src.plotting import TrajectoryPlotter, TrajectoryData

import threading
import torch

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)
from pathlib import Path
from src.dynamics import AircraftOpts

plt.ion()
default_solver_options = {'ipopt': {'max_iter': 10000,
                                    'tol': 1e-2,
                                    'acceptable_tol': 1e-2,
                                    'acceptable_obj_change_tol': 1e-2,
                                    'hessian_approximation': 'limited-memory'
                                    },
                        'print_time': 10,
                        # 'expand' : True
                        }

def cumulative_distances(waypoints:np.ndarray, VERBOSE:bool = False):
    """
    Given a set of waypoints, calculate the distance between each waypoint.

    Parameters
    ----------
    waypoints : np.array
        Array of waypoints. (d x n) where d is the dimension of the waypoints 
        and n is the number of waypoints. The first waypoint is taken as the 
        initial position.

    Returns
    -------
    distance : np.array
        Cumulative distance between waypoints.
    
    """
    differences = np.diff(waypoints, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    distance = np.cumsum(distances)

    if VERBOSE: 
        print("Cumulative waypoint distances: ", distance)
    return distance

class ControlProblem:
    def __init__(
        self, 
        opti:ca.Opti, 
        aircraft:Aircraft,
        trajectory_config:TrajectoryConfiguration,
        num_control_nodes: int,
        VERBOSE:bool = True

        ):
        self.opti = opti
        self.aircraft = aircraft
        self.state_dim = aircraft.num_states
        self.control_dim = aircraft.num_controls

        self.dynamics = aircraft.state_update

        

        self.nodes = num_control_nodes
        self.waypoints = trajectory_config.waypoints()
        self.current_waypoint = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.next_waypoint = self.waypoints[1]
        else:
            self.next_waypoint = None
        self.trajectory = trajectory_config
        self.num_waypoints = self.waypoints.shape[0] - 1
        self.distances = cumulative_distances(self.waypoints)
        self.switch_var = np.array(
            self.nodes * np.array(self.distances) 
            / self.distances[-1], dtype = int
            )
        
        self.VERBOSE = VERBOSE
        pass

    @dataclass
    class Node:
        index:int
        state:ca.MX
        state_next:ca.MX
        control:ca.MX

    def setup_opti_vars(self, 
                        scale_state = ca.vertcat(
                            [1, 1, 1, 1],
                            [1e3, 1e3, 1e3],
                            [1e2, 1e2, 1e2],
                            [1, 1, 1]
                            ), 
                        scale_control = ca.vertcat(
                            5, 5, 5,
                            [1e2, 1e2, 1e2],
                            [1, 1, 1],
                            [1e2, 1e2, 1e2]
                            ), 
                        scale_time = 1,
                        ):
        
        opti = self.opti
        self.time = scale_time * opti.variable()
        self.dt = self.time / self.nodes

        
        state_list = []
        control_list = []

        for i in range(self.nodes + 1):

            state_list.append(ca.DM(scale_state) * 
                              opti.variable(self.state_dim))

            if i < self.nodes:
                control_list.append(ca.DM(scale_control) *          
                            opti.variable(self.control_dim))
                

        self.state = ca.hcat(state_list)
        self.control = ca.hcat(control_list)

    def control_constraint(self, node:Node, fix_com:bool = True):
        control_envelope = self.trajectory.control
        opti = self.opti
        com = self.trajectory.aircraft.aero_centre_offset

        opti.subject_to(opti.bounded(control_envelope.lb[:6],
                node.control[:6], control_envelope.ub[:6]))
        
        opti.subject_to(opti.bounded(np.zeros(node.control[9:].shape),
                node.control[9:], np.zeros(node.control[9:].shape)))
        
        if fix_com:
            opti.subject_to(node.control[6:9]==com)



    def state_constraint(self, node:Node, dt:ca.MX):
        
        state_envelope = self.trajectory.state
        opti = self.opti
        dynamics = self.dynamics

        alpha = self.aircraft.alpha
        beta = self.aircraft.beta
        airspeed = self.aircraft.airspeed

        opti.subject_to(opti.bounded(state_envelope.alpha.lb,
            alpha(node.state, node.control), state_envelope.alpha.ub))

        opti.subject_to(opti.bounded(state_envelope.beta.lb,
            beta(node.state, node.control), state_envelope.beta.ub))

        opti.subject_to(opti.bounded(state_envelope.airspeed.lb,
            airspeed(node.state, node.control), state_envelope.airspeed.ub))
        
        opti.subject_to(node.state_next == dynamics(node.state, node.control, dt))


    def loss(self, state:Optional[ca.MX] = None, control:Optional[ca.MX] = None, 
             time:Optional[ca.MX] = None):
        return time ** 2





    def setup(self):
        opti = self.opti
        trajectory = self.trajectory

        _, time_guess = self.state_guess(trajectory)
        self.setup_opti_vars(scale_time=1/time_guess)
        nodes = self.nodes
        time = self.time
        state = self.state
        dt = self.dt
        control = self.control
        
        waypoint_info = trajectory.waypoints
        num_waypoints = self.num_waypoints
        waypoints = waypoint_info.waypoints
        waypoint_indices = np.array(waypoint_info.waypoint_indices)
        final_waypoint = waypoint_info.final_position[waypoint_indices]

        
        opti.subject_to(time > 0)

        if waypoint_info.initial_state is not None:
            initial_state = waypoint_info.initial_state
            opti.subject_to(state[4:, 0] == initial_state[4:])

        opti.subject_to(ca.dot(state[:4, 0], state[:4, 0]) == 1)

        # waypoint_node = 0
        for index in range(nodes):

            node_data = self.Node(
                index=index,
                state_next = state[:, index + 1],
                state=state[:, index],
                control = control[:, index]
            )
                
            self.state_constraint(node_data, dt)
            
            self.control_constraint(node_data)
            # waypoint_node = self.waypoint_constraint(node_data, waypoint_node)
        
        if self.VERBOSE:
            print("Initial State: ", initial_state)
            print("Predicted Switching Nodes: ", self.switch_var)

        self.opti.subject_to(
            self.state[4 + waypoint_indices, -1] ==  final_waypoint)

        self.initialise()

        opti.minimize(self.loss(state = state, control = control, time = time))

    def solve(self, 
                opts:dict = default_solver_options,
                warm_start:Union[ca.OptiSol, ca.Opti] = (None, None),
                filepath:str = None
                ):
        
        self.sol_state_list = []
        self.sol_control_list = []
        self.final_times = []

        if filepath is not None:
            if os.path.exists(filepath):
                os.remove(filepath)
        # plt.ion()
        # plotter = TrajectoryPlotter(self.aircraft)
        # plt.show(block = False)
        # TODO: investigate fig.add_subfigure for better plotting
        self.opti.solver('ipopt', opts)
        # self.opti.callback(lambda i: self.callback(plotter, i, filepath))
        # plt.show()

        if warm_start != (None, None):
            warm_sol, warm_opti = warm_start
            self.opti.set_initial(warm_sol.value_variables())
            # lam_g0 = warm_sol.value(warm_opti.lam_g)
            # self.opti.set_initial(self.opti.lam_g, lam_g0)
        sol = self.opti.solve()
        # plt.ioff()
        # plt.show(block=True)        
        return (sol, self.opti)



    def initialise(self):

        x_guess, time_guess = self.state_guess(self.trajectory)

        control_guess = np.zeros(self.control.shape)

        control_guess[6:9, :] = np.repeat([self.trajectory.aircraft.aero_centre_offset], 
                                          self.control.shape[1], axis = 0).T

        if self.VERBOSE:
            print("State Trajectory Guess: ", x_guess)

        self.opti.set_initial(self.state, x_guess)
        self.opti.set_initial(self.time, time_guess)
        self.opti.set_initial(self.control, control_guess)

    def state_guess(self, trajectory:TrajectoryConfiguration):
        """
        Initial guess for the state variables.
        """
        

        state_dim = self.aircraft.num_states
        initial_pos = trajectory.waypoints.initial_position
        velocity_guess = trajectory.waypoints.default_velocity
        waypoints = self.waypoints[1:, :]
        
        x_guess = np.zeros((state_dim, self.nodes + 1))
        distance = self.distances
    
        self.r_glide = 10
        

        direction_guess = (waypoints[0, :] - initial_pos)
        vel_guess = velocity_guess *  direction_guess / np.linalg.norm(direction_guess)

        if self.VERBOSE:
            print("Cumulative Waypoint Distances: ", distance)
            print("Predicted Switching Nodes: ", self.switch_var)
            print("Direction Guess: ", direction_guess)
            print("Velocity Guess: ", vel_guess)
            print("Initial Position: ", initial_pos)
            print("Waypoints: ", waypoints)

        x_guess[:3, 0] = initial_pos
        x_guess[3:6, 0] = vel_guess

        z_flip = R.from_euler('x', 180, degrees=True)

        for i, waypoint in enumerate(waypoints):
            if len(self.trajectory.waypoints.waypoint_indices) < 3:
                    waypoint[2] += self.distances[i] / self.r_glide
        i_wp = 0
        for i in range(self.nodes):
            # switch condition
            if i > self.switch_var[i_wp]:
                i_wp += 1
                
            if i_wp == 0:
                wp_last = initial_pos
            else:
                wp_last = waypoints[i_wp-1, :]
            wp_next = waypoints[i_wp, :]

            if i_wp > 0:
                interpolation = (i - self.switch_var[i_wp-1]) / (self.switch_var[i_wp] - self.switch_var[i_wp-1])
            else:
                interpolation = i / self.switch_var[0]

            

            # extend position guess
            pos_guess = (1 - interpolation) * wp_last + interpolation * wp_next

            x_guess[4:7, i + 1] = np.reshape(pos_guess, (3,))
            

            direction = (wp_next - wp_last) / ca.norm_2(wp_next - wp_last)
            vel_guess = velocity_guess * direction
            x_guess[7:10, i + 1] = np.reshape(velocity_guess * direction, (3,))

            rotation, _ = R.align_vectors(np.array(direction).reshape(1, -1), [[1, 0, 0]])

            # Check if the aircraft is moving in the opposite direction
            if np.dot(direction.T, [1, 0, 0]) < 0:
                flip_y = R.from_euler('y', 180, degrees=True)
                rotation = rotation * flip_y

            x_guess[:4, i + 1] = (rotation * z_flip).as_quat()

        # x_guess = self.smooth_trajectory(x_guess)

        time_guess = distance[-1] / velocity_guess
        # if self.VERBOSE:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection = '3d')
        #     ax.plot(x_guess[4, :], x_guess[5, :], x_guess[6, :])
        #     plt.show(block = True)
        
        
        return x_guess, time_guess
    

def main():

    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
    model_path = Path(NETWORKPATH) / 'model-dynamics.pth'

    # opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)
    opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)#, physical_integration_substeps=10)

    aircraft = Aircraft(opts = opts)


    opti = ca.Opti()

    num_control_nodes = 50
    # aircraft = Aircraft(traj_dict['aircraft'], model)#, LINEAR=True)
    problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    problem.setup()
    (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control.hdf5'))

    # _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    # problem.plot_convergence(ax, sol)
    
    # sol_traj = sol.value(problem.state)
    # opti = ca.Opti()
    # aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=False)
    # problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    # problem.setup()
    # # problem.opti.set_initial(problem.state, sol_traj)
    # (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control_nn.hdf5'), warm_start=(sol, opti))

    # _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    # problem.plot_convergence(ax, sol)
    plotter = TrajectoryPlotter(aircraft)
    trajectory_data = TrajectoryData(
                state = np.array(sol.value(problem.state))[:, 1:],
                control = np.array(sol.value(problem.control)),
                time = np.array(sol.value(problem.time))
            )
            
    plotter.plot(trajectory_data = trajectory_data)
    plt.show(block = True)

    return sol

if __name__ == "__main__":
    main()