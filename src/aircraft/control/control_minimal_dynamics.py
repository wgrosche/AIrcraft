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

from aircraft.dynamics.dynamics import Aircraft, AircraftOpts
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from aircraft.utils.utils import TrajectoryConfiguration, load_model
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
import h5py
from scipy.interpolate import CubicSpline
from aircraft.plotting_minimal import TrajectoryPlotter, TrajectoryData

import threading
import torch

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)
from pathlib import Path

plt.ion()
default_solver_options = {'ipopt': {'max_iter': 10000,
                                    'tol': 1e-2,
                                    'acceptable_tol': 1e-2,
                                    'acceptable_obj_change_tol': 1e-2,
                                    'hessian_approximation': 'exact',
                                    'linear_solver': 'mumps',
                                    'mumps_mem_percent': 10000,      # Increase memory allocation percentage
                                    'mumps_pivtol': 1e-6,           # Pivot tolerance (can help with numerical stability)
                                    'mumps_pivtolmax': 1e-2,        # Maximum pivot tolerance
                                    'mumps_permuting_scaling': 7,   # Use a more robust scaling strategy
                                    'max_cpu_time': 1e4             # Increase the maximum CPU time
                                    },
                        'print_time': 10,
                        'expand': True

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

        self.dynamics = aircraft.state_update.expand()

        

        self.num_nodes = num_control_nodes
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
            self.num_nodes * np.array(self.distances) 
            / self.distances[-1], dtype = int
            )
        
        self.VERBOSE = VERBOSE
        pass

    @dataclass
    class Node:
        index:int
        state:ca.MX
        control:ca.MX
        lam:ca.MX
        mu:ca.MX
        nu:ca.MX

    def control_constraint(self, node:Node):
        self.opti.subject_to(
            self.opti.bounded(
                -5, 
                node.control[0], 
                5
                )
            )
        self.opti.subject_to(
            self.opti.bounded(
                -5, 
                node.control[1], 
                5
                )
            )



    def state_constraint(self, node:Node, next:Node, dt:ca.MX):
        """
        Constraints on the state variables.

        node:Node - current node
        next:Node - next node
        dt:ca.MX - time step
        """
        self.opti.subject_to(
            self.opti.bounded(
                20, 
                self.aircraft.airspeed(node.state, node.control), 
                80
                )
            )

        self.opti.subject_to(
            self.opti.bounded(
                -np.deg2rad(10), 
                self.aircraft.beta(node.state, node.control), 
                np.deg2rad(10)
                )
            )

        self.opti.subject_to(
            self.opti.bounded(
                -np.deg2rad(20), 
                self.aircraft.alpha(node.state, node.control), 
                np.deg2rad(20)
                )
            )
        
        self.opti.subject_to(
            next.state == self.dynamics(node.state, node.control, dt)
            )

        self.opti.subject_to(
            next.state[2] > 0
            )



    def waypoint_constraint(self, node:Node, next:Node):#, waypoint_node:int):
        """
        Waypoint constraint implementation from:
        https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf
        """
        tolerance = self.trajectory.waypoints.tolerance
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

    def loss(self, state:Optional[ca.MX] = None, control:Optional[ca.MX] = None, 
            time:Optional[ca.MX] = None):
        lambda_rate = 1.0
        rate_penalty = 0
        for i in range(self.num_nodes - 1):
            rate_penalty += ca.sumsqr(self.control[:, i+1] - self.control[:, i])
        
        final_waypoint = self.trajectory.waypoints.final_position[self.trajectory.waypoints.waypoint_indices]

        print(final_waypoint.shape)
        print(state[self.trajectory.waypoints.waypoint_indices, -1].shape)
        final_waypoint_loss = (state[self.trajectory.waypoints.waypoint_indices, -1] - final_waypoint)

        print(final_waypoint_loss.shape)

        return time ** 2 + lambda_rate * rate_penalty + ca.dot(final_waypoint_loss, final_waypoint_loss)

    def setup(self):
        opti = self.opti
        trajectory = self.trajectory

        scale_state = ca.vertcat(
                            [1e3, 1e3, 1e3],
                            [1e2, 1e2, 1e2],
                            [1, 1, 1, 1],
                            [1, 1, 1]
                            )
        scale_control = ca.vertcat(5, 5, 5)

        x_guess, time_guess = self.state_guess(trajectory)
        self.time = time_guess * opti.variable()
        opti.subject_to(self.time > 0)
        opti.set_initial(self.time, time_guess)

        self.dt = self.time / self.num_nodes

        waypoint_info = trajectory.waypoints
        num_waypoints = self.num_waypoints
        waypoints = waypoint_info.waypoints
        waypoint_indices = np.array(waypoint_info.waypoint_indices)
        final_waypoint = trajectory.waypoints.final_position[waypoint_indices]

        print("Final Waypoint: ", final_waypoint)

        
        current_node = self.Node(
            index=0,
            state=ca.DM(scale_state) * opti.variable(self.state_dim),
            control=ca.DM(scale_control) * opti.variable(self.control_dim),
            lam=None, #opti.variable(self.num_waypoints),
            mu=None, #opti.variable(self.num_waypoints),
            nu=None, #opti.variable(self.num_waypoints)
        )

        # if waypoint_info.initial_state is not None:
        initial_state = waypoint_info.initial_state
        opti.subject_to(current_node.state == initial_state)
        opti.set_initial(current_node.state, x_guess[:, 0])
        opti.set_initial(current_node.control, np.zeros(self.control_dim))

        opti.subject_to(ca.dot(current_node.state[6:10], current_node.state[6:10]) == 1)

        # opti.subject_to(current_node.lam == [1] * num_waypoints)

        self.state = [current_node.state]
        self.control = [current_node.control]
        # self.lam = [current_node.lam]
        # self.mu = [current_node.mu]
        # self.nu = [current_node.nu]
        

        for index in range(1, self.num_nodes + 1):
            next_node = self.Node(
                index=index,
                state = ca.DM(scale_state) * opti.variable(self.state_dim),
                control = ca.DM(scale_control) * opti.variable(self.control_dim),
                lam=None,#opti.variable(self.num_waypoints),
                mu=None,#opti.variable(self.num_waypoints),
                nu=None,#opti.variable(self.num_waypoints)
            )
            
            self.state_constraint(current_node, next_node, self.dt)
            self.control_constraint(current_node)
            # self.waypoint_constraint(current_node, next_node)

            current_node = next_node

            opti.set_initial(current_node.state, x_guess[:, index])
            opti.set_initial(current_node.control, np.zeros(self.control_dim))

            self.state.append(current_node.state)
            # self.lam.append(current_node.lam)
            if index < self.num_nodes:
                self.control.append(current_node.control)
                # self.mu.append(current_node.mu)
                # self.nu.append(current_node.nu)

        # self.opti.subject_to(current_node.state[waypoint_indices] == final_waypoint)
        
        # self.opti.subject_to(current_node.lam == [0] * self.num_waypoints)

        self.state = ca.hcat(self.state)
        print("State Shape: ", self.state.shape)
        self.control = ca.hcat(self.control)
        # self.lam = ca.hcat(self.lam)
        # self.mu = ca.hcat(self.mu)
        # self.nu = ca.hcat(self.nu)
        
        # if self.VERBOSE:
        #     print("Initial State: ", initial_state)
        #     print("Waypoints: ", waypoints)
        #     print("Waypoint Indices: ", waypoint_indices)
        #     print("Final Waypoint: ", final_waypoint)
        #     print("Predicted Switching Nodes: ", self.switch_var)

        print("State Shape: ", self.state.shape)

        # self.initialise()

        opti.minimize(self.loss(state = self.state, control = self.control, time = self.time))


    def save_progress(self, filepath, iteration):
        if filepath is not None:
            # save the state, control and time to a file
            with h5py.File(filepath, "a") as h5file:
                for i, (X, U, time, lam, mu, nu) in enumerate(zip(
                    self.sol_state_list[-10:], 
                    self.sol_control_list[-10:], 
                    self.final_times[-10:], 
                    self.lam_list[-10:], 
                    self.mu_list[-10:], 
                    self.nu_list[-10:]
                    )):
                    grp = h5file.create_group(f'iteration_{iteration - 10 + i}')
                    grp.create_dataset('state', data=X)
                    grp.create_dataset('control', data=U)
                    grp.create_dataset('time', data=time)
                    # grp.create_dataset('lam', data=lam)
                    # grp.create_dataset('mu', data=mu)
                    # grp.create_dataset('nu', data=nu)

    def plot_convergence(self, ax:plt.axes, sol:ca.OptiSol):
        ax.semilogy(sol.stats()['iterations']['inf_du'], label="Dual infeasibility")
        ax.semilogy(sol.stats()['iterations']['inf_pr'], label="Primal infeasibility")

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Infeasibility (log scale)')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show(block = True)

    def callback(self, plotter:TrajectoryPlotter, iteration:int, filepath:str):
        if iteration % 10 == 5:
            trajectory_data = TrajectoryData(
                state = np.array(self.opti.debug.value(self.state))[:, 1:],
                control = np.array(self.opti.debug.value(self.control)),
                time = np.array(self.opti.debug.value(self.time)),
                lam = None,#np.array(self.opti.debug.value(self.lam)),
                mu = None,#np.array(self.opti.debug.value(self.mu)),
                nu = None,#np.array(self.opti.debug.value(self.nu))
            )
            
            plotter.plot(trajectory_data = trajectory_data)
            # plt.pause(0.001)
            plt.draw()
            plotter.figure.canvas.start_event_loop(0.0002)


            # self.plot_sparsity(axs[0])
            # self.plot_trajectory(axs[1])

        if filepath is not None:
            self.sol_state_list.append(self.opti.debug.value(self.state))
            self.sol_control_list.append(self.opti.debug.value(self.control))
            self.final_times.append(self.opti.debug.value(self.time))
            # self.lam_list.append(self.opti.debug.value(self.lam))
            # self.mu_list.append(self.opti.debug.value(self.mu))
            # self.nu_list.append(self.opti.debug.value(self.nu))
            if iteration % 10 == 0:
                self.save_progress(filepath, iteration)

    def solve(self, 
                opts:dict = default_solver_options,
                warm_start:Union[ca.OptiSol, ca.Opti] = (None, None),
                filepath:str = None
                ):
        
        self.sol_state_list = []
        self.sol_control_list = []
        self.final_times = []
        self.lam_list = []
        self.mu_list = []
        self.nu_list = []

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
        
        (lambda_guess, mu_guess, nu_guess) = self.waypoint_variable_guess()

        # x_guess, time_guess = self.state_guess(self.trajectory)

        # control_guess = np.zeros(self.control.shape)

        # control_guess[6:9, :] = np.repeat([self.trajectory.aircraft.aero_centre_offset], 
        #                                   self.control.shape[1], axis = 0).T

        # if self.VERBOSE:
        #     print("State Trajectory Guess: ", x_guess.shape)


        self.opti.set_initial(self.nu, nu_guess)
        self.opti.set_initial(self.lam, lambda_guess)
        self.opti.set_initial(self.mu, mu_guess)
        # print("Size of state: ", self.state.shape, "Size of guess: ", x_guess.shape)
        # self.opti.set_initial(self.state, x_guess)
        # self.opti.set_initial(self.time, time_guess)
        # self.opti.set_initial(self.control, control_guess)
    
    def waypoint_variable_guess(self):

        num_waypoints = self.num_waypoints

        lambda_guess = np.zeros((num_waypoints, self.num_nodes + 1))
        mu_guess = np.zeros((num_waypoints, self.num_nodes))
        nu_guess = np.zeros((num_waypoints, self.num_nodes))

        i_wp = 0
        for i in range(1, self.num_nodes):
            if i > self.switch_var[i_wp]:
                i_wp += 1

            if ((i_wp == 0) and (i + 1 >= self.switch_var[0])) or i + 1 - self.switch_var[i_wp-1] >= self.switch_var[i_wp]:
                mu_guess[i_wp, i] = 1

            for j in range(num_waypoints):
                if i + 1 >= self.switch_var[j]:
                    lambda_guess[j, i] = 1

        return (lambda_guess, mu_guess, nu_guess)
    
    # def smooth_trajectory(self, x_guess):
    #         # Extract the points along the trajectory
    #         x_vals = x_guess[4, :]  # x-coordinates
    #         y_vals = x_guess[5, :]  # y-coordinates
    #         z_vals = x_guess[6, :]  # z-coordinates

    #         # Create a parameter t for the trajectory points
    #         t = np.linspace(0, 1, len(x_vals))

    #         # Fit cubic splines to the trajectory points
    #         spline_x = CubicSpline(t, x_vals)
    #         spline_y = CubicSpline(t, y_vals)
    #         spline_z = CubicSpline(t, z_vals)

    #         # Evaluate the splines at finer intervals for a smoother trajectory
    #         t_fine = t#np.linspace(0, 1, len(x_vals) * 10)  # Increase resolution by 10x
    #         x_smooth = spline_x(t_fine)
    #         y_smooth = spline_y(t_fine)
    #         z_smooth = spline_z(t_fine)

    #         # Update x_guess with the smoothed values (optional, for visualization)
    #         x_guess[4, :] = np.interp(np.linspace(0, len(x_vals)-1, len(x_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), x_smooth)
    #         x_guess[5, :] = np.interp(np.linspace(0, len(y_vals)-1, len(y_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), y_smooth)
    #         x_guess[6, :] = np.interp(np.linspace(0, len(z_vals)-1, len(z_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), z_smooth)
    #         return x_guess

    def state_guess(self, trajectory:TrajectoryConfiguration):
        """
        Initial guess for the state variables.
        """
        state_dim = self.aircraft.num_states
        initial_pos = trajectory.waypoints.initial_position
        initial_orientation = trajectory.waypoints.initial_state[6:10]
        velocity_guess = trajectory.waypoints.default_velocity
        waypoints = self.waypoints[1:, :]
        
        x_guess = np.zeros((state_dim, self.num_nodes + 1))
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


        rotation, _ = R.align_vectors(np.array(direction_guess).reshape(1, -1), [[1, 0, 0]])

        # Check if the aircraft is moving in the opposite direction
        if np.dot(direction_guess.T, [1, 0, 0]) < 0:
            flip_y = R.from_euler('y', 180, degrees=True)
            rotation = rotation * flip_y

        # Get the euler angles
        euler = rotation.as_euler('xyz')
        print("Euler: ", euler)
        # If roll is close to 180, apply correction
        # if abs(euler[0]) >= np.pi/2: 
            # Create rotation around x-axis by 180 degrees
        roll_correction = R.from_euler('x', 180, degrees=True)
        
        x_guess[6:10, 0] = (rotation).as_quat()

        # z_flip = R.from_euler('x', 180, degrees=True)

        for i, waypoint in enumerate(waypoints):
            if len(self.trajectory.waypoints.waypoint_indices) < 3:
                    waypoint[2] = initial_pos[2] + self.distances[i] / self.r_glide
        i_wp = 0
        for i in range(self.num_nodes):
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

            x_guess[:3, i + 1] = np.reshape(pos_guess, (3,))
            

            direction = (wp_next - wp_last) / ca.norm_2(wp_next - wp_last)
            vel_guess = velocity_guess * direction
            x_guess[3:6, i + 1] = np.reshape(velocity_guess * direction, (3,))

            rotation, _ = R.align_vectors(np.array(direction).reshape(1, -1), [[1, 0, 0]])

            # Check if the aircraft is moving in the opposite direction
            if np.dot(direction.T, [1, 0, 0]) < 0:
                flip_y = R.from_euler('y', 180, degrees=True)
                rotation = rotation * flip_y

            # Get the euler angles
            euler = rotation.as_euler('xyz')
            # print("Euler: ", euler)
            # If roll is close to 180, apply correction
            # if abs(euler[0]) >= np.pi/2: 
                # Create rotation around x-axis by 180 degrees
            # roll_correction = R.from_euler('x', 180, degrees=True)
                # Apply correction
            # rotation = rotation * roll_correction


            x_guess[6:10, i + 1] = (rotation).as_quat()

        # x_guess = self.smooth_trajectory(x_guess)

        time_guess = distance[-1] / velocity_guess
        # if self.VERBOSE:
        #     print("State Guess: ", x_guess)
        #     plotter = TrajectoryPlotter(self.aircraft)
        #     trajectory_data = TrajectoryData(
        #         state = np.array(x_guess),
        #         # time = np.array(time_guess)
        #     )
            
        #     plotter.plot(trajectory_data = trajectory_data)
        #     plt.pause(0.001)
        #     # fig = plt.figure()
        #     # ax = fig.add_subplot(111, projection = '3d')
        #     # ax.plot(x_guess[4, :], x_guess[5, :], x_guess[6, :])
            
        #     plt.show(block = True)
        
        
        return x_guess, time_guess
    

def main():

    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
    model_path = Path(NETWORKPATH) / 'model-dynamics.pth'

    poly_path = Path(NETWORKPATH) / "fitted_models_casadi.pkl"

    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)

    # opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)
    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)

    # [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 
    aircraft.com = [0.0131991, -1.78875e-08, 0.00313384]


    opti = ca.Opti()

    num_control_nodes = 1000
    # aircraft = Aircraft(traj_dict['aircraft'], model)#, LINEAR=True)
    problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    problem.setup()
    (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control.hdf5'))

    # save sol
    # import pickle

    # with(open('solution.pkl', 'wb')) as f:
    #     pickle.dump(sol, f)

    plotter = TrajectoryPlotter(aircraft)
    trajectory_data = TrajectoryData(
                state = np.array(sol.value(problem.state))[:, 1:],
                control = np.array(sol.value(problem.control)),
                time = np.array(sol.value(problem.time))
            )
            
    plotter.plot(trajectory_data = trajectory_data)

    plt.show(block = True)


    # _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    # problem.plot_convergence(ax, sol)
    
    # # sol_traj = sol.value(problem.state)
    # opti = ca.Opti()
    # aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=False)
    # problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    # problem.setup()
    # # problem.opti.set_initial(problem.state, sol_traj)
    # (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control_nn.hdf5'), warm_start=(sol, opti))

    # _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    # problem.plot_convergence(ax, sol)

    return sol

if __name__ == "__main__":
    main()