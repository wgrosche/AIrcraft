import casadi as ca
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
import os
import sys

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASEPATH)
sys.path.append(BASEPATH)

from src.dynamics import Aircraft, load_model
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from src.utils import TrajectoryConfiguration
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
import h5py
from scipy.interpolate import CubicSpline

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
        lam:ca.MX
        lam_next:ca.MX
        mu:ca.MX
        nu:ca.MX

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
        lam_list = []
        mu_list = []
        nu_list = []

        for i in range(self.nodes + 1):

            state_list.append(ca.DM(scale_state) * 
                              opti.variable(self.state_dim))
            lam_list.append(opti.variable(self.num_waypoints))

            if i < self.nodes:
                control_list.append(ca.DM(scale_control) *          
                            opti.variable(self.control_dim))
                mu_list.append(opti.variable(self.num_waypoints))
                nu_list.append(opti.variable(self.num_waypoints))
                

        self.state = ca.hcat(state_list)
        self.control = ca.hcat(control_list)
        self.lam = ca.hcat(lam_list)
        self.mu = ca.hcat(mu_list)
        self.nu = ca.hcat(nu_list)

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

        alpha = self.aircraft._alpha
        beta = self.aircraft._beta
        airspeed = self.aircraft._airspeed

        opti.subject_to(opti.bounded(state_envelope.alpha.lb,
            alpha(node.state, node.control), state_envelope.alpha.ub))

        opti.subject_to(opti.bounded(state_envelope.beta.lb,
            beta(node.state, node.control), state_envelope.beta.ub))

        opti.subject_to(opti.bounded(state_envelope.airspeed.lb,
            airspeed(node.state, node.control), state_envelope.airspeed.ub))
        
        opti.subject_to(node.state_next == dynamics(node.state, node.control, dt))


    def waypoint_constraint(self, node:Node, waypoint_node:int):
        """
        Waypoint constraint implementation from:
        https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf
        """
        tolerance = self.trajectory.waypoints.tolerance
        waypoint_indices = np.array(self.trajectory.waypoints.waypoint_indices)
        num_waypoints = self.num_waypoints
        waypoints = self.waypoints[1:, waypoint_indices]
        opti = self.opti
        
        if node.index > self.switch_var[waypoint_node]:
            waypoint_node += 1
        
        for j in range(num_waypoints):
            opti.subject_to(node.lam_next[j] - node.lam[j] + node.mu[j] == 0)
            opti.subject_to(node.mu[j] >= 0)
            if j < num_waypoints - 1:
                opti.subject_to(node.mu[j] - node.mu[j + 1] <= 0)

            diff = node.state[4 + waypoint_indices] - waypoints[j, waypoint_indices]
            opti.subject_to(opti.bounded(0, node.nu[j], tolerance**2))
            opti.subject_to(node.mu[j] * (ca.dot(diff, diff) - node.nu[j]) == 0)

        return waypoint_node

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
        lam = self.lam
        mu = self.mu
        nu = self.nu

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

        opti.subject_to(lam[:, 0] == [1] * num_waypoints)

        waypoint_node = 0
        for index in range(nodes):

            node_data = self.Node(
                index=index,
                state_next = state[:, index + 1],
                state=state[:, index],
                control = control[:, index],
                lam=lam[:, index],
                lam_next=lam[:, index + 1],
                mu=mu[:, index],
                nu=nu[:, index]
            )
                
            self.state_constraint(node_data, dt)
            
            self.control_constraint(node_data)
            
            waypoint_node = self.waypoint_constraint(node_data, waypoint_node)
        
        if self.VERBOSE:
            print("Initial State: ", initial_state)
            print("Waypoints: ", waypoints)
            print("Waypoint Indices: ", waypoint_indices)
            print("Final Waypoint: ", final_waypoint)
            print("Predicted Switching Nodes: ", self.switch_var)

        self.opti.subject_to(
            self.state[4 + waypoint_indices, -1] ==  final_waypoint)
        
        self.opti.subject_to(self.mu[:, -1] == [0] * self.num_waypoints)

        self.initialise()

        opti.minimize(self.loss(state = state, control = control, time = time))

        if self.VERBOSE:
            constraints = opti.g
            print(f"Constraint 545: {constraints[576]}")

    def plot_sparsity(self, ax:plt.axes):
        jacobian = self.opti.debug.value(
            ca.jacobian(self.opti.g, self.opti.x)
            ).toarray()
        
        ax.clear()
        ax.spy(jacobian)
        plt.draw()
        plt.pause(0.01)

    def plot_trajectory(self, ax:plt.axes):
        state = self.opti.debug.value(self.state)
        ax.clear()
        ax.plot(state[4, :], state[5, :], state[6, :])
        plt.draw()
        plt.pause(0.01)

    def save_progress(self, filepath, iteration):
        if filepath is not None:
            # save the state, control and time to a file
            with h5py.File(filepath, "a") as h5file:
                for i, (X, U, time) in enumerate(zip(self.sol_state_list[-10:], self.sol_control_list[-10:], self.final_times[-10:])):
                    grp = h5file.create_group(f'iteration_{iteration - 10 + i}')
                    grp.create_dataset('state', data=X)
                    grp.create_dataset('control', data=U)
                    grp.create_dataset('time', data=time)

    def plot_convergence(self, ax:plt.axes, sol:ca.OptiSol):
        ax.semilogy(sol.stats()['iterations']['inf_du'], label="Dual infeasibility")
        ax.semilogy(sol.stats()['iterations']['inf_pr'], label="Primal infeasibility")

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Infeasibility (log scale)')
        ax.grid(True)

        plt.tight_layout()
        plt.show(block = True)

    def callback(self, axs:List[plt.axes], iteration:int, filepath:str):
        if iteration % 10 == 5:
            self.plot_sparsity(axs[0])
            self.plot_trajectory(axs[1])

        if filepath is not None:
            self.sol_state_list.append(self.opti.debug.value(self.state))
            self.sol_control_list.append(self.opti.debug.value(self.control))
            self.final_times.append(self.opti.debug.value(self.time))
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

        if filepath is not None:
            if os.path.exists(filepath):
                os.remove(filepath)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, projection = '3d')
        # TODO: investigate fig.add_subfigure for better plotting
        self.opti.solver('ipopt', opts)
        self.opti.callback(lambda i: self.callback([ax, ax2], i, filepath))
        plt.show()

        if warm_start != (None, None):
            warm_sol, warm_opti = warm_start
            self.opti.set_initial(warm_sol.value_variables())
            # lam_g0 = warm_sol.value(warm_opti.lam_g)
            # self.opti.set_initial(self.opti.lam_g, lam_g0)
        sol = self.opti.solve()
        
        return (sol, self.opti)



    def initialise(self):
        
        (lambda_guess, mu_guess, nu_guess) = self.waypoint_variable_guess()

        x_guess, time_guess = self.state_guess(self.trajectory)

        control_guess = np.zeros(self.control.shape)

        control_guess[6:9, :] = np.repeat([self.trajectory.aircraft.aero_centre_offset], 
                                          self.control.shape[1], axis = 0).T

        if self.VERBOSE:
            print("State Trajectory Guess: ", x_guess)


        self.opti.set_initial(self.nu, nu_guess)
        self.opti.set_initial(self.lam, lambda_guess)
        self.opti.set_initial(self.mu, mu_guess)
        self.opti.set_initial(self.state, x_guess)
        self.opti.set_initial(self.time, time_guess)
        self.opti.set_initial(self.control, control_guess)
    

    def waypoint_variable_guess(self):

        num_waypoints = self.num_waypoints

        lambda_guess = np.zeros((num_waypoints, self.nodes + 1))
        mu_guess = np.zeros((num_waypoints, self.nodes))
        nu_guess = np.zeros((num_waypoints, self.nodes))

        i_wp = 0
        for i in range(1, self.nodes):
            if i > self.switch_var[i_wp]:
                i_wp += 1

            if ((i_wp == 0) and (i + 1 >= self.switch_var[0])) or i + 1 - self.switch_var[i_wp-1] >= self.switch_var[i_wp]:
                mu_guess[i_wp, i] = 1

            for j in range(num_waypoints):
                if i + 1 >= self.switch_var[j]:
                    lambda_guess[j, i] = 1

        return (lambda_guess, mu_guess, nu_guess)
    
    def smooth_trajectory(self, x_guess):
            # Extract the points along the trajectory
            x_vals = x_guess[4, :]  # x-coordinates
            y_vals = x_guess[5, :]  # y-coordinates
            z_vals = x_guess[6, :]  # z-coordinates

            # Create a parameter t for the trajectory points
            t = np.linspace(0, 1, len(x_vals))

            # Fit cubic splines to the trajectory points
            spline_x = CubicSpline(t, x_vals)
            spline_y = CubicSpline(t, y_vals)
            spline_z = CubicSpline(t, z_vals)

            # Evaluate the splines at finer intervals for a smoother trajectory
            t_fine = t#np.linspace(0, 1, len(x_vals) * 10)  # Increase resolution by 10x
            x_smooth = spline_x(t_fine)
            y_smooth = spline_y(t_fine)
            z_smooth = spline_z(t_fine)

            # Update x_guess with the smoothed values (optional, for visualization)
            x_guess[4, :] = np.interp(np.linspace(0, len(x_vals)-1, len(x_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), x_smooth)
            x_guess[5, :] = np.interp(np.linspace(0, len(y_vals)-1, len(y_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), y_smooth)
            x_guess[6, :] = np.interp(np.linspace(0, len(z_vals)-1, len(z_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), z_smooth)
            return x_guess


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

            rotation = R.align_vectors(np.array(direction).reshape(1, -1), [[1, 0, 0]])[0]

            # Check if the aircraft is moving in the opposite direction
            if np.dot(direction.T, [1, 0, 0]) < 0:
                flip_y = R.from_euler('y', 180, degrees=True)
                rotation = rotation * flip_y

            x_guess[:4, i + 1] = (rotation * z_flip).as_quat()

        # x_guess = self.smooth_trajectory(x_guess)

        time_guess = distance[-1] / velocity_guess
        if self.VERBOSE:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.plot(x_guess[4, :], x_guess[5, :], x_guess[6, :])
            plt.show(block = True)
        
        
        return x_guess, time_guess
    

def main():
    opti = ca.Opti()
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))
    trajectory_config = TrajectoryConfiguration(traj_dict)

    num_control_nodes = 40
    aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=True)
    problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    problem.setup()
    (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control.hdf5'))

    _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    problem.plot_convergence(ax, sol)
    
    # sol_traj = sol.value(problem.state)
    opti = ca.Opti()
    aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=False)
    problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    problem.setup()
    # problem.opti.set_initial(problem.state, sol_traj)
    (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control_nn.hdf5'), warm_start=(sol, opti))

    _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    problem.plot_convergence(ax, sol)

    return sol

if __name__ == "__main__":
    main()