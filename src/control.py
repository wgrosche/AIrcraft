import casadi as ca
import numpy as np
from typing import List, Optional

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


def cumulative_waypoint_distances(
        waypoints:np.ndarray,
        VERBOSE:bool = False
        ):
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
    differences = np.diff(waypoints, axis=1)
    distances = np.linalg.norm(differences, axis=0)
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
        num_control_nodes: int

        ):
        self.opti = opti
        self.aircraft = aircraft
        self.num_control_nodes = num_control_nodes
        self.waypoints = trajectory_config.waypoints()
        self.trajectory = trajectory_config
        self.num_waypoints = self.waypoints.shape[0] - 1
        self.cumulative_distances = cumulative_waypoint_distances(self.waypoints)
        self.switching_variable = np.array(
            num_control_nodes * np.array(self.cumulative_distances) 
            / self.cumulative_distances[-1], dtype = int
            )

        pass


    def setup_opti_vars(self, 
                        scale_state = ca.vertcat(
                            [1, 1, 1, 1],
                            [1e-3, 1e-3, 1e-3],
                            [1e-2, 1e-2, 1e-2],
                            [1, 1, 1]
                            ), 
                        scale_control = ca.vertcat(
                            0.2,
                            0.2,
                            0.2,
                            [1e-2, 1e-2, 1e-2],
                            [1e-2, 1e-2, 1e-2],
                            [1, 1, 1],
                            [1e1, 1e1, 1e1]
                            ), 
                        scale_time = 1):
        
        # Time and timestep variables
        self.time = scale_time * self.opti.variable()
        self.dt = self.time / self.num_control_nodes
        
        # Initialize lists to store the interleaved variables for each time step
        state_list = []
        control_list = []
        tau_list = []
        lam_list = []
        mu_list = []

        for i in range(self.num_control_nodes + 1):  # For states, we need num_control_nodes + 1
            # State variables for each time step i
            state_t = scale_state * self.opti.variable(self.aircraft.num_states)
            state_list.append(state_t)

            if i < self.num_control_nodes:
                # Control variables for each time step i
                control_t = scale_control * self.opti.variable(self.aircraft.num_controls)
                tau_t = self.opti.variable(self.num_waypoints)
                lam_t = self.opti.variable(self.num_waypoints)
                mu_t = self.opti.variable(self.num_waypoints)

                # Append control and other variables only for the control nodes
                control_list.append(control_t)
                tau_list.append(tau_t)
                lam_list.append(lam_t)
                mu_list.append(mu_t)

        # Convert lists to matrices (CasADi MX form)
        self.state = ca.hcat(state_list)     # State matrix: [num_states x (num_control_nodes + 1)]
        print(state_list)
        self.control = ca.hcat(control_list) # Control matrix: [num_controls x num_control_nodes]
        self.tau = ca.hcat(tau_list)         # Tau matrix: [num_waypoints x num_control_nodes]
        self.lam = ca.hcat(lam_list)         # Lambda matrix: [num_waypoints x num_control_nodes]
        self.mu = ca.hcat(mu_list)           # Mu matrix: [num_waypoints x num_control_nodes]





    # def setup_opti_vars(self, 
    #                     scale_state, 
    #                     scale_control, 
    #                     scale_time):
        
        
    #     self.time = scale_time * self.opti.variable()
    #     self.dt = self.time /  self.num_control_nodes
    #     self.state = scale_state * self.opti.variable(
    #         self.aircraft.num_states, 
    #         self.num_control_nodes + 1
    #         )
    #     self.control = scale_control * self.opti.variable(
    #         self.aircraft.num_controls, 
    #         self.num_control_nodes)
        
    #     self.tau = self.opti.variable(self.num_waypoints, self.num_control_nodes)

    #     self.lam = self.opti.variable(self.num_waypoints, self.num_control_nodes)

    #     self.mu = self.opti.variable(self.num_waypoints, self.num_control_nodes)


    # def node(self, node, control, state, control_envelope, state_envelope):
    #     control_node = control[:, node]
    #     state_node = state[:, node]

    #     self.control_constraint(control_node, control_envelope)
    #     self.state_constraint(state_node, control_node, state_envelope)

    def control_constraint(self, control_node, control_envelope):
        self.opti.subject_to(
            self.opti.bounded(
                control_envelope.lb,
                control_node[:9],
                control_envelope.ub
                )
        )
        self.opti.subject_to(
            self.opti.bounded(
                np.zeros(control_node[9:].shape),
                control_node[9:],
                np.zeros(control_node[9:].shape)
                )
        )
        pass

    def state_constraint(self, state_node, next_state, control_node, state_envelope, dt):
        self.opti.subject_to(
            self.opti.bounded(
                state_envelope.alpha.lb,
                self.aircraft._alpha(state_node, control_node),
                state_envelope.alpha.ub
            )
        )

        self.opti.subject_to(
            self.opti.bounded(
                state_envelope.beta.lb,
                self.aircraft._beta(state_node, control_node),
                state_envelope.beta.ub
            )
        )

        self.opti.subject_to(
            self.opti.bounded(
                state_envelope.airspeed.lb,
                self.aircraft._airspeed(state_node, control_node),
                state_envelope.airspeed.ub
            )
        )
        self.opti.subject_to(next_state == self.aircraft.state_update(state_node, control_node, dt))
        pass

    def waypoint_constraint(
            self,
            mu:ca.MX,
            tau:ca.MX,
            lam:ca.MX,
            state_node,
            node:int,
            waypoint_node:int,
            switching_variable:List[int],
            waypoint_tolerance:float = 1e-2
            ):
        

        num_waypoints = self.num_waypoints# - 1
        
        waypoints = self.waypoints[:, 1:]

        if node > switching_variable[waypoint_node]:
            waypoint_node += 1

        self.opti.subject_to(self.opti.bounded(0, tau, waypoint_tolerance**2))
        self.opti.subject_to(self.opti.bounded(0, lam, 1))
        self.opti.subject_to(self.opti.bounded(0, mu, 1))
        
        for j in range(num_waypoints):
            diff = state_node[4:7] - waypoints[:,j]
            self.opti.subject_to(self.opti.bounded(0.0, lam[j, node] * (ca.dot(diff, diff) - tau[j, node]), 0.01))

        self.opti.subject_to(mu[:, node] - lam[:, node] - mu[:, node-1] == [0] * self.num_waypoints)
        
        for j in range(num_waypoints - 1):
            self.opti.subject_to(self.opti.bounded(0, mu[j + 1, node] - mu[j, node], 1))

    def loss(self, state:Optional[ca.MX] = None, control:Optional[ca.MX] = None, time:Optional[ca.MX] = None):
        return time ** 2


    def setup(self):
        _, time_guess = self.state_guess(self.trajectory)
        self.setup_opti_vars(scale_time=1/time_guess)
        self.opti.subject_to(self.time > 0)


        
    
        self.opti.subject_to(ca.dot(self.state[4:7, 0], self.state[4:7, 0]) == 0)
        self.opti.subject_to(ca.dot(self.state[10:, 0], self.state[10:, 0]) < 0.1)
        self.opti.subject_to(ca.dot(self.state[7:10, 0], self.state[7:10, 0]) == 50**2)


        self.opti.subject_to(self.mu[:, 0] == [1] * self.num_waypoints)

        waypoint_node = 0
        for node in range(self.num_control_nodes):
            self.state_constraint(self.state[:, node], self.state[:, node + 1], self.control[:, node], self.trajectory.state, self.dt)
            self.control_constraint(self.control[:, node], self.trajectory.control)
            self.waypoint_constraint(
                self.mu,
                self.tau,
                self.lam,
                self.state[:, node],
                node,
                waypoint_node,
                self.switching_variable
            )
            # print(node)

        self.opti.subject_to(self.state[4, -1] ==  self.waypoints[-1,:])
        self.opti.subject_to(self.mu[:, -1] == [0] * self.num_waypoints)

        self.initialise(self.tau, self.mu, self.lam, self.state, self.control, self.time)

        self.opti.minimize(self.loss(time = self.time))

    def plot_sparsity(self, ax:plt.axes):
        jacobian = self.opti.debug.value(ca.jacobian(self.opti.g,self.opti.x)).toarray()
        # print(jacobian)
        ax.spy(jacobian)
        plt.draw()
        plt.pause(10.0)

    def solve(
            self, 
            opts:dict = {
                        'ipopt': {
                            'max_iter': 10000,
                            'tol': 1e-2,
                            'acceptable_tol': 1e-2,
                            'acceptable_obj_change_tol': 1e-2,
                            'hessian_approximation': 'limited-memory'
                        },
                        'print_time': 10,
                        },
            save:bool = True
            ):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        self.opti.solver('ipopt', opts)
        self.opti.callback(lambda i: self.plot_sparsity(ax))
        
        sol = self.opti.solve()

        # TODO: Save functionality

        return sol



    def initialise(
            self,
            tau:ca.MX,
            mu:ca.MX,
            lam:ca.MX,
            state:ca.MX,
            control:ca.MX,
            time:ca.MX
            ):
        (tau_guess, lambda_guess, mu_guess) = self.waypoint_variable_guess()
        x_guess, time_guess = self.state_guess(self.trajectory)
        control_guess = np.zeros(control.shape)
        control_guess[6:9, :] = np.repeat([self.trajectory.aircraft.aero_centre_offset], control.shape[1], axis = 0).T
        print(x_guess)
        print("State update at initialisation", self.aircraft.state_update(x_guess[:,0], control_guess[:,0], 0.01))

        self.opti.set_initial(tau, tau_guess)
        self.opti.set_initial(lam, lambda_guess)
        self.opti.set_initial(mu, mu_guess)
        self.opti.set_initial(state, x_guess)
        self.opti.set_initial(time, time_guess)
        self.opti.set_initial(control, control_guess)
    

    def waypoint_variable_guess(self,):

        num_waypoints = self.num_waypoints
        lambda_guess = np.zeros((num_waypoints, self.num_control_nodes))
        tau_guess = np.zeros((num_waypoints, self.num_control_nodes))
        mu_guess = np.ones((num_waypoints, self.num_control_nodes))
        i_wp = 0
        for i in range(1, self.num_control_nodes):
            print(i, i_wp)
            # switch condition
            if i > self.switching_variable[i_wp]:
                i_wp += 1
            # progress variables
            if ((i_wp == 0) and (i + 1 >= self.switching_variable[0])) or i + 1 - self.switching_variable[i_wp-1] >= self.switching_variable[i_wp]:
                lambda_guess[i_wp, i] = 1

            for j in range(num_waypoints):
                if i + 1 < self.switching_variable[j]:
                    mu_guess[j, i] = 0

        return (tau_guess, lambda_guess, mu_guess)


    def state_guess(self, trajectory:TrajectoryConfiguration):
        # def x_guess(state:ca.MX, waypoints:np.ndarray, initial_pos:np.ndarray, velocity_guess:float):
        """
        Initial guess for the state variables.
        """
        state_dim = self.aircraft.num_states
        initial_pos = trajectory.waypoints.initial_position
        velocity_guess = trajectory.waypoints.default_velocity
        waypoints = self.waypoints[:, 1:]
        # convert initial pos to numpy array:
        if not isinstance(initial_pos, np.ndarray):
            initial_pos = initial_pos.full().flatten()
        if isinstance(velocity_guess, ca.MX) or isinstance(velocity_guess, ca.DM):
            velocity_guess = velocity_guess.full().flatten()
        
        x_guess = np.zeros((state_dim, self.num_control_nodes + 1))
        distance = self.cumulative_distances
        i_switch = np.array((self.num_control_nodes + 1) * np.array(distance) / distance[-1], dtype=int)

        direction_guess = (waypoints[:, 0] - initial_pos)
        vel_guess = velocity_guess *  direction_guess / np.linalg.norm(direction_guess)

        x_guess[:3, 0] = initial_pos
        x_guess[3:6, 0] = vel_guess

        z_flip = R.from_euler('x', 180, degrees=True)

        i_wp = 0
        for i in range(self.num_control_nodes + 1):
            # switch condition
            if i > i_switch[i_wp]:
                i_wp += 1
            if i_wp == 0:
                wp_last = initial_pos
            else:
                wp_last = waypoints[:,i_wp-1]
            wp_next = waypoints[:,i_wp]

            if i_wp > 0:
                interpolation = (i - i_switch[i_wp-1]) / (i_switch[i_wp] - i_switch[i_wp-1])
            else:
                interpolation = i / i_switch[0]

            # extend position guess
            pos_guess = (1 - interpolation) * wp_last + interpolation * wp_next
            x_guess[4:4+pos_guess.shape[0], i] = np.reshape(((1 - interpolation) * wp_last + interpolation * wp_next), (len(pos_guess),))

            direction = (wp_next - wp_last) / ca.norm_2(wp_next - wp_last)
            vel_guess = velocity_guess * direction
            x_guess[7:7 + vel_guess.shape[0], i] = np.reshape(velocity_guess * direction, (vel_guess.shape[0],))

            x_guess[:4, i] = (R.align_vectors(np.array(direction).T, [[1, 0, 0]])[0] * z_flip).as_quat()

        time_guess = distance[-1] / velocity_guess
        
        
        return x_guess, time_guess
    

def main():
    opti = ca.Opti()
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'),)
    trajectory_config = TrajectoryConfiguration(traj_dict)
    num_control_nodes = 9
    aircraft = Aircraft(traj_dict['aircraft'], model)
    problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    problem.setup()
    sol = problem.solve()
    return 1#sol

if __name__ == "__main__":
    main()