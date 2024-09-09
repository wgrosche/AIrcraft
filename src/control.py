import casadi as ca
import numpy as np
from typing import List

import os
import sys

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASEPATH)
sys.path.append(BASEPATH)

from src.dynamics import Aircraft, load_model
from collections import namedtuple
from scipy.spatial.transform import Rotation as R



ControlEnvelope = namedtuple('control', ['lb', 'ub'])
StateEnvelope
Envelope = namedtuple('envelope', ['control', 'state'], [ControlEnvelope, StateEnvelope])


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
        waypoints:np.ndarray,
        num_control_nodes: int

        ):
        self.opti = opti
        self.aircraft = aircraft
        self.num_control_nodes = num_control_nodes
        self.waypoints = waypoints

        self.num_waypoints = waypoints.shape[0] - 1
        self.cumulative_distances = cumulative_waypoint_distances(waypoints)
        self.switching_variable = np.array(
            num_control_nodes * np.array(self.cumulative_distances) 
            / self.cumulative_distances[-1], dtype = int
            )

        pass

    def setup_opti_vars(self, 
                        scale_state, 
                        scale_control, 
                        scale_time):
        
        scale_state = ca.repmat(ca.vertcat(
            [1, 1, 1, 1],
            [1e-3, 1e-3, 1e-3],
            [1e-2, 1e-2, 1e-2],
            [1, 1, 1]
            ), 1, self.num_control_nodes+1)
        
        scale_control = ca.repmat(ca.vertcat(
            0.2,
            0.2,
            0.2,
            [1e-2, 1e-2, 1e-2],
            [1e-2, 1e-2, 1e-2],
            [1, 1, 1],
            [1e1, 1e1, 1e1]
            ), 1, self.num_control_nodes)
        
        self.time = scale_time * self.opti.variable()
        self.state = scale_state * self.opti.variable(
            self.aircraft.num_states, 
            self.num_control_nodes + 1
            )
        self.control = scale_control * self.opti.variable(
            self.aircraft.num_controls, 
            self.num_control_nodes)
        
        self.tau = self.opti.variable(self.num_waypoints, self.num_control_nodes)

        self.lam = self.opti.variable(self.num_waypoints, self.num_control_nodes)

        self.mu = self.opti.variable(self.num_waypoints, self.num_control_nodes)


    def node(self, node, control, state, control_envelope, state_envelope):
        control_node = control[:, node]
        state_node = state[:, node]

        self.control_constraint(control_node, control_envelope)
        self.state_constraint(state_node, control_node, state_envelope)

    def control_constraint(self, control_node, control_envelope):
        self.opti.subject_to(
            self.opti.bounded(
                control_envelope.lb,
                control_node,
                control_envelope.ub
                )
        )
        pass

    def state_constraint(self, state_node, control_node, state_envelope):
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
        if node > switching_variable[waypoint_node]:
            waypoint_node += 1

        self.opti.subject_to(self.opti.bounded(0, tau, waypoint_tolerance**2))
        self.opti.subject_to(self.opti.bounded(0, lam, 1))
        self.opti.subject_to(self.opti.bounded(0, mu, 1))
        
        for j in range(self.num_waypoints):
            diff = state_node[4:4 + len(self.waypoints[:, j]), node] - self.waypoints[:,j]
            self.opti.subject_to(self.opti.bounded(0.0, lam[j, node] * (ca.dot(diff, diff) - tau[j, node]), 0.01))

        self.opti.subject_to(mu[:, node] - lam[:, node] - mu[:, node-1] == [0] * self.num_waypoints)
        
        for j in range(self.num_waypoints - 1):
            self.opti.subject_to(self.opti.bounded(0, mu[j + 1, node] - mu[j, node], 1))


    def setup(self, mu):

        self.opti.subject_to(mu[:, 0] == [1] * self.num_waypoints)
        self.opti.subject_to(mu[:, -1] == [0] * self.num_waypoints)
        waypoint_node = 0
        for node in range(self.num_control_nodes):
            self.state_constraint(self.state[:, node], self.control[:, node], self.envelope.state)
            self.control_constraint(self.control[:, node], self.envelope.control)
            self.waypoint_constraint(
                self.mu,
                self.tau,
                self.lam,
                self.state[:, node],
                node,
                waypoint_node,
                self.switching_variable
            )


    def initialise(
            self,
            tau:ca.MX,
            mu:ca.MX,
            lam:ca.MX
            ):
        (tau_guess, lambda_guess, mu_guess) = self.waypoint_variable_guess()

        self.opti.set_initial(tau, tau_guess)
        self.opti.set_initial(lam, lambda_guess)
        self.opti.set_initial(mu, mu_guess)

    

    def waypoint_variable_guess(self,):
        lambda_guess = np.zeros((self.num_waypoints, self.num_control_nodes))
        tau_guess = np.zeros((self.num_waypoints, self.num_control_nodes))
        mu_guess = np.ones((self.num_waypoints, self.num_control_nodes))
        i_wp = 0
        for i in range(1, self.num_control_nodes):
            # switch condition
            if i > self.switching_variable[i_wp]:
                i_wp += 1
            # progress variables
            if ((i_wp == 0) and (i + 1 >= self.switching_variable[0])) or i + 1 - self.switching_variable[i_wp-1] >= self.switching_variable[i_wp]:
                lambda_guess[i_wp, i] = 1

            for j in range(self.num_waypoints):
                if i + 1 < self.switching_variable[j]:
                    mu_guess[j, i] = 0

        return (tau_guess, lambda_guess, mu_guess)


    def state_guess(self):
        # def x_guess(state:ca.MX, waypoints:np.ndarray, initial_pos:np.ndarray, velocity_guess:float):
        """
        Initial guess for the state variables.
        """
        # convert initial pos to numpy array:
        if not isinstance(initial_pos, np.ndarray):
            initial_pos = initial_pos.full().flatten()
        if isinstance(velocity_guess, ca.MX) or isinstance(velocity_guess, ca.DM):
            velocity_guess = velocity_guess.full().flatten()
        
        x_guess = np.zeros(self.state.shape)
        distance = self.cumulative_distances(self.waypoints, initial_pos, VERBOSE = True)
        i_switch = np.array(self.state.shape[1] * np.array(distance) / distance[-1], dtype=int)

        direction_guess = (self.waypoints[:, 0] - initial_pos)
        vel_guess = velocity_guess *  direction_guess / np.linalg.norm(direction_guess)

        x_guess[:3, 0] = initial_pos
        x_guess[3:6, 0] = vel_guess

        i_wp = 0
        for i in range(self.state.shape[1]):
            # switch condition
            if i > i_switch[i_wp]:
                i_wp += 1
            if i_wp == 0:
                wp_last = initial_pos
            else:
                wp_last = self.waypoints[:,i_wp-1]
            wp_next = self.waypoints[:,i_wp]

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
            x_guess[:4, i] = R.align_vectors(np.array(direction).T, [[1, 0, 0]])[0].as_quat()

        time_guess = distance[-1] / velocity_guess
        
        
        return x_guess, time_guess