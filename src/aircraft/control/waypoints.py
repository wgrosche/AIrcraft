import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R
from typing import List


from aircraft.control.base import ControlNode, ControlProblem
from typing import List, Optional
from dataclasses import dataclass

from aircraft.utils.utils import TrajectoryConfiguration

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
            control = node.control,
            lam = lam,
            mu = mu,
            nu = nu
        )
    


class WaypointControl(ControlProblem):
    """
    Implements waypoint traversal via complementarity constraint.

    TODO: Modify so that it only implements waypoint constraints to create a family tree of control classes
    """
    def __init__(self, dynamics:ca.Function, trajectory_config:TrajectoryConfiguration, opts:Optional[dict] = {}):
        """
        To be implemented
        """
        max_control_nodes = opts.get('max_control_nodes', 100)
        # to calculate the num of nodes needed we use the dubins path from initialisation.py and add a tolerance
        num_nodes = 100
        self.waypoint_tolerance = 100
        self.trajectory = trajectory_config
        self.num_waypoints = len(trajectory_config.waypoints.waypoints)

        super().__init__(dynamics, num_nodes, opts)


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
    
    def _initialise_waypoints(self):
        """
        TODO: Implement
        """
        pass
    

def waypoint_distances(waypoints:np.ndarray, 
                       p_initial:np.ndarray, 
                       verbose:bool = False):
    """
    Given a set of waypoints, calculate the distance between each waypoint.

    Parameters
    ----------
    waypoints : np.array
        Array of waypoints. (d x n) where d is the dimension of the waypoints and n is the number of waypoints.
    p_initial : np.array
        Initial position of the aircraft.

    Returns
    -------
    distance : np.array
        Cumulative distance between waypoints.
    
    """
    print(len(waypoints[:, 0]))
    differences = np.diff(np.insert(waypoints, 0, p_initial[:len(waypoints[:, 0])], axis=1), axis=1)
    distances = np.linalg.norm(differences, axis=0)
    distance = np.cumsum(distances)

    if verbose: 
        print("Cumulative waypoint distances: ", distance)
    return distance



def setup_progress_vars(
        opti:ca.Opti, 
        num_nodes:int, 
        waypoints:np.ndarray, 
        state:ca.MX, 
        initial_pos:np.ndarray, 
        tolerance:float = 1e-2,
        verbose:bool = False
        ):
    
    if verbose:
        print('Setting up progress variables...')

    num_waypoints = waypoints.shape[0] - 1

    if num_waypoints == 0:
        return
    
    distance = waypoint_distances(waypoints, initial_pos)
    # switching variable (nodes at which we anticipate a change in objective (targeted waypoint))
    i_switch = np.array(num_nodes * np.array(distance) / distance[-1], dtype=int)

    if verbose:
        print('Switching nodes: ', i_switch)

    # Progress variables
    tau = opti.variable(num_waypoints, num_nodes)
    opti.subject_to(opti.bounded(0, tau, tolerance**2))

    lam = opti.variable(num_waypoints, num_nodes)
    opti.subject_to(opti.bounded(0, lam, 1))

    mu = opti.variable(num_waypoints, num_nodes)
    opti.subject_to(opti.bounded(0, mu, 1))
    opti.subject_to(mu[:, 0] == [1] * num_waypoints)

    # Initial guess for the progress variables
    lambda_guess = np.zeros((num_waypoints, num_nodes))
    tau_guess = np.zeros((num_waypoints, num_nodes))
    mu_guess = np.ones((num_waypoints, num_nodes))
    i_wp = 0
    for i in range(1, num_nodes):
        # switch condition
        if i > i_switch[i_wp]:
            i_wp += 1
        # progress variables
        if ((i_wp == 0) and (i + 1 >= i_switch[0])) or i + 1 - i_switch[i_wp-1] >= i_switch[i_wp]:
            lambda_guess[i_wp, i] = 1
        for j in range(num_waypoints):
            diff = state[4:4 + len(waypoints[:, j]), i] - waypoints[:,j]
            opti.subject_to(opti.bounded(0.0, lam[j, i] * (ca.dot(diff, diff) - tau[j, i]), 0.01))
        
        opti.subject_to(mu[:, i] - lam[:, i] - mu[:, i-1] == [0] * num_waypoints)

        for j in range(num_waypoints):
            if i + 1 < i_switch[j]:
                mu_guess[j, i] = 0

        for j in range(num_waypoints - 1):
            opti.subject_to(opti.bounded(0, mu[j + 1, i] - mu[j, i], 1))

    opti.subject_to(mu[:, -1] == [0] * num_waypoints)

    opti.set_initial(tau, tau_guess)
    opti.set_initial(lam, lambda_guess)      
    opti.set_initial(mu, mu_guess)

def x_guess(state:ca.MX, waypoints:np.ndarray, initial_pos:np.ndarray, velocity_guess:float):
    """
    Initial guess for the state variables.
    """
    # convert initial pos to numpy array:
    if not isinstance(initial_pos, np.ndarray):
        initial_pos = initial_pos.full().flatten()
    if isinstance(velocity_guess, ca.MX) or isinstance(velocity_guess, ca.DM):
        velocity_guess = velocity_guess.full().flatten()
    
    x_guess = np.zeros(state.shape)
    distance = waypoint_distances(waypoints, initial_pos, VERBOSE = True)
    i_switch = np.array(state.shape[1] * np.array(distance) / distance[-1], dtype=int)

    direction_guess = (waypoints[:, 0] - initial_pos)
    vel_guess = velocity_guess *  direction_guess / np.linalg.norm(direction_guess)

    x_guess[:3, 0] = initial_pos
    x_guess[3:6, 0] = vel_guess

    i_wp = 0
    for i in range(state.shape[1]):
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
        x_guess[:4, i] = R.align_vectors(np.array(direction).T, [[1, 0, 0]])[0].as_quat()

    time_guess = distance[-1] / velocity_guess
    
    
    return x_guess, time_guess