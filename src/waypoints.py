import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R

def waypoint_distances(waypoints, p_initial, VERBOSE = False):
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

    if VERBOSE: 
        print("Waypoint distances: ", distance) 
    return distance



def setup_progress_vars(opti, num_nodes, waypoints, X, initial_pos, tolerance = 1e-2):
    print('Setting up progress variables...')
    num_waypoints = waypoints.shape[1] -1

    if num_waypoints == 0:
        return
    distance = waypoint_distances(waypoints, initial_pos)
    # switching variable (nodes at which we anticipate a change in objective (targeted waypoint))
    i_switch = np.array(num_nodes * np.array(distance) / distance[-1], dtype=int)
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
            diff = X[4:4 + len(waypoints[:, j]), i] - waypoints[:,j]
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

def x_guess(aircraft, num_nodes, waypoints, initial_pos, velocity_guess):
    """
    Initial guess for the state variables.
    """
    # convert initial pos to numpy array:
    if not isinstance(initial_pos, np.ndarray):
        initial_pos = initial_pos.full().flatten()
    if isinstance(velocity_guess, ca.MX) or isinstance(velocity_guess, ca.DM):
        velocity_guess = velocity_guess.full().flatten()
    
    x_guess = np.zeros((aircraft.num_states, num_nodes))
    distance = waypoint_distances(waypoints, initial_pos, VERBOSE = True)
    i_switch = np.array(num_nodes * np.array(distance) / distance[-1], dtype=int)

    direction_guess = (waypoints[:, 0] - initial_pos)
    vel_guess = velocity_guess *  direction_guess / np.linalg.norm(direction_guess)

    x_guess[:3, 0] = initial_pos
    x_guess[3:6, 0] = vel_guess

    i_wp = 0
    for i in range(num_nodes):
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
        # x_guess[10:, i] = np.zeros_like(x_guess[10:, i])
    time_guess = distance[-1] / velocity_guess
    # print('Initial guess for state variables:', x_guess)
    
    
    return x_guess, time_guess