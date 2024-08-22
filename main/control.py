""" 
TODO - Scale Variables
TODO - Speed up sim
TODO - Improve Convergence
"""

import numpy as np
import torch
import os
import pandas as pd
import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
matplotlib.use('TkAgg')
import sys
import h5py
import json
from pylab import spy

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASEPATH)
sys.path.append(BASEPATH)

from src.dynamics import Aircraft, load_model
from src.waypoints import waypoint_distances, setup_progress_vars, x_guess
# from src.preprocessing import get_airplane_params
from src.models import ScaledModel, MiniModel
# from src.visualisation import plot
from src.plotting import plot, debug
from src.utils import Control, State, AircraftParameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))

DATAPATH = os.path.join(BASEPATH, 'data')
PARAMSPATH = os.path.join(DATAPATH, 'parameters')
NETWORKPATH = os.path.join(DATAPATH, 'networks', 'model-dynamics.pth')
USING_WAYPOINTS = True

# TOLERANCE
TOLERANCE = 1e-4

# NUM NODES
NUM_NODES = 30

# DEBUG MODE
DEBUG = False

aircraft_params = AircraftParameters(params = json.load(open(os.path.join(
    BASEPATH, 
    'data', 
    'glider', 
    'glider_fs.json'
    ))))



def check_constraints(opti:ca.Opti):
    """ Function to evaluate the constraints and their Jacobian """
    try:
        g = opti.g
        g_val = opti.debug.value(g)
        jac_g_val = ca.jacobian(opti.g, opti.x) 
        
        print("Constraints values at initial guess:")
        # print(g_val)
        
        print("Jacobian of constraints at initial guess:")
        # print(jac_g_val)
    except Exception as e:
        print(f"Error while evaluating constraints: {e}")
    

def flight_envelope_constraints(
        opti: ca.Opti, 
        envelope: dict, 
        state: ca.MX, 
        control: ca.MX, 
        aircraft: Aircraft
        ):
    """
    Apply flight envelope constraints for each node in the state and control 
    variables.

    Parameters:
    - opti: The optimization problem.
    - envelope: Dictionary containing envelope limits for 'airspeed', 'alpha', 
                and 'beta'.
    - state: num_states x num_nodes dimensional array representing the state.
    - control: num_controls x num_nodes dimensional array representing the 
                control.
    - aircraft: Aircraft object with methods to compute airspeed, alpha, 
                and beta.

    """
    # Get the number of nodes
    _, num_nodes = control.shape

    for node in range(num_nodes):
        # Extract state and control for the current node
        state_node = state[:, node]
        control_node = control[:, node]

        # Limit airspeed
        airspeed = ca.dot(aircraft._v_frd_rel(state_node, control_node), 
                          aircraft._v_frd_rel(state_node, control_node))
        opti.subject_to(
            opti.bounded(
                envelope['airspeed'][0],
                airspeed,
                envelope['airspeed'][1]
            )
        )

        # Limit angle of attack
        alpha = aircraft._alpha(state_node, control_node)
        opti.subject_to(
            opti.bounded(
                envelope['alpha'][0],
                alpha,
                envelope['alpha'][1]
            )
        )

        # Limit angle of yaw
        beta = aircraft._beta(state_node, control_node)
        opti.subject_to(
            opti.bounded(
                envelope['beta'][0],
                beta,
                envelope['beta'][1]
            )
        )

    return None


def control_constraints(
        opti:ca.Opti, 
        control:ca.MX, 
        control_limits:dict
        ):
    """
    opti: casadi.Opti object
    control: control input vector
    control_limits: dictionary containing the limits for each control input
    """
    # control constraints
    opti.subject_to(
        opti.bounded(
            control_limits['aileron'][0],
            control[0,:],
            control_limits['aileron'][1]
            )
        )
    opti.subject_to(
        opti.bounded(
            control_limits['elevator'][0],
            control[1,:],
            control_limits['elevator'][1]
            )
        )
    opti.subject_to(
        opti.bounded(
            control_limits['rudder'][0],
            control[2,:],
            control_limits['rudder'][1]
            )
        ) # TODO: IMPLEMENT CONTROL LIMITS
    opti.subject_to(control[3:6,:] == control_limits['throttle'][0])
        # opti.bounded(
        #     control_limits['throttle'][0],
        #     control[3:6,:],
        #     control_limits['throttle'][1]
        #     )
        # )
    
    opti.subject_to(control[6:9,:] == [0,0,0]
        # opti.bounded(
        #     [0, 0, 0],
        #     control[6:9,:],
        #     [0, 0, 0]
        #     )
        )
    
    # opti.subject_to(
        # opti.bounded(
        #     [0, 0, 0],
        #     control[6:9,:],
        #     [0, 0, 0]
        #     )
        # )
    
    opti.subject_to(control[9:12,:] == [0,0,0])
        # opti.bounded(
        #     [0, 0, 0],
        #     control[9:12,:],
        #     [0, 0, 0]
        #     )
        # )

    return None


def main():
    initial_state = State(
    orientation = np.array([1, 0, 0, 0]), 
    velocity = np.array([60, 0, 0])
        )

    initial_controls = Control(
        centre_of_mass=aircraft_params['aero_centre_offset']
        )

    waypoints = np.array([[100]])
    opti = ca.Opti()

    model = load_model()
    
    aircraft = Aircraft(aircraft_params, model)

    time = opti.variable()
    dt = time/NUM_NODES
    opti.subject_to(time >= 0)

    scale_state = ca.repmat(ca.vertcat(
        [1, 1, 1, 1],
        [1e-3, 1e-3, 1e-3],
        [1e-2, 1e-2, 1e-2],
        [1, 1, 1]
        ), 1, NUM_NODES+1)
    
    scale_control = ca.repmat(ca.vertcat(
        0.2,
        0.2,
        0.2,
        [1e-2, 1e-2, 1e-2],
        [1e-2, 1e-2, 1e-2],
        [1, 1, 1],
        [1e1, 1e1, 1e1]
        ), 1, NUM_NODES)

    state = scale_state * opti.variable(aircraft.num_states, NUM_NODES + 1)
    control = scale_control * opti.variable(aircraft.num_controls, NUM_NODES)

    flight_envelope_constraints(
        opti, 
        flight_envelope, 
        state, 
        control, 
        aircraft
        )
    
    control_constraints(
        opti, 
        control, 
        control_limits
        )
    
    setup_progress_vars(
        opti, 
        NUM_NODES, 
        waypoints, 
        state[4:7, :], 
        initial_state[4:7], 
        TOLERANCE
        )

    # trajectory guess
    trajectory_guess, time_guess = x_guess(aircraft, 
                                           NUM_NODES + 1, 
                                           waypoints.T, 
                                           initial_state[4:7], 
                                           50)


    opti.set_initial(state, trajectory_guess)
    opti.set_initial(time, time_guess)
    opti.set_initial(control, ca.repmat(initial_controls, 1, NUM_NODES))

    # TODO: adapt for ndim waypoints
    opti.subject_to(state[4, -1] ==  waypoints[-1,:])
    
    opti.subject_to(ca.dot(state[4:7, 0], state[4:7, 0]) == 0)
    opti.subject_to(ca.dot(state[10:, 0], state[10:, 0]) < 0.1)
    opti.subject_to(ca.dot(state[7:10, 0], state[7:10, 0]) == 70**2)
    for i in range(NUM_NODES + 1):
        opti.subject_to(ca.dot(state[:4, i], state[:4, i]) == 1)

    opti.subject_to(control[-3:, :] == initial_controls[-3:])

    cost_fn = time ** 2
    opti.minimize(cost_fn)

    
    for k in range(NUM_NODES):

        opti.subject_to(state[:,k+1] == aircraft.state_update(state[:,k], control[:,k], dt))

    opts = {
        'ipopt': {
            'max_iter': 10000,
            'tol': 1e-2,
            'acceptable_tol': 1e-2,
            'acceptable_obj_change_tol': 1e-2,
            'hessian_approximation': 'limited-memory'
        },
        'print_time': 10,
    }

    opti.solver('ipopt', opts)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    filepath = os.path.join(DATAPATH, "trajectories", "physical_solution_new.h5")
    
    # if the file exists, delete it
    if os.path.exists(filepath):
        os.remove(filepath)
    if DEBUG:
        # opti.callback(lambda i: debug(
        #     opti, 
        #     opti.debug.value(state), 
        #     opti.debug.value(control), 
        #     aircraft, 
        #     opti.debug.value(opti.debug.g)
        #     ))
        opti.callback(lambda i: spy(opti.debug.value(ca.jacobian(opti.g,opti.x))))
    else:
        opti.callback(lambda i: plot(
            ax, 
            opti.debug.value(state), 
            waypoints, 
            i, 
            ax2, 
            opti.debug.value(control), 
            opti.debug.value(time), 
            None, 
            filepath = None
            ))
        
    sol = opti.solve()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)

    plot(
        ax, 
        sol.value(state),
        waypoints, 
        1, 
        ax2, 
        sol.value(control), 
        sol.value(time), 
        None, 
        filepath = None
        )
    
    spy(sol.value(ca.jacobian(opti.g,opti.x)))

    pickle_filename = "optimisation_control.pkl"

    with open(pickle_filename, 'wb') as f:
        pickle.dump((sol, state, time), f)

    print(f"Data has been saved to {pickle_filename}")

import pickle

if __name__=="__main__":
    main()

    