import numpy as np
import os
import sys
import json
import casadi as ca
from pathlib import Path
from liecasadi import Quaternion

# Set up paths and imports
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
sys.path.append(BASEPATH)

from src.dynamics_trimming import Aircraft, AircraftOpts
from src.utils import load_model, aero_to_state, TrajectoryConfiguration

# Wrap functions for aerodynamic derivatives
def wrap_aero(func, control):
    """
    Wraps a function so that it can take aerodynamic variables as input instead
    of the full state.
    """
    # define symbolic aerodynamic variables
    alpha = ca.MX.sym('alpha')
    beta = ca.MX.sym('beta')
    q_bar = ca.MX.sym('q_bar')

    wrapped_function = ca.Function(
        f'{func.name()}_aerodynamic',
        [q_bar, alpha, beta, control],
        [func(aero_to_state(q_bar, alpha, beta), control)]
    )

    return wrapped_function

def jacobian_wrapper(
        function:ca.Function,  
                     state:ca.MX, 
                     control:ca.MX,
                     aero:bool = False, 
                     finite_differences:bool = True, 
                     eps:float = 1e-6
                     ):
    """
    Jacobians with respect to the aerodynamic variables
    """
    # define symbolic aerodynamic variables
    alpha = ca.MX.sym('alpha')
    beta = ca.MX.sym('beta')
    q_bar = ca.MX.sym('q_bar')

    if aero:
        wrapped_function = wrap_aero(function, control)

        if finite_differences:
            wrapped_jac = ca.MX.zeros(wrapped_function.size1_out(0), 
                                      ca.vertcat(alpha, beta).size1())

            wrapped_jac[:, 0] = (wrapped_function(q_bar, alpha + eps, beta, control)\
                                - wrapped_function(q_bar, alpha, beta, control)) / eps
            wrapped_jac[:, 1] = (wrapped_function(q_bar, alpha, beta + eps, control)\
                                - wrapped_function(q_bar, alpha, beta, control)) / eps

        else:
            # generate symbolic jacobian
            wrapped_jac = ca.jacobian(wrapped_function(q_bar, alpha, beta, control), ca.vertcat(alpha, beta))

        # wrap as function so we can evaluate the jacobian at a chosen point
        wrapped_jac_function = ca.Function(
            f'{wrapped_function.name()}_jacobian',
            [q_bar, alpha, beta, control],
            [wrapped_jac]
            )
        
    else:
        wrapped_function = function
        if finite_differences:
            wrapped_jac = ca.MX.zeros(wrapped_function.size1_out(0), state.size1())

            for i in range(state.size1()):
                x_plus = ca.MX(state)
                x_plus[i] += eps
                wrapped_jac[:, i] = (wrapped_function(x_plus, control) - wrapped_function(state, control)) / eps

        else:
            # generate symbolic jacobian
            wrapped_jac = ca.jacobian(wrapped_function(state, control), state)
        # wrap as function so we can evaluate the jacobian at a chosen point
        wrapped_jac_function = ca.Function(
            f'{wrapped_function.name()}_jacobian',
            [state, control],
            [wrapped_jac]
            )

    return wrapped_jac_function


def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft
    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)

    f = aircraft.state_derivative(aircraft.state, aircraft.control)
    
    # Initialize state and control variables
    x0 = np.zeros(3)
    v0 = np.array([60, 0, 0])
    q0 = np.array([0, 0, 0, 1])
    omega0 = np.array([0, 0, 0])
    state = ca.vertcat(x0, v0, q0, omega0)
    control = np.zeros(aircraft.num_controls)
    control[-3:] = traj_dict['aircraft']['aero_centre_offset']

    

    # jacobian_sym = ca.jacobian(f(aircraft.state, aircraft.control), ca.vertcat(aircraft.state, aircraft.control))

    # wrap as function so we can evaluate the jacobian at a chosen point
    # wrapped_jac_function = ca.Function('dynamics_jacobian', [state, control], [jacobian_sym])

    constraints = ca.vertcat(
        aircraft.state,
        aircraft.control
        
    )

    lower_bounds = ca.vertcat(
        state,

        # control: aileron, elevator, thrust, com
        ca.vertcat(
            0,
            0,
            0, 
            -0.5, 
            -0.01, 
            -0.05
            )
        
    )

    upper_bounds = ca.vertcat(
        state,

        # control: aileron, elevator, thrust, com
        ca.vertcat(
            0,
            0,
            0,
            +0.5, 
            +0.01, 
            +0.05
            )

    )

    initial_guess = ca.vertcat(state, control)

    opts = {
        'ipopt': {
            'max_iter': 10000,
            # 'print_level': 2,
            'tol': 1e-4,
            'acceptable_tol': 1e-4,
            'acceptable_obj_change_tol': 1e-4,
            'hessian_approximation': 'limited-memory',  # Set Hessian approximation method here
        },
        'print_time': 10
    }

    desired_derivative = ca.vertcat(
                                v0,
                                np.array([0,0,0]),
                                ca.vertcat(omega0, 0),
                                np.array([0,0,0])
                                )
    nlp = {
        'x': ca.vertcat(aircraft.state, aircraft.control),
        'f': ca.dot(f - desired_derivative, f - desired_derivative),
        'g': constraints
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve for the trim condition
    sol = solver(x0=initial_guess, lbg=lower_bounds, ubg=upper_bounds)
    trim_state_control_aero = sol['x']

    print("Trim state and control:", trim_state_control_aero)

    trim_state = trim_state_control_aero[:aircraft.num_states]
    trim_control = trim_state_control_aero[aircraft.num_states:]

    print(f"Forces at trim: {aircraft.forces_frd(trim_state, trim_control)}")
    print(f"Moments at trim: {aircraft.moments_frd(trim_state, trim_control)}")
    # jac_print_exact = jacobian_wrapper(aircraft._moments_frd, aircraft.state, aircraft.control, aero = True, finite_differences=False)(aircraft._qbar(trim_state, trim_control), 
    #                          aircraft._alpha(trim_state, trim_control), 
    #                          aircraft._beta(trim_state, trim_control), 
    #                          trim_control)
    
    # jac_print = moments_aero(aircraft._qbar(trim_state, trim_control), 
    #                          aircraft._alpha(trim_state, trim_control), 
    #                          aircraft._beta(trim_state, trim_control), 
    #                          trim_control)
    
    # print(f"Initial Jacobian: {moments_aero(aircraft._qbar(state, control), aircraft._alpha(state, control), aircraft._beta(state, control), control)}")

    # print(f"Exact Jacobian at trim: {jac_print_exact}")

    # print(f"Finite Diffs Jacobian at trim: {jac_print}")

if __name__ == "__main__":
    main()