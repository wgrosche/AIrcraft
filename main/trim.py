"""
Determines location of Centre of Mass such that the aircraft is statically stable
for a given trim condition.

"""

import casadi as ca
import os
import numpy as np
import torch
import sys
import pandas as pd
import json

# Define the base path and append it to sys.path
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)

# Import necessary modules from your project
from src.dynamics import Aircraft#, LinearisedAircraft
from src.models import ScaledModel
import matplotlib.pyplot as plt

def setup_model(aircraft_params):
    
    # Load the neural network model and its scaler
    DEVICE = 'cpu'
    NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
    checkpoint = torch.load(os.path.join(NETWORKPATH, 'model-dynamics.pth'), map_location=DEVICE)
    scaler = checkpoint['input_mean'], checkpoint['input_std'], checkpoint['output_mean'], checkpoint['output_std']
    model = ScaledModel(5, 6, scaler=scaler)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize the aircraft model
    aircraft = Aircraft(aircraft_params, model)
    return aircraft

import casadi as ca
import numpy as np
import json
import os

def main():
    aircraft_params = json.load(open(os.path.join(BASEPATH, 'data', 'glider', 'glider_sim_freestream.json')))

    goal_vel = 50
    aircraft = setup_model(aircraft_params)

    state = ca.DM(np.zeros(aircraft.num_states))
    state[7] = goal_vel
    state[3] = 1

    control = ca.DM(np.zeros(aircraft.num_controls))

    def residuals(state, control):
        aero_loads = aircraft._loads_wf(state, control)
        return aero_loads

    decision_vars = ca.vertcat(aircraft.state, aircraft.control)

    constraints = ca.vertcat(
        aircraft.airspeed - goal_vel,
        aircraft.angular_velocity,
        aircraft.throttle
    )

    nlp = {
        'x': decision_vars,
        'f': ca.sum1(residuals(aircraft.state, aircraft.control)**2),
        'g': constraints
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp)

    lbg = ca.DM.zeros(constraints.size())
    ubg = ca.DM.zeros(constraints.size())

    # Set a small tolerance for airspeed constraint
    lbg[0] = -10.0  # Allow airspeed to deviate slightly from target
    ubg[0] = 10.0   # Allow airspeed to deviate slightly from target
    # ubg[1] = 5
    # Provide initial guesses
    initial_guess = ca.vertcat(state, control)

    # Solve for the trim condition
    sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg)
    trim_state_control_aero = sol['x']

    print("Trim state and control:", trim_state_control_aero)

    # # Extract the trim state and control
    # trim_state = trim_state_control_aero[:aircraft.num_states]
    # trim_control = trim_state_control_aero[aircraft.num_states:]

    # # Calculate Cm at the trim condition
    # Cm_trim = aircraft.Cm(trim_state, trim_control)
    # # Symbolic variables for state and control
    # sym_state = ca.MX.sym('state', aircraft.num_states)
    # sym_control = ca.MX.sym('control', aircraft.num_controls)

    # sym_alpha = ca.atan2(-sym_state[2], sym_state[0])
    # # Define a function for the derivative of Cm with respect to alpha (atan2(-w, u))
    # dAlpha_dState = ca.jacobian(aircraft.alpha_function(sym_state), sym_state)
    # dCM_dState = ca.jacobian(aircraft.Cm(sym_state, sym_control), sym_state)
    # dCm_dAlpha = ca.mtimes(dCM_dState, ca.pinv(dAlpha_dState))

    
    # # Define the gradient of dCm/dAlpha with respect to the center of mass (control[4:])
    # dCm_dAlpha_func = ca.Function('dCm_dAlpha_func', [sym_state, sym_control], [dCm_dAlpha])
    # grad_dCm_dAlpha = ca.jacobian(dCm_dAlpha_func(sym_state, sym_control), sym_control)

    # # evaluate the gradient at the trim condition
    # dCm_dAlpha_func_evaluated = dCm_dAlpha_func(trim_state, trim_control)
    # print("Gradient of Cm with respect to alpha:", dCm_dAlpha_func_evaluated)

    # # Gradient at the trim condition
    # grad_dCm_dAlpha_evaluated = ca.Function('grad', [sym_state, sym_control], [grad_dCm_dAlpha])(trim_state, trim_control).full().flatten()
    # print("Gradient of Cmalpha with respect to center of mass:", grad_dCm_dAlpha_evaluated)



    # # Gradient descent parameters
    # learning_rate = 0.01
    # max_iterations = 0
    # CmAlphaValues = []
    # trim_control_list = [trim_control.full().flatten()]

    # # Gradient descent loop to adjust center of mass for stability
    # for _ in range(max_iterations):
    #     # Evaluate the gradient
    #     gradient_evaluated = ca.Function('grad', [sym_state, sym_control], [grad_dCm_dAlpha])(trim_state, trim_control)[4:].full().flatten()
        
    #     # Update the center of mass using gradient descent
    #     trim_control[4:] -= learning_rate * gradient_evaluated
    #     trim_control_list.append(trim_control.full().flatten())

    #     # Recalculate the trim condition with the updated center of mass
    #     initial_guess = ca.vertcat(trim_state, trim_control)
    #     sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg, verbose = False)
    #     trim_state_control_aero = sol['x']
    #     trim_state = trim_state_control_aero[:aircraft.num_states]
    #     trim_control = trim_state_control_aero[aircraft.num_states:]

    #     # Recalculate dCm/dAlpha at the new trim condition
    #     dCm_dAlpha_value = dCm_dAlpha_func(trim_state, trim_control)
    #     CmAlphaValues.append(dCm_dAlpha_value.full().flatten())
    #     print("CmAlpha:", dCm_dAlpha_value)
        

    #     # Check for static stability
    #     if dCm_dAlpha_value < 0:
    #         print("The aircraft is statically stable.")
    #         break
    # else:
    #     print("Unable to achieve static stability within the maximum number of iterations.")
    # # Plot the progress of CmAlpha values
    # plt.figure(figsize=(10, 6))
    # plt.plot(CmAlphaValues)
    # plt.xlabel('Iteration')
    # plt.ylabel('CmAlpha')
    # plt.title('Progress of CmAlpha during Gradient Descent')
    # plt.grid(True)
    # plt.show()

    # # Plot the progress of the center of mass
    # trim_control_list = np.array(trim_control_list)
    # plt.figure(figsize=(10, 6))
    # plt.plot(trim_control_list[:, 4:])
    # plt.xlabel('Iteration')
    # plt.ylabel('Center of Mass Offset')
    # plt.title('Progress of Center of Mass Offset during Gradient Descent')
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()

# def main():
#     aircraft_params = json.load(open(os.path.join(BASEPATH, 'data', 
#                             'glider', 'glider_sim_freestream.json')))

#     goal_vel = 100
#     aircraft = setup_model(aircraft_params)

#     state = ca.DM(np.array([
#         0, 0, 0,
#         goal_vel, 0, 0,
#         0, 0, 0,
#         0, 0, 0
#     ]))

#     control = ca.DM(np.array([0, 0, 0, 0, aircraft_params["mass"], 
#                               *aircraft_params['aero_centre_offset']]))

#     def residuals(state, control):
#         aero_loads = aircraft.loads_function(state, control)
#         return aero_loads

#     decision_vars = ca.vertcat(aircraft.state, aircraft.control)

#     constraints = ca.vertcat(
#         aircraft.airspeed - goal_vel,
#         aircraft.mass - aircraft_params["mass"],
#         ca.vertcat(aircraft.p, aircraft.q, aircraft.r),
#         aircraft.control[3]
#     )

#     lbg = ca.DM.zeros(constraints.size())
#     ubg = ca.DM.zeros(constraints.size())

#     lbg[0] = -0.1
#     ubg[0] = 0.1
#     ubg[1] = 5


#     nlp = {
#         'x': decision_vars,
#         'f': ca.sum1(residuals(aircraft.state, aircraft.control)**2),
#         'g': constraints
#     }

#     solver = ca.nlpsol('solver', 'ipopt', nlp)

    

#     # Provide initial guesses
#     initial_guess = ca.vertcat(state, control)

#     # Solve for the trim condition and center of mass location
#     sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg)
#     trim_state_control_aero = sol['x']

# # Extract the trim state, control, and aero center offset
# trim_state = trim_state_control_aero[:12]
# trim_control = trim_state_control_aero[12:17]
# trim_aero_centre_offset = trim_state_control_aero[17:]

# print("Trim state and control:", trim_state_control_aero)
# print("Trim aero center offset:", trim_aero_centre_offset)

# # Update the aircraft parameters with the new center of mass location
# # aircraft_params["aero_centre_offset"] = ca.DM(trim_aero_centre_offset).toarray().flatten().tolist()
# # aircraft.update_params(aircraft_params)

# # Define the state and control symbolic variables
# state_sym = aircraft.state
# control_sym = aircraft.control

# # Compute the Jacobian matrices at the trim condition
# state_derivative_eq = aircraft.state_derivative_function(state_sym, control_sym)
# A = ca.jacobian(state_derivative_eq, state_sym)
# B = ca.jacobian(state_derivative_eq, control_sym)

# A_eval = ca.Function('A_eval', [state_sym, control_sym], [A])
# B_eval = ca.Function('B_eval', [state_sym, control_sym], [B])
# def objective(state, control):
#     evs = np.linalg.eigvals(A_eval(trim_state, ca.vertcat(trim_control, trim_aero_centre_offset)).full())
#     return 
# # Evaluate the Jacobian matrices at the trim condition
# A_eval = ca.Function('A_eval', [state_sym, control_sym], [A])(trim_state, ca.vertcat(trim_control, trim_aero_centre_offset))
# B_eval = ca.Function('B_eval', [state_sym, control_sym], [B])(trim_state, ca.vertcat(trim_control, trim_aero_centre_offset))

# print("A_eval:", A_eval)
# print("B_eval:", B_eval)

# # Compute the eigenvalues of the Jacobian matrix A
# eigenvalues = np.linalg.eigvals(A_eval.full())

# # Analyze the eigenvalues
# print("Eigenvalues:", eigenvalues)

# # Check stability
# stable = np.all(np.real(eigenvalues) < 0)
# if stable:
#     print("The system is locally stable.")
# else:
#     print("The system is not locally stable.")
