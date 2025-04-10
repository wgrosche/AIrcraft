from aircraft.control.aircraft import TrajectoryConfiguration
from pathlib import Path
from aircraft.config import NETWORKPATH
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
import numpy as np
from tqdm import tqdm
import casadi as ca
import json
from aircraft.control.quadrotor import QuadrotorControl
from aircraft.dynamics.quadrotor import Quadrotor
from aircraft.control.initialisation import DubinsInitialiser
from aircraft.plotting.plotting import TrajectoryData, TrajectoryPlotter
import matplotlib.pyplot as plt

def main():
    """
    Minimal test of quadrotor control class
    """
    
    quad = Quadrotor()
    # create guess with dubins initialiser
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    # poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    
    

    trajectory_config = TrajectoryConfiguration(traj_dict)
    # aircraft_config = trajectory_config.aircraft

    # opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)
    # aircraft = Aircraft(opts = opts)

    plotter = TrajectoryPlotter(quad)
            
    
    initialiser = DubinsInitialiser(trajectory_config)
    initial_guess = initialiser.guess

    initial_guess = np.zeros((101, 13))
    initial_guess[:, 10] = 1
    initial_guess = np.concat((initial_guess, np.ones((initial_guess.shape[0],quad.num_controls ))), axis = 1).T
    num_nodes = initial_guess.shape[1] - 1
    i = 0
    state_list = initial_guess
    state = ca.DM(initial_guess[:quad.num_states, 0])
    time_intervals = [5/num_nodes for _ in range(num_nodes)]#initialiser.time_intervals
    # dt = 5/200
    for interval in tqdm(time_intervals, desc = 'Simulating Trajectory:'):
        if np.isnan(state[0]):
            print('Aircraft crashed')
            break
        else:
            state_list[:quad.num_states, i] = state.full().flatten()
            state = quad.state_update(state, initial_guess[quad.num_states:, i], interval)
        i += 1
    
    print(state_list[:, 0])
    print(quad.num_states)
    print(state_list[:quad.num_states, :].shape)
    # plotter.plot(TrajectoryData(state = state_list[:quad.num_states, :], control = state_list[quad.num_states:, :]))
    # plt.show(block = True)
    # return None

    print(num_nodes)
    control_problem = QuadrotorControl(quad.state_update, num_nodes)
    print(initialiser.guess.T[:control_problem.state_dim, 0])
    # control_problem.setup(initial_guess, target = initial_guess.T[:3, -1])
    control_problem.setup(state_list, target = state_list.T[:3, -1])

    try:
        sol = control_problem.solve()
    except Exception as e:
        print(f"Solver failed with error: {e}")
        
        # Try to access debug information
        try:
            debug = control_problem.opti.debug
            
            # Check for NaN values in the state and control
            state_vals = debug.value(control_problem.state)
            control_vals = debug.value(control_problem.control)
            
            if np.any(np.isnan(state_vals)):
                print("NaN values detected in state variables")
                nan_indices = np.where(np.isnan(state_vals))
                print(f"NaN state indices: {list(zip(*nan_indices))}")
                
            if np.any(np.isnan(control_vals)):
                print("NaN values detected in control variables")
                nan_indices = np.where(np.isnan(control_vals))
                print(f"NaN control indices: {list(zip(*nan_indices))}")
            
            # For constraints, use proper CasADi methods to access elements
            # Instead of iterating over debug.g directly, use CasADi's methods
            g = debug.g
            n_g = g.numel()  # Get number of elements
            print(f"Length of constraint defs: {(control_problem.constraint_descriptions)}")
            print(f"Number of constraints: {n_g}")
            
            # Check each constraint individually
            for i in range(n_g):
                try:
                    # Use CasADi's element access method
                    g_i = g[i]
                    value = debug.value(g_i)
                    if i >= len(control_problem.constraint_descriptions):
                        break
                    if np.isnan(value):
                        print(f"NaN detected in constraint {i}, {control_problem.constraint_descriptions[i]}")
                    # distinguish types
                    elif control_problem.constraint_descriptions[i][2] == '==':
                        if abs(value) > 1e-6:  # Check for violations
                            print(f"Constraint {i} {control_problem.constraint_descriptions[i]} violated: {value}")
                    elif control_problem.constraint_descriptions[i][2] == '<':
                        if value > 0:
                            print(f"Constraint {i} {control_problem.constraint_descriptions[i]} violated: {value}")
                    elif control_problem.constraint_descriptions[i][2] == '>':
                        if value < 0:
                            print(f"Constraint {i} {control_problem.constraint_descriptions[i]} violated: {value}")
                except Exception as inner_e:
                    print(f"Could not evaluate constraint {i}: {inner_e}")
                    
        except Exception as debug_e:
            print(f"Could not access debug information: {debug_e}")





if __name__ =="__main__":
    main()