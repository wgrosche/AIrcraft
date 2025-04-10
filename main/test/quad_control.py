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


def debug_jacobian(control_problem):
    """Extract and analyze the Jacobian of constraints to find NaN values."""
    opti = control_problem.opti
    
    # Get the symbolic constraints and variables
    g = opti.g
    x = opti.x
    
    # Create a function to compute the Jacobian
    jac_g = ca.jacobian(g, x)
    jac_g_func = ca.Function('jac_g_func', [x], [jac_g])
    
    try:
        # Try to evaluate the Jacobian at the current point
        x_val = opti.debug.value(x)
        jac_g_val = jac_g_func(x_val)
        
        # Check for NaN values
        if np.any(np.isnan(jac_g_val)):
            print("NaN values found in the Jacobian matrix")
            
            # Find the indices of NaN values
            nan_indices = np.where(np.isnan(np.array(jac_g_val)))
            for row, col in zip(nan_indices[0], nan_indices[1]):
                print(f"NaN at position (row {row}, col {col})")
                
                # If this matches the error message location
                if row == 21 and col == 0:
                    print("This matches the location in the error message!")
                    
                    # Try to identify which constraint and variable this corresponds to
                    if row < g.size1():
                        print(f"Constraint: g[{row}] = {g[row]}")
                    
                    if col < x.size1():
                        print(f"Variable: x[{col}]")
                        
                        # Try to determine what this variable represents
                        # This requires knowledge of how variables are organized in your problem
                        if hasattr(control_problem, 'state') and hasattr(control_problem, 'control'):
                            state_size = control_problem.state.size1() * control_problem.state.size2()
                            control_size = control_problem.control.size1() * control_problem.control.size2()
                            
                            if col < state_size:
                                # This is a state variable
                                state_flat = ca.reshape(control_problem.state, -1, 1)
                                print(f"This is a state variable: {state_flat[col]}")
                            elif col < state_size + control_size:
                                # This is a control variable
                                control_flat = ca.reshape(control_problem.control, -1, 1)
                                print(f"This is a control variable: {control_flat[col - state_size]}")
        
        # Save the Jacobian to a file for further analysis
        np.savetxt('jacobian_matrix.csv', np.array(jac_g_val), delimiter=',')
        print("Jacobian matrix saved to 'jacobian_matrix.csv'")
        
        # For a more detailed analysis, you can also output the sparsity pattern
        print("\nJacobian sparsity pattern:")
        jac_sparsity = jac_g.sparsity()
        print(f"Number of non-zeros: {jac_sparsity.nnz()}")
        
        # Output the row and column for each non-zero element
        for k in range(jac_sparsity.nnz()):
            row = jac_sparsity.row(k)
            col = jac_sparsity.col(k)
            print(f"Non-zero at (row {row}, col {col})")
            
            # Check if this matches the problematic index
            if row == 21 and col == 0:
                print("This is the problematic element mentioned in the error!")
        
        return jac_g_val
        
    except Exception as e:
        print(f"Error evaluating Jacobian: {e}")
        
        # Try a different approach - evaluate row by row
        print("\nAttempting to evaluate Jacobian row by row:")
        for i in range(min(g.size1(), 30)):  # Limit to first 30 rows to avoid excessive output
            try:
                # Create a function for just this row of the Jacobian
                jac_row_i = ca.jacobian(g[i], x)
                jac_row_func = ca.Function(f'jac_row_{i}', [x], [jac_row_i])
                
                # Evaluate
                jac_row_val = jac_row_func(x_val)
                
                # Check for NaN
                if np.any(np.isnan(jac_row_val)):
                    print(f"Row {i} contains NaN values")
                    
                    # Find which columns have NaN
                    nan_cols = np.where(np.isnan(np.array(jac_row_val)))[1]
                    print(f"NaN in columns: {nan_cols}")
                    
                    # If this is row 21, pay special attention
                    if i == 21:
                        print("This is the problematic row mentioned in the error!")
            except Exception as row_e:
                print(f"Error evaluating row {i}: {row_e}")
        
        return None

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

    initial_guess = np.zeros((21, 13))
    initial_guess[:, 6] = 1
    initial_guess[:, 5] = -10
    initial_guess = np.concat((initial_guess, np.ones((initial_guess.shape[0],quad.num_controls ))), axis = 1).T
    num_nodes = initial_guess.shape[1] - 1
    i = 0
    state_list = initial_guess
    state = ca.DM(initial_guess[:quad.num_states, 0])
    time_intervals = [1/num_nodes for _ in range(num_nodes + 1)]#initialiser.time_intervals
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
    control_problem.setup(state_list, target = [10, 10, 10])
    # control_problem.setup(state_list, target = state_list.T[:3, -1])
    try:
        sol = control_problem.solve()
    except Exception as e:
        print(f"Solver failed with error: {e}")
        
        # Debug the Jacobian
        jac_g_val = debug_jacobian(control_problem)
        
        # If you want to visualize the Jacobian (if it's not too large)
        if jac_g_val is not None and jac_g_val.size1() < 100 and jac_g_val.size2() < 100:
            try:
                import matplotlib.pyplot as plt
                
                # Convert to numpy array and replace NaN with a specific value for visualization
                jac_np = np.array(jac_g_val)
                jac_np_no_nan = np.nan_to_num(jac_np, nan=-999)  # Replace NaN with -999
                
                plt.figure(figsize=(10, 8))
                plt.imshow(jac_np_no_nan, cmap='viridis')
                plt.colorbar(label='Jacobian Value')
                plt.title('Jacobian Matrix (NaN values shown as -999)')
                plt.xlabel('Variable Index')
                plt.ylabel('Constraint Index')
                
                # Highlight the problematic element
                plt.plot(0, 21, 'rx', markersize=10)
                
                plt.savefig('jacobian_visualization.png')
                print("Jacobian visualization saved to 'jacobian_visualization.png'")
            except Exception as viz_e:
                print(f"Error visualizing Jacobian: {viz_e}")

    # try:
    #     sol = control_problem.solve()
    # except Exception as e:
    #     print(f"Solver failed with error: {e}")
        
    #     # Try to access debug information
    #     try:
    #         debug = control_problem.opti.debug
            
    #         # Check for NaN values in the state and control
    #         state_vals = debug.value(control_problem.state)
    #         control_vals = debug.value(control_problem.control)
            
    #         if np.any(np.isnan(state_vals)):
    #             print("NaN values detected in state variables")
    #             nan_indices = np.where(np.isnan(state_vals))
    #             print(f"NaN state indices: {list(zip(*nan_indices))}")
                
    #         if np.any(np.isnan(control_vals)):
    #             print("NaN values detected in control variables")
    #             nan_indices = np.where(np.isnan(control_vals))
    #             print(f"NaN control indices: {list(zip(*nan_indices))}")
            
    #         # For constraints, use proper CasADi methods to access elements
    #         # Instead of iterating over debug.g directly, use CasADi's methods
    #         g = debug.g
    #         n_g = g.numel()  # Get number of elements
    #         print(f"Length of constraint defs: {len(control_problem.constraint_descriptions)}")
    #         print(f"Number of constraints: {n_g}")

    #         for i in range(n_g):
    #             try:
    #                 g_i = g[i]
    #                 value = debug.value(g_i)

    #                 print(f"{i}: {control_problem.constraint_descriptions[i]}")

    #                 if i >= len(control_problem.constraint_descriptions):
    #                     print(f"Constraint {i} has no description.")
    #                     continue

    #                 desc = control_problem.constraint_descriptions[i]

    #                 if np.isnan(value):
    #                     print(f"NaN detected in constraint {i}, {desc}")
    #                     continue

    #                 # Equality constraint check
    #                 if "==" in desc:
    #                     if abs(value) > 1e-6:
    #                         print(f"[Violation] Equality constraint {i}: {desc} --> Residual: {value}")

    #                 # Inequality (<= or >=)
    #                 elif "<=" in desc or ">=" in desc:
    #                     if value > 1e-6:  # Positive residual → violation for ≤
    #                         print(f"[Violation] Inequality constraint {i}: {desc} --> Residual: {value}")

    #                 # Bounded constraints — try to detect chained comparisons
    #                 elif "<=" in desc and ">=" in desc or "<=" in desc and "opti" in desc:
    #                     if value > 1e-6:
    #                         print(f"[Violation] Bounded constraint {i}: {desc} --> Residual: {value}")

    #                 # Greater-than constraint (`>`), interpreted as `-expr <= 0`
    #                 elif ">" in desc:
    #                     if value < -1e-6:
    #                         print(f"[Violation] '>' constraint {i}: {desc} --> Residual: {value}")

    #                 else:
    #                     # Fallback if type can't be determined
    #                     if abs(value) > 1e-6:
    #                         print(f"[Violation] Unknown-type constraint {i}: {desc} --> Residual: {value}")

    #             except Exception as e:
    #                 print(f"Could not evaluate constraint {i}: {e}")

            
    #         # # Check each constraint individually
    #         # for i in range(n_g):
    #         #     try:
    #         #         # Use CasADi's element access method
    #         #         g_i = g[i]
    #         #         value = debug.value(g_i)
    #         #         if i >= len(control_problem.constraint_descriptions):
    #         #             break
    #         #         if np.isnan(value):
    #         #             print(f"NaN detected in constraint {i}, {control_problem.constraint_descriptions[i]}")
    #         #         # distinguish types
    #         #         elif '==' in control_problem.constraint_descriptions[i]:
    #         #             if abs(value) > 1e-6:  # Check for violations
    #         #                 print(f"Constraint {i} {control_problem.constraint_descriptions[i]} violated: {value}")
    #         #         elif '<=' in control_problem.constraint_descriptions[i]:
    #         #             if value > 0:
    #         #                 print(f"Constraint {i} {control_problem.constraint_descriptions[i]} violated: {value}")
    #         #         elif '>' in control_problem.constraint_descriptions[i][2]:
    #         #             if value < 0:
    #         #                 print(f"Constraint {i} {control_problem.constraint_descriptions[i]} violated: {value}")

    #         #         else:
    #         #             print(control_problem.constraint_descriptions[i])
    #         #     except Exception as inner_e:
    #         #         print(f"Could not evaluate constraint {i}: {inner_e}")
                    
    #     except Exception as debug_e:
    #         print(f"Could not access debug information: {debug_e}")





if __name__ =="__main__":
    main()