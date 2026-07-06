import json
import numpy as np
import casadi as ca
from tqdm import tqdm

from aircraft.utils import TrajectoryConfiguration
from aircraft.control.quadrotor import QuadrotorControl
from aircraft.dynamics.quadrotor import Quadrotor


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


def initialise_guess(quad, initial_state, initial_control, num_nodes, dt):
    """Roll out dynamics to create an initial guess, matching control.py style."""
    guess = np.zeros((quad.num_states + quad.num_controls, num_nodes + 1))

    state = ca.DM(initial_state)
    control = np.array(initial_control, dtype=float)
    dyn = quad.state_update

    for i in tqdm(range(num_nodes), desc='Initialising Trajectory:'):
        guess[:quad.num_states, i] = state.full().flatten()
        guess[quad.num_states:, i] = control
        state = dyn(state, control, dt)

    guess[:quad.num_states, -1] = state.full().flatten()
    guess[quad.num_states:, -1] = control

    return guess

def main():
    """
    Minimal quadrotor control setup following the control.py flow.
    """

    quad = Quadrotor()
    quad.physical_integration_substeps = 1

    traj_dict = json.load(open('data/glider/problem_definition.json'))
    trajectory_config = TrajectoryConfiguration(traj_dict)

    # Build an initial state from trajectory metadata and the quad state layout.
    initial_position = np.array(trajectory_config.waypoints.initial_position)
    initial_velocity = np.array([5.0, 0.0, 0.0])
    initial_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    initial_rates = np.array([0.0, 0.0, 0.0])
    initial_state = np.concatenate(
        [initial_position, initial_velocity, initial_quaternion, initial_rates]
    )

    initial_control = np.ones(quad.num_controls)

    num_nodes = 200
    dt = 0.05

    guess = initialise_guess(
        quad=quad,
        initial_state=initial_state,
        initial_control=initial_control,
        num_nodes=num_nodes,
        dt=dt,
    )

    controller_opts = {
        'time': 'fixed',
        'quaternion': 'integration',
        'integration': 'explicit',
    }

    control_problem = QuadrotorControl(quad, num_nodes=num_nodes, dt=dt, opts=controller_opts)
    target = np.array([100.0, 100.0, -200.0])
    control_problem.setup(guess, target=target)

    try:
        sol = control_problem.solve()
        final_state = control_problem.opti.debug.value(control_problem.state)[:, -1]
        final_control = control_problem.opti.debug.value(control_problem.control)[:, -1]
        final_time = control_problem.opti.debug.value(control_problem.times)[-1]
        print(
            "Final State:", final_state,
            "Final Control:", final_control,
            "Final Time:", final_time,
        )
    except Exception as e:
        print(f"Solver failed with error: {e}")
        debug_jacobian(control_problem)

if __name__ =="__main__":
    main()