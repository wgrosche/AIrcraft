from aircraft.dynamics.aircraft import Aircraft, AircraftOpts
from aircraft.utils.utils import load_model, TrajectoryConfiguration, AircraftConfiguration, perturb_quaternion
from aircraft.config import NETWORKPATH, DATAPATH, DEVICE
from pathlib import Path
from aircraft.plotting.plotting import TrajectoryPlotter
import json
import casadi as ca
from typing import Optional
import numpy as np

import matplotlib.pyplot as plt


perturbation = False




# state around which to calculate the stability
trim_state_and_control = [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]

def main(mode:Optional[int] = None) -> None:
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    if mode == 0:
        model_path = Path(NETWORKPATH) / 'model-dynamics.pth'
        opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    elif mode == 1:
        poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
        opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)
    elif mode == 2:
        linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
        opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)
    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
    control = np.zeros(aircraft.num_controls)
    control[:3] = trim_state_and_control[aircraft.num_states:-3]
    control[0] = 0
    control[1] = 0
    aircraft.com = np.array(trim_state_and_control[-3:])

    # Define f(state, control) (e.g., the dynamics function)
    f = aircraft.state_update(aircraft.state, aircraft.control, aircraft.dt_sym)

    # Compute the Jacobian of f w.r.t state
    J = ca.jacobian(f, aircraft.state)

    # Create a CasADi function for numerical evaluation
    J_func = ca.Function('J', [aircraft.state, aircraft.control, aircraft.dt_sym], [J])

    # Evaluate J numerically for a specific state and control
    # J_val = J_func(state, control, .01)

    # Define perturbations (adjust as needed)
    state_perturbations = np.linspace(-0.1, 0.1, num=5)  # Small deviations
    control_perturbations = np.linspace(-0.0, 0.0, num=5)

    # Storage for eigenvalues
    eigenvalues_list = []
    condition_numbers = []
    for dx in state_perturbations:
        for du in control_perturbations:
            perturbed_state = state + dx
            perturbed_quaternion = perturb_quaternion(state[6:10].toarray().flatten())
            perturbed_state[6:10] = perturbed_quaternion
            perturbed_control = control + du
            
            # Evaluate the discrete-time Jacobian at perturbed states
            J_val = J_func(perturbed_state, perturbed_control, 0.01)
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(J_val)
            eigenvalues_list.append(eigenvalues)
            # Compute condition number
            condition_number = np.linalg.cond(J_val)
            condition_numbers.append(condition_number)


    # Convert to NumPy array for easier analysis
    eigenvalues_array = np.array(eigenvalues_list)
    condition_numbers_array = np.array(condition_numbers)
    print(condition_numbers_array)

    def plot_control_surface_condition_numbers(state, control, control_ranges=(-5, 5), num_points=20):
        """Plot condition numbers for different control surface deflections"""
        # Create meshgrid for aileron and elevator deflections
        aileron_range = np.linspace(control_ranges[0], control_ranges[1], num_points)
        elevator_range = np.linspace(control_ranges[0], control_ranges[1], num_points)
        ail_mesh, ele_mesh = np.meshgrid(aileron_range, elevator_range)
        
        # Calculate condition numbers for each combination
        condition_numbers = np.zeros_like(ail_mesh)
        for i in range(num_points):
            for j in range(num_points):
                control[0] = ail_mesh[i,j]  # aileron
                control[1] = ele_mesh[i,j]  # elevator
                J_val = J_func(state, control, 0.01)
                condition_numbers[i,j] = max(np.abs(np.linalg.eigvals(J_val)))#np.linalg.cond(J_val)

        
        # Create 3D surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(ail_mesh, ele_mesh, condition_numbers, cmap='viridis')
        
        ax.set_xlabel('Aileron Deflection (deg)')
        ax.set_ylabel('Elevator Deflection (deg)')
        ax.set_zlabel('Max Eigenvalue')
        fig.colorbar(surf)
        
        plt.show(block=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 12))


    # First subplot - Eigenvalues
    unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
    ax1.add_patch(unit_circle)

    for eigvals in eigenvalues_array:
        ax1.scatter(eigvals.real, eigvals.imag, color='blue', alpha=0.5)

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_xlabel("Real")
    ax1.set_ylabel("Imaginary")
    ax1.set_title("Eigenvalues Under Perturbed States & Controls")
    ax1.grid()

    # Define timestep range (log scale for better resolution)
    timesteps = np.logspace(-4, 0, num=20)  # From very small to larger dt values
    max_eigenvalues = []

    for dt in timesteps:
        # Compute discrete-time Jacobian at this timestep
        J_val = J_func(state, control, dt)  # Get continuous Jacobian

        # Compute eigenvalues and store the largest norm
        eigvals = np.linalg.eigvals(J_val)
        max_eigenvalues.append(max(np.abs(eigvals)))

    # Second subplot - Max Eigenvalue vs Timestep
    ax2.plot(timesteps, max_eigenvalues, marker='o', linestyle='-')
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axhline(1, color='r', linestyle='--', label="Unit Circle Bound")
    ax2.set_xlabel("Timestep (Δt)")
    ax2.set_ylabel("Max Eigenvalue Norm")
    ax2.set_title("Max Eigenvalue Norm vs. Timestep")
    ax2.legend()
    ax2.grid()

    # Condition Numbers Plot

    ax3.plot(timesteps, max_eigenvalues, marker='o', linestyle='-')
    ax3.set_ylabel("Condition Number")
    ax3.set_title("Condition of Perturbed Jacobian")
    ax3.legend()
    ax3.grid()

    # Control Deflected Condition Numbers Plot

    fig.tight_layout()
    plt.show(block = True)


    plot_control_surface_condition_numbers(state, control)


if __name__ == '__main__':
    main(mode=1)

    # dyn = aircraft.state_update
    # dt = .01
    # tf = 5
    # state_list = np.zeros((aircraft.num_states, int(tf / dt)))
    # # investigate stiffness:

    # # Define f(state, control) (e.g., the dynamics function)
    

    # # Compute eigenvalues using numpy
    # eigvals = np.linalg.eigvals(np.array(J_val))

    # print(eigvals)

    


    



    # dt = 0.001
    # t = 0
    # ele_pos = True
    # ail_pos = True
    # control_list = np.zeros((aircraft.num_controls, int(tf / dt)))
    # for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
    #     if np.isnan(state[0]):
    #         print('Aircraft crashed')
    #         break
    #     else:
    #         state_list[:, i] = state.full().flatten()
    #         control_list[:, i] = control
    #         state = dyn(state, control, dt)
                    
    #         t += 1
    # # print(state)
    # # J_val = J_func(state, control)
    # # eigvals = np.linalg.eigvals(np.array(J_val))

    # # print(eigvals)
    # first = None
    # t -=10
    # def save(filepath):
    #     with h5py.File(filepath, "a") as h5file:
    #         grp = h5file.create_group('iteration_0')
    #         grp.create_dataset('state', data=state_list[:, :t])
    #         grp.create_dataset('control', data=control_list[:, :t])
    
    
    # filepath = os.path.join("data", "trajectories", "simulation.h5")
    # if os.path.exists(filepath):
    #     os.remove(filepath)
    # save(filepath)

    # plotter = TrajectoryPlotter(aircraft)
    # plotter.plot(filepath=filepath)
    # plt.show(block = True)