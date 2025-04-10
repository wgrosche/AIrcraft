from aircraft.control.aircraft import TrajectoryConfiguration
from pathlib import Path
from aircraft.config import NETWORKPATH
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
import numpy as np
from tqdm import tqdm
import casadi as ca
import json
from aircraft.control.aircraft import AircraftControl


    

def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    
    num_nodes = 30
    time_guess = 10
    dt = time_guess / (num_nodes)

    guess =  np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))

    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
    control = np.zeros(aircraft.num_controls)
    control[:len(trim_state_and_control) - aircraft.num_states - 3] = trim_state_and_control[aircraft.num_states:-3]
    aircraft.com = np.array(trim_state_and_control[-3:])
    
    dyn = aircraft.state_update

    # Initialize trajectory with debugging prints
    for i in tqdm(range(num_nodes + 1), desc='Initialising Trajectory:'):
        guess[:aircraft.num_states, i] = state.full().flatten()
        # control = control + 1 * (rng.random(len(control)) - 0.5)
        guess[aircraft.num_states:, i] = control
        next_state = dyn(state, control, dt)
        # print(f"Node {i}: State = {state}, Control = {control}, Next State = {next_state}")
        state = next_state


    # Second loop: Validate initial guess
    guess2 = np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))
    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])

    for i in tqdm(range(num_nodes + 1), desc='Validating Trajectory:'):
        guess2[:aircraft.num_states, i] = state.full().flatten()
        control = guess[aircraft.num_states:, i]  # Use controls from guess1
        next_state = dyn(state, control, dt)
        guess2[aircraft.num_states:, i] = control
        state = next_state


    
    for i in range(num_nodes + 1):
        print(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i])
        assert np.allclose(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i]), f"Problem in node {i}"




    control_problem = AircraftControl(aircraft, num_nodes, time_guess = time_guess)    
    control_problem.setup(guess)
    control_problem.solve()
    


def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    
    num_nodes = 100
    time_guess = 10
    dt = time_guess / (num_nodes)

    guess =  np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))

    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
    control = np.zeros(aircraft.num_controls)
    control[:len(trim_state_and_control) - aircraft.num_states - 3] = trim_state_and_control[aircraft.num_states:-3]
    aircraft.com = np.array(trim_state_and_control[-3:])
    aircraft.STEPS = 1
    dyn = aircraft.state_update

    # Initialize trajectory with debugging prints
    for i in tqdm(range(num_nodes + 1), desc='Initialising Trajectory:'):
        guess[:aircraft.num_states, i] = state.full().flatten()
        # control = control + 1 * (rng.random(len(control)) - 0.5)
        guess[aircraft.num_states:, i] = control
        next_state = dyn(state, control, dt)
        # print(f"Node {i}: State = {state}, Control = {control}, Next State = {next_state}")
        state = next_state


    # Second loop: Validate initial guess
    guess2 = np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))
    state = ca.vertcat(trim_state_and_control[:aircraft.num_states])

    for i in tqdm(range(num_nodes + 1), desc='Validating Trajectory:'):
        guess2[:aircraft.num_states, i] = state.full().flatten()
        control = guess[aircraft.num_states:, i]  # Use controls from guess1
        next_state = dyn(state, control, dt)
        guess2[aircraft.num_states:, i] = control
        state = next_state


    
    for i in range(num_nodes + 1):
        print(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i])
        assert np.allclose(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i]), f"Problem in node {i}"




    control_problem = AircraftControl(aircraft, num_nodes, time_guess = time_guess)    
    control_problem.setup(guess)
    control_problem.solve()


if __name__ =="__main__":
    main()