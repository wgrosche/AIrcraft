from aircraft.utils.utils import TrajectoryConfiguration
from pathlib import Path
from aircraft.config import NETWORKPATH
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
import numpy as np
from tqdm import tqdm
import casadi as ca
import json
from typing import Any
from aircraft.control.aircraft import AircraftControl,  ControlNode
from aircraft.control.base import SaveMixin#, VariableTimeMixin

from aircraft.control.variable_time import ProgressTimeMixin
from aircraft.config import DATAPATH
import matplotlib.pyplot as plt
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData


class Controller(AircraftControl):#, SaveMixin):#, ProgressTimeMixin):
    def __init__(self, *, aircraft:Aircraft, num_nodes:int=298, dt:float=.01, opts:dict = {}, filepath:str|Path = '', **kwargs:Any):
        super().__init__(aircraft=aircraft, num_nodes=num_nodes, opts = opts, dt = dt, **kwargs)
        if filepath:
            self.save_path = filepath
        # SaveMixin._init_saving(self, self.save_path, force_overwrite=True)
        # ProgressTimeMixin._init_progress_time(self, self.opti, num_nodes)
        

    def loss(self, nodes:ControlNode, time:ca.MX) -> ca.MX:
        control_loss = ca.sumsqr(self.control[: 1:] - self.control[:, -1])
        self.constraint(ca.sumsqr(nodes[-1].state[:3] - [0, 30, -180])==0)
        return control_loss + ca.sum(self.times)#1000*ca.sumsqr(nodes[-1].state[:3] - [0, 30, -180])# + control_loss# + time - ca.sumsqr(self.state[:, 3])/100 - ca.sumsqr(self.state[:, 2])/100#time
        # return 1000*ca.sumsqr(nodes[-1].state[:3] - [0, 100, -180]) + control_loss + time #time
    
    def initialise(self, initial_state):
        """
        Initialize the optimization problem with initial state.
        """
        guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
        guess[13, :] = 1
        guess[14, :] = 1
        guess[:self.state_dim, 0] = initial_state.toarray().flatten()
        # dt_initial = self.dt
        # dt_initial = 0.01#2 / self.num_nodes
        # self.opti.set_initial(self.time, 2)
        # Propagate forward using nominal dynamics
        for i in range(self.num_nodes):
            guess[:self.state_dim, i + 1] = self.dynamics(
                guess[:self.state_dim, i],
                guess[self.state_dim:, i],
                self.dt 
                
            ).toarray().flatten()


        return guess
    

    def callback(self, iteration: int):
            """
            Callback method for tracking optimization progress and visualizing trajectory.
    
            This method is called during each iteration of the optimization process. It performs two main tasks:
            1. Prints the quaternion norm for each node to help diagnose potential quaternion normalization issues
            2. Periodically plots the trajectory at specified intervals (every 50 iterations)
    
            Args:
                iteration (int): Current iteration number of the optimization process
    
            Notes:
                - Quaternion norm is calculated using numpy's linalg.norm function
                - Extracts quaternion components from state vector (indices 6:10)
                - Plots trajectory using TrajectoryPlotter when iteration is a multiple of 50
            """
            super().callback(iteration)

def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    # poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
    opts = AircraftOpts(coeff_model_type='default', coeff_model_path='', aircraft_config=aircraft_config, physical_integration_substeps=1)

    aircraft = Aircraft(opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    aircraft.com = np.array(trim_state_and_control[-3:])
    filepath = Path(DATAPATH) / 'trajectories' / 'basic_test.h5'

    # controller_opts = {'time':'fixed', 'quaternion':'', 'integration':'explicit'}
    controller_opts = {'time':'variable', 'quaternion':'', 'integration':'explicit'}
    controller = Controller(aircraft=aircraft, filepath=filepath, opts = controller_opts)
    guess = controller.initialise(ca.DM(trim_state_and_control[:aircraft.num_states]))
    controller.setup(guess)
    controller.logging = False
    
    controller.solve()
    final_state = controller.opti.debug.value(controller.state)[:, -1]
    final_control = controller.opti.debug.value(controller.control)[:, -1]
    final_time = controller.opti.debug.value(controller.times)[-1]
    print("Final State: ", final_state, " Final Control: ", final_control, " Final Forces: ", aircraft.forces_frd(final_state, final_control), " Final Time: ", final_time)
    plt.show(block = True)

    
    
    
if __name__ =="__main__":
    main()
#     num_nodes = 30
#     time_guess = 10
#     dt = time_guess / (num_nodes)

#     guess =  np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))

#     state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
#     control = np.zeros(aircraft.num_controls)
#     control[:len(trim_state_and_control) - aircraft.num_states - 3] = trim_state_and_control[aircraft.num_states:-3]
#     aircraft.com = np.array(trim_state_and_control[-3:])
    
#     dyn = aircraft.state_update

#     # Initialize trajectory with debugging prints
#     for i in tqdm(range(num_nodes + 1), desc='Initialising Trajectory:'):
#         guess[:aircraft.num_states, i] = state.full().flatten()
#         # control = control + 1 * (rng.random(len(control)) - 0.5)
#         guess[aircraft.num_states:, i] = control
#         next_state = dyn(state, control, dt)
#         # print(f"Node {i}: State = {state}, Control = {control}, Next State = {next_state}")
#         state = next_state


#     # Second loop: Validate initial guess
#     guess2 = np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))
#     state = ca.vertcat(trim_state_and_control[:aircraft.num_states])

#     for i in tqdm(range(num_nodes + 1), desc='Validating Trajectory:'):
#         guess2[:aircraft.num_states, i] = state.full().flatten()
#         control = guess[aircraft.num_states:, i]  # Use controls from guess1
#         next_state = dyn(state, control, dt)
#         guess2[aircraft.num_states:, i] = control
#         state = next_state


    
#     for i in range(num_nodes + 1):
#         print(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i])
#         assert np.allclose(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i]), f"Problem in node {i}"




#     control_problem = AircraftControl(aircraft, num_nodes, time_guess = time_guess)    
#     control_problem.setup(guess)
#     control_problem.solve()
    


# def main():
#     traj_dict = json.load(open('data/glider/problem_definition.json'))

#     trajectory_config = TrajectoryConfiguration(traj_dict)

#     aircraft_config = trajectory_config.aircraft

#     poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
#     opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

#     aircraft = Aircraft(opts = opts)
#     trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    
#     num_nodes = 100
#     time_guess = 10
#     dt = time_guess / (num_nodes)

#     guess =  np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))

#     state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
#     control = np.zeros(aircraft.num_controls)
#     control[:len(trim_state_and_control) - aircraft.num_states - 3] = trim_state_and_control[aircraft.num_states:-3]
#     aircraft.com = np.array(trim_state_and_control[-3:])
#     aircraft.STEPS = 10
#     dyn = aircraft.state_update

#     # Initialize trajectory with debugging prints
#     for i in tqdm(range(num_nodes + 1), desc='Initialising Trajectory:'):
#         guess[:aircraft.num_states, i] = state.full().flatten()
#         # control = control + 1 * (rng.random(len(control)) - 0.5)
#         guess[aircraft.num_states:, i] = control
#         next_state = dyn(state, control, dt)
#         # print(f"Node {i}: State = {state}, Control = {control}, Next State = {next_state}")
#         state = next_state


#     # Second loop: Validate initial guess
#     guess2 = np.zeros((aircraft.num_states + aircraft.num_controls, num_nodes + 1))
#     state = ca.vertcat(trim_state_and_control[:aircraft.num_states])

#     for i in tqdm(range(num_nodes + 1), desc='Validating Trajectory:'):
#         guess2[:aircraft.num_states, i] = state.full().flatten()
#         control = guess[aircraft.num_states:, i]  # Use controls from guess1
#         next_state = dyn(state, control, dt)
#         guess2[aircraft.num_states:, i] = control
#         state = next_state


    
#     for i in range(num_nodes + 1):
#         print(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i])
#         assert np.allclose(guess[:aircraft.num_states, i], guess2[:aircraft.num_states, i]), f"Problem in node {i}"




#     control_problem = AircraftControl(aircraft, num_nodes, time_guess = time_guess)    
#     control_problem.setup(guess)
#     control_problem.solve()

