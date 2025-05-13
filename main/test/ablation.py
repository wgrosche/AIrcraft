"""
Study comparing convergence impact of:
 - Quaternion normalisation approach
 - Model type
 - Variable time implementation
"""

from aircraft.control.aircraft import TrajectoryConfiguration
from pathlib import Path
from aircraft.config import NETWORKPATH
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
import numpy as np
from tqdm import tqdm
import casadi as ca
import json
from aircraft.control.aircraft import AircraftControl, WaypointControl
from aircraft.control.base import SaveMixin#, VariableTimeMixin

from aircraft.control.variable_time import ProgressTimeMixin
from aircraft.config import DATAPATH
import matplotlib.pyplot as plt
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData


class Controller(AircraftControl, SaveMixin):#, ProgressTimeMixin):
    def __init__(self, *, aircraft, num_nodes=298, dt=.01, opts = {}, filepath:str = '', **kwargs):
        super().__init__(aircraft=aircraft, num_nodes=num_nodes, opts = opts, dt = dt, **kwargs)
        if filepath:
            self.save_path = filepath
        SaveMixin._init_saving(self, self.save_path, force_overwrite=True)
        # ProgressTimeMixin._init_progress_time(self, self.opti, num_nodes)
        self.plotter = TrajectoryPlotter(aircraft)
        

    def loss(self, nodes, time, goal:list = []):
        control_loss = ca.sumsqr(self.control[: 1:] - self.control[:, -1])
        if goal:
            indices = len(goal)
            distance_loss = 1000*ca.sumsqr(nodes[-1].state[:indices] - goal)


        height_loss = 
        # self.constraint(ca.sumsqr(nodes[-1].state[:3] - [0, 30, -180])==0)
        return distance_loss + control_loss + time - ca.sumsqr(self.state[:, 3])/100 - ca.sumsqr(self.state[:, 2])/100#time
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
            state_traj = self.opti.debug.value(self.state)  # shape (n_states, N+1)
            # quat_norms = np.linalg.norm(state_traj[6:10, :], axis=0)
            # print(f"Iteration: {iteration}", " Quaternion Norm: ", quat_norms)
            if self.plotter and iteration % 50 == 0:
                print("plotting")
                trajectory_data = TrajectoryData(
                    state=np.array(state_traj)[:, 1:],
                    control=np.array(self.opti.debug.value(self.control)),
                    time=np.array(self.opti.debug.value(self.time))
                )
                self.plotter.plot(trajectory_data=trajectory_data)
                plt.draw()
                self.plotter.figure.canvas.start_event_loop(0.0002)
                plt.show()
        
                super().callback(iteration)

def main():
    traj_dict = json.load(open('data/glider/problem_definition.json'))
    trajectory_config = TrajectoryConfiguration(traj_dict)
    aircraft_config = trajectory_config.aircraft
    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'

    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)


    aircraft = Aircraft(opts = opts)
    aircraft.STEPS = 1
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    aircraft.com = np.array(trim_state_and_control[-3:])

    filepath = Path(DATAPATH) / 'trajectories' / 'ablation.h5'

    controller = Controller(aircraft=aircraft, filepath=filepath, implicit=False, progress = False)
    guess = controller.initialise(ca.DM(trim_state_and_control[:aircraft.num_states]))
    controller.setup(guess)
    
    controller.solve()
    final_state = controller.opti.debug.value(controller.state)[:, -1]
    final_control = controller.opti.debug.value(controller.control)[:, -1]


    print("Final State: ", final_state, " Final Control: ", final_control, " Final Forces: ", aircraft.forces_frd(final_state, final_control))

    plt.show(block = True)

    
    
    
if __name__ =="__main__":
    main()