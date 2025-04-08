import numpy as np
from aircraft.control.base import ControlNode, ControlProblem
from aircraft.dynamics.quadrotor import Quadrotor
import matplotlib.pyplot as plt
from typing import List, Optional
import casadi as ca

class QuadrotorControl(ControlProblem):
    def __init__(self, dynamics):
        self.num_nodes = 100
        super().__init__(dynamics, self.num_nodes)

    def setup(self, initial_pos:np.ndarray = np.zeros(3)):
        guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
        guess[9, :] = 1

        # dt = .01
        # tf = 5
        # state_list = np.zeros((quad.num_states, int(tf / dt)))
        # t = 0
        # control_list = np.zeros((quad.num_controls, int(tf / dt)))
        # for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
        #     if np.isnan(state[0]):
        #         print('quad crashed')
        #         break
        #     else:

        #         state_list[:, i] = state.full().flatten()
        #         control_list[:, i] = control
        #         state = dyn(state, control, dt)
                        
        #         t += 1
        super().setup(guess)

    def control_constraint(self, node):
        opti = self.opti

        opti.subject_to(opti.bounded(0, node.control, 5))
        return super().control_constraint(node)
    
    def _setup_objective(self, nodes):
        self.opti.subject_to(self.state[:3, -1] ==  [100, 100, 100])
        self.opti.minimize(self.loss(time = self.time))

    def solve(self, warm_start:Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start = warm_start)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sol.value(self.state[0, :]), sol.value(self.state[1, :]), sol.value(self.state[2, :]))
        plt.show(block = True)
    