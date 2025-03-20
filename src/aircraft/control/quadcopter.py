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

    def setup(self, _ = 0):
        guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
        guess[6, :] = 1
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
    
def main():
    """
    Minimal test of quadrotor control class
    """
    quad = Quadrotor()
    control_problem = QuadrotorControl(quad.step())

    control_problem.setup()
    control_problem.solve()

    
if __name__ == '__main__':
    main()