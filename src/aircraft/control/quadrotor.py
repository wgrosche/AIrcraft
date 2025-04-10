import numpy as np
from aircraft.control.base import ControlNode, ControlProblem
from aircraft.dynamics.quadrotor import Quadrotor
import matplotlib.pyplot as plt
from typing import List, Optional
import casadi as ca

class QuadrotorControl(ControlProblem):
    def control_constraint(self, node):
        opti = self.opti

        opti.subject_to(opti.bounded(0, node.control, 100))
        self.constraint_descriptions.append(('thrust constraint', node.control, '><'))
        return super().control_constraint(node)
    
    def setup(self, guess, target=None):
        super().setup(guess)
        if target is not None:
            self.opti.minimize(ca.sumsqr(self.state[:3, -1] -  target))
            self.opti.subject_to(self.state[:3, -1] ==  target)
            self.constraint_descriptions.append(('goal constraint', self.state[:3, -1], '=='))
            
            
    


    def solve(self, warm_start:Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start = warm_start)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sol.value(self.state[0, :]), sol.value(self.state[1, :]), sol.value(self.state[2, :]))
        plt.show(block = True)
    