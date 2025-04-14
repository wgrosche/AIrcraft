import numpy as np
from aircraft.control.base import ControlNode, ControlProblem
from aircraft.dynamics.quadrotor import Quadrotor
import matplotlib.pyplot as plt
from typing import List, Optional
import casadi as ca
from aircraft.plotting.plotting import TrajectoryData, TrajectoryPlotter
class QuadrotorControl(ControlProblem):
    def __init__(self, quad, **kwargs):
        dynamics = quad.state_update
        super().__init__(dynamics=dynamics, **kwargs)
        self.plotter = TrajectoryPlotter(quad)
        
    def control_constraint(self, node):
        opti = self.opti
        self.constraint(opti.bounded(0, node.control, 10))
        # opti.subject_to(opti.bounded(0, node.control, 100))
        # self.constraint_descriptions.append(('thrust constraint', node.control, '><'))
        return super().control_constraint(node)
    
    def setup(self, guess, target=None):
        super().setup(guess)
        if target is not None:
            self.opti.minimize(ca.sumsqr(self.state[:3, -1] -  target))
            self.constraint(self.state[:3, -1] ==  target)
            # self.opti.subject_to(self.state[:3, -1] ==  target)
            # self.constraint_descriptions.append(('goal constraint', self.state[:3, -1], '=='))
            
    
    def callback(self, iteration: int):
        # Call the parent class callback to handle saving progress
        super().callback(iteration)

        # Plotting
        if self.plotter and iteration % 10 == 5:
            trajectory_data = TrajectoryData(
                state=np.array(self.opti.debug.value(self.state))[:, 1:],
                control=np.array(self.opti.debug.value(self.control)),
                time=np.array(self.opti.debug.value(self.time))
            )
            self.plotter.plot(trajectory_data=trajectory_data)
            plt.draw()
            self.plotter.figure.canvas.start_event_loop(0.0002)

    def solve(self, warm_start:Optional[ca.OptiSol] = None):
        super().solve(warm_start=warm_start)

        trajectory_data = TrajectoryData(
                state=np.array(self.opti.debug.value(self.state))[:, 1:],
                control=np.array(self.opti.debug.value(self.control)),
                time=np.array(self.opti.debug.value(self.time))
            )
        self.plotter.plot(trajectory_data=trajectory_data)
        plt.draw()
        self.plotter.figure.canvas.start_event_loop(0.0002)

        plt.show(block = True)
    


    # def solve(self, warm_start:Optional[ca.OptiSol] = None):
    #     sol = super().solve(warm_start = warm_start)
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(sol.value(self.state[0, :]), sol.value(self.state[1, :]), sol.value(self.state[2, :]))
    #     plt.show(block = True)
    