import numpy as np
from dataclasses import dataclass
import casadi as ca
from aircraft.control.base import ControlProblem, ControlNode
from aircraft.control.aircraft import AircraftControl
from dataclasses import dataclass
from typing import Optional, List
from aircraft.control.initialisation import DubinsInitialiser

@dataclass
class MPCCNode(ControlNode):
    progress:Optional[ca.MX] = None

class MHTT(ControlProblem):
    def __init__(self, *, track, dt, **kwargs):
        super().__init__(**kwargs)
        self.track = track
        self.dt = dt
        self.time = dt * self.num_nodes
    
    def loss(self, nodes, time = None):
        return self.progress_loss

    def _setup_time(self):
        """
        Override as we don't need variable timestepping here. 
        """
        pass

    def _setup_initial_node(self, guess):
        """
        set up the progress variable.
        """
        current_node = super()._setup_initial_node(guess)
        progress = self.opti.variable()
        self.progress = [progress]
        tracking_error = ca.sumsqr(self.track(progress) - current_node.state[:3])
        self.progress_loss = 10 * tracking_error - 100 * progress
        return current_node
    
    def _setup_step(self, index, current_node, guess):
        print(current_node)
        current_node = super()._setup_step(index = index, current_node = current_node, guess = guess)
        progress = self.opti.variable()
        self.progress.append(progress)
        tracking_error = ca.sumsqr(self.track(progress) - current_node.state[:3])
        self.progress_loss += 10 * tracking_error - 100 * progress
        self.constraint(self.progress[-1] >= self.progress[-2], description="Progress Constraint")
        return current_node

    def _setup_variables(self, nodes:List[ControlNode]):
        super()._setup_variables(nodes)
        self.progress = ca.vcat(self.progress)
        if self.verbose:
            print("Progress Shape: ", self.progress.shape)

    def initialise(self, initial_state):
        guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
        guess[:self.state_dim, 0] = initial_state.toarray().flatten()
        progress = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            guess[:self.state_dim, i + 1] = self.dynamics(guess[:self.state_dim, i], guess[self.state_dim:, i], self.dt).toarray().flatten()

        return guess, progress
    
    def callback(self, iteration:int):
        """
        To be implemented
        """
        pass

