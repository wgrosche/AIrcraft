# import numpy as np
# from dataclasses import dataclass
# import casadi as ca
# from aircraft.control.base import ControlProblem, ControlNode
# from aircraft.control.aircraft import AircraftControl
# from dataclasses import dataclass
# from typing import Optional, List
# from aircraft.control.initialisation import DubinsInitialiser

# @dataclass
# class MPCCNode(ControlNode):
#     progress:Optional[ca.MX] = None

# class MHTT(ControlProblem):
#     def __init__(self, *, track, track_length, dt, **kwargs):
#         super().__init__(**kwargs)
#         self.track = track
#         self.dt = dt
#         self.time = dt * self.num_nodes
#         self.track_length = track_length
#         self.initial_progress_param = self.opti.parameter()
    
#     def loss(self, nodes, time = None):
#         return self.progress_loss

#     def _setup_time(self):
#         """
#         Override as we don't need variable timestepping here. 
#         """
#         pass

#     def _setup_initial_node(self, guess):
#         """
#         set up the progress variable.
#         """
#         current_node = super()._setup_initial_node(guess)
#         progress = self.opti.variable()
#         self.progress = [progress]
#         tracking_error = ca.sumsqr(self.track(progress) - current_node.state[:3])
#         self.constraint(progress == self.initial_progress, description="Initial Progress Lock")

#         self.progress_loss = 1000 * tracking_error - 100 * progress

#         return current_node
    
#     def _setup_step(self, index, current_node, guess):
#         print(current_node)
#         current_node = super()._setup_step(index = index, current_node = current_node, guess = guess)
#         progress = self.opti.variable()
#         self.progress.append(progress)
#         tracking_error = ca.sumsqr(self.track(progress) - current_node.state[:3])
#         self.progress_loss += 100 * tracking_error - 10 * progress
#         self.constraint(self.progress[-1] >= self.progress[-2], description="Progress Constraint")
#         self.constraint(self.opti.bounded(0, self.progress[-1], 1), description="Progress Constraint")
#         return current_node

#     def _setup_variables(self, nodes:List[ControlNode]):
#         super()._setup_variables(nodes)
#         self.progress = ca.vcat(self.progress)

#         if self.verbose:
#             print("Progress Shape: ", self.progress.shape)

#     def setup(self, guess, progress):
#         super().setup(guess)
#         self.opti.set_initial(self.progress, progress)

#     # def initialise(self, initial_state):
#     #     guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
#     #     guess[:self.state_dim, 0] = initial_state.toarray().flatten()

#     #     for i in range(self.num_nodes):
#     #         guess[:self.state_dim, i + 1] = self.dynamics(guess[:self.state_dim, i], guess[self.state_dim:, i], self.dt).toarray().flatten()

#     #     progress = [0.0]
#     #     for i in range(1, self.num_nodes + 1):
#     #         dx = guess[:3, i] - guess[:3, i - 1]
#     #         progress.append(progress[-1] + np.linalg.norm(dx))
#     #     progress = np.array(progress)
#     #     progress /= progress[-1]  # Normalize to [0, 1]


#     #     return guess, progress

#     def initialise(self, initial_state, current_progress):
#         guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
#         guess[:self.state_dim, 0] = initial_state.toarray().flatten()
        
#         # Propagate forward using nominal dynamics
#         for i in range(self.num_nodes):
#             guess[:self.state_dim, i + 1] = self.dynamics(
#                 guess[:self.state_dim, i],
#                 guess[self.state_dim:, i],
#                 self.dt
#             ).toarray().flatten()

#         # Estimate progress over the horizon, starting from current_progress
#         progress = [current_progress]
#         for i in range(1, self.num_nodes + 1):
#             dx = guess[:3, i] - guess[:3, i - 1]
#             delta = np.linalg.norm(dx) / self.track_length  # normalize relative to full path
#             progress.append(progress[-1] + delta)
#         self.initial_progress = current_progress

#         return guess, np.array(progress)

    
#     def callback(self, iteration:int):
#         """
#         To be implemented
#         """
#         pass

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
    progress: Optional[ca.MX] = None

class MHTT(ControlProblem):
    def __init__(self, *, track, track_length, dt, **kwargs):
        super().__init__(**kwargs)
        self.track = track
        self.dt = dt
        self.time = dt * self.num_nodes
        self.track_length = track_length
        
        # Create parameters for changing values between iterations
        self.initial_progress_param = self.opti.parameter()
        self.initial_state_param = self.opti.parameter(self.state_dim)
    
    def loss(self, nodes, time=None):
        return self.progress_loss

    def _setup_time(self):
        """
        Override as we don't need variable timestepping here. 
        """
        pass

    def _setup_initial_node(self, guess):
        """
        Set up the progress variable and use parameter for initial state.
        """
        current_node = super()._setup_initial_node(guess)
        
        # Constrain initial state to parameter value
        self.constraint(current_node.state == self.initial_state_param, description="Initial State Lock")
        
        # Use parameter for progress
        progress = self.opti.variable()
        self.progress = [progress]
        tracking_error = ca.sumsqr(self.track(progress) - current_node.state[:3])
        self.constraint(progress == self.initial_progress_param, description="Initial Progress Lock")

        self.progress_loss = 1000 * tracking_error - 100 * progress

        return current_node
    
    def _setup_step(self, index, current_node, guess):
        current_node = super()._setup_step(index=index, current_node=current_node, guess=guess)
        progress = self.opti.variable()
        self.progress.append(progress)
        tracking_error = ca.sumsqr(self.track(progress) - current_node.state[:3])
        self.progress_loss += 100 * tracking_error - 10 * progress
        self.constraint(self.progress[-1] >= self.progress[-2], description="Progress Constraint")
        self.constraint(self.opti.bounded(0, self.progress[-1], 1), description="Progress Constraint")
        return current_node

    def _setup_variables(self, nodes: List[ControlNode]):
        super()._setup_variables(nodes)
        self.progress = ca.vcat(self.progress)

        if self.verbose:
            print("Progress Shape: ", self.progress.shape)

    def setup(self, guess, progress, initial_state=None):
        """
        Set up the optimization problem with initial guesses and parameter values.
        """
        super().setup(guess)
        self.opti.set_initial(self.progress, progress)
        
        # Set parameter values if provided
        if initial_state is not None:
            self.opti.set_value(self.initial_state_param, initial_state)
        self.opti.set_value(self.initial_progress_param, progress[0])

    def initialise(self, initial_state, current_progress):
        """
        Initialize the optimization problem with initial state and progress.
        """
        guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
        guess[:self.state_dim, 0] = initial_state.toarray().flatten()
        
        # Propagate forward using nominal dynamics
        for i in range(self.num_nodes):
            guess[:self.state_dim, i + 1] = self.dynamics(
                guess[:self.state_dim, i],
                guess[self.state_dim:, i],
                self.dt
            ).toarray().flatten()

        # Estimate progress over the horizon, starting from current_progress
        progress = [current_progress]
        for i in range(1, self.num_nodes + 1):
            dx = guess[:3, i] - guess[:3, i - 1]
            delta = np.linalg.norm(dx) / self.track_length  # normalize relative to full path
            progress.append(progress[-1] + delta)
            
        # Set parameter values
        self.opti.set_value(self.initial_state_param, initial_state)
        self.opti.set_value(self.initial_progress_param, current_progress)

        return guess, np.array(progress)
    
    def update_parameters(self, initial_state, current_progress):
        """
        Update parameters between MPC iterations.
        """
        self.opti.set_value(self.initial_state_param, initial_state)
        self.opti.set_value(self.initial_progress_param, current_progress)
    
    def callback(self, iteration: int):
        """
        To be implemented
        """
        pass
