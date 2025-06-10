import numpy as np
from dataclasses import dataclass
import casadi as ca
from aircraft.control.base import ControlProblem, ControlNode
from dataclasses import dataclass
from typing import Optional, List, Sequence
from aircraft.control.initialisation import DubinsInitialiser
from aircraft.plotting.plotting import TrajectoryData
import matplotlib.pyplot as plt


@dataclass
class ProgressNode(ControlNode):
    track_progress: Optional[ca.MX] = None
    progress_rate:Optional[ca.MX] = None
    tracking_error: Optional[ca.MX] = None

    @classmethod
    def from_control_node(cls, control_node: ControlNode, 
                          track_progress: Optional[ca.MX] = None, 
                          progress_rate: Optional[ca.MX] = None,
                          tracking_error: Optional[ca.MX] = None):
        return cls(
            index=control_node.index,
            state=control_node.state,
            control=control_node.control,
            progress=control_node.progress,
            track_progress=track_progress,
            progress_rate = progress_rate,
            tracking_error = tracking_error
        )

class MHTT(ControlProblem):
    def __init__(self, *, track:DubinsInitialiser, dt, **kwargs):
        assert kwargs.get('opts').get('time', 'fixed') == 'fixed', 'can only run mhtt with fixed time'
        super().__init__(**kwargs)
        self.track = track
        self.dt = dt
        self.track_length = self.track.length()
        self.progress_guess_parameter = self.opti.parameter(self.num_nodes + 1)
        self.state_memory = []
        self.control_memory = []
        self.time_memory = 0

    def loss(self, nodes, time=None):
        loss = super().loss(nodes, time)
        
        # Weights
        w_tracking = 10.0
        w_progress = 5.0
        w_progress_rate = 2.0
        w_backward = 50.0
        
        tracking_loss = 0
        progress_reward = 0
        progress_rate_reward = 0
        backward_penalty = 0
        
        for node in nodes[1:]:
            # Tracking error
            if hasattr(node, 'tracking_error'):
                tracking_loss += node.tracking_error
            
            # Progress reward
            progress_reward += node.track_progress
            
            # Progress rate reward and backward penalty
            if hasattr(node, 'progress_rate'):
                progress_rate_reward += node.progress_rate
                backward_penalty += ca.fmax(0, -node.progress_rate)**2
        
        loss += (w_tracking * tracking_loss 
                - w_progress * progress_reward 
                - w_progress_rate * progress_rate_reward
                + w_backward * backward_penalty)
        
        return loss

    def _setup_initial_node(self, guess):
        """
        Set up the progress variable and use parameter for initial state.
        """
        current_node = self._make_node(index=0, guess=guess, enforce_state_constraint=True)

        return current_node
    
    def _make_node(self, index: int, guess: np.ndarray, enforce_state_constraint: bool = False) -> ProgressNode:
        control_node = super()._make_node(index, guess, enforce_state_constraint)

        progress_node = ProgressNode.from_control_node(control_node, track_progress=self.opti.variable())
        if enforce_state_constraint:
            self.constraint(progress_node.track_progress == self.progress_guess_parameter[index])

        return progress_node
    
    def _setup_step(self, index, current_node: ProgressNode, guess):
        next_node: ProgressNode = super()._setup_step(index=index, current_node=current_node, guess=guess)
        
        # Get track information at current progress
        tangent = self.track.eval_tangent(current_node.track_progress)
        tangent_norm = tangent / (ca.norm_2(tangent) + 1e-6)
        track_pos = self.track.eval(current_node.track_progress)
        
        # Current state
        pos = current_node.state[:3]
        vel = current_node.state[3:6]
        
        # Progress dynamics
        s_dot = ca.dot(vel, tangent_norm) / self.track_length
        
        # Optional: position-based correction for robustness
        pos_err = pos - track_pos
        delta_s_correction = ca.dot(pos_err, tangent_norm) / self.track_length
        
        # Combined progress update
        predicted_next_progress = (current_node.track_progress + 
                                s_dot * self.dt + 
                                0.05 * delta_s_correction)  # small correction weight
        
        # Constraints
        self.constraint(next_node.track_progress == predicted_next_progress)
        self.constraint(
            self.opti.bounded(0, next_node.track_progress, 1),
            description="Progress Bounds"
        )
        
        # Store for loss computation
        next_node.progress_rate = s_dot
        next_node.tracking_error = ca.sumsqr(pos - track_pos)
        
        return next_node

    def _setup_variables(self, nodes: Sequence[ProgressNode]):
        super()._setup_variables(nodes)
        assert isinstance(nodes[0], ProgressNode)
        self.track_progress = ca.vcat([node.track_progress for node in nodes])

        if self.verbose:
            print("Progress Shape: ", self.track_progress.shape)

    def setup(self, guess, initial_state=None):
        """
        Set up the optimization problem with initial guesses and parameter values.
        """
        self.opti.set_value(self.progress_guess_parameter, guess[-1, :])
        super().setup(guess)

    def initialise(self, initial_state, current_progress):
        guess = np.zeros((self.state_dim + self.control_dim + 1, self.num_nodes + 1))
        guess[:self.state_dim, 0] = initial_state.toarray().flatten()
        
        # Forward propagate dynamics
        for i in range(self.num_nodes):
            guess[:self.state_dim, i + 1] = self.dynamics(
                guess[:self.state_dim, i],
                guess[self.state_dim:self.state_dim+self.control_dim, i],
                self.dt
            ).toarray().flatten()

        # Better progress initialization
        guess[-1, 0] = current_progress
        
        # Estimate progress based on velocity projection
        for i in range(1, self.num_nodes + 1):
            pos = guess[:3, i-1]
            vel = guess[3:6, i-1]
            s_current = guess[-1, i-1]
            
            # Get tangent at current progress
            try:
                tangent = self.track.eval_tangent(s_current)
                tangent_norm = tangent / np.linalg.norm(tangent)
                
                # Estimate progress increment
                s_dot = np.dot(vel, tangent_norm) / self.track_length
                delta_s = s_dot * self.dt
                
                guess[-1, i] = np.clip(s_current + delta_s, 0.0, 1.0)
            except:
                print("Fallback")
                # Fallback to uniform spacing
                guess[-1, i] = min(current_progress + i * 0.01, 1.0)

        return guess
    def update_parameters(self, initial_state, current_progress):
        """
        Update parameters between MPC iterations.
        """
        pass
    
    def callback(self, iteration: int):
        if iteration % 10 == 0:
            print(self.opti.debug.value(self.track_progress))


    def solve(self, warm_start: Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start=warm_start)

        return sol
    

#     import numpy as np
# from dataclasses import dataclass
# import casadi as ca
# from aircraft.control.base import ControlProblem, ControlNode
# from dataclasses import dataclass
# from typing import Optional, List, Sequence
# from aircraft.control.initialisation import DubinsInitialiser
# from aircraft.plotting.plotting import TrajectoryData
# import matplotlib.pyplot as plt


# @dataclass
# class ProgressNode(ControlNode):
#     track_progress: Optional[ca.MX] = None
#     progress_rate:Optional[ca.MX] = None
#     tracking_error: Optional[ca.MX] = None

#     @classmethod
#     def from_control_node(cls, control_node: ControlNode, 
#                           track_progress: Optional[ca.MX] = None, 
#                           progress_rate: Optional[ca.MX] = None,
#                           tracking_error: Optional[ca.MX] = None):
#         return cls(
#             index=control_node.index,
#             state=control_node.state,
#             control=control_node.control,
#             progress=control_node.progress,
#             track_progress=track_progress,
#             progress_rate = progress_rate,
#             tracking_error = tracking_error
#         )

# class MHTT(ControlProblem):
#     def __init__(self, *, track:DubinsInitialiser, dt, **kwargs):
#         assert kwargs.get('opts').get('time', 'fixed') == 'fixed', 'can only run mhtt with fixed time'
#         super().__init__(**kwargs)
#         self.track = track
#         self.dt = dt
#         self.track_length = self.track.length()
        
#         # self.initial_state_param = self.opti.parameter(self.state_dim)
#         self.progress_guess_parameter = self.opti.parameter(self.num_nodes + 1)
#         self.state_memory = []
#         self.control_memory = []
#         self.time_memory = 0
    
#     # def loss(self, nodes, time=None):
#     #     loss = super().loss(nodes, time)
#     #     # Add progress loss
#     #     weights = ca.linspace(0, 100, self.num_nodes + 1)
#     #     tracking_error = 0
#     #     for i in range(self.num_nodes + 1):
#     #         tracking_error += ca.sumsqr(self.state[:3, i] - self.track.eval(self.track_progress[i]))
#     #     # evaluated_track = ca.vertcat(self.track.eval(ca.vertcat(self.track_progress)))

#     #     # tracking_error = ca.sumsqr(self.state - evaluated_track)
#     #     progress_loss = -ca.sum1(self.track_progress) #-weights.T @ 

#     #     loss += progress_loss
#     #     loss += tracking_error
#     #     # loss += ca.sumsqr(self.state[-1, :2] - ca.DM([50, 50]).T)
#     #     return loss

#     def loss(self, nodes, time=None):
#         loss = super().loss(nodes, time)
        
#         # Weights
#         w_tracking = 10.0
#         w_progress = 5.0
#         w_progress_rate = 2.0
#         w_backward = 50.0
        
#         tracking_loss = 0
#         progress_reward = 0
#         progress_rate_reward = 0
#         backward_penalty = 0
        
#         for node in nodes[1:]:
#             # Tracking error
#             if hasattr(node, 'tracking_error'):
#                 tracking_loss += node.tracking_error
            
#             # Progress reward
#             progress_reward += node.track_progress
            
#             # Progress rate reward and backward penalty
#             if hasattr(node, 'progress_rate'):
#                 progress_rate_reward += node.progress_rate
#                 backward_penalty += ca.fmax(0, -node.progress_rate)**2
        
#         loss += (w_tracking * tracking_loss 
#                 - w_progress * progress_reward 
#                 - w_progress_rate * progress_rate_reward
#                 + w_backward * backward_penalty)
        
#         return loss

#     def _setup_initial_node(self, guess):
#         """
#         Set up the progress variable and use parameter for initial state.
#         """
#         current_node = self._make_node(index=0, guess=guess, enforce_state_constraint=False)
        
#         # Constrain initial state to parameter value
#         # self.constraint(current_node.state == self.initial_state_param, description="Initial State Lock")

#         # self.constraint(current_node.track_progress == self.initial_progress_param, description="Initial Progress Lock")

#         return current_node
    
#     def _make_node(self, index: int, guess: np.ndarray, enforce_state_constraint: bool = False) -> ProgressNode:
#         control_node = super()._make_node(index, guess, enforce_state_constraint)

#         progress_node = ProgressNode.from_control_node(control_node, track_progress=self.opti.variable())
#         if enforce_state_constraint:
#             self.constraint(progress_node.track_progress == self.progress_guess_parameter[index])

#         return progress_node
    
#     # def _setup_step(self, index, current_node:ProgressNode, guess):
#     #     next_node:ProgressNode = super()._setup_step(index=index, current_node=current_node, guess=guess)


#     #     # Progress calculation
#     #     # Get tangent and position at current progress
#     #     tangent = self.track.eval_tangent(current_node.track_progress)
#     #     tangent_norm = tangent / ca.norm_2(tangent)
#     #     track_pos = self.track.eval(current_node.track_progress)
#     #     position = current_node.state[:3]

#     #     # Position error
#     #     pos_err =position  - track_pos  # shape (3,)
#     #     vel = current_node.state[3:6]  # shape (3,)

#     #     # Project error onto tangent to estimate delta_s
#     #     # delta_s = ca.dot(pos_err, tangent) / self.track_length
#     #     delta_s = ca.dot(pos_err, tangent) / self.track_length
#     #     s_dot = ca.dot(vel, tangent_norm) / self.track_length
#     #     # Constrain next progress to be close to projected value
#     #     # predicted_next_progress = current_node.track_progress + delta_s
#     #     predicted_next_progress = current_node.track_progress + s_dot * self.dt + 0.05 * delta_s
#     #     self.constraint(next_node.track_progress == predicted_next_progress)

#     #     # Optional: constrain progress to stay within bounds
#     #     self.constraint(
#     #         self.opti.bounded(0, next_node.track_progress, 1),
#     #         description="Progress Bounds"
#     #     )

#     #     next_node.progress_rate = s_dot
#     #     next_node.tracking_error = ca.sumsqr(position - track_pos)

#     #     return next_node
#     def _setup_step(self, index, current_node: ProgressNode, guess):
#         next_node: ProgressNode = super()._setup_step(index=index, current_node=current_node, guess=guess)
        
#         # Get track information at current progress
#         tangent = self.track.eval_tangent(current_node.track_progress)
#         tangent_norm = tangent / ca.norm_2(tangent)  # L2 norm
#         track_pos = self.track.eval(current_node.track_progress)
        
#         # Current state
#         pos = current_node.state[:3]
#         vel = current_node.state[3:6]
        
#         # Progress dynamics
#         s_dot = ca.dot(vel, tangent_norm) / self.track_length
        
#         # Optional: position-based correction for robustness
#         pos_err = pos - track_pos
#         delta_s_correction = ca.dot(pos_err, tangent_norm) / self.track_length
        
#         # Combined progress update
#         predicted_next_progress = (current_node.track_progress + 
#                                 s_dot * self.dt + 
#                                 0.05 * delta_s_correction)  # small correction weight
        
#         # Constraints
#         self.constraint(next_node.track_progress == predicted_next_progress)
#         self.constraint(
#             self.opti.bounded(0, next_node.track_progress, 1),
#             description="Progress Bounds"
#         )
        
#         # Store for loss computation
#         next_node.progress_rate = s_dot
#         next_node.tracking_error = ca.sumsqr(pos - track_pos)
        
#         return next_node

#     def _setup_variables(self, nodes: Sequence[ProgressNode]):
#         super()._setup_variables(nodes)
#         assert isinstance(nodes[0], ProgressNode)
#         self.track_progress = ca.vcat([node.track_progress for node in nodes])

#         if self.verbose:
#             print("Progress Shape: ", self.track_progress.shape)

#     def setup(self, guess, initial_state=None):
#         """
#         Set up the optimization problem with initial guesses and parameter values.
#         """
#         self.opti.set_value(self.progress_guess_parameter, guess[-1, :])
#         super().setup(guess)
#         # self.opti.set_initial(self.track_progress, progress)
        
#         # Set parameter values if provided
#         # if initial_state is not None:
#         #     self.opti.set_value(self.initial_state_param, initial_state)
#         # self.opti.set_value(self.initial_progress_param, guess[-1, 0])

#     def initialise(self, initial_state, current_progress):
#         """
#         Initialize the optimization problem with initial state and progress.
#         """
#         guess = np.zeros((self.state_dim + self.control_dim + 1, self.num_nodes + 1))
#         guess[:self.state_dim, 0] = initial_state.toarray().flatten()
        
#         # Propagate forward using nominal dynamics
#         for i in range(self.num_nodes):
#             guess[:self.state_dim, i + 1] = self.dynamics(
#                 guess[:self.state_dim, i],
#                 guess[self.state_dim:self.state_dim+self.control_dim, i],
#                 self.dt
#             ).toarray().flatten()

#         # Estimate progress over the horizon, starting from current_progress
#         # progress = [current_progress]
#         guess[-1, 0] = current_progress
#         for i in range(1, self.num_nodes + 1):
#             dx = guess[:3, i] - guess[:3, i - 1]
#             delta = np.linalg.norm(dx) / self.track_length  # normalize relative to full path
#             guess[-1, i] = guess[-1, i-1] + delta
            
#         # Set parameter values
#         # self.opti.set_value(self.initial_state_param, initial_state)
#         # self.opti.set_value(self.initial_progress_param, current_progress)

#         return guess
    
#     def update_parameters(self, initial_state, current_progress):
#         """
#         Update parameters between MPC iterations.
#         """
#         # self.opti.set_value(self.initial_state_param, initial_state)
#         # self.opti.set_value(self.initial_progress_param, current_progress)
    
#     def callback(self, iteration: int):
#         if iteration % 10 == 0:
#             print(self.opti.debug.value(self.track_progress))
#         return None
#         # Plot full trajectory every N iterations
#         if self.plotter and iteration % 10 == 5:
#             current_state = self.opti.debug.value(self.state)[:, 1:]
#             current_control = self.opti.debug.value(self.control)
#             current_time = np.linspace(
#                 iteration * self.dt,
#                 (iteration + self.num_nodes) * self.dt,
#                 self.num_nodes
#             )

#             # Combine all past and current trajectories
#             all_states = self.state_memory + [current_state]
#             all_controls = self.control_memory + [current_control]
#             all_times = self.time_memory + self.time#[current_time]

#             full_state = np.hstack(all_states)  # shape: (state_dim, total_steps)
#             full_control = np.hstack(all_controls)  # shape: (control_dim, total_steps)
#             full_time = all_times#np.hstack(all_times)  # shape: (total_steps,)

#             trajectory_data = TrajectoryData(
#                 state=full_state,
#                 control=full_control,
#                 time=full_time
#             )

#             self.plotter.plot(trajectory_data=trajectory_data)
#             plt.draw()
#             self.plotter.figure.canvas.start_event_loop(0.0002)


#     def solve(self, warm_start: Optional[ca.OptiSol] = None):
#         sol = super().solve(warm_start=warm_start)
        
#         # Store current horizon (excluding the first overlapping node)
#         # state_val = self.opti.debug.value(self.state)[:, 1:21]
#         # control_val = self.opti.debug.value(self.control[:, :20])

#         # self.state_memory.append(state_val)
#         # self.control_memory.append(control_val)
#         # self.time_memory += self.time

#         return sol