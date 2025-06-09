import numpy as np
from dataclasses import dataclass
import casadi as ca
from aircraft.control.base import ControlProblem, ControlNode
from dataclasses import dataclass
from typing import Optional, List
from aircraft.control.initialisation import DubinsInitialiser
from aircraft.plotting.plotting import TrajectoryData
import matplotlib.pyplot as plt


@dataclass
class ProgressNode(ControlNode):
    track_progress: Optional[ca.MX] = None

    @classmethod
    def from_control_node(cls, control_node: ControlNode, track_progress: Optional[ca.MX] = None):
        return cls(
            index=control_node.index,
            state=control_node.state,
            control=control_node.control,
            progress=control_node.progress,
            track_progress=track_progress
        )

class MHTT(ControlProblem):
    def __init__(self, *, track:DubinsInitialiser, dt, **kwargs):
        assert kwargs.get('opts').get('time', 'fixed') == 'fixed', 'can only run mhtt with fixed time'
        super().__init__(**kwargs)
        self.track = track
        self.dt = dt
        self.time = dt * self.num_nodes
        self.track_length = self.track.length()
        
        self.initial_progress_param = self.opti.parameter()
        self.initial_state_param = self.opti.parameter(self.state_dim)

        self.state_memory = []
        self.control_memory = []
        self.time_memory = 0
    
    def loss(self, nodes, time=None):
        loss = super().loss(nodes, time)
        # Add progress loss
        progress_loss = ca.sumsqr(self.track_progress)

        loss += progress_loss
        return loss

    def _setup_initial_node(self, guess):
        """
        Set up the progress variable and use parameter for initial state.
        """
        current_node = self._make_node(index=0, guess=guess, enforce_state_constraint=False)
        
        # Constrain initial state to parameter value
        self.constraint(current_node.state == self.initial_state_param, description="Initial State Lock")

        self.constraint(current_node.track_progress == self.initial_progress_param, description="Initial Progress Lock")

        return current_node
    
    def _make_node(self, index: int, guess: np.ndarray, enforce_state_constraint: bool = False) -> ProgressNode:
        control_node = super()._make_node(index, guess, enforce_state_constraint)

        progress_node = ProgressNode.from_control_node(control_node, track_progress=self.opti.variable())

        return progress_node
    
    def _setup_step(self, index, current_node:ProgressNode, guess):
        next_node:ProgressNode = super()._setup_step(index=index, current_node=current_node, guess=guess)

        # Progress calculation
        # Get tangent and position at current progress
        tangent = self.track.eval_tangent(current_node.track_progress)
        track_pos = self.track.eval(current_node.track_progress)

        # Position error
        pos_err = current_node.state[:3] - track_pos  # shape (3,)

        # Project error onto tangent to estimate delta_s
        delta_s = ca.dot(pos_err, tangent) / self.track_length

        # Constrain next progress to be close to projected value
        predicted_next_progress = current_node.track_progress + delta_s
        self.constraint(next_node.track_progress == predicted_next_progress)

        # Optional: constrain progress to stay within bounds
        self.constraint(
            self.opti.bounded(0, next_node.track_progress, 1),
            description="Progress Bounds"
        )

        return current_node

    def _setup_variables(self, nodes: List[ProgressNode]):
        super()._setup_variables(nodes)
        self.track_progress = ca.vcat([node.track_progress for node in nodes])

        if self.verbose:
            print("Progress Shape: ", self.track_progress.shape)

    def setup(self, guess, progress, initial_state=None):
        """
        Set up the optimization problem with initial guesses and parameter values.
        """
        super().setup(guess)
        self.opti.set_initial(self.track_progress, progress)
        
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
        pass
        # Plot full trajectory every N iterations
        if self.plotter and iteration % 10 == 5:
            current_state = self.opti.debug.value(self.state)[:, 1:]
            current_control = self.opti.debug.value(self.control)
            current_time = np.linspace(
                iteration * self.dt,
                (iteration + self.num_nodes) * self.dt,
                self.num_nodes
            )

            # Combine all past and current trajectories
            all_states = self.state_memory + [current_state]
            all_controls = self.control_memory + [current_control]
            all_times = self.time_memory + self.time#[current_time]

            full_state = np.hstack(all_states)  # shape: (state_dim, total_steps)
            full_control = np.hstack(all_controls)  # shape: (control_dim, total_steps)
            full_time = all_times#np.hstack(all_times)  # shape: (total_steps,)

            trajectory_data = TrajectoryData(
                state=full_state,
                control=full_control,
                time=full_time
            )

            self.plotter.plot(trajectory_data=trajectory_data)
            plt.draw()
            self.plotter.figure.canvas.start_event_loop(0.0002)


    def solve(self, warm_start: Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start=warm_start)
        
        # Store current horizon (excluding the first overlapping node)
        state_val = self.opti.debug.value(self.state)[:, 1:21]
        control_val = self.opti.debug.value(self.control[:, :20])

        self.state_memory.append(state_val)
        self.control_memory.append(control_val)
        self.time_memory += self.time

        return sol