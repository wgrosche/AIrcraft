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
from aircraft.plotting.plotting import TrajectoryData
import matplotlib.pyplot as plt
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

        self.state_memory = []
        self.control_memory = []
        self.time_memory = 0
    
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

        self.progress_loss = 10 * tracking_error - 1* progress

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
    
    # def callback(self, iteration: int):
    #     state_val = self.opti.debug.value(self.state)[:,1:]
    #     control_val = self.opti.debug.value(self.control)
    #     time_val = self.time

    #     # Store entire horizon at this iteration
    #     self.state_memory.append(state_val)
    #     self.control_memory.append(control_val)
    #     self.time_memory.append(np.linspace(iteration * self.dt, (iteration + self.num_nodes) * self.dt, self.num_nodes + 1))

    #     # Plot only current trajectory if desired
    #     if self.plotter and iteration % 10 == 5:
    #         trajectory_data = TrajectoryData(
    #             state=state_val,
    #             control=control_val,
    #             time=self.time_memory[-1][-1]
    #         )
    #         self.plotter.plot(trajectory_data=trajectory_data)
    #         plt.draw()
    #         self.plotter.figure.canvas.start_event_loop(0.0002)
    def callback(self, iteration: int):
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

    # def callback(self, iteration: int):

        

    #     # Plot full trajectory every N iterations
    #     if self.plotter and iteration % 10 == 5:
    #         # state_val = self.opti.debug.value(self.state)[:, 1:]
    #         # control_val = self.opti.debug.value(self.control)
    #         # # Concatenate all horizons into a single long trajectory
    #         # states = [self.state_memory[0]] + [s[:, 1:] for s in self.state_memory[1:]]
    #         # controls = self.control_memory
    #         # times = [self.time_memory[0]] + [t[1:] for t in self.time_memory[1:]]
    #         if self.state_memory:
    #             full_state = np.hstack([self.state_memory, self.opti.debug.value(self.state)[:, 1:]])
    #         else:
    #             full_state = self.opti.debug.value(self.state)[:, 1:]
    #         if self.control_memory:  
    #             full_control = np.hstack([self.control_memory, self.opti.debug.value(self.control)])
    #         else:
    #             full_control = self.opti.debug.value(self.control)
    #         full_time = self.time_memory + self.time

    #         trajectory_data = TrajectoryData(
    #             state=full_state,
    #             control=full_control,
    #             time=full_time
    #         )

    #         self.plotter.plot(trajectory_data=trajectory_data)
    #         plt.draw()
    #         self.plotter.figure.canvas.start_event_loop(0.0002)


    def solve(self, warm_start: Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start=warm_start)
        
        # Store current horizon (excluding the first overlapping node)
        state_val = self.opti.debug.value(self.state)[:, 1:21]
        control_val = self.opti.debug.value(self.control[:, :20])

        self.state_memory.append(state_val)
        self.control_memory.append(control_val)
        self.time_memory += self.time

        return sol