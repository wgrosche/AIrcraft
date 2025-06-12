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

    # def loss(self, nodes, time=None):
    #     loss = super().loss(nodes, time)
        
    #     # Weights
    #     w_tracking = 10.0
    #     w_progress = 5.0
    #     w_progress_rate = 2.0
    #     w_backward = 50.0
        
    #     tracking_loss = 0
    #     progress_reward = 0
    #     progress_rate_reward = 0
    #     backward_penalty = 0
        
    #     for node in nodes[1:]:
    #         # Tracking error
    #         if hasattr(node, 'tracking_error'):
    #             tracking_loss += node.tracking_error
            
    #         # Progress reward
    #         progress_reward += node.track_progress
            
    #         # Progress rate reward and backward penalty
    #         if hasattr(node, 'progress_rate'):
    #             progress_rate_reward += node.progress_rate
    #             backward_penalty += ca.fmax(0, -node.progress_rate)**2
        
    #     loss += (w_tracking * tracking_loss 
    #             - w_progress * progress_reward 
    #             - w_progress_rate * progress_rate_reward
    #             + w_backward * backward_penalty)
        
    #     return loss

    def loss(self, nodes, time=None):
        loss = super().loss(nodes, time)

        # Weights (tune as needed)
        w_tracking = 10.0
        w_progress = 5.0
        w_progress_rate = 2.0
        w_backward = 50.0
        w_terminal_align = 20.0
        w_low_velocity = 10.0
        w_control = 1.0

        # Accumulators
        tracking_loss = 0
        progress_reward = 0
        progress_rate_reward = 0
        backward_penalty = 0
        low_velocity_penalty = 0
        control_effort = 0

        for i, node in enumerate(nodes[1:], 1):  # skip first node for rate
            # Tracking error (if defined)
            if hasattr(node, 'tracking_error'):
                tracking_loss += node.tracking_error

            # Decaying progress reward (earlier progress is more valuable)
            decay_weight = ca.exp(-0.05 * i)  # or use 1 - i / num_nodes
            progress_reward += decay_weight * node.track_progress

            # Progress rate reward and backward penalty
            if hasattr(node, 'progress_rate'):
                progress_rate_reward += node.progress_rate
                backward_penalty += ca.fmax(0, -node.progress_rate) ** 2

            # Penalize low forward velocity
            velocity = ca.norm_2(node.state[3:6])  # assuming [vx, vy, vz]
            low_velocity_penalty += ca.fmax(0.1 - velocity, 0) ** 2

            # Penalize control effort
            control_effort += ca.sumsqr(node.control)

        # Terminal alignment with final track point
        final_pos = nodes[-1].state[:3]
        goal_pos = self.track.eval(1.0)  # assumes Dubins track parameterized in [0,1]
        terminal_tracking_error = ca.norm_2(final_pos - goal_pos)

        # Add all to loss
        loss += (
            w_tracking * tracking_loss
            - w_progress * progress_reward
            - w_progress_rate * progress_rate_reward
            + w_backward * backward_penalty
            + w_low_velocity * low_velocity_penalty
            + w_terminal_align * terminal_tracking_error
            + w_control * control_effort
        )

        return loss
    
    def set_initial_from_array(self, guess_array: np.ndarray):
        """
        Set initial guesses for the existing variables.
        """
        self.opti.set_initial(self.state, guess_array[:self.state_dim, :])
        self.opti.set_initial(self.control, guess_array[self.state_dim:self.state_dim+self.control_dim, :-1])
        self.opti.set_initial(self.track_progress, guess_array[-1, :])


    def _setup_initial_node(self, guess):
        """
        Set up the progress variable and use parameter for initial state.
        """
        # print("Initial state here: ", guess[:, 0])
        return_node = self._make_node(index=0, guess=guess, enforce_state_constraint=True)
        
        
        # print("Constraint descriptions: ", self.constraint_descriptions)
        return return_node
    
    def _make_node(self, index: int, guess: np.ndarray, enforce_state_constraint: bool = False) -> ProgressNode:
        
        
        control_node = super()._make_node(index, guess, enforce_state_constraint)

        progress_node = ProgressNode.from_control_node(control_node, track_progress=self.opti.variable())
        self.constraint(
            self.opti.bounded(0, progress_node.track_progress, 1),
            description="Progress Bounds"
        )
        if enforce_state_constraint:
            self.constraint(progress_node.track_progress == self.progress_guess_parameter[index])

        return progress_node
    
    def _setup_step(self, index, current_node: ProgressNode, guess):
        next_node: ProgressNode = super()._setup_step(index=index, current_node=current_node, guess=guess)
        
        # Get track information at current progress
        tangent = self.track.eval_tangent(current_node.track_progress)
        norm = ca.norm_2(tangent)
        norm_safe = ca.if_else(norm > 1e-3, norm, 1.0)
        tangent_norm = tangent / norm_safe
        track_pos = self.track.eval(current_node.track_progress)
        
        # Current state
        pos = current_node.state[:3]
        vel = current_node.state[3:6]
        assert self.track_length > 1e-6
        
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
        self.constraint(next_node.track_progress <= predicted_next_progress)
        self.opti.set_initial(next_node.track_progress, guess[-1, index])
        
        # Store for loss computation
        next_node.progress_rate = s_dot
        next_node.tracking_error = ca.sumsqr(pos_err)
        
        return next_node

    def _setup_variables(self, nodes: Sequence[ProgressNode]):
        super()._setup_variables(nodes)
        assert isinstance(nodes[0], ProgressNode)
        self.track_progress = ca.vcat([node.track_progress for node in nodes])
        self.nodes = nodes
        self.progress_rates = ca.vcat([node.progress_rate for node in nodes[1:]])
        self.tracking_errors = ca.vcat([node.tracking_error for node in nodes[1:]])

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
                tangent_norm = tangent / ca.norm_2(tangent)
                
                # Estimate progress increment
                s_dot = np.dot(vel, tangent_norm) / self.track_length
                delta_s = s_dot * self.dt
                
                guess[-1, i] = np.clip(s_current + delta_s, 0, 1.0)
            except:
                print("Fallback")
                # Fallback to uniform spacing
                guess[-1, i] = min(current_progress + i * 0.01, 1.0)

        return guess
    def update_parameters(self, guess):
        """
        Update parameters between MPC iterations.
        """
        self.opti.set_value(self.progress_guess_parameter, guess[-1, :])
        self.opti.set_value(self.state_guess_parameter, guess[:self.state_dim, :])
        self.opti.set_value(self.control_guess_parameter, guess[self.state_dim:self.state_dim + self.control_dim, :])

    
    def callback(self, iteration: int):
        pass
        # if iteration % 10 == 0:
        #     print("Progress: ", self.opti.debug.value(self.track_progress))
        #     print("Tracking errors: ", self.opti.debug.value(self.tracking_errors))
        #     print("Progress rates: ", self.opti.debug.value(self.progress_rates))
        #     print("Positions: ", self.opti.debug.value(self.state)[2,:])
        #     print("Track evals", [self.track.eval_tangent(s) for s in self.opti.debug.value(self.track_progress)])


    def solve(self, warm_start: Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start=warm_start)

        return sol
    

#  TODO: Investigate caching the nlp

# import casadi as ca

# # 1. Extract the NLP
# nlp_dict = {
#     'x': opti.x,
#     'f': opti.f,
#     'g': opti.g
# }

# # 2. Extract solver options
# opts = {
#     "ipopt.print_level": 0,
#     "print_time": False
# }

# # 3. Create the solver only once and reuse
# solver = ca.nlpsol("solver", "ipopt", nlp_dict, opts)

# # 4. Solve by feeding values into arguments
# arg = {
#     "x0": opti.debug.value(opti.x),
#     "p": opti.debug.value(opti.p) if opti.p.numel() > 0 else [],
#     "lbx": opti.lbx,
#     "ubx": opti.ubx,
#     "lbg": opti.lbg,
#     "ubg": opti.ubg,
# }

# res = solver(**arg)
