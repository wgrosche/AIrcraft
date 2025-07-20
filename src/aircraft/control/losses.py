import casadi as ca

    def loss(self, nodes, time=None):
        loss = super().loss(nodes, time)

        # Weights (tune as needed)
        w_tracking = 10.0
        w_progress = 5.0
        w_progress_rate = 2.0
        w_backward = 50.0
        w_terminal_align = 20.0
        w_low_velocity = 10.0
        w_control = 100.0
        w_discontinuity = 1000  # Penalty for discontinuities in progress
        # Accumulators
        tracking_loss = 0
        progress_reward = 0
        progress_rate_reward = 0
        backward_penalty = 0
        low_velocity_penalty = 0
        control_effort = 0
        discontinuity_loss = 0

        for i, node in enumerate(nodes[1:], 1):  # skip first node for rate
            # Tracking error (if defined)
            if hasattr(node, 'tracking_error'):
                tracking_loss += node.tracking_error

            # Decaying progress reward (earlier progress is more valuable)
            decay_weight = 1#ca.exp(-0.05 * i)  # or use 1 - i / num_nodes
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

            # discontinuity = (nodes[i - 1].track_progress - nodes[i].track_progress) * self.num_nodes
            # discontinuity_loss += discontinuity**2
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
            + w_discontinuity * discontinuity_loss
        )

        return loss

class Loss:
    def __init__(self, config:dict, num_nodes:int):
        self.num_nodes = num_nodes
        self.config = config

    def control(self, control:ca.MX, actuation:bool = True, change:bool = True) -> ca.MX:
        """
        Calculate the control loss.
        """
        w_actuation = self.config.get('w_actuation', 1.0)
        w_change = self.config.get('w_change', 1.0)

        control_scaling = self.config.get('control_scaling', 10.0)
        
        actuation_loss = ca.sumsqr(control / control_scaling) / self.num_nodes
        change_loss = ca.sumsqr(control[:, 1:] / control_scaling - control[:, :-1] / control_scaling) / self.num_nodes
        return w_actuation * actuation_loss + w_change * change_loss
    
    def goal(self, state:ca.MX, goal:ca.MX) -> ca.MX:
        """
        Calculate the goal loss.
        """
        w_goal = self.config.get('w_goal', 1.0)
        return w_goal * ca.sumsqr(state[:3] - goal)

    def __call__(self, state:ca.MX, goal:ca.MX, control:ca.MX, actuation:bool = True, change:bool = True) -> ca.MX:
        loss = 0
        if actuation:
            loss += self.control(control)
        if change:
            loss += self.change(control)
        loss += self.goal(state, goal)
        return loss
