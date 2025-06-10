import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ProgressTrackingController:
    def __init__(self, track_eval, track_eval_tangent, track_length):
        self.track_eval = track_eval
        self.track_eval_tangent = track_eval_tangent
        self.track_length = track_length
        
        # Cost weights
        self.w_track = 10.0      # tracking error weight
        self.w_progress = 5.0    # progress reward weight
        self.w_control = 0.1     # control effort weight
        self.w_backward = 100.0  # backward motion penalty
        
    def dynamics(self, state, control, dt):
        """
        State: [x, y, z, vx, vy, vz, s]
        Control: [ax, ay, az] (acceleration commands)
        """
        pos = state[:3]
        vel = state[3:6]
        s = state[6]
        
        # Point mass dynamics
        pos_next = pos + vel * dt
        vel_next = vel + control * dt
        
        # Progress dynamics
        track_tangent = self.track_eval_tangent(s)
        if np.linalg.norm(track_tangent) > 1e-8:
            track_tangent_normalized = track_tangent / np.linalg.norm(track_tangent)
            # Progress rate = component of velocity along track
            s_dot = np.dot(vel, track_tangent_normalized) / self.track_length
        else:
            s_dot = 0.0
            
        s_next = np.clip(s + s_dot * dt, 0.0, 1.0)
        
        return np.concatenate([pos_next, vel_next, [s_next]])
    
    def stage_cost(self, state, control):
        """Compute stage cost for current state and control"""
        pos = state[:3]
        vel = state[3:6]
        s = state[6]
        
        # Reference position at current progress
        ref_pos = self.track_eval(s)
        
        # Tracking error
        tracking_error = np.linalg.norm(pos - ref_pos)**2
        
        # Progress reward (we want to maximize s)
        progress_reward = s
        
        # Control effort
        control_cost = np.linalg.norm(control)**2
        
        # Progress rate reward
        track_tangent = self.track_eval_tangent(s)
        if np.linalg.norm(track_tangent) > 1e-8:
            track_tangent_normalized = track_tangent / np.linalg.norm(track_tangent)
            s_dot = np.dot(vel, track_tangent_normalized) / self.track_length
            progress_rate_reward = s_dot
            backward_penalty = np.maximum(0, -s_dot)**2
        else:
            progress_rate_reward = 0.0
            backward_penalty = 0.0
        
        return (self.w_track * tracking_error 
                - self.w_progress * progress_rate_reward
                + self.w_control * control_cost
                + self.w_backward * backward_penalty)
    
    def terminal_cost(self, state):
        """Terminal cost - reward reaching the end of track"""
        s = state[6]
        # Big reward for completing the track
        completion_reward = 100.0 * s
        
        # Small penalty for final tracking error
        pos = state[:3]
        ref_pos = self.track_eval(s)
        final_tracking_error = np.linalg.norm(pos - ref_pos)**2
        
        return -completion_reward + 0.1 * final_tracking_error
    
    def solve_mpc(self, initial_state, horizon=10, dt=0.1):
        """Simple MPC implementation"""
        n_states = 7  # [x, y, z, vx, vy, vz, s]
        n_controls = 3  # [ax, ay, az]
        
        def objective(decision_vars):
            # Reshape decision variables
            controls = decision_vars.reshape(horizon, n_controls)
            
            # Forward simulate
            state = initial_state.copy()
            total_cost = 0.0
            
            for k in range(horizon):
                # Add stage cost
                total_cost += self.stage_cost(state, controls[k])
                
                # Propagate dynamics
                state = self.dynamics(state, controls[k], dt)
            
            # Add terminal cost
            total_cost += self.terminal_cost(state)
            
            return total_cost
        
        # Initial guess (zero acceleration)
        x0 = np.zeros(horizon * n_controls)
        
        # Control bounds (e.g., acceleration limits)
        control_bounds = [(-5.0, 5.0)] * (horizon * n_controls)
        
        # Solve optimization
        result = minimize(objective, x0, bounds=control_bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_controls = result.x.reshape(horizon, n_controls)
            return optimal_controls[0]  # Return first control action
        else:
            print("Optimization failed:", result.message)
            return np.zeros(n_controls)

# Example usage
def example_track():
    """Create a simple 3D helical track"""
    def track_eval(s):
        t = s * 4 * np.pi  # 2 full turns
        return np.array([
            np.cos(t),
            np.sin(t),
            s * 2  # rising helix
        ])
    
    def track_eval_tangent(s):
        t = s * 4 * np.pi
        return np.array([
            -4 * np.pi * np.sin(t),
            4 * np.pi * np.cos(t),
            2.0
        ])
    
    # Approximate track length
    n_samples = 1000
    s_vals = np.linspace(0, 1, n_samples)
    positions = np.array([track_eval(s) for s in s_vals])
    track_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    
    return track_eval, track_eval_tangent, track_length

# Example simulation
if __name__ == "__main__":
    # Create track
    track_eval, track_eval_tangent, track_length = example_track()
    
    # Create controller
    controller = ProgressTrackingController(track_eval, track_eval_tangent, track_length)
    
    # Initial state: [x, y, z, vx, vy, vz, s]
    # Start slightly off the track at s=0
    initial_state = np.array([1.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Simulate
    dt = 0.1
    T_sim = 15.0
    n_steps = int(T_sim / dt)
    
    states = [initial_state]
    controls = []
    
    for step in range(n_steps):
        current_state = states[-1]
        
        # Compute optimal control
        u_opt = controller.solve_mpc(current_state, horizon=10, dt=dt)
        controls.append(u_opt)
        
        # Apply control and simulate
        next_state = controller.dynamics(current_state, u_opt, dt)
        states.append(next_state)
        
        print(f"Step {step}: s={current_state[6]:.3f}, pos={current_state[:3]}")
        
        # Stop if we've completed the track
        if current_state[6] >= 0.99:
            print("Track completed!")
            break
    
    states = np.array(states)
    print(f"Final progress: {states[-1, 6]:.3f}")
    print(f"Final position: {states[-1, :3]}")