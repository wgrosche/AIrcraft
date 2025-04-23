import numpy as np
from collections import deque

class CurriculumLearning:
    def __init__(self, 
                 initial_goal_distance=10.0,  # Start with closer goals
                 max_goal_distance=150.0,     # Maximum goal distance
                 success_threshold=0.8,       # Success rate to increase difficulty
                 window_size=10,              # Number of episodes to consider for success rate
                 difficulty_increment=10.0,   # How much to increase goal distance
                 min_success_episodes=5):     # Minimum episodes before allowing progression
        
        self.current_goal_distance = initial_goal_distance
        self.max_goal_distance = max_goal_distance
        self.success_threshold = success_threshold
        self.success_history = deque(maxlen=window_size)
        self.difficulty_increment = difficulty_increment
        self.min_success_episodes = min_success_episodes
        self.episode_count = 0
    
    def get_goal(self, initial_position):
        """Generate a goal based on current difficulty level"""
        # Generate random direction (in horizontal plane)
        angle = np.random.uniform(0, 2 * np.pi)
        dx = self.current_goal_distance * np.cos(angle)
        dy = self.current_goal_distance * np.sin(angle)
        
        # Keep same altitude as starting position
        goal = np.array([initial_position[0] + dx, initial_position[1] + dy, initial_position[2]])
        return goal
    
    def update_difficulty(self, success):
        """Update curriculum difficulty based on agent performance"""
        self.episode_count += 1
        self.success_history.append(1 if success else 0)
        
        # Only consider increasing difficulty after minimum episodes
        if self.episode_count < self.min_success_episodes:
            return
        
        # Calculate success rate over recent episodes
        if len(self.success_history) >= self.success_history.maxlen:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            # Increase difficulty if agent is succeeding consistently
            if success_rate >= self.success_threshold:
                old_distance = self.current_goal_distance
                self.current_goal_distance = min(
                    self.current_goal_distance + self.difficulty_increment,
                    self.max_goal_distance
                )
                
                # Reset success history after increasing difficulty
                if old_distance != self.current_goal_distance:
                    self.success_history.clear()
                    return True  # Indicates difficulty was increased
        return False
    
    def get_current_difficulty(self):
        """Return current curriculum difficulty metrics"""
        return {
            "goal_distance": self.current_goal_distance,
            "success_rate": sum(self.success_history) / max(1, len(self.success_history)),
            "episodes_at_current_level": len(self.success_history),
            "total_episodes": self.episode_count
        }
    
# Initialize curriculum learning
curriculum = CurriculumLearning(
    initial_goal_distance=10.0,
    max_goal_distance=150.0,
    success_threshold=0.8,
    window_size=10,
    difficulty_increment=10.0
)

# Define success criteria
def is_success(states, goal, tolerance=1.0):
    """Determine if the episode was successful"""
    final_position = states[0][:2]  # Assuming first two elements are x,y
    distance_to_goal = np.linalg.norm(final_position - goal[:2])
    return distance_to_goal < tolerance and not is_crash(states)

def is_crash(states):
    """Determine if the aircraft crashed"""
    # Implement crash detection logic - using pitch, speed, etc.
    # Example:
    pitch = aircraft.theta(states[0])
    speed = aircraft.v_frd_rel(states[0], np.zeros(action_dim))[0]
    return abs(pitch) > np.deg2rad(60) or speed < 20

# Main training loop
for i_episode in range(1, num_episodes+1):
    # Get initial state
    states = np.array([initial_state.full().flatten() for _ in range(num_agents)])
    
    # Generate goal based on current curriculum difficulty
    goal = curriculum.get_goal(states[0][:3])  # Assuming first 3 elements are x,y,z
    
    # Reset the agent
    agent.reset()
    
    # Run episode...
    # (your existing episode code)
    # ...
    
    # After episode completion
    episode_success = is_success(states, goal)
    difficulty_increased = curriculum.update_difficulty(episode_success)
    
    # Log curriculum progress
    difficulty_info = curriculum.get_current_difficulty()
    print(f"Episode {i_episode} - Goal distance: {difficulty_info['goal_distance']:.1f}, " 
          f"Success rate: {difficulty_info['success_rate']:.2f}")
    
    if difficulty_increased:
        print(f"Difficulty increased! New goal distance: {difficulty_info['goal_distance']:.1f}")
        
    # Save networks if significantly improved
    # (your existing save code)


def calculate_rewards(states, actions, goal, previous_actions):
    """Calculate rewards with curriculum-appropriate scaling"""
    rewards = np.zeros(num_agents)
    
    for i in range(num_agents):
        # Distance-based reward (scaled to current goal distance)
        distance_to_goal = np.linalg.norm(states[i][:2] - goal[:2])
        # Use curriculum's current distance to scale reward appropriately
        distance_reward = -distance_to_goal / curriculum.current_goal_distance * 10
        rewards[i] += distance_reward
        
        # Speed reward/penalty
        speed = aircraft.v_frd_rel(states[i], actions[i, :])[0]
        min_speed = 40
        max_speed = 80
        if speed < min_speed:
            # Gentle gradient for speeds approaching minimum
            speed_penalty = -5 * (min_speed - speed) / min_speed
        elif speed > max_speed:
            # Penalty for excessive speed
            speed_penalty = -2 * (speed - max_speed) / max_speed
        else:
            # Positive reward for good speed range
            speed_penalty = 0.5
        rewards[i] += speed_penalty
        
        # Attitude stability rewards
        pitch = aircraft.theta(states[i])
        roll = aircraft.phi(states[i])
        
        # Penalize extreme attitudes
        attitude_penalty = -2 * (abs(pitch) / np.pi + abs(roll) / np.pi)
        rewards[i] += attitude_penalty
        
        # Control smoothness (penalize large control changes)
        if previous_actions is not None:
            control_change = np.linalg.norm(actions[i] - previous_actions[i])
            control_penalty = -0.5 * control_change
            rewards[i] += control_penalty
    
    return rewards