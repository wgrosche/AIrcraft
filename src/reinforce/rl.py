"""
Starting on rl implementation for trajectory optimisation

"""
from abc import abstractmethod, ABC
from ..dynamics_minimal import Aircraft
import numpy as np

class RL(ABC):
    def __init__(self, aircraft:Aircraft):
        self.dynamics = aircraft.state_update

        self.state = None
        self.control = None

        pass

    def step(self, action):
        # Apply dynamics x_t+1 = f(x_t, u_t)
        self.state = self.dynamics(self.state, action)  

        # Compute reward
        reward = self.reward_function(self.state, action)

        # Check if episode is done (e.g., reaching goal, max steps)
        done = self.check_termination(self.state)

        # Optional additional info (for debugging/logging)
        info = {}

        return self.state, reward, done, info
    
    def reset(self):
        self.state = np.array([0, 0, 0, 50, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        return self.state


    @abstractmethod
    def reset(self):
        pass



    @abstractmethod
    def reward(self, state):
        pass

    @abstractmethod
    def forward(self):
        pass

def wrapper():
    """
    dynamics wrapper that 
    """