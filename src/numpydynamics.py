import numpy as np
from abc import ABC, abstractmethod


class SixDOF(ABC):
    def __init__(self):
        pass

    
    @abstractmethod
    @property
    def controls(self):
        pass

    @property
    def state(self):
        if not hasattr(self, '_state_initialized') or not self._state_initialized:
            # Define _symbolic state variables once
            self._p_ned = np.zeros(3)
            self._q_frd_ned = np.zeros(4)
            self._v_ned = np.zeros(3)
            self._omega_frd_ned = np.zeros(4)

            self._state = np.concatenate((self._p_ned, self._q_frd_ned, self._v_ned, self._omega_frd_ned))

            self.num_states = self._state.size()[0]
            
            # Set the flag to indicate initialization
            self._state_initialized = True
        pass

    @abstractmethod
    def dynamics(self):
        pass