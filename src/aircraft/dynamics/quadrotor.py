import casadi as ca
from liecasadi import Quaternion
from aircraft.dynamics.base import SixDOF


class Quadrotor(SixDOF):
    def __init__(self):
        """
        RPG Time Optimal Quadrotor Simplification for testing
        """
        super().__init__()
        self.mass = 1.0
        self.com = ca.vertcat(0, 0, 0)

        self.STEPS = 1
        self.state
        self.control

    @property
    def inertia_tensor(self):
        """
        Inertia Tensor around the Centre of Mass
        """

        inertia_tensor = ca.MX.eye(3)

        return inertia_tensor 
    


    @property
    def control(self):
        if not hasattr(self, '_control_initialized') or not self._control_initialized:
            self._thrust = ca.MX.sym('thrust', 4)

            self._control = self._thrust
            self.num_controls = self._control.size()[0]

            self._control_initialized = True

        return self._control
    
    @property
    def _forces_frd(self):
        return ca.vertcat(0, 0, ca.sum1(self._thrust))
    

    @property
    def _moments_aero_frd(self):
        T = self._thrust
        moments_aero = ca.vertcat(
                        (T[0]-T[1]-T[2]+T[3]),
                        (-T[0]-T[1]+T[2]+T[3]),
                        0.5*(T[0]-T[1]+T[2]-T[3]))

        return moments_aero