import casadi as ca
from liecasadi import Quaternion
from aircraft.dynamics.base import SixDOF


class Quadrotor(SixDOF):
    def __init__(self):
        """
        RPG Time Optimal Quadrotor Simplification for testing
        """
        self.I = ca.DM([(1, 0, 0), (0, 1, 0), (0, 0, 1)])    # Inertia
        self.I_inv = ca.inv(self.I) 
        self.g = -9.81
        self.m = 1.0

    @property
    def dynamics(self):
        
        p = ca.MX.sym('p', 3)
        v = ca.MX.sym('v', 3)
        q = ca.MX.sym('q', 4)
        w = ca.MX.sym('w', 3)
        T = ca.MX.sym('thrust', 4)

        x = ca.vertcat(p, v, q, w)
        u = ca.vertcat(T)

        g = ca.DM([0, 0, -self.g])
        q = Quaternion(q)
        thrust_quat = Quaternion(ca.vertcat(0, 0, (T[0]+T[1]+T[2]+T[3]), 0))
        rotated_thrust = (q * thrust_quat * q.inverse()).coeffs()[:3]
        ang_acc_quat = Quaternion(ca.vertcat(w, 0))
        x_dot = ca.vertcat(
                        v,
                        rotated_thrust + g - v,
                        0.5 * q * ang_acc_quat * q.inverse(),
                        ca.mtimes(self.I_inv, ca.vertcat(
                                    (T[0]-T[1]-T[2]+T[3]),
                                    (-T[0]-T[1]+T[2]+T[3]),
                                    0.5*(T[0]-T[1]+T[2]-T[3]))
                        -ca.cross(w,ca.mtimes(self.I,w)))
        )
        fx = ca.Function('f',  [x, u], [x_dot], ['x', 'u'], ['x_dot'])
        return fx
    
    
    def step(self):
        dt = ca.MX.sym('dt', 1)
        x = ca.MX.sym('x', self.dynamics.size1_in(0))
        u = ca.MX.sym('u', self.dynamics.size1_in(1))
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + dt/2 * k1, u)
        k3 = self.dynamics(x + dt/2 * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        integrator = ca.Function('integrator',
            [x, u, dt],
            [x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)],
            ['x', 'u', 'dt'], ['xn'])
        return integrator