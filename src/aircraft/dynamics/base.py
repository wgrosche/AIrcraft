
import casadi as ca
from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
from liecasadi import Quaternion
from dataclasses import dataclass

@dataclass
class SixDOFOpts:
    epsilon:float = 1e-6
    physical_integration_substeps:int = 10
    gravity:ca.DM = ca.vertcat(0, 0, 9.81)
    mass:float = 1.0

class SixDOF(ABC):
    """
    Baseclass for 6DOF Dynamics Simulation in a NED system
    """
    num_controls:int
    
    com:Union[ca.DM, np.ndarray, ca.MX, list[float]]
    static_inertia_tensor:ca.DM
    def __init__(self, *, opts:Optional[SixDOFOpts] = None, lqr:bool = False, setpoint:Optional[np.ndarray]=None, **kwargs) -> None:
        if opts is None:
            opts = SixDOFOpts()
        self.opts = opts
        self.gravity = self.opts.gravity
        self.dt_sym:ca.MX = ca.MX.sym('dt') # type: ignore[arg-type]
        self.epsilon = self.opts.epsilon
        self.mass = self.opts.mass
        self.lqr = lqr
        self.normalise = False
        self.physical_integration_substeps:int = self.opts.physical_integration_substeps
        if self.lqr:
            assert setpoint is not None, "Cannot perform LQR without a valid setpoint"
            self.setpoint = setpoint

        super().__init__(**kwargs)

    def _initialise_LQR(self, setpoint:np.ndarray) -> None:
        """
        LQR around steady state setpoint
        """
        from scipy import linalg
        assert hasattr(self, 'num_controls')
        if self.lqr and not hasattr(self, 'LQR_mat'):
            x = self.state
            u = self.control
            f = self.state_update(x, u)
            A = ca.Function("A", [x, u], [ca.jacobian(f, x)])
            B = ca.Function("B", [x, u], [ca.jacobian(f, u)])
            self.x_ss = setpoint[:self.num_states]
            self.u_ss = setpoint[self.num_states:self.num_states+self.num_controls]
            a_ss = A(self.x_ss, self.u_ss)
            b_ss = B(self.x_ss, self.u_ss)
            Q = np.eye(self.num_states)
            R = np.eye(self.num_controls)

            P = linalg.solve_continuous_are(a_ss, b_ss, Q, R)
            assert isinstance(b_ss, Union[ca.DM, ca.MX])
            self.K = np.linalg.inv(R) @ b_ss.T @ P

    @property
    def state_update_LQR(self) -> ca.Function:
        setpoint = self.setpoint
        if not hasattr(self, 'K'):
            self._initialise_LQR(setpoint=setpoint)
        u_lqr = self.control -self.K @ (self.state - self.x_ss) + self.u_ss
        dt = self.dt_sym
        res = self.state_update(self.state, u_lqr, dt)
        return ca.Function(
            'state_update', 
            [self.state, self.control, dt], 
            [res]
            )

    def _ensure_initialized(self, *properties) -> None:
        """Ensure properties are initialized by calling them"""
        for prop in properties:
            if not hasattr(self, f"_{prop}"):
                getattr(self, prop)

    @property
    def state(self) -> ca.MX:
        if not hasattr(self, '_state_initialized') or not self._state_initialized:
            # Define _symbolic state variables once
            self._q_frd_ned = ca.MX.sym('q_frd_ned', 4)# type: ignore[arg-type]
            self._p_ned = ca.MX.sym('p_ned', 3)# type: ignore[arg-type]
            self._v_ned = ca.MX.sym('v_ned', 3)# type: ignore[arg-type]
            self._omega_frd_ned = ca.MX.sym('omega_frd_ned', 3)# type: ignore[arg-type]
            
            # Bundle state variables together
            self._state = ca.vertcat(
            self._p_ned, 
            self._v_ned, 
            self._q_frd_ned, 
            self._omega_frd_ned
            )

            self.num_states = self._state.size()[0]
            
            # Set the flag to indicate initialization
            self._state_initialized = True
        assert isinstance(self._state, ca.MX)
        return self._state
    
    
    @property
    @abstractmethod
    def control(self) -> ca.MX:
        """Must initialize self.num_controls (int) and self._control (ca.MX)"""
        ...
    
    
    @property
    def inertia_tensor(self) -> ca.MX:
        """
        Inertia Tensor around the Centre of Mass
        """
        assert isinstance(self.mass, Union[float, ca.MX])
        assert isinstance(self.com, (ca.DM, np.ndarray, ca.MX, list))
        assert isinstance(self.static_inertia_tensor, ca.DM)

        com = self.com
        
        x, y, z = com[0], com[1], com[2]

        com_term = ca.vertcat(
            ca.horzcat(y**2 + z**2, -x*y, -x*z),
            ca.horzcat(-y*x, x**2 + z**2, -y*z),
            ca.horzcat(-z*x, -z*y, x**2 + y**2)
        )

        inertia_tensor = self.static_inertia_tensor + self.mass * com_term

        return inertia_tensor 
    
    @property
    def inverse_inertia_tensor(self) -> ca.MX:
        """
        Inverted Inertia Tensor around Centre of Mass
        """
        return ca.inv(self.inertia_tensor)
    

    @property
    def v_frd_rel(self) -> ca.Function:
        q_frd_ned = Quaternion(self._q_frd_ned)
        
        if hasattr(self, 'v_wind'):
            vel = self._v_ned + self.v_wind # type: ignore[arg-type]
        else:
            vel = self._v_ned

        v_ned = Quaternion(ca.vertcat(vel, 0)) # type: ignore[arg-type]

        self._v_frd_rel = (q_frd_ned.inverse() * (v_ned) 
                            * q_frd_ned).coeffs()[:3] + self.epsilon
        
        return ca.Function('v_frd_rel', [self.state, self.control], 
            [self._v_frd_rel])
    
    @property
    def airspeed(self) -> ca.Function:
        self._ensure_initialized('v_frd_rel')
        self._airspeed = ca.sqrt(ca.sumsqr(self._v_frd_rel) + self.epsilon)

        return ca.Function('airspeed', [self.state, self.control], 
            [self._airspeed])
    
    @property
    def alpha(self) -> ca.Function:
        self._ensure_initialized('v_frd_rel')
        v_frd_rel = self._v_frd_rel
        self._alpha = ca.atan2(v_frd_rel[2], v_frd_rel[0] + self.epsilon)
        return ca.Function('alpha', [self.state, self.control], [self._alpha])    
    
    @property
    def phi(self) -> ca.Function:
        x, y, z, w = ca.vertsplit(self._q_frd_ned)
        self._phi = ca.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        return ca.Function('phi', [self.state], [self._phi])

    @property
    def theta(self) -> ca.Function:
        x, y, z, w = ca.vertsplit(self._q_frd_ned)
        self._theta = ca.asin(2 * (w * y - z * x))
        return ca.Function('theta', [self.state], [self._theta])

    @property
    def psi(self) -> ca.Function:
        x, y, z, w = ca.vertsplit(self._q_frd_ned)
        self._psi = ca.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return ca.Function('psi', [self.state], [self._psi])

    @property
    def phi_dot(self) -> ca.Function:
        self._ensure_initialized('phi', 'theta')

        phi = self._phi
        theta = self._theta
        p, q, r = ca.vertsplit(self._omega_frd_ned)

        self._phi_dot = p + ca.sin(phi) * ca.tan(theta) * q + ca.cos(phi) * ca.tan(theta) * r

        return ca.Function('phi_dot', [self.state], [self._phi_dot])

    @property
    def theta_dot(self) -> ca.Function:
        self._ensure_initialized('phi')
        phi = self._phi
        _, q, r = ca.vertsplit(self._omega_frd_ned)

        self._theta_dot = ca.cos(phi) * q - ca.sin(phi) * r
        return ca.Function('theta_dot', [self.state], [self._theta_dot])

    @property
    def psi_dot(self) -> ca.Function:

        self._ensure_initialized('phi', 'theta')
        phi = self._phi
        theta = self._theta
        _, q, r = ca.vertsplit(self._omega_frd_ned)

        self._psi_dot = (ca.sin(phi) / ca.cos(theta)) * q + (ca.cos(phi) / ca.cos(theta)) * r
        return ca.Function('psi_dot', [self.state], [self._psi_dot])

    @property
    def beta(self) -> ca.Function:
        self._ensure_initialized('v_frd_rel', 'airspeed')

        v_frd_rel = self._v_frd_rel
        self._beta = ca.asin(v_frd_rel[1] / self._airspeed)
        return ca.Function('beta', [self.state, self.control], [self._beta])

    @property
    def qbar(self) -> ca.Function:
        self._ensure_initialized('v_frd_rel')
        self._qbar = 0.5 * 1.225 * ca.dot(self._v_frd_rel, self._v_frd_rel)
        return ca.Function('qbar', [self.state, self.control], [self._qbar])

    @property
    @abstractmethod
    def _forces_frd(self) -> ca.MX:
        pass

    @property
    def forces_frd(self) -> ca.Function:
        return ca.Function('forces_frd',
            [self.state, self.control], [self._forces_frd])

    @property
    def _moments_from_forces_frd(self) -> ca.MX:
        return ca.cross(self.com, self._forces_frd)

    @property
    def moments_from_forces_frd(self) -> ca.Function:
        return ca.Function('moments_from_forces_frd',
            [self.state, self.control], [self._moments_from_forces_frd])

    @property
    @abstractmethod
    def _moments_aero_frd(self) -> ca.MX:
        pass

    @property
    def moments_aero_frd(self) -> ca.Function:
        return ca.Function('moments_aero_frd',
            [self.state, self.control], [self._moments_aero_frd])

    @property
    def moments_frd(self) -> ca.Function:
        self._ensure_initialized('moments_aero_frd', 'moments_from_forces_frd')
        self._moments_frd = self._moments_aero_frd + self._moments_from_forces_frd
        
        return ca.Function('moments_frd', 
            [self.state, self.control], [self._moments_frd])
    
    @property
    def forces_ned(self) -> ca.Function:
        if not hasattr(self, '_forces_frd'):
            self.forces_frd
        forces_frd = Quaternion(ca.vertcat(self._forces_frd, 0)) # type: ignore[arg-type]
        q_frd_ned = Quaternion(self._q_frd_ned)
        self._forces_ned = (q_frd_ned * forces_frd * q_frd_ned.inverse()).coeffs()[:3]
        return ca.Function('forces_ned', 
            [self.state, self.control], [self._forces_ned])
        
    @property
    def q_frd_ned_dot(self) -> ca.Function:
        q_frd_ecf = Quaternion(self._q_frd_ned)
        omega_frd_ned = Quaternion(ca.vertcat(self._omega_frd_ned, 0)) # type: ignore[arg-type]

        self._q_frd_ned_dot = (0.5 * q_frd_ecf * omega_frd_ned).coeffs()
        return ca.Function('q_frd_ecf_dot', 
            [self.state, self.control], [self._q_frd_ned_dot])
    
    @property
    def q_frd_ned_update(self) -> ca.Function:
        dt:ca.MX = self.dt_sym
        q_frd_ned = Quaternion(self._q_frd_ned).coeffs()
        omega_frd_ned = self._omega_frd_ned

        # Compute the exponential map using the Rodrigues' rotation formula
        half_theta = 0.5 * dt * ca.norm_fro(omega_frd_ned)
        sin_half_theta = ca.sin(half_theta)
        cos_half_theta = ca.cos(half_theta)

        # Compute the exponential map terms (quaternion)
        exp_q:ca.MX = ca.vertcat(sin_half_theta * omega_frd_ned, cos_half_theta) # type: ignore[arg-type]

        # Compute the updated quaternion
        q_next = Quaternion.product(exp_q, q_frd_ned)

        return ca.Function('q_frd_ned_update', [self.state, self.control, dt], [q_next])
    
    @property
    def p_ned_dot(self) -> ca.Function:
        self._p_ned_dot = self._v_ned
        
        return ca.Function('p_ecf_cm_O_dot', 
            [self.state, self.control], [self._v_ned])
    
    @property
    def v_ned_dot(self) -> ca.Function:
        if not hasattr(self, '_forces_ned'):
            self.forces_ned
        forces = self._forces_ned
        grav = self.gravity
        mass = self.mass

        self._v_ned_dot =  forces / mass + grav
        
        return ca.Function(
            'v_ecf_cm_e_dot', 
            [self.state, self.control], 
            [self._v_ned_dot]
            )

    @property
    def omega_frd_ned_dot(self) -> ca.Function:
        if not hasattr(self, '_moments_frd'):
            self.moments_frd
        
        omega_frd_ned = self._omega_frd_ned
        moments = self._moments_frd


        self._omega_frd_ned_dot = ca.mtimes(self.inverse_inertia_tensor, (moments 
             - ca.cross(omega_frd_ned, ca.mtimes(self.inertia_tensor, omega_frd_ned))))
        
        return ca.Function(
            'omega_frd_ned_dot', 
            [self.state, self.control], 
            [self._omega_frd_ned_dot]
            )

    @property
    def state_derivative(self) -> ca.Function:
        if not hasattr(self, '_q_frd_ned_dot'):
            _ = self.q_frd_ned_dot
        if not hasattr(self, '_p_ned_dot'):
            _ = self.p_ned_dot
        if not hasattr(self, '_v_ned_dot'):
            _ = self.v_ned_dot
        if not hasattr(self, '_omega_frd_ned_dot'):
            _ = self.omega_frd_ned_dot
        self._state_derivative = ca.vertcat(
            self._p_ned_dot, 
            self._v_ned_dot, 
            self._q_frd_ned_dot, 
            self._omega_frd_ned_dot
            )
        
        return ca.Function(
            'dynamics', 
            [self.state, self.control], 
            [self._state_derivative], ['x', 'u'], ['x_dot']
            )
    
    def state_step(
        self, 
        state:ca.MX, 
        control:ca.MX, 
        dt_scaled:Union[float, ca.MX], 
        normalise_quaternion:bool=False
        ) -> ca.MX:
        """ 
        Runge-Kutta step for state update.
        Due to the multiplicative nature of the quaternion integration,
        we cannot rely on conventional integrators.
        """
        state_derivative = self.state_derivative

        # Precompute scaled timestep constants
        half_dt = dt_scaled / 2
        sixth_dt = dt_scaled / 6

        # Runge-Kutta intermediate steps
        k1 = state_derivative(state, control)
        state_k1 = state + half_dt * k1 # type: ignore[arg-type]

        k2 = state_derivative(state_k1, control)
        state_k2 = state + half_dt * k2 # type: ignore[arg-type]

        k3 = state_derivative(state_k2, control)
        state_k3 = state + dt_scaled * k3 # type: ignore[arg-type]

        k4 = state_derivative(state_k3, control)

        # Aggregate RK4 step
        state = state + sixth_dt * (k1 + 2 * k2 + 2 * k3 + k4) # type: ignore[arg-type]

        # Quaternion update to maintain unit norm
        # state[6:10] = self.q_frd_ned_update(state, control, dt_scaled)
        if normalise_quaternion:
            state[6:10] = Quaternion(state[6:10]).normalize()

        return state



    @property
    def state_update(self) -> ca.Function:
        """
        Runge Kutta integration with quaternion update, for loop over self.STEPS
        """
        dt = self.dt_sym
        state = self.state
        control_sym = self.control
        if self.physical_integration_substeps is None:
            num_steps = self.opts.physical_integration_substeps
        else:
            num_steps = self.physical_integration_substeps

        if num_steps == 1:
            state = self.state_step(state, control_sym, dt, normalise_quaternion=self.normalise)
        else:
            dt_scaled = dt / num_steps
            input_to_fold = ca.vertcat(self.state, self.control, dt)
            fold_output = ca.vertcat(self.state_step(state, control_sym, dt_scaled), control_sym, dt)
            folded_update = ca.Function('folder', [input_to_fold], [fold_output])
            
            F = folded_update.fold(num_steps)
            state = F(input_to_fold)[:self.num_states]
            if self.normalise:
                state[6:10] = Quaternion(state[6:10]).normalize()

        return ca.Function(
            'state_update', 
            [self.state, self.control, self.dt_sym], 
            [state]
            )
