import casadi as ca
from aircraft.dynamics.aircraft import Aircraft

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki 
        self.kd = kd
        self.dt = dt
        
        self.previous_error = 0
        self.integral = 0
        
    def reset(self):
        self.previous_error = 0
        self.integral = 0
        
    def update(self, setpoint, measured_value):
        # Calculate error
        error = setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative term 
        derivative = (error - self.previous_error) / self.dt
        d_term = self.kd * derivative
        
        # Update previous error
        self.previous_error = error
        
        # Calculate total control output
        control = p_term + i_term + d_term
        
        return control

class PIDAircraft(Aircraft):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize PID controllers for different control channels
        self.roll_controller = PIDController(kp=0.5, ki=0.1, kd=0.1, dt=0.01)
        self.pitch_controller = PIDController(kp=0.5, ki=0.1, kd=0.1, dt=0.01)
        
    def compute_control(self, state, desired_roll, desired_pitch):
        # Get current angles from aircraft state
        current_roll = self.phi(state)
        current_pitch = self.theta(state)
        
        # Compute control surface deflections
        aileron = self.roll_controller.update(desired_roll, current_roll)
        elevator = self.pitch_controller.update(desired_pitch, current_pitch)
        
        return aileron, elevator

    @property
    def state_update(self) -> ca.Function:
        """
        Runge Kutta integration with quaternion update, PID controller for angle of attack (α).
        """
        # Symbols for inputs
        dt = self.dt_sym
        alpha_desired = ca.MX.sym('alpha_desired') # type: ignore[arg-type]
        Kp, Ki, Kd = ca.MX.sym('Kp'), ca.MX.sym('Ki'), ca.MX.sym('Kd') # type: ignore[arg-type]
        # PID error states
        error = ca.MX.sym('error')  # type: ignore[arg-type]
        int_error = ca.MX.sym('int_error')  # type: ignore[arg-type]

        num_steps = self.physical_integration_substeps
        dt_scaled = dt / num_steps

        # PID Controller
        def pid_controller(alpha_current, int_error, dt_scaled):
            d_error = (alpha_desired - alpha_current) / dt_scaled  # Derivative of error
            control = Kp * error + Ki * int_error + Kd * d_error  # PID control law
            int_error_new = int_error + error * dt_scaled  # Update integral of error
            return control, int_error_new

        def step_with_pid():

            # Compute control from PID controller
            control, int_error_new = pid_controller(self.alpha, int_error, dt_scaled)

            # Apply control to dynamics
            updated_state = self.state_step(self.state, self.control, dt_scaled)
            return ca.vertcat(updated_state, control, dt_scaled, error, int_error_new)

        # dynamics with PID control
        state = step_with_pid()

        return ca.Function(
            'state_update_with_pid',
            [self.state, self.control, dt, alpha_desired, Kp, Ki, Kd],
            [state],
        )
