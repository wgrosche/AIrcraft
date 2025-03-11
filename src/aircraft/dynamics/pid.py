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

class AircraftController:
    def __init__(self, aircraft):
        self.aircraft = aircraft
        
        # Initialize PID controllers for different control channels
        self.roll_controller = PIDController(kp=0.5, ki=0.1, kd=0.1, dt=0.01)
        self.pitch_controller = PIDController(kp=0.5, ki=0.1, kd=0.1, dt=0.01)
        
    def compute_control(self, state, desired_roll, desired_pitch):
        # Get current angles from aircraft state
        current_roll = self.aircraft.phi(state)
        current_pitch = self.aircraft.theta(state)
        
        # Compute control surface deflections
        aileron = self.roll_controller.update(desired_roll, current_roll)
        elevator = self.pitch_controller.update(desired_pitch, current_pitch)
        
        return aileron, elevator


@property
def state_update_with_pid(self, normalisation_interval: int = 10):
    """
    Runge Kutta integration with quaternion update, PID controller for angle of attack (α).
    """
    # Symbols for inputs
    dt = ca.MX.sym('dt')
    state = self.state
    control_sym = self.control  # This includes control inputs like elevator deflection
    alpha_desired = ca.MX.sym('alpha_desired')  # Desired angle of attack
    Kp, Ki, Kd = ca.MX.sym('Kp'), ca.MX.sym('Ki'), ca.MX.sym('Kd')  # PID gains

    # PID error states
    error = ca.MX.sym('error')  # Current error
    int_error = ca.MX.sym('int_error')  # Integral of error

    num_steps = self.STEPS
    dt_scaled = dt / num_steps

    # PID Controller
    def pid_controller(alpha_current, int_error, dt_scaled):
        d_error = (alpha_desired - alpha_current) / dt_scaled  # Derivative of error
        control = Kp * error + Ki * int_error + Kd * d_error  # PID control law
        int_error_new = int_error + error * dt_scaled  # Update integral of error
        return control, int_error_new

    # Folding logic with PID integration
    input_to_fold = ca.vertcat(state, control_sym, dt, error, int_error)
    def step_with_pid(input_sym):
        state, control, dt_scaled, error, int_error = (
            input_sym[:self.num_states],
            input_sym[self.num_states:self.num_states + self.num_controls],
            input_sym[-2],
            input_sym[-1],
        )
        # Extract current angle of attack (α) from state
        alpha_current = self.extract_alpha(state)

        # Compute control from PID controller
        control, int_error_new = pid_controller(alpha_current, int_error, dt_scaled)

        # Apply control to dynamics
        updated_state = self.state_step(state, control, dt_scaled)

        # Normalize quaternion if required
        if (i % normalisation_interval == 0) or (i == num_steps - 1):
            updated_state[:4] = Quaternion(updated_state[:4]).normalize().coeffs()

        return ca.vertcat(updated_state, control, dt_scaled, error, int_error_new)

    # Folding dynamics with PID control
    folder = ca.Function('folder', [input_to_fold], [step_with_pid(input_to_fold)])
    F = folder.fold(num_steps)
    state = F(input_to_fold)[:self.num_states]

    return ca.Function(
        'state_update_with_pid',
        [self.state, self.control, dt, alpha_desired, Kp, Ki, Kd],
        [state],
    )
