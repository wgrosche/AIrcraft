"""
Moving horizon estimator (MHE)

"""



class MHE:
    def __init__(self, aircraft, dt, horizon, initial_state, initial_state_covariance,
                 measurement_noise_covariance, process_noise_covariance,
                 control_input_matrix, measurement_matrix, control_input_vector):
        self.aircraft = aircraft
        self.dt = dt
        self.horizon = horizon
        self.initial_state