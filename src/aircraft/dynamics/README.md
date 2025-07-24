# Dynamics

Dynamics are broken into multiple parts for clarity/readability.

- `base.py` : Baseclass for 6DOF dynamics. Implements:
    - state vector
    - reference frame conversions (body to inertial and back)
    - attitude conversions (quaternion representation to euler angles)
    - attitude rates (euler rates from angular velocity)
    - state derivatives (p_dot, v_dot, q_dot, omega_dot)
    - quaternion update based on exponential map
    - runge kutta integrator (with optional normalisation)
    - experimental LQR wrapper to preprocess control inputs before passing through model

- `aircraft.py` : aircraft specific class that additionally implements;
    - control inputs for the aircraft
    - inertia tensor dependent on centre of mass placement
    - effective aerodynamic angles at control lifting surfaces
    - aerodynamic coefficient stall calculation
    - aerodynamic forces and moments from coefficients
    - deprecated trim class that allows determining trim com for level flight

- `coefficient_models.py` : implements:
    - default: hardcoded linear model
    - linear: fitted linear coefficients
    - polynomial: fitted polynomial coefficients
    - neural: fitted neural model with l4casadi to port it
