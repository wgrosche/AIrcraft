"""
## Positioning the centre of mass and finding trim


This is a constrained optimisation problem. We want to minimise $|\vec{F}|$ and $|vec{M}|$ while constraining what the aircraft is doing:

The orientation should still be valid so $|\text{state}[:4]| = 1$

The position is w.l.o.g $\text{state}[4:7] = (0,0,0)$

The velocity should be within the region of validity for our CFD results $30 < |\text{state}[7:10]| < 70$ and we want to be flying forwards $\text{state}[7] > 0$

The attitude of our aircraft should not be changing so our angular rates should be 0. $|state[10:]| = 0$. This may be hard to enforce as a hard constraint so we will replace it with  $|state[10:]| < \epsilon$



Further, we want to satisfy the conditions for stability. This involves constraints on the derivatives with respect to the aerodynamic variables $(\alpha, \beta)$ and the angular rates $(p, q, r)$.

These constraints are as follows:

To ensure longitudinal stability we require $\frac{\partial C_m}{\partial \alpha} < 0$. We will be working with the moments directly, this derivative should be unaffected in sign by this change. Similar treatment for the other forces and moments yields.

$\frac{\partial D}{\partial q} > 0$

$\frac{\partial Y}{\partial r} > 0$

$\frac{\partial L}{\partial q} < 0$

$\frac{\partial l}{\partial \beta} < 0$

$\frac{\partial m}{\partial \alpha} < 0$

$\frac{\partial n}{\partial \beta} > 0$

$\frac{\partial l}{\partial p} < 0$

$\frac{\partial m}{\partial q} < 0$

$\frac{\partial n}{\partial r} < 0$


The method for optimisation is solving a constrained NLP using CasADi and IPOPT.

To handle the lack of derivatives in the neural net we use a finite differences approximation for the jacobian.


For the Cessna 172 aerodynamic data include the following coefficients:

---

| **Weight** | W = 2300 lb |
|------------|-------------|
| **Geometry** | S = 174.0 ft²  |
| | b = 35.8 ft |
| | c̅ = 4.90 ft |

| **Control surface limits** | -28° < δₑ < 23° |
| | -16° < δᵣ < 16° |
| | -20° < δₐ < 20° |

| **Lift coefficient** | C_L₀ = 0.31 |
| | C_Lα = 5.143 rad⁻¹ |
| | C_Lq = 3.9 rad⁻¹ |
| | C_L,δₑ = 0.43 rad⁻¹ |

| **Drag coefficient** | C_D = C_D₀ + kC_L² |
| | C_D₀ = 0.031 |
| | k = 0.054 |

| **Pitching moment coefficient** | C_m₀ = -0.015 |
| | C_m,α = -0.89 rad⁻¹ |
| | C_m,q = -12.4 rad⁻¹ |
| | C_m,δₑ = -1.28 rad⁻¹ |

| **Side force coefficient** | C_Y₀ = 0 |
| | **C_Y,β = -0.31 rad⁻¹** |
| | C_Yₚ = -0.037/(rad/s) |
| | C_Y,ᵣ = 0.21/(rad/s) |
| | C_Y,δₐ = 0 |
| | C_Y,δᵣ = 0.187/rad |

| **Rolling moment coefficient** | C_l₀ = 0 |
| | C_l,β = -0.089 rad⁻¹ |
| | C_lₚ = -0.47/(rad/s) |
| | C_l,ᵣ = 0.096/(rad/s) |
| | C_l,δₐ = -0.178/(rad/s) |
| | C_l,δᵣ = 0.0147/rad |

| **Yawing moment coefficient** | C_n₀ = 0 |
| | **C_n,β = 0.065 rad⁻¹** |
| | C_nₚ = -0.03/(rad/s) |
| | C_n,ᵣ = -0.053/(rad/s) |
| | C_n,δₐ = -0.0657/rad |
| | C_n,δᵣ = -0.099/(rad/s) |

--- 

From this we derive the following required gradients for stable flight


Due to our sign conventions (alpha flipped, z flipped wrt lift)

C_X_alpha
C_X_beta

C_Y_alpha
C_Y_beta < 0

C_Z_alpha > 0
C_Z_beta

C_l_alpha
C_l_beta

C_m_alpha > 0
C_m_beta

C_n_alpha
C_n_beta < 0
"""



import numpy as np
import os
import sys
import json
import casadi as ca

# Set up paths and imports
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)

from src.dynamics import Aircraft
from src.utils import load_model, aero_to_state, TrajectoryConfiguration


# Wrap functions for aerodynamic derivatives
def wrap_aero(func, control):
    """
    Wraps a function so that it can take aerodynamic variables as input instead
    of the full state.

    """
    # define symbolic aerodynamic variables
    alpha = ca.MX.sym('alpha')
    beta = ca.MX.sym('beta')
    q_bar = ca.MX.sym('q_bar')

    wrapped_function = ca.Function(
        f'{func.name()}_aerodynamic',
        [q_bar, alpha, beta, control],
        [func(aero_to_state(q_bar, alpha, beta), control)]
    )

    return wrapped_function

def jacobian_wrapper(
        function:ca.Function,  
                     state:ca.MX, 
                     control:ca.MX,
                     aero:bool = False, 
                     finite_differences:bool = True, 
                     eps:float = 1e-6
                     ):
    """
    Jacobians with respect to the aerodynamic variables
    """
    # define symbolic aerodynamic variables
    alpha = ca.MX.sym('alpha')
    beta = ca.MX.sym('beta')
    q_bar = ca.MX.sym('q_bar')

    if aero:
        wrapped_function = wrap_aero(function, control)

        if finite_differences:
            wrapped_jac = ca.MX.zeros(wrapped_function.size1_out(0), 
                                      ca.vertcat(alpha, beta).size1())

            wrapped_jac[:, 0] = (wrapped_function(q_bar, alpha + eps, beta, control)\
                                - wrapped_function(q_bar, alpha, beta, control)) / eps
            wrapped_jac[:, 1] = (wrapped_function(q_bar, alpha, beta + eps, control)\
                                - wrapped_function(q_bar, alpha, beta, control)) / eps

        else:
            # generate symbolic jacobian
            wrapped_jac = ca.jacobian(wrapped_function(q_bar, alpha, beta, control), ca.vertcat(alpha, beta))

        # wrap as function so we can evaluate the jacobian at a chosen point
        wrapped_jac_function = ca.Function(
            f'{wrapped_function.name()}_jacobian',
            [q_bar, alpha, beta, control],
            [wrapped_jac]
            )
        
    else:
        wrapped_function = function
        if finite_differences:
            wrapped_jac = ca.MX.zeros(wrapped_function.size1_out(0), state.size1())

            for i in range(state.size1()):
                x_plus = ca.MX(state)
                x_plus[i] += eps
                wrapped_jac[:, i] = (wrapped_function(x_plus, control) - wrapped_function(state, control)) / eps

        else:
            # generate symbolic jacobian
            wrapped_jac = ca.jacobian(wrapped_function(state, control), state)
        # wrap as function so we can evaluate the jacobian at a chosen point
        wrapped_jac_function = ca.Function(
            f'{wrapped_function.name()}_jacobian',
            [state, control],
            [wrapped_jac]
            )

    return wrapped_jac_function


def main():
    # Load aircraft parameters and model
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))
    trajectory_config = TrajectoryConfiguration(traj_dict)


    aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=False)
    forces_func = aircraft._forces_frd

    # Initialize state and control variables
    x0 = np.zeros(3)
    v0 = np.array([60, 0, 0])
    q0 = np.array([0, 0, 0, 1])
    omega0 = np.array([0, 0, 0])
    state = ca.vertcat(q0, x0, v0, omega0)
    control = np.zeros(aircraft.num_controls)
    control[-3:] = traj_dict['aircraft']['aero_centre_offset']

    # define jacobians, we use the notation F_x to mean jacobian of F wrt. x
    moments_aero = jacobian_wrapper(aircraft._moments_frd, aircraft.state, aircraft.control, aero = True)
    # moments_state = jacobian_wrapper(aircraft._moments_frd, aircraft.state, aircraft.control, aero = False)

    forces_aero = jacobian_wrapper(aircraft._forces_frd, aircraft.state, aircraft.control, aero = True)
    # forces_state = jacobian_wrapper(aircraft._forces_frd, aircraft.state, aircraft.control, aero = False)

    constraints = ca.vertcat(
        # ca.vec(moments_aero(aircraft._qbar(aircraft.state, aircraft.control), 
        #                     aircraft._alpha(aircraft.state, aircraft.control), 
        #                     aircraft._beta(aircraft.state, aircraft.control), 
        #                     aircraft.control)),

        # ca.vec(forces_aero(aircraft._qbar(aircraft.state, aircraft.control), 
        #                     aircraft._alpha(aircraft.state, aircraft.control), 
        #                     aircraft._beta(aircraft.state, aircraft.control), 
        #                     aircraft.control)),
        

        ca.dot(aircraft.state[:4], aircraft.state[:4]) - 1,# - [0, 0, 0, 1], # Orientation
        ca.dot(aircraft._v_frd_rel(aircraft.state, aircraft.control), 
               aircraft._v_frd_rel(aircraft.state, aircraft.control)), # Velocity
        aircraft.state[10:], # Angular rates
        aircraft.throttle,
        aircraft.com,
        aircraft.v_wind_ned
        
    )

    lower_bounds = ca.vertcat(
        # Moment derivatives aerodynamic angles
        # ca.vertcat(
        #     -ca.inf, -ca.inf,
        #     0, -ca.inf,
        #     -ca.inf, -ca.inf
        # ),

        # forces
        # ca.vertcat(
        #     -ca.inf, -ca.inf,
        #     -ca.inf, -ca.inf,
        #     0,       -ca.inf,
        # ),
        # -ca.inf,
        # 0.,
        # -ca.inf,

        # # Moment derivatives rates
        # -ca.inf,
        # -ca.inf,
        # -ca.inf,

        # # Force derivatives rates
        # 0, 
        # 0,
        # -ca.inf,

        # Orientation
        0,#[0,0,0,0],

        # # Airspeed
        30**2,

        # # Angular rates
        ca.vertcat(-1e-2, -1e-2, -1e-2),


        ca.vertcat(0,0,0),
        ca.vertcat(-0.5, -0.01, -0.05),
        ca.vertcat(0,0,0)

        
    )

    upper_bounds = ca.vertcat(
        # Moment derivatives aerodynamic angles
        # ca.vertcat(
        #     ca.inf, ca.inf,
        #     ca.inf,      ca.inf,
        #     ca.inf, 0
        # ),
        # forces
        # ca.vertcat(
        #     ca.inf, ca.inf,
        #     ca.inf, 0,
        #     ca.inf, ca.inf
        # ),
        # 0,
        # ca.inf,
        # 0,

        # # Moment derivatives rates
        # 0,
        # 0,
        # 0,

        # # Force derivatives rates
        # ca.inf,
        # ca.inf,
        # 0,

        # Orientation
        0, #[0,0,0,0],

        # # Airspeed
        100**2,

        # # Angular rates
        ca.vertcat(1e-2, 1e-2, 1e-2),

        # # Controls
        ca.vertcat(0,0,0),
        ca.vertcat(0.5, 0.01, 0.05),
        ca.vertcat(0,0,0)

    )

    decision_vars = ca.vertcat(aircraft.state, aircraft.control)
    initial_guess = ca.vertcat(state, control)

    opts = {
        'ipopt': {
            'max_iter': 10000,
            # 'print_level': 2,
            'tol': 1e-4,
            'acceptable_tol': 1e-4,
            'acceptable_obj_change_tol': 1e-4,
            'hessian_approximation': 'limited-memory',  # Set Hessian approximation method here
        },
        'print_time': 10
    }

    nlp = {
        'x': decision_vars,
        'f': ca.dot(aircraft._v_ned_dot(aircraft.state, aircraft.control), 
                    aircraft._v_ned_dot(aircraft.state, aircraft.control)) 
            + ca.dot(aircraft._omega_frd_ned_dot(aircraft.state, aircraft.control), 
                    aircraft._omega_frd_ned_dot(aircraft.state, aircraft.control)),
        'g': constraints
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve for the trim condition
    sol = solver(x0=initial_guess, lbg=lower_bounds, ubg=upper_bounds)
    trim_state_control_aero = sol['x']

    print("Trim state and control:", trim_state_control_aero)

    trim_state = trim_state_control_aero[:aircraft.num_states]
    trim_control = trim_state_control_aero[aircraft.num_states:]

    print(f"Forces at trim: {aircraft._forces_frd(trim_state, trim_control)}")
    print(f"Moments at trim: {aircraft._moments_frd(trim_state, trim_control)}")
    jac_print_exact = jacobian_wrapper(aircraft._moments_frd, aircraft.state, aircraft.control, aero = True, finite_differences=False)(aircraft._qbar(trim_state, trim_control), 
                             aircraft._alpha(trim_state, trim_control), 
                             aircraft._beta(trim_state, trim_control), 
                             trim_control)
    
    jac_print = moments_aero(aircraft._qbar(trim_state, trim_control), 
                             aircraft._alpha(trim_state, trim_control), 
                             aircraft._beta(trim_state, trim_control), 
                             trim_control)
    
    print(f"Initial Jacobian: {moments_aero(aircraft._qbar(state, control), aircraft._alpha(state, control), aircraft._beta(state, control), control)}")

    print(f"Exact Jacobian at trim: {jac_print_exact}")

    print(f"Finite Diffs Jacobian at trim: {jac_print}")

if __name__ == "__main__":
    main()