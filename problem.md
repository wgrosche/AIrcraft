# Questions to the reader

- Is the dynamics model sound?
- Can this formulation be extended to deformable wings (large control dimensionality) or is this too expensive?
- Is the proposed MHE approach reasonable?

## A summary of the work thus far

The problem under consideration is the trajectory optimisation of a fixed-wing aircraft for a waypoint navigation task. The goal is for the approach to be extensible to deformable wings but currently the focus is on a particular glider for which the aerodynamic data exists.
Given a glider whose dynamics are modeled $\dot{\vec{x}} = \vec{f}(\vec{x}, \vec{u})$, where $\vec{x}$ is the state vector and $\vec{u}$ is the control vector, the goal is to find the optimal trajectory $\vec{x}(t)$ that minimises a cost function $J(\vec{x}, \vec{u})$.

## The dynamics model

The state vector for the dynamics model is: $\vec{x} = (\vec{x}_{i}, \vec{v}_{i}, q^{i}_{b}, \vec{\omega}_{b})^T$. The control vector is: $\vec{u} = (\delta_a, \delta_e, \delta_r, \delta_{t_x}, \delta_{t_y}, \delta_{t_z})^T$.

The dynamics model attempts to shift as much heavy lifting as possible away from the data-driven approach. This is done due to the limited available data, especially in the form of dynamic simulations. The data are used to generate three models:

1. A linearised model of the aerodynamic coefficients.
2. An MLP that predicts the aerodynamic coefficients directly.
3. A residual model that predicts the error between the linearised model and the measured data.

The aerodynamic coefficients are used to generate the aerodynamic forces and moments in the body frame:
$$
\vec{F} = \begin{bmatrix}
    F_X \\
    F_Y \\
    F_Z \\
    M_l \\
    M_m \\
    M_n
    \end{bmatrix}
    = S\cdot \bar{q}
    \begin{bmatrix}
    C_{X_0} \\
    C_{Y_0} \\
    C_{Z_0} \\
    C_{l_0}\cdot b\\
    C_{m_0}\cdot c\\
    C_{n_0}\cdot b
    \end{bmatrix}
$$

Where $S$ is the wing area, $\bar{q}$ is the dynamic pressure, $b$ is the span, $c$ is the chord, and $C_{X_0}, C_{Y_0}, C_{Z_0}, C_{l_0}, C_{m_0}, C_{n_0}$ are the aerodynamic coefficients. Since we don't have data for the rate coefficients I misappropriated some from a model of the Cessna that I found on the Microsoft Flight Simulator forums. These were scaled down but are approximate, any assistance on determining these would be appreciated. We also lack data in the stall regime, during control this shouldn't be reached due to constraints on the flight envelope but I use a sigmoid saturation model above a certain angle of attack to prevent the model from blowing up during simulation.

The forces are transformed into the inertial frame according to:
$$
\vec{F}_{i} = q_{i}^{b} \vec{F}_{b} q_{b}^{i}
$$
Note that when I multiply a vector by a quaternion I am transforming the vector to a quaternion and setting the scalar component  to 0.

### State derivative

- $\dot{\vec{x}_i} = \vec{v}_i$

- $\dot{\vec{v}} = \vec{F}_i / m + \vec{g}$

- $\dot{q} = \frac{1}{2} \cdot q_{i}^{b} \cdot \vec{\omega}_{b}$

- $\dot{\vec{\omega}} = I^{-1} \cdot (\vec{M}_b - \vec{\omega}_b \times I \cdot \vec{\omega}_b)$

Where m is the mass, $I$ is the inertia tensor, $\vec{g}$ is the gravitational acceleration, and $\vec{M}_b$ is the moment vector in the body frame.

### The models

All NN models take as input the angle of attack $\alpha = \arctan{v_{b,z}/ v_{b,x}}$, the sideslip angle $\beta = \arcsin{v_{b,y}/|v_b|}$, the dynamic pressure $\bar{q}$, and the control surface deflections $\delta_a, \delta_e$. They predict the coefficients directly and consist of an MLP with pre- and post-scaling. They are processed using L4CasADi to integrate them into the dynamics model with their gradients.

## Simulation

To get from dynamics to simulation we integrate the dynamics using RungeKutta45. Since I noticed very little quaternion drift I am currently not normalising the quaternions after each integration step, nor am I resorting to the quaternion exponential map (very  little difference to be seen and very expensive computationally). We now have a function $F(x, u, \delta t)$ that takes the state, control, and time step and returns the next state.

## Control

Up until this point we have been working from [dynamics.py](src/dynamics.py). The next step is to define a controller.The controller is defined in [control.py](src/control.py). It implements the complementary waypoint constraint formulation from [Time-Optimal Planning for Quadrotor Waypoint Flight by Philipp Foehn et al.](https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf)

Currently we optimise over the entire trajectory at once. Since this is expensive I want to switch to moving horizon estimation instead. My plan for doing this is to consider only the next up to 2 waypoints that are within a distance reachable in N timesteps from the current position. The final waypoint in this window is a fixed end point constraint while the others are implemented as before using the complementary waypoint constraint formulation. If none are reachable the goal becomes minimising the distance to the next waypoint in N timesteps. This trajectory is optimised and the path up to the first waypoint is locked in, we then move the horizon and repeat.
