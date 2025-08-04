# Integrators

We provide two default integration schemes compatible with both fixed- and minimum-time planning.

## RK4

The explicit Runge Kutta 4 scheme. This is used for simulations by default and is stable for most (state, control) configurations provided a timestep under 0.01s is chosen.

## Implicit Euler

The implicit integration scheme: $x_{n+1} = x_n + \delta t f(x_{n + 1}, u_n) is inherently stable. 

## Aside: Quaternion Normalisation

We implement 4 optional avenues for preserving quaternion norm:
- Exponential map update for quaternions:
    Uses the exponential map: $
    \[
\delta \theta = \| \bm{\omega} \| \Delta t, \quad 
\mathbf{u} = \frac{\bm{\omega}}{\| \bm{\omega} \|} \quad \text{if } \| \bm{\omega} \| > 0
\]

\[
\delta \mathbf{q} = 
\begin{bmatrix}
\cos\left(\frac{\delta \theta}{2}\right) \\
\mathbf{u} \sin\left(\frac{\delta \theta}{2}\right)
\end{bmatrix}
\]

- In integration normalisation:
    Normalise after `normalisation_interval <= aircraft.STEPS` simulation steps. 

- Baumgarte quaternion normalisation:
            # x_dot_q = self.x_dot(node.state, node.control)[6:10]
            # phi_dot = 2 * ca.dot(node.state[6:10], x_dot_q)

            # alpha = 2.0  # damping
            # beta = 2.0   # stiffness

            # phi = ca.dot(node.state[6:10], node.state[6:10]) - 1
            # stabilized_phi = 2 * alpha * phi_dot + beta**2 * phi

            # self.constraint(stabilized_phi == 0, description="Baumgarte quaternion normalization")

- Explicit norm constraint:
    Hard constraint on quaternion norm


### Notes:
Currently the quaternion norm constraint is not significantly violated even without the above measures.

[return to main](../README.md)