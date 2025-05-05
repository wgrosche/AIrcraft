# NLP

We define the NLP for [state](/docs/state.md) $\vec{x}$, [control](/docs/control.md) $\vec{u}$ underlying the dynamics defined in [dynamics.md](/docs/dynamics.md). We use $v_i = 1/\delta t_i$ as a decision variable to represent the timesteps. See [time discretisation](/docs/time.md) for the reasoning.

## Objective Function

The aim is to minimise objective $J(\underline{x}, \underline{u}, \underline{\delta t})$. We define various terms that may apply in different cases.

### Minimum Time Case

The simplest minimum time objective function $J_T = \sum_i \delta t_i$

### Final Position

Minimise distance to position $\vec{x}_g$: $J_F = (\vec{x}_F - \vec{x}_g)_p^2$

### Control Effort

Minimal Control Effort $J_C = \sum_i \vec{u}_i^2$

### Control Change

Limiting shifts in control inputs $J_C = \sum_i (\vec{u}_{i + 1} - \vec{u}_i)^2$

## Equality Constraints

### Dynamics Constraint

$x_{n+1} == \text{integrator}(x_n, u_n, \delta t)$ , $\text{integrator}$ is defined in [integrators.md](/docs/integratos.md).

### Quaternion Normalisation (Optional)

$<q, q> = 1$, Further information in [integrators.md](/docs/integrators.md).

### 

## Inequality Constraints

### State Envelope

We limit the state envelope to keep within the region of dataset validity

- $0 \leq \delta t \leq 0.1$ (timestep)
- $-30\degree \leq \alpha \leq 30\degree$ (angle of attack)
- $-30\degree \leq \beta \leq 30\degree$ (sideslip angle)
- $20ms^{-1} \leq v_x \leq 80ms{-1}$ (forward speed)


### Control Limits

We limit the control surface deflections
- $-5 \leq \delta a \leq 5$
- $-5 \leq \delta e \leq 5$
- $-5 \leq \delta r \leq 5$


### 


[return to main](../README.md)