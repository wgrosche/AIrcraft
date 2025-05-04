# Time

## Fixed Time Discretisation

$dt = T_{total} / N$,

where $T_{total}$ is the total time of the simulation and $N$ is the number of control steps.

## Minimum Time Discretisation



## Variable timestep Discretisation

To improve the sparsity of the resulting nlp problem we switch to a per node timestep variable. These can be combined with 

$dt = \frac{1}{v_i}$


### Error Surrogate

J = jacobian(f(x, u), x)
error_surrogate = alpha * (1 / v_i**2) * dot(J @ f(x, u), J @ f(x, u))
constraints += [error_surrogate <= tol]
loss += weight * v_i


### Comparison Results

[return to main](../README.md)