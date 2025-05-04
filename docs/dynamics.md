
# Dynamics

## Control Vector

The control vector contains all variables relevant to the dynamics that are not
governed by the dynamics of the system, control inputs, wind, rotation of the earth.

- Aileron Deflection
- Elevator Deflection
- Rudder Deflection
- Throttle (3-dimensional force vector)
- Wind
- $\vec{\omega} _{i\rightarrow e}$

## Perturbation

The simulation in [dynamics](../src/dynamics.py) contains perturbation in the
wind vecor to test dynamic stability.

[return to main](../README.md)
