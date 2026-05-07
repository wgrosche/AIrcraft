
# AIrcraft

Model Predictive control for fixed wing drones.

## Data

The surrogate models used to represent the drone are derived from CFD and Windtunnel data. Information on data acquisition and processing can be found [here](/docs/data.md)

## Models

 Information on the models used to model the system dynamics can be found [here](/docs/models.md)

## Dynamics

The dynamics simulation implementation decisions are based on [Aircraft Control and Simulation by Brian L. Stevens, Frank L. Lewis, Eric N. Johnson](onlinelibrary.wiley.com/doi/book/10.1002/9781119174882). Elaboration on specifics, independent derivations and stability [here](/docs/dynamics.md)

## Control

The initial control implementation is a reformulation of the NLP from [Minimum-Time Quadrotor Waypoint Flight in Cluttered Environments](https://github.com/uzh-rpg/sb_min_time_quadrotor_planning). Further design decisions [here](/docs/control.md)

## Examples

[here](/docs/examples.md)

# This might be useful

[Post on how to remove jupyter notebook output from git commits.](https://gist.github.com/33eyes/431e3d432f73371509d176d0dfb95b6e)

## TBR Links

AerodynamicShapeOp

## Install

install with `pip install -e .`



