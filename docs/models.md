
# Models

We use the following approaches to model our system dynamics:

## Direct MLP

We model the aerodynamic coefficients directly using a simple MLP architecture.
The inputs for the model are the aerodynamic state, characterised by the control
surface deflections (aileron deflection, elevator deflection); the dynamic
pressure $\bar{q} = \rho S V_{rel}$; and the aerodynamic angles, $\alpha$ and
$\beta$. Further information on the definitions can be found in the
[dynamics](dynamics.md) page.

## Linearised

TODO

## Residuals MLP

We use a linearised model augmented with an MLP that predicts the error between
the linearised dynamics and those measured in the windtunnel.

## Visualising Results

The dynamics surrogates can be visualised using [this script](../main/visualise/visualise_forces.py).
