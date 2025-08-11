import casadi as ca
import numpy as np
from typing import Optional
from aircraft.dynamics.aircraft import Aircraft
import matplotlib.pyplot as plt
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData
from aircraft.control.base import ControlNode, ControlProblem
from dataclasses import dataclass
from contextlib import contextmanager

@contextmanager
def reverse_dynamics_override(aircraft, forces_var, moments_var, state_dim, control_dim):
    """
    Temporarily override forces_ned and moments_frd to return decision variables.
    Restores originals on exit.
    """
    # Save originals
    original_forces = aircraft._forces_ned
    original_moments = aircraft._moments_frd

    # Save originals to call later if needed
    aircraft._original_forces_ned = original_forces
    aircraft._original_moments_frd = original_moments

    # Create symbolic placeholders for the override
    state_sym = ca.MX.sym('state', state_dim)
    control_sym = ca.MX.sym('control', control_dim)

    # Override with decision variable functions
    aircraft._forces_ned = forces_var
    aircraft._moments_frd = moments_var

    try:
        yield aircraft
    finally:
        # Restore originals
        aircraft._forces_ned = original_forces
        aircraft._moments_frd = original_moments

@dataclass
class ReverseNode(ControlNode):
    forces: Optional[ca.MX] = None
    moments:Optional[ca.MX] = None


class Reverse(ControlProblem):

    
    def __init__(self, *, aircraft: Aircraft, **kwargs) -> None:
        super().__init__(aircraft=aircraft, **kwargs)
        self.aircraft = aircraft
        self.aircraft_forces = aircraft.forces_ned
        self.aircraft_moments = aircraft.moments_frd
        

    
    def _setup_initial_node(self, guess: np.ndarray) -> ControlNode:
        node = self._make_node(index=0, guess=guess, enforce_state_constraint=True)
        node = ReverseNode.from_parent(control_node = node, forces = self.opti.variable(3), moments = self.opti.variable(3))
        return node


    def _setup_step(self, index: int, current_node: ControlNode, guess: np.ndarray) -> ControlNode:
        next_node = self._make_node(index=index, guess=guess)
        next_node = ReverseNode.from_parent(control_node = next_node, forces = self.opti.variable(3), moments = self.opti.variable(3))
        self.state_constraint(current_node, next_node)
        self.control_constraint(current_node)
        return next_node


    def state_constraint(self, node: ReverseNode, next: ReverseNode) -> None:
        # opti = self.opti
        
        dt_i = 1.0 / node.progress**2 if self.opts.get('time', 'fixed') in ['progress', 'fixed'] else ca.MX(node.progress)**2
        # state_sym = ca.MX.sym('state', self.state_dim)
        # control_sym = ca.MX.sym('control', self.control_dim)

        with reverse_dynamics_override(self.aircraft, node.forces, node.moments,
                                self.state_dim, self.control_dim):

            # Integration now sees decision variables
            next_state = self.dynamics(node.state, node.control, dt_i)
            self.constraint(next.state - next_state == 0,
                            description=f"state dynamics constraint at node {node.index}")

        # Now you can add constraints using the original model outputs
        self.constraint(
            self.aircraft.forces_ned(node.state, node.control) - node.forces == 0,
            description="force constraint"
        )
        self.constraint(
            self.aircraft.moments_frd(node.state, node.control) - node.moments == 0,
            description="moment constraint"
        )

        # self.aircraft.forces_ned = ca.Function('forces_ned_override', [state_sym, control_sym], [node.forces])
        # self.aircraft.moments_frd = ca.Function('moments_frd_override', [state_sym, control_sym], [node.moments])
        # next_state = self.dynamics(node.state, node.control, dt_i)
        # self.constraint(next.state - next_state == 0, description=f"state dynamics constraint at node {node.index}")


        # # dynamics constraint

        # self.constraint(self.aircraft_forces(node.state, node.control) - node.forces == 0, description="force constraint")
        # self.constraint(self.aircraft_moments(node.state, node.control) - node.moments == 0, description="moment constraint")

        if self.opts.get('quaternion', None) == 'constraint':
            self.constraint(ca.dot(next.state[6:10], next.state[6:10]) == 1, description=f"quaternion norm constraint at node {node.index}")

        elif self.opts.get('quaternion', None) == 'baumgarte':
            x_dot = self.x_dot(next.state, next.control)
            assert isinstance(x_dot, ca.MX)
            assert x_dot.size()[0] == self.state_dim, f"x_dot: {x_dot.size()[0]} state: {self.state_dim}"
            x_dot_q = x_dot[6:10]
            phi_dot = 2 * ca.dot(next.state[6:10], x_dot_q)

            alpha = 2.0  # damping
            beta = 2.0   # stiffness

            phi = ca.dot(next.state[6:10], next.state[6:10]) - 1
            stabilized_phi = 2 * alpha * phi_dot + beta**2 * phi

            self.constraint(stabilized_phi == 0, description="Baumgarte quaternion normalization")

        if self.opts.get('time', 'fixed') in ['progress', 'variable']:
            self.constraint(next.progress == node.progress)#self.max_jump)

        elif self.opts.get('time', 'fixed') == 'adaptive':
            assert next.progress is not None, "cannot run adaptive without progress variable"
            alpha = 1e-2
            adaptive_weight = 1.0
            tol = 1e-2
            func_state = self.dynamics(next.state, next.control)
            J = ca.jacobian(func_state, next.state)
            prod = J @ func_state
            error_surrogate = alpha * (1 / next.progress**4) * ca.dot(prod, J @ prod)
            self.constraint(error_surrogate <= tol, description="Error bound for adaptive timestepping")
            self.opti.minimize(adaptive_weight * next.progress**2)