import casadi as ca
import numpy as np
from typing import Optional
from aircraft.dynamics.aircraft import Aircraft
import matplotlib.pyplot as plt
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData
from aircraft.control.base import ControlNode, ControlProblem

class AircraftControl(ControlProblem):
    """
    Class that implements constraints upon state and control for an aircraft
    """

    def __init__(self, *, aircraft: Aircraft, **kwargs) -> None:

        self.aircraft = aircraft
        self.plotter = TrajectoryPlotter(aircraft)
        self.control_limits = kwargs.get('control_limits', {"aileron": [-5, 5], "elevator": [-5, 5], "rudder": [-5, 5]})

        super().__init__(system=aircraft, **kwargs)
        
        
    def control_constraint(self, node: ControlNode) -> None:
        super().control_constraint(node)
        self.constraint(
            self.opti.bounded(self.control_limits["aileron"][0], node.control[0], self.control_limits["aileron"][1]), 
            description="Aileron Constraint")
        self.constraint(
            self.opti.bounded(self.control_limits["elevator"][0], node.control[1], self.control_limits["elevator"][1]), 
            description="Elevator Constraint")
        self.constraint(
            self.opti.bounded(self.control_limits["rudder"][0], node.control[2], self.control_limits["rudder"][1]), 
            description="Rudder Constraint")
        

    def state_constraint(self, node: ControlNode, next: ControlNode) -> None:
        super().state_constraint(node, next)
        v_rel = self.aircraft.v_frd_rel(node.state, node.control)
        self.constraint(
            self.opti.bounded(20**2, ca.dot(v_rel, v_rel), 80**2), # type: ignore[arg-type]
            description="Speed constraint")
        self.constraint(
            self.opti.bounded(-np.deg2rad(90), self.aircraft.phi(node.state), np.deg2rad(90)), 
            description="Roll constraint")
        self.constraint(
            self.opti.bounded(-np.deg2rad(10), self.aircraft.beta(node.state, node.control), np.deg2rad(10)), 
            description="Sideslip constraint")
        self.constraint(
            self.opti.bounded(-np.deg2rad(20), self.aircraft.alpha(node.state, node.control), np.deg2rad(20)), 
            description="Attack constraint")
        self.constraint(node.state[2] < 0, description="Height constraint")




    def log(self, iteration:int) -> None:
        super().log(iteration)

        aircraft = self.aircraft
        f = aircraft.state_update(aircraft.state, aircraft.control, aircraft.dt_sym)
        
        # Compute the Jacobian of f w.r.t state
        jacobian = ca.jacobian(f, aircraft.state)
        
        # Create a CasADi function for numerical evaluation
        jacobian_func = ca.Function('J', [aircraft.state, aircraft.control, aircraft.dt_sym], [jacobian])
        evaluated_jacobian = np.array(jacobian_func(self.opti.debug.value(self.state)[:, 1:],
                                           self.opti.debug.value(self.control),
                                           self.opti.debug.value(self.time)/self.num_nodes))
        condition_numbers = np.linalg.cond(evaluated_jacobian)
        
        self.logger.info(f"Condition numbers: {condition_numbers}")
        # Get constraint values and dual variables
        g = self.opti.debug.value(self.opti.g)
        lam_g = self.opti.debug.value(self.opti.lam_g)
        
        # Check which constraints are active (close to bounds)
        tolerance = 1e-6
        active_constraints = np.nonzero(abs(g) > tolerance)[0]
        # Check dynamics violations
        dynamics_violations = []
        for i in range(self.num_nodes):
            current_state = self.opti.debug.value(self.state)[:, i]
            current_control = self.opti.debug.value(self.control)[:, i]
            dt = self.opti.debug.value(self.time)/self.num_nodes
            
            predicted_next = self.dynamics(current_state, current_control, dt)
            actual_next = self.opti.debug.value(self.state)[:, i+1]
            
            dynamics_violation = np.linalg.norm(predicted_next - actual_next)
            if dynamics_violation > 1e-3:
                dynamics_violations.append((i, dynamics_violation))
                self.logger.warning(f"Large dynamics violation at node {i}: {dynamics_violation}")
        
        if not dynamics_violations:
            self.logger.info("No significant dynamics violations detected")
        
        # Log active constraints
        if len(active_constraints) > 0:
            self.logger.info(f"Active constraints: {len(active_constraints)} constraints")
            self.logger.info(f"Constraint values: {g[active_constraints]}")
            self.logger.info(f"Dual variables: {lam_g[active_constraints]}")
        else:
            self.logger.info("No active constraints")
        
        # Log control limits
        control_values = self.opti.debug.value(self.control)
        aileron_limit = abs(control_values[0]) >= 5.0
        elevator_limit = abs(control_values[1]) >= 5.0
        self.logger.info(f"Control limits active - Aileron: {aileron_limit}, Elevator: {elevator_limit}")
        
        state = self.opti.debug.value(self.state)[:, :-1]
        control = self.opti.debug.value(self.control)
        
        airspeed_values = self.aircraft.airspeed(state, control)
        alpha_values = self.aircraft.alpha(state, control)
        beta_values = self.aircraft.beta(state, control)
        
        self.logger.info("State constraints:")
        self.logger.info(f"Airspeed values: {airspeed_values}")
        self.logger.info(f"Alpha values: {alpha_values}")
        self.logger.info(f"Beta values: {beta_values}")
    
    def callback(self, iteration: int):
        super().callback(iteration)
        if self.plotter and iteration % 10 == 5:
            trajectory_data = TrajectoryData(
                state=np.array(self.opti.debug.value(self.state))[:, 1:],
                control=np.array(self.opti.debug.value(self.control)),
                time=np.array(self.opti.debug.value(self.time))
            )
            self.plotter.plot(trajectory_data=trajectory_data)
            plt.draw()
            self.plotter.figure.canvas.start_event_loop(0.0002)


    def solve(self, warm_start: Optional[ca.OptiSol] = None):
        sol = super().solve(warm_start=warm_start)
        trajectory_data = TrajectoryData(
                state=np.array(self.opti.debug.value(self.state))[:, :-1],
                control=np.array(self.opti.debug.value(self.control)),
                time=np.array(self.opti.debug.value(self.time))
            )
        self.plotter.plot(trajectory_data=trajectory_data)
        plt.draw()
        self.plotter.figure.canvas.start_event_loop(0.0002)
        return sol