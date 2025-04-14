import casadi as ca
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import h5py
from typing import Type, List
import logging
import os
from datetime import datetime
from pathlib import Path
import time
from aircraft.config import default_solver_options, BASEPATH, NETWORKPATH, DATAPATH, DEVICE, rng
from abc import ABC, abstractmethod

plt.ion()

@dataclass
class ControlNode:
    index:Optional[int] = None
    state:Optional[ca.MX] = None
    control:Optional[ca.MX] = None


def plot_convergence(self, ax:plt.axes, sol:ca.OptiSol):
    ax.semilogy(sol.stats()['iterations']['inf_du'], label="Dual infeasibility")
    ax.semilogy(sol.stats()['iterations']['inf_pr'], label="Primal infeasibility")

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Infeasibility (log scale)')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show(block = True)



class ControlProblem(ABC):
    """
    Control Problem parent class
    """
    

    def __init__(self, *, dynamics:ca.Function, num_nodes:int, opts:Optional[dict] = {}, **kwargs):

        self.opti = ca.Opti('nlp')
        self.state_dim = dynamics.size1_in(0)
        self.control_dim = dynamics.size1_in(1)
        self.num_nodes = num_nodes
        self.verbose = opts.get('verbose', False)
        self.dynamics = dynamics

        self.scale_state = opts.get('scale_state', None)
        self.scale_control = opts.get('scale_control', None)
        self.timescale = None
        self.initial_state = opts.get('initial_state', None)
        self.solver_opts = opts.get('solver_options', default_solver_options)

        self.filepath = opts.get('savefile', None)
        if self.filepath:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            self.h5file = h5py.File(self.filepath, "a")

        self.sol_state_list = []
        self.sol_control_list = []
        self.final_times = []
        self.constraint_descriptions = []
        super().__init__(**kwargs)

    def constraint(self, expr, description=None) -> None:
        """
        Wrapper for adding constraints to the opti stack with optional auto-description.

        Args:
            expr: Constraint expression (scalar or vector). Can be bounded, equality, or inequality.
            description: Optional human-readable description string.
        """
        self.opti.subject_to(expr)
        num_constraints = expr.numel()
        # Auto-generate a fallback description if not provided
        if description is None:
            # Try to infer the type
            if hasattr(expr, 'is_equal') and expr.is_equal():  # CasADi equality
                inferred_type = "equality"
            elif hasattr(expr, 'is_lte') and expr.is_lte():    # CasADi less-than
                inferred_type = "inequality"
            else:
                inferred_type = "constraint"

            # Use the symbolic expression as a fallback string
            raw_str = str(expr)

            # Limit to 1 line and truncate if too long
            if "\n" in raw_str:
                raw_str = raw_str.split("\n")[0]
            if len(raw_str) > 80:
                raw_str = raw_str[:77] + "..."

            description = f"{inferred_type}: {raw_str}"

        # Expand descriptions if vector-valued
        if num_constraints == 1:
            self.constraint_descriptions.append(description)
        else:
            self.constraint_descriptions.extend(
                [f"{description} (dim {i})" for i in range(num_constraints)]
            )

    @abstractmethod
    def control_constraint(self, node:ControlNode):
        pass

    def state_constraint(self, node:ControlNode, next:ControlNode, dt:ca.MX) -> None:
        self.constraint(next.state - self.dynamics(node.state, node.control, dt) == 0)

    @abstractmethod
    def loss(self, nodes:List[ControlNode], time:Optional[ca.MX] = None) -> ca.MX:
        pass

    def _setup_step(self, index:int, current_node:ControlNode, guess:np.ndarray):
        opti = self.opti

        if self.scale_state and self.scale_control:
            next_node = ControlNode(
                index=0,
                state=ca.DM(self.scale_state) * opti.variable(self.state_dim),
                control=ca.DM(self.scale_control) * opti.variable(self.control_dim),
            )
        else:
            next_node = ControlNode(
                index=0,
                state=opti.variable(self.state_dim),
                control=opti.variable(self.control_dim),
            )
            
        self.state_constraint(current_node, next_node, self.dt)
        self.control_constraint(current_node)

        # initial_state = opti.parameter(13) TODO: investigate this for speeding up initialisation esp for MHE

        opti.set_initial(next_node.state, guess[:self.state_dim, index])
        opti.set_initial(next_node.control, guess[self.state_dim:, index])
        return next_node
    
    def _setup_time(self) -> None:
        opti = self.opti
        if not self.timescale:
            self.timescale = 1
        self.time = self.timescale * opti.variable()
        self.constraint(self.time > 0, description="positive time constraint")
        opti.set_initial(self.time, self.timescale)
        self.dt = self.time / self.num_nodes

    def _setup_initial_node(self, guess:np.ndarray) -> ControlNode:
        opti = self.opti
        if self.scale_state and self.scale_control:
            current_node = ControlNode(
                index=0,
                state=ca.DM(self.scale_state) * opti.variable(self.state_dim),
                control=ca.DM(self.scale_control) * opti.variable(self.control_dim),
            )
        else:
            current_node = ControlNode(
                index=0,
                state=opti.variable(self.state_dim),
                control=opti.variable(self.control_dim),
            )
        state_guess = guess[:self.state_dim, :]
        control_guess = guess[self.state_dim:self.state_dim + self.control_dim, :]
        opti.set_initial(current_node.state, state_guess[:, 0])
        self.constraint(current_node.state == state_guess[:, 0])
        opti.set_initial(current_node.control, control_guess[:, 0])

        return current_node
    
    def _setup_variables(self, nodes:List[ControlNode]) -> None:
        self.state = ca.hcat([nodes[i].state for i in range(self.num_nodes + 1)])
        self.control = ca.hcat([nodes[i].control for i in range(self.num_nodes)])

        if self.verbose:
            print("State Shape: ", self.state.shape)
            print("Control Shape: ", self.control.shape)


    def _setup_objective(self, nodes):
        self.opti.minimize(self.loss(nodes = nodes, time = self.time))

    def setup(self, guess:np.ndarray):
        self._setup_time()
        current_node = self._setup_initial_node(guess)
        nodes = [current_node]
        
        for index in range(1, self.num_nodes + 1):
            nodes.append(self._setup_step(index, nodes[-1], guess))

        self._setup_variables(nodes)
        self._setup_objective(nodes)

    def save_progress(self, iteration, states, controls, time_vals, save_interval:int = 10):
        if self.h5file is not None and iteration % save_interval == 0:
            try:
                for i, (state, control, time_val) in enumerate(zip(states[-10:], controls[-10:], time_vals[-10:])):
                    grp_name = f'iteration_{iteration - 10 + i}'
                    grp = self.h5file.require_group(grp_name)
                    grp.attrs['timestamp'] = time.time()

                    for name, data in zip(['state', 'control', 'time'], [state, control, time_val]):
                        if name in grp:
                            del grp[name]  # Overwrite if dataset already exists
                        grp.create_dataset(name, data=data, compression='gzip')
            except Exception as e:
                print(f"Error saving progress: {e}")

    def callback(self, iteration: int):
        self.sol_state_list.append(self.opti.debug.value(self.state))
        self.sol_control_list.append(self.opti.debug.value(self.control))
        self.final_times.append(self.opti.debug.value(self.time))
        self.save_progress(iteration, self.sol_state_list, self.sol_control_list, self.final_times)
        self.log(iteration)

    def solve(self, warm_start:Optional[ca.OptiSol] = None) -> ca.OptiSol:
        self.opti.solver('ipopt', self.solver_opts)
        self.opti.callback(lambda iteration: self.callback(iteration))
        if warm_start:
            self.opti.set_initial(warm_start.value_variables())
        sol = self.opti.solve()
        return sol
    
    def log(self, iteration):
        # Set up logging if not already configured
        if not hasattr(self, 'logger'):
            # Create logs directory if it doesn't exist
            log_dir = Path(DATAPATH) / 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            # Create a unique log file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f'aircraft_control_{timestamp}.log'
            
            # Configure logger
            self.logger = logging.getLogger('aircraft_control')
            self.logger.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            
            self.logger.info("Starting optimization logging")

        self.logger.info(f"Iteration {iteration}")
        self.logger.info(f"Final time: {self.opti.debug.value(self.time)[-1]}")
        self.logger.info(f"Final position: {self.opti.debug.value(self.state)[:, -1]}")
        self.logger.info(f"Final velocity: {self.opti.debug.value(self.state)[2, -1]}")
        self.logger.info(f"Final control: {self.opti.debug.value(self.control)[:, -1]}")
        self.logger.info(f"Final control rate: {self.opti.debug.value(self.control)[:, -1] - self.opti.debug.value(self.control)[:, -2]}")

