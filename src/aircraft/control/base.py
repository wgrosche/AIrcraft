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



class VariableTimeMixin:
    def _init_variable_time(self, opti: ca.Opti, num_nodes: int, timescale: float = 1.0):
        self._variable_time_enabled = True
        self._num_nodes = num_nodes
        self._timescale = timescale

        self.timescale_parameter = opti.parameter()
        opti.set_value(self.timescale_parameter, self._timescale)

        self.time = self.timescale_parameter * opti.variable()
        opti.subject_to(self.time > 0)

        if hasattr(self, "constraint_descriptions"):
            self.constraint_descriptions.append("positive time constraint")

        self._dt = self.time / self._num_nodes

    @property
    def dt(self):
        return self._dt




class ControlProblem(ABC):
    """
    Abstract base class for defining optimal control problems using CasADi's Opti framework.

    This class provides a generic structure for setting up, solving, and managing trajectory 
    optimization or model predictive control problems involving continuous dynamics. It supports
    scaling, logging, parameterization, and flexible constraint/variable setup over a discrete time horizon.

    Subclasses must implement:
        - `control_constraint(node)`: Enforce control-specific constraints (e.g., bounds).
        - `loss(nodes, time)`: Define the cost function to minimize.

    Attributes:
        opti (ca.Opti): The underlying CasADi optimization object.
        dynamics (ca.Function): A function f(x, u, dt) → x_next describing the system's dynamics.
        num_nodes (int): Number of control nodes (time steps) in the horizon.
        state_dim (int): Dimension of the system state vector.
        control_dim (int): Dimension of the control input vector.
        dt (ca.MX or float): Time step size (can be symbolic if time is optimized).
        time (ca.MX): Total time of the horizon if optimized.
        verbose (bool): Print debug info during setup if True.
        solver_opts (dict): Options for the underlying solver (default: IPOPT).
        scale_state (Optional[list[float]]): Optional scaling factors for states.
        scale_control (Optional[list[float]]): Optional scaling factors for controls.
        constraint_descriptions (list[str]): Human-readable descriptions of constraints (for debugging/logging).
        sol_state_list (list[np.ndarray]): List of solved states over iterations.
        sol_control_list (list[np.ndarray]): List of solved controls over iterations.
        final_times (list[float]): Final time values from each solve (for adaptive time problems).
        h5file (Optional[h5py.File]): Optional HDF5 file for logging progress to disk.

    Initialization Args:
        dynamics (ca.Function): Dynamics function f(x, u, dt).
        num_nodes (int): Number of control steps.
        opts (dict, optional): Dictionary of optional flags and configuration:
            - 'verbose' (bool): Print debug output.
            - 'scale_state' (list): State scaling factors.
            - 'scale_control' (list): Control scaling factors.
            - 'initial_state' (np.ndarray): Initial state for warm starting.
            - 'solver_options' (dict): Options passed to IPOPT.
            - 'savefile' (str): Path to an HDF5 file for saving progress.

    Usage (in subclasses):
        - Call `setup(guess)` to build the problem.
        - Call `solve()` to solve it.
        - Override `control_constraint()` and `loss()` to define problem-specific behavior.
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
        self.timescale_parameter = self.opti.parameter()
        self.state_guess_parameter = self.opti.parameter(self.state_dim, self.num_nodes + 1)
        self.control_guess_parameter = self.opti.parameter(self.control_dim, self.num_nodes + 1)
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

    def _scale_variable(self, var, scale):
        return ca.DM(scale) * var if scale else var

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
        self.constraint(next.state - self.dynamics(node.state, node.control, dt) == 0, description=f"state dynamics constraint at node {node.index}")

    @abstractmethod
    def loss(self, nodes:List[ControlNode], time:Optional[ca.MX] = None) -> ca.MX:
        pass

    def _setup_step(self, index:int, current_node:ControlNode, guess:np.ndarray):
        opti = self.opti

        next_node = ControlNode(
            index=0,
            state=self._scale_variable(opti.variable(self.state_dim), self.scale_state),
            control=self._scale_variable(opti.variable(self.control_dim), self.scale_control),
            )

            
        self.state_constraint(current_node, next_node, self.dt)
        self.control_constraint(current_node)

        # initial_state = opti.parameter(13) TODO: investigate this for speeding up initialisation esp for MHE
        
        opti.set_initial(next_node.state, self.state_guess_parameter[:, index])
        opti.set_initial(next_node.control, self.control_guess_parameter[:, index])
        return next_node
    
    # def _setup_time(self) -> None:
    #     opti = self.opti
    #     if not self.timescale:
    #         self.timescale = 1
    #     opti.set_value(self.timescale_parameter, self.timescale)
    #     self.time = self.timescale_parameter * opti.variable()
    #     self.constraint(self.time > 0, description="positive time constraint")
    #     opti.set_initial(self.time, self.timescale_parameter)
        
    #     self.dt = self.time / self.num_nodes

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

        opti.set_initial(current_node.state, self.state_guess_parameter[:, 0])
        self.constraint(current_node.state == self.state_guess_parameter[:, 0])
        opti.set_initial(current_node.control, self.control_guess_parameter[:, 0])

        return current_node
    
    def _setup_variables(self, nodes:List[ControlNode]) -> None:
        self.state = ca.hcat([nodes[i].state for i in range(self.num_nodes + 1)])
        self.control = ca.hcat([nodes[i].control for i in range(self.num_nodes)])

        if self.verbose:
            print("State Shape: ", self.state.shape)
            print("Control Shape: ", self.control.shape)


    def _setup_objective(self, nodes):
        self.opti.minimize(self.loss(nodes = nodes, time = self.time))

    # def setup(self, guess: Optional[np.ndarray] = None):
    #     guess = guess if guess is not None else np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))

    #     self.opti.set_value(self.state_guess_parameter, guess[:self.state_dim, :])
    #     self.opti.set_value(self.control_guess_parameter, guess[self.state_dim:self.state_dim + self.control_dim, :])
    #     self._setup_time()
    #     current_node = self._setup_initial_node(guess)
    #     nodes = [current_node]
        
    #     for index in range(1, self.num_nodes + 1):
    #         nodes.append(self._setup_step(index, nodes[-1], guess))

    #     self._setup_variables(nodes)
    #     self._setup_objective(nodes)

    def setup(self, guess: np.ndarray):
        guess = guess if guess is not None else np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))

        self.opti.set_value(self.state_guess_parameter, guess[:self.state_dim, :])
        self.opti.set_value(self.control_guess_parameter, guess[self.state_dim:self.state_dim + self.control_dim, :])

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

    def get_solution(self, sol: ca.OptiSol) -> dict:
        return {
            "state": sol.value(self.state),
            "control": sol.value(self.control),
            "time": sol.value(self.time) if hasattr(self, 'time') else None
        }
