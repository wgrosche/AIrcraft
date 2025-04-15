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

__all__ = ['ControlNode', 'SaveMixin', 'ControlProblem', 'VariableTimeMixin']

@dataclass
class ControlNode:
    """
    A data structure representing a single node in the control horizon.

    Attributes:
        index (Optional[int]): Index of the node in the trajectory.
        state (Optional[ca.MX]): Symbolic representation of the state at this node.
        control (Optional[ca.MX]): Symbolic representation of the control at this node.
    """
    index:Optional[int] = None
    state:Optional[ca.MX] = None
    control:Optional[ca.MX] = None

class SaveMixin:
    """
    A mixin that enables saving progress of an optimization problem to an HDF5 file.

    Methods:
        _init_saving(filepath, force_overwrite): Initializes saving and opens the file.
        _save_progress(iteration, states, controls, times): Writes a snapshot of the latest progress to the HDF5 file.
    """
    def _init_saving(self, filepath: Optional[str], force_overwrite: bool = True):
        self._save_enabled = filepath is not None
        self._save_interval = 10  # can expose as parameter later

        self.h5file = None
        self._save_path = filepath

        if self._save_enabled:
            if force_overwrite and Path(filepath).exists():
                Path(filepath).unlink()

            self.h5file = h5py.File(filepath, "a")

    def _save_progress(
        self,
        iteration: int,
        states: list,
        controls: list,
        times: list
    ):
        if not self._save_enabled or self.h5file is None:
            return

        try:
            for i, (state, control, time_val) in enumerate(zip(states[-10:], controls[-10:], times[-10:])):
                grp_name = f'iteration_{iteration - 10 + i}'
                grp = self.h5file.require_group(grp_name)
                grp.attrs['timestamp'] = time.time()

                for name, data in zip(['state', 'control', 'time'], [state, control, time_val]):
                    if name in grp:
                        del grp[name]
                    grp.create_dataset(name, data=data, compression='gzip')
        except Exception as e:
            print(f"[SaveMixin] Error saving progress: {e}")


class VariableTimeMixin:
    """
    A mixin that enables optimization over variable total time.

    Attributes:
        _variable_time_enabled (bool): Whether variable time is enabled.
        timescale_parameter (ca.MX): CasADi parameter for scaling time.
        time (ca.MX): Symbolic total time variable.
        _dt (ca.MX): Time step duration, computed as time / num_nodes.

    Methods:
        _init_variable_time(opti, num_nodes, timescale): Setup symbolic variable time.
        dt: Property returning symbolic time step size.
    """
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
        opti (ca.Opti): CasADi optimization problem object.
        dynamics (ca.Function): Dynamics function f(x, u, dt).
        num_nodes (int): Number of time steps in the horizon.
        state_dim (int): Dimension of the system state.
        control_dim (int): Dimension of the control input.
        dt (ca.MX or float): Time step size (symbolic if time is optimized).
        time (ca.MX): Total horizon time (if variable time enabled).
        verbose (bool): Verbose setup output.
        solver_opts (dict): IPOPT or other solver options.
        scale_state (Optional[list[float]]): Optional scaling for state variables.
        scale_control (Optional[list[float]]): Optional scaling for control variables.
        constraint_descriptions (list): Text descriptions of constraints.
        sol_state_list (list): History of solved state trajectories.
        sol_control_list (list): History of solved control trajectories.
        final_times (list): Solved final time values.
        h5file (Optional[h5py.File]): Handle to HDF5 file if saving is enabled.

    Methods:
        setup(guess): Build and initialize problem from an initial guess.
        solve(): Solve the control problem.
        get_solution(sol): Extract values from a CasADi solution.
        log(iteration): Log iteration data to file.
        callback(iteration): Store data and trigger logging.
        constraint(expr, description): Register constraint with optional description.
    """

    def __init__(self, *, dynamics:ca.Function, num_nodes:int, opts:Optional[dict] = {}, **kwargs):
        """
        Initialize the control problem.

        Args:
            dynamics (ca.Function): Dynamics function f(x, u, dt).
            num_nodes (int): Number of control nodes (time steps).
            opts (dict, optional): Dictionary of configuration options:
                - 'verbose' (bool): Enable debug output.
                - 'scale_state' (list): Scaling for state variables.
                - 'scale_control' (list): Scaling for control inputs.
                - 'initial_state' (np.ndarray): Warm start initial state.
                - 'solver_options' (dict): IPOPT solver options.
                - 'savefile' (str): HDF5 save path.
        """

        self.opti = ca.Opti('nlp')

        self.state_dim = dynamics.size1_in(0)
        self.control_dim = dynamics.size1_in(1)

        self.num_nodes = num_nodes
        self.verbose = opts.get('verbose', False)
        self.dynamics = dynamics

        self.scale_state = opts.get('scale_state', None)
        self.scale_control = opts.get('scale_control', None)

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
        """
        Abstract method for applying control-related constraints.
        Should be implemented in subclasses.
        """
        pass

    def state_constraint(self, node:ControlNode, next:ControlNode, dt:ca.MX) -> None:
        self.constraint(next.state - self.dynamics(node.state, node.control, dt) == 0, description=f"state dynamics constraint at node {node.index}")

    @abstractmethod
    def loss(self, nodes:List[ControlNode], time:Optional[ca.MX] = None) -> ca.MX:
        """
        Abstract method for defining the objective function.
        Should return the scalar loss to minimize.

        Args:
            nodes (List[ControlNode]): List of control nodes.
            time (Optional[ca.MX]): Optional total time variable.

        Returns:
            ca.MX: Symbolic expression for the loss.
        """
        pass

    def _setup_step(self, index:int, current_node:ControlNode, guess:np.ndarray):
        opti = self.opti

        next_node = ControlNode(
            index=0,
            state=self._scale_variable(opti.variable(self.state_dim), self.scale_state),
            control=self._scale_variable(opti.variable(self.control_dim), self.scale_control),
            )

        state_guess = guess[:self.state_dim, index]
        control_guess = guess[self.state_dim:self.state_dim + self.control_dim, index]
        self.state_constraint(current_node, next_node, self.dt)
        self.control_constraint(current_node)
        
        opti.set_initial(next_node.state, state_guess)
        opti.set_initial(next_node.control, control_guess)
        return next_node

    def _setup_initial_node(self, guess:np.ndarray) -> ControlNode:
        opti = self.opti

        current_node = ControlNode(
            index=0,
            state=self._scale_variable(opti.variable(self.state_dim), self.scale_state),
            control=self._scale_variable(opti.variable(self.control_dim), self.scale_control),
            )

        state_guess = guess[:self.state_dim, 0]
        control_guess = guess[self.state_dim:self.state_dim + self.control_dim, 0]

        self.constraint(current_node.state == self.state_guess_parameter[:, 0])

        opti.set_initial(current_node.state, state_guess)
        opti.set_initial(current_node.control, control_guess)

        return current_node
    
    def _setup_variables(self, nodes:List[ControlNode]) -> None:
        self.state = ca.hcat([nodes[i].state for i in range(self.num_nodes + 1)])
        self.control = ca.hcat([nodes[i].control for i in range(self.num_nodes)])

        if self.verbose:
            print("State Shape: ", self.state.shape)
            print("Control Shape: ", self.control.shape)


    def _setup_objective(self, nodes):
        self.opti.minimize(self.loss(nodes = nodes, time = self.time))

    def setup(self, guess: np.ndarray):
        """
        Set up the optimization problem using an initial guess.

        Args:
            guess (np.ndarray): Initial guess array of shape (state_dim + control_dim, num_nodes + 1).
        """
        guess = guess if guess is not None else np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))

        self.opti.set_value(self.state_guess_parameter, guess[:self.state_dim, :])
        self.opti.set_value(self.control_guess_parameter, guess[self.state_dim:self.state_dim + self.control_dim, :])

        current_node = self._setup_initial_node(guess)
        nodes = [current_node]

        for index in range(1, self.num_nodes + 1):
            nodes.append(self._setup_step(index, nodes[-1], guess))

        self._setup_variables(nodes)
        self._setup_objective(nodes)

    def callback(self, iteration: int):
        self.sol_state_list.append(self.opti.debug.value(self.state))
        self.sol_control_list.append(self.opti.debug.value(self.control))
        self.final_times.append(self.opti.debug.value(self.time))

        if hasattr(self, "_save_progress"):
            self._save_progress(iteration, self.sol_state_list, self.sol_control_list, self.final_times)

        self.log(iteration)

    def solve(self, warm_start:Optional[ca.OptiSol] = None) -> ca.OptiSol:
        self.opti.solver('ipopt', self.solver_opts)
        self.opti.callback(lambda iteration: self.callback(iteration))
        if warm_start:
            self.opti.set_initial(warm_start.value_variables())
        sol = self.opti.solve()
        return sol
    
    def log(self, iteration:int):
        """
        Log solver progress

        Args:
            iteration (int): Current iteration number.
        """
        if not hasattr(self, 'logger'):
            log_dir = Path(DATAPATH) / 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f'aircraft_control_{timestamp}.log'
            
            self.logger = logging.getLogger('aircraft_control')
            self.logger.setLevel(logging.INFO)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            
            self.logger.info("Starting optimization logging")

        self.logger.info(f"Iteration {iteration}")
        self.logger.info(f"Final time: {self.opti.debug.value(self.time)[-1]}")
        self.logger.info(f"Final position: {self.opti.debug.value(self.state)[:, -1]}")
        self.logger.info(f"Final velocity: {self.opti.debug.value(self.state)[2, -1]}")
        self.logger.info(f"Final control: {self.opti.debug.value(self.control)[:, -1]}")
        self.logger.info(f"Final control rate: {self.opti.debug.value(self.control)[:, -1] - self.opti.debug.value(self.control)[:, -2]}")

    def get_solution(self, sol: ca.OptiSol) -> dict:
        """
        Get the solution from the optimization problem.

        Args:
            sol (ca.OptiSol): Solution object.
        """
        return {
            "state": sol.value(self.state),
            "control": sol.value(self.control),
            "time": sol.value(self.time) if hasattr(self, 'time') else None
        }
