import casadi as ca
import numpy as np
from dataclasses import dataclass
import os
import h5py
from typing import List, Protocol, runtime_checkable, Optional, Union, Sequence, Generic, TypeVar
import logging
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm

from aircraft.config import default_solver_options, DATAPATH
from aircraft.dynamics.base import SixDOF

__all__ = ['ControlNode', 'SaveMixin', 'ControlProblem']

@dataclass
class ControlNode:
    """
    A data structure representing a single node in the control horizon.

    Attributes:
        index (Optional[int]): Index of the node in the trajectory.
        state (Optional[ca.MX]): Symbolic representation of the state at this node.
        control (Optional[ca.MX]): Symbolic representation of the control at this node.
    """
    index:int
    state:ca.MX
    control:ca.MX
    progress:Union[ca.MX, float] # sqrt of the progress/time variable (to assure positivity)

@runtime_checkable
class SaveProgressProtocol(Protocol):
    def _save_progress(self, iteration: int, states:list[ca.MX], controls:list[ca.MX], times:list[ca.MX]) -> None: ...

class SaveMixin(SaveProgressProtocol):
    """
    A mixin that enables saving progress of an optimization problem to an HDF5 file.

    Methods:
        _init_saving(filepath, force_overwrite): Initializes saving and opens the file.
        _save_progress(iteration, states, controls, times): Writes a snapshot of the latest progress to the HDF5 file.
    """
    def _init_saving(self, filepath:Union[str, Path], force_overwrite: bool = True) -> None:
        self._save_enabled:bool = True
        self._save_interval:int = 10

        if force_overwrite and Path(filepath).exists():
            Path(filepath).unlink()

        self.h5file = h5py.File(filepath, "a")
        assert isinstance(self.h5file, h5py.File)

    def _save_progress(
        self,
        iteration: int,
        states: list[ca.MX],
        controls: list[ca.MX],
        times: list[ca.MX]
    ):
        if not self._save_enabled or self.h5file is None:
            return

        try:
            recent_data = zip(states[-10:], controls[-10:], times[-10:])
            for i, (state, control, time_val) in enumerate(recent_data):
                self._save_iteration_data(
                    group_name=f'iteration_{iteration - 10 + i}',
                    state=state,
                    control=control,
                    time_val=time_val
                )
        except Exception as e:
            print(f"[SaveMixin] Error saving progress: {e}")


    def _save_iteration_data(
        self,
        group_name: str,
        state: ca.MX,
        control: ca.MX,
        time_val: ca.MX
    ):
        assert self.h5file is not None
        grp = self.h5file.require_group(group_name)
        grp.attrs['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")

        data_items = {'state': state, 'control': control, 'time': time_val}
        for name, data in data_items.items():
            if name in grp:
                del grp[name]
            self._create_dataset_safely(grp, name, data)


    def _create_dataset_safely(self, group, name: str, data):
        try:
            if hasattr(data, 'size') and isinstance(data.size, int) and data.size > 1:
                group.create_dataset(name, data=data, compression='gzip')
            else:
                group.create_dataset(name, data=data)
        except Exception as e:
            print(f"[SaveMixin] Failed to create dataset '{name}': {e}")

class PlottingMixin:
    def __init__(self):
        pass

    def _plot(self):
        pass

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

    def __init__(self, *, system:SixDOF, dt:float = 0.01, num_nodes:int, 
                 opts:Optional[dict] = None, progress:bool = False, **kwargs):
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

        self.opts = opts if opts else {}

        if self.opts['quaternion'] == 'integration':
            system.normalise = True
        else:
            system.normalise = False

        self.dynamics:ca.Function = system.state_update
        self.state_dim:int = self.dynamics.size1_in(0)
        self.control_dim:int = self.dynamics.size1_in(1)
        self.x_dot:ca.Function = system.state_derivative

        self.num_nodes = num_nodes

        self.verbose = self.opts.get('verbose', False)
        self.constraint_descriptions = []

        self.progress = self.opts.get('time', 'fixed') in ['progress', 'variable']
        self.dt = dt
        self.dt_bounds = self.opts.get('dt_bounds', (1e-4, 1e-2))  # Default bounds for dt if not specified
        self.max_jump = 0.05
        
        self.scale_state = self.opts.get('scale_state', None)
        self.scale_control = self.opts.get('scale_control', None)


        self.state_guess_parameter = self.opti.parameter(self.state_dim, self.num_nodes + 1)
        self.control_guess_parameter = self.opti.parameter(self.control_dim, self.num_nodes + 1)
        self.initial_state = self.opts.get('initial_state', None)

        self.solver_opts = self.opts.get('solver_options', default_solver_options)
        self.logging = False

        self.filepath = self.opts.get('savefile', None)
        if self.filepath:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            self.h5file = h5py.File(self.filepath, "a")

        self.sol_state_list = []
        self.sol_control_list = []
        self.final_times = []
        
        super().__init__()#**kwargs)

    def _scale_variable(self, var, scale):
        return ca.MX(scale) * var if scale else var

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
        
    def state_constraint(self, node: ControlNode, next: ControlNode) -> None:
        dt_i = 1.0 / node.progress**2 if self.opts.get('time', 'fixed') in ['progress', 'fixed'] else ca.MX(node.progress)**2
        # print(node.progress.is_symbolic())
        if self.opts.get('integration', 'explicit') == 'explicit':
            next_state = self.dynamics(node.state, node.control, dt_i)
            self.constraint(next.state - next_state == 0, description=f"state dynamics constraint at node {node.index}")

        elif self.opts.get('integration', 'explicit') == 'implicit':
            next_state = node.state + dt_i * self.x_dot(next.state, node.control)
            self.constraint(next.state - next_state == 0, description=f"state dynamics constraint at node {node.index}")
        else:
            raise NotImplementedError("Must choose integration mode from ['implicit', 'explicit']")
        
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
            
        
                
    @abstractmethod
    def loss(self, nodes:List[ControlNode], time:Optional[Union[ca.MX, float]] = None) -> ca.MX:
        """
        Abstract method for defining the objective function.
        Should return the scalar loss to minimize.

        Args:
            nodes (List[ControlNode]): List of control nodes.
            time (Optional[ca.MX]): Optional total time variable.

        Returns:
            ca.MX: Symbolic expression for the loss.
        """
        loss = 0
        return loss

    def _make_node(self, index: int, guess: np.ndarray, enforce_state_constraint: bool = False) -> ControlNode:
        opti = self.opti
        # print(type(self.scale_control))
        state = self._scale_variable(opti.variable(self.state_dim), self.scale_state)
        control = self._scale_variable(opti.variable(self.control_dim), self.scale_control)
        # print(self.progress)
        if self.progress:
            # print(type(np.sqrt(1 / self.dt)))
            progress = self._scale_variable(opti.variable(1), ca.MX(1 / self.dt))
            # assert progress.is_symbolic()
        else:
            progress =  ca.sqrt(1 / self.dt)
        
        node = ControlNode(index=index, state=state, control=control, progress=progress)

        # Set initial guesses
        state_guess = guess[:self.state_dim, index]
        control_guess = guess[self.state_dim:self.state_dim + self.control_dim, index]
        opti.set_initial(node.state, state_guess)
        opti.set_initial(node.control, control_guess)

        # Progress constraint
        if self.opts.get('time', 'fixed') == 'progress':
            opti.set_initial(node.progress, ca.sqrt(1 / self.dt))
            self.constraint(
                opti.bounded(
                    ca.sqrt(1 / self.dt_bounds[1]), 
                    node.progress, 
                    ca.sqrt(1 / self.dt_bounds[0])
                    ),  # type: ignore[arg-type]
                description=f"positive progress rate at node {index}"
            )

        elif self.opts.get('time', 'fixed') == 'variable':
            opti.set_initial(node.progress, ca.sqrt(self.dt))
            self.constraint(
                opti.bounded(ca.sqrt(self.dt_bounds[0]), 
                             node.progress, 
                             ca.sqrt(self.dt_bounds[1])),  # type: ignore[arg-type]
                description=f"positive progress rate at node {index}"
            )

        # State constraint only on initial node
        if enforce_state_constraint:
            self.constraint(node.state == self.state_guess_parameter[:, index])

        return node


    def _setup_initial_node(self, guess: np.ndarray) -> ControlNode:
        return self._make_node(index=0, guess=guess, enforce_state_constraint=True)


    def _setup_step(self, index: int, current_node: ControlNode, guess: np.ndarray) -> ControlNode:
        next_node = self._make_node(index=index, guess=guess)
        self.state_constraint(current_node, next_node)
        self.control_constraint(current_node)
        return next_node

    
    def _setup_variables(self, nodes:Sequence[ControlNode]) -> None:
        self.state = ca.hcat([nodes[i].state for i in range(self.num_nodes + 1)])
        self.control = ca.hcat([nodes[i].control for i in range(self.num_nodes)])
        if self.opts.get('time', 'fixed') in ['fixed', 'progress']:
            self.times = ca.cumsum(ca.vertcat(*[1 / nodes[i].progress**2 for i in range(self.num_nodes)]))
        else:
            self.times = ca.cumsum(ca.vertcat(*[nodes[i].progress**2 for i in range(self.num_nodes)]))
        
        if self.progress:
            self.time = ca.sum1(ca.vertcat(*[ca.MX(1.0) / nodes[i].progress**2 for i in range(self.num_nodes)]))
            assert isinstance(self.time, ca.MX)

        else:
            self.time = self.dt * self.num_nodes
            assert isinstance(self.time, float)

        if self.verbose:
            print("State Shape: ", self.state.shape)
            print("Control Shape: ", self.control.shape)


    def _setup_objective(self, nodes):
        self.opti.minimize(self.loss(nodes = nodes, time = self.time))

    def setup(self, guess: np.ndarray) -> None:
        """
        Set up the optimization problem using an initial guess.

        Args:
            guess (np.ndarray): Initial guess array of shape (state_dim + control_dim, num_nodes + 1).
        """
        guess = guess if guess is not None else np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))

        
        self.opti.set_value(self.state_guess_parameter, guess[:self.state_dim, :])
        self.opti.set_value(self.control_guess_parameter, guess[self.state_dim:self.state_dim + self.control_dim, :])
        print("Guess in base setup: ", guess[:, 0])
        current_node = self._setup_initial_node(guess)
        nodes = [current_node]

        for index in tqdm(range(1, self.num_nodes + 1), desc="Setting up nodes..."):
            nodes.append(self._setup_step(index, nodes[-1], guess))

        self._setup_variables(nodes)
        self._setup_objective(nodes)

    def callback(self, iteration: int) -> None:
        # self.sol_state_list.append(self.opti.debug.value(self.state))
        # self.sol_control_list.append(self.opti.debug.value(self.control))
        # self.final_times.append(self.opti.debug.value(self.time))

        # if isinstance(self, SaveProgressProtocol):
        #     self._save_progress(iteration, self.sol_state_list, self.sol_control_list, self.final_times)
        if self.logging == True:
            self.log(iteration)

    def setup_solver(self):
        """One-time solver initialization."""
        if getattr(self, 'solver_is_setup', False):
            return
        self.opti.solver('ipopt', self.solver_opts)
        self.opti.callback(lambda iteration: self.callback(iteration))
        self.solver_is_setup = True

    def solve(self, warm_start:Optional[ca.OptiSol] = None) -> ca.OptiSol:
        if not getattr(self, 'solver_is_setup', False):
            self.setup_solver()
        if warm_start:
            self.opti.set_initial(warm_start.value_variables())
        try:
            sol = self.opti.solve()
            success = True
        except RuntimeError as e:
            print("Solver failed:", e)
            sol = self.opti.debug
            success = False
        stats = self.opti.stats()
        print(self.extract_solver_metrics(stats, success))
        return sol
    
    def extract_solver_metrics(self, stats:dict, success:bool):
        metrics = {}

        # IPOPT-specific stats
        ipopt_stats = stats.get('iter_count', None)
        solver_stats = stats.get('solver_stats', {})

        if 't_wall_total' in solver_stats:
            metrics['total_time'] = solver_stats['t_wall_total']
        if 't_proc_total' in solver_stats:
            metrics['cpu_time'] = solver_stats['t_proc_total']
        if 'return_status' in solver_stats:
            metrics['status'] = solver_stats['return_status']
        if 'iterations' in solver_stats:
            metrics['iterations'] = solver_stats['iterations']
        if 'objective' in solver_stats:
            metrics['final_objective'] = solver_stats['objective']
        if 'dual_inf' in solver_stats:
            metrics['dual_inf'] = solver_stats['dual_inf']
        if 'primal_inf' in solver_stats:
            metrics['primal_inf'] = solver_stats['primal_inf']
        if 'kkt' in solver_stats:
            metrics['kkt_error'] = solver_stats['kkt']

        # Include success flag
        metrics['success'] = success

        return metrics
    
    def log(self, iteration:int) -> None:
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
        if hasattr(self, "_init_variable_time"):
            self.logger.info(f"Final time: {self.opti.debug.value(self.time)}")
        self.logger.info(f"Final position: {self.opti.debug.value(self.state)[:, -1]}")
        self.logger.info(f"Final velocity: {self.opti.debug.value(self.state)[3, -1]}")
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
            "times": sol.value(self.times) if hasattr(self, 'times') else None
        }


