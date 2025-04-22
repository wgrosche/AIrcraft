import casadi as ca
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass


from aircraft.dynamics.aircraft import Aircraft
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from aircraft.utils.utils import TrajectoryConfiguration, load_model
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
from scipy.interpolate import CubicSpline
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData
from tqdm import tqdm
from typing import Type, List
import logging
import os
from datetime import datetime
from pathlib import Path
from aircraft.dynamics.aircraft import AircraftOpts
from aircraft.control.initialisation import cumulative_distances
from abc import abstractmethod
from aircraft.config import default_solver_options, BASEPATH, NETWORKPATH, DATAPATH, DEVICE, rng
from aircraft.control.base import ControlNode, ControlProblem

class ProgressTimeMixin:
    """
    A mixin that enables minimum-time trajectory optimization with per-node local time steps (dt_i).

    Attributes:
        v (List[ca.MX]): List of symbolic progress rate variables (1 / dt_i) for each node.
        _dt (List[ca.MX]): List of symbolic time step durations (dt_i = 1 / v_i).
    
    Methods:
        _init_progress_time(opti, num_nodes): Setup per-node progress rate variables.
        dt(i): Get symbolic dt_i for node i.
        total_time(): Symbolic expression for total time.
    """
    def _init_progress_time(self, opti: ca.Opti, num_nodes: int):
        self._use_progress_time = True
        self._num_nodes = num_nodes

        self.v = [opti.variable() for _ in range(num_nodes + 1)]  # progress rates (1 / dt_i)
        for i, v_i in enumerate(self.v):
            opti.set_initial(v_i, 1e2)
            self.constraint(v_i >= 1e-3, description=f"positive progress rate at node {i}")

        self._dt = [1.0 / v_i for v_i in self.v]  # per-node local time steps
        self.time = self.total_time

    def dt(self, i: int) -> ca.MX:
        return self._dt[i]

    @property
    def total_time(self) -> ca.MX:
        return ca.sumsqr(ca.vertcat(*[1.0 / v_i for v_i in self.v]))

    def progress_state_constraint(self, node: ControlNode, next: ControlNode, i: int) -> None:
        """
        Override this in your ControlProblem if using progress-based time. Assumes self.x_dot is defined.
        """
        if hasattr(self, 'x_dot'):
            print("Running with progress")
            dt_i = self.dt(i)
            self.constraint(
                next.state - node.state - dt_i * self.x_dot(next.state, node.control) == 0,
                description=f"implicit state dynamics constraint at node {node.index}"
            )
            self.constraint(ca.sumsqr(node.state[6:10]) == 1, description=f"quaternion norm constraint at node {node.index}")
        else:
            raise NotImplementedError("ProgressTimeMixin assumes implicit dynamics with x_dot defined.")

# @dataclass
# class TimeNode(ControlNode):
#     time:Optional[ca.MX] = None

# class ProgressTimeMixin:
#     """
#     A mixin that enables sparse minimum-time optimization by introducing per-segment
#     progress rates v_i = 1/dt_i as decision variables.

#     Attributes:
#         progress_rates (list of ca.MX): Per-segment progress rate variables (v_i).
#         dt_list (list of ca.MX): Computed local time steps (1/v_i).

#     Methods:
#         _init_progress_time(opti, num_segments): Initialize symbolic variables.
#         local_dt(i): Get the dt for segment i.
#         total_time(): Sum of all 1/v_i for objective.
#     """

#     def _init_progress_time(self, opti: ca.Opti, num_segments: int):
#         self._num_segments = num_segments
#         self.progress_rates = [opti.variable() for _ in range(num_segments)]

#         for i, v in enumerate(self.progress_rates):
#             opti.set_initial(v, 1.0)  # Initial guess (uniform progress rate)
#             opti.subject_to(v >= 1e-3)  # Avoid division by zero

#         # Cache dt = 1/v for convenience
#         self.dt_list = [1.0 / v for v in self.progress_rates]

#     def local_dt(self, i):
#         """Return the local time step for segment i."""
#         return self.dt_list[i]

#     def total_time(self):
#         """Return the total time (to be minimized)."""
#         return sum(self.dt_list)

#     def progress_state_constraint(self, node, next_node, i):
#         """
#         Add the dynamics constraint using progress-based local dt.
#         Override this in your controller.
#         """
#         dt_i = self.local_dt(i)

#         if hasattr(self, 'x_dot'):
#             dyn = next_node.state - node.state - dt_i * self.x_dot(next_node.state, node.control)
#         else:
#             dyn = next_node.state - self.dynamics(node.state, node.control, dt_i)

#         self.constraint(dyn == 0, description=f"progress-based dynamics at node {i}")

#         # If needed, quaternion norm constraint
#         if hasattr(self, 'quaternion_index'):
#             q = node.state[self.quaternion_index]
#             self.constraint(ca.sumsqr(q) == 1, description=f"quaternion norm at node {i}")


# # class VariableTimeMixin(ControlProblem):
# #     """
# #     A mixin for variable time per node, preserving sparsity.

# #     Attributes:
# #         dt_vars (List[ca.MX]): Per-node time steps (dt_i).
# #         total_time (ca.MX): Optional total time for minimization or constraints.
# #     """

# #     def __init__(self, *, opti:ca.Opti, dt_initial:float = 0.01, **kwargs):
# #         super().__init__(**kwargs)
# #         self.dt_initial = 0.01

# #     def _setup_step(self, index, current_node, guess):
# #         next_node = TimeNode(super()._setup_step(index, current_node, guess))
# #         next_node.time = self.opti.variable()
# #         self.opti.set_initial(next_node.time, self.dt_initial)

# #         return 
    
# #     _init_variable_time(self, opti: ca.Opti, num_nodes: int, timescale: float = 1.0):
# #         self._variable_time_enabled = True
# #         self._num_nodes = num_nodes
# #         self._timescale = timescale
# #         self.dt = None

# #         # Individual time step variables per segment (node to next)
# #         self.dt_vars = [opti.variable() for _ in range(num_nodes - 1)]
# #         for i, dt in enumerate(self.dt_vars):
# #             self.constraint(dt >= 1e-3, description=f"positive time step constraint for segment {i}")

# #         # Optional: total time as sum of dt_i
# #         self.total_time = ca.sumsqr(ca.vertcat(*self.dt_vars))  # or ca.sum1 if you want total duration
# #         self.time = self.total_time
# #         # Optional: constraint or objective on total_time

# #     def get_dt(self, index: int) -> ca.MX:
# #         """Returns symbolic dt for node index (from node to next)."""
# #         return self.dt_vars[index]