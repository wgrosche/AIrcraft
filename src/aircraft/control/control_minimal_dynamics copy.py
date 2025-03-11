"""
Waypoint constraint formulation for moving horizon mpc:

While a waypoint is not reached all waypoints other than the next and the one after that aren't considered.

Minimising time to next waypoint while maintaining waypoint constraint on current waypoint.

Once waypoint constraint is satisfied soften constraint on new current waypoint and introduce hard constraint on new next waypoint.

Switching will be non-differentiable if naively implemented.

How to handle case where all or one of the next 2 waypoints are out of horizon?

Minimise distances instead of imposing final state constraint.

Formulating the moving horizon mpc:

Num control nodes = 10
Max dt (don't know if this should be flexible) = 0.25s (human reaction time for realistic control input)

waypoints = [...]
current_waypoint = waypoints[0]
next_waypoint = waypoints[1]

state = state_0

def check_waypoint_reached(state_list):
    check whether the waypoint condition is met for any state in the state list


while not final_waypoint_reached:
    if check_waypoint_reached(state_list):
        waypoint_index = i+1
        current_waypoint = next_waypoint
        next_waypoint = waypoints[i]
    
    opti problem with warm start?




"""




import casadi as ca
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
import os
import sys

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASEPATH)
sys.path.append(BASEPATH)

from aircraft.dynamics.dynamics import Aircraft, AircraftOpts
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from aircraft.utils.utils import TrajectoryConfiguration, load_model
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
import h5py
from scipy.interpolate import CubicSpline
from aircraft.plotting.plotting_minimal import TrajectoryPlotter, TrajectoryData

import threading
import torch

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)
from pathlib import Path

plt.ion()
default_solver_options = {'ipopt': {'max_iter': 10000,
                                    'tol': 1e-2,
                                    'acceptable_tol': 1e-2,
                                    'acceptable_obj_change_tol': 1e-2,
                                    'hessian_approximation': 'exact'
                                    },
                        'print_time': 10,
                        # 'expand': True

                        }


class ControlProblem:
    def __init__(
        self, 
        opti:ca.Opti, 
        aircraft:Aircraft,
        trajectory_config:TrajectoryConfiguration,
        num_control_nodes: int,
        VERBOSE:bool = True
        ):

        self.opti = opti
        self.aircraft = aircraft
        self.state_dim = aircraft.num_states
        self.control_dim = aircraft.num_controls
        dt_sym = ca.MX.sym('dt')
        self.dynamics = ca.Function('step', [aircraft.state, aircraft.control, dt_sym], [aircraft.state_step(aircraft.state, aircraft.control, dt_sym)]).expand()
        # aircraft.state_update.expand()

        

        self.num_nodes = num_control_nodes

        self.trajectory = trajectory_config
        self.VERBOSE = VERBOSE
        pass

    @dataclass
    class Node:
        index:int
        state:ca.MX
        control:ca.MX

    def control_constraint(self, node:Node):
        self.opti.subject_to(
            self.opti.bounded(
                -5, 
                node.control[0], 
                5
                )
            )
        self.opti.subject_to(
            self.opti.bounded(
                -5, 
                node.control[1], 
                5
                )
            )



    def state_constraint(self, node:Node, next:Node, dt:ca.MX):
        """
        Constraints on the state variables.

        node:Node - current node
        next:Node - next node
        dt:ca.MX - time step
        """
        self.opti.subject_to(
            self.opti.bounded(
                20, 
                self.aircraft.airspeed(node.state, node.control), 
                80
                )
            )

        self.opti.subject_to(
            self.opti.bounded(
                -np.deg2rad(10), 
                self.aircraft.beta(node.state, node.control), 
                np.deg2rad(10)
                )
            )

        self.opti.subject_to(
            self.opti.bounded(
                -np.deg2rad(20), 
                self.aircraft.alpha(node.state, node.control), 
                np.deg2rad(20)
                )
            )
        
        self.opti.subject_to(
            next.state == self.dynamics(node.state, node.control, dt)
            )

        self.opti.subject_to(
            next.state[2] < 0
            )



    def loss(
            self, 
            state:Optional[ca.MX] = None, 
            control:Optional[ca.MX] = None, 
            time:Optional[ca.MX] = None
            ):
        lambda_rate = 1.0
        rate_penalty = 0
        for i in range(self.num_nodes - 1):
            rate_penalty += ca.sumsqr(self.control[:, i+1] - self.control[:, i])

        final_state_diff = state[:3, -1] - self.trajectory.waypoints.final_position

        final_pos_loss = ca.dot(final_state_diff, final_state_diff)
         

        return time ** 2 + lambda_rate * rate_penalty + 10**5 * final_pos_loss

    def setup(self):
        opti = self.opti
        trajectory = self.trajectory

        scale_state = ca.vertcat(
                            [1e3, 1e3, 1e3],
                            [1e2, 1e2, 1e2],
                            [1, 1, 1, 1],
                            [1, 1, 1]
                            )
        scale_control = ca.vertcat(5, 5, 5)

        x_guess, time_guess = self.state_guess(trajectory)
        self.time = time_guess * opti.variable()
        opti.subject_to(self.time > 0)
        opti.set_initial(self.time, time_guess)

        self.dt = self.time / self.num_nodes
        
        current_node = self.Node(
            index=0,
            state=ca.DM(scale_state) * opti.variable(self.state_dim),
            control=ca.DM(scale_control) * opti.variable(self.control_dim),
        )

        # if waypoint_info.initial_state is not None:
        initial_state = self.trajectory.waypoints.initial_state
        opti.subject_to(current_node.state == initial_state)
        opti.set_initial(current_node.state, x_guess[:, 0])
        opti.set_initial(current_node.control, np.zeros(self.control_dim))

        # opti.subject_to(ca.dot(current_node.state[6:10], current_node.state[6:10]) == 1)

        self.state = [current_node.state]
        self.control = [current_node.control]
        

        for index in range(1, self.num_nodes + 1):
            next_node = self.Node(
                index=index,
                state = ca.DM(scale_state) * opti.variable(self.state_dim),
                control = ca.DM(scale_control) * opti.variable(self.control_dim),
            )
            
            self.state_constraint(current_node, next_node, self.dt)
            self.control_constraint(current_node)
            # self.waypoint_constraint(current_node, next_node)

            current_node = next_node

            opti.set_initial(current_node.state, x_guess[:, index])
            opti.set_initial(current_node.control, np.zeros(self.control_dim))

            self.state.append(current_node.state)
            # self.lam.append(current_node.lam)
            if index < self.num_nodes:
                self.control.append(current_node.control)
                # self.mu.append(current_node.mu)
                # self.nu.append(current_node.nu)
        
        # self.opti.subject_to(current_node.lam == [0] * self.num_waypoints)

        self.state = ca.hcat(self.state)
        print("State Shape: ", self.state.shape)
        self.control = ca.hcat(self.control)
        print("State Shape: ", self.state.shape)

        opti.minimize(self.loss(state = self.state, control = self.control, time = self.time))


    def solve(self, 
                opts:dict = default_solver_options,
                warm_start:Union[ca.OptiSol, ca.Opti] = (None, None),
                filepath:str = None
                ):
        
        self.sol_state_list = []
        self.sol_control_list = []
        self.final_times = []

        if filepath is not None:
            if os.path.exists(filepath):
                os.remove(filepath)
        # plt.ion()
        # plotter = TrajectoryPlotter(self.aircraft)
        # plt.show(block = False)
        # TODO: investigate fig.add_subfigure for better plotting
        self.opti.solver('ipopt', opts)
        # self.opti.callback(lambda i: self.callback(plotter, i, filepath))
        # plt.show()

        if warm_start != (None, None):
            warm_sol, warm_opti = warm_start
            self.opti.set_initial(warm_sol.value_variables())
            # lam_g0 = warm_sol.value(warm_opti.lam_g)
            # self.opti.set_initial(self.opti.lam_g, lam_g0)
        sol = self.opti.solve()
        # plt.ioff()
        # plt.show(block=True)        
        return (sol, self.opti)

    def state_guess(self, trajectory:TrajectoryConfiguration):
        """
        Initial guess for the state variables.
        """
        state_dim = self.aircraft.num_states
        aircraft = self.aircraft
        initial_state = trajectory.waypoints.initial_state
        
        x_guess = np.zeros((state_dim, self.num_nodes + 1))

        dyn = aircraft.state_update
        tf =  np.linalg.norm(trajectory.waypoints.final_position - trajectory.waypoints.initial_position) / trajectory.waypoints.default_velocity


        dt = tf / self.num_nodes

        state_list = np.zeros((aircraft.num_states, self.num_nodes + 1))
        state_list[:, 0] = initial_state
        state = ca.DM(initial_state)
        control_list = np.zeros((aircraft.num_controls, self.num_nodes))
        for i in range(self.num_nodes):
            print(i)
            state_list[:, i + 1] = state.full().flatten()
            random_control = [2 * np.random.random() - 1, 2 * np.random.random() - 1, 0]
            control_list[:, i] = random_control
            state = dyn(state, random_control, dt)
                        
            # t += 1

        
        
        return state_list, tf
    

def main():

    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
    model_path = Path(NETWORKPATH) / 'model-dynamics.pth'

    poly_path = Path(NETWORKPATH) / "fitted_models_casadi.pkl"

    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=5)

    # opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)
    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)

    # [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 
    aircraft.com = [0.0131991, -1.78875e-08, 0.00313384]


    opti = ca.Opti()

    num_control_nodes = 200
    # aircraft = Aircraft(traj_dict['aircraft'], model)#, LINEAR=True)
    problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    problem.setup()
    (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control.hdf5'))

    # save sol
    # import pickle

    # with(open('solution.pkl', 'wb')) as f:
    #     pickle.dump(sol, f)

    plotter = TrajectoryPlotter(aircraft)
    trajectory_data = TrajectoryData(
                state = np.array(sol.value(problem.state))[:, 1:],
                control = np.array(sol.value(problem.control)),
                time = np.array(sol.value(problem.time))
            )
            
    plotter.plot(trajectory_data = trajectory_data)

    plt.show(block = True)


    # _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    # problem.plot_convergence(ax, sol)
    
    # # sol_traj = sol.value(problem.state)
    # opti = ca.Opti()
    # aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=False)
    # problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

    # problem.setup()
    # # problem.opti.set_initial(problem.state, sol_traj)
    # (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control_nn.hdf5'), warm_start=(sol, opti))

    # _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
    # problem.plot_convergence(ax, sol)

    return sol

if __name__ == "__main__":
    main()