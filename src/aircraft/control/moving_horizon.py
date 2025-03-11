# """
# Waypoint constraint formulation for moving horizon mpc:

# While a waypoint is not reached all waypoints other than the next and the one after that aren't considered.

# Minimising time to next waypoint while maintaining waypoint constraint on current waypoint.

# Once waypoint constraint is satisfied soften constraint on new current waypoint and introduce hard constraint on new next waypoint.

# Switching will be non-differentiable if naively implemented.

# How to handle case where all or one of the next 2 waypoints are out of horizon?

# Minimise distances instead of imposing final state constraint.

# Formulating the moving horizon mpc:

# Num control nodes = 10
# Max dt (don't know if this should be flexible) = 0.25s (human reaction time for realistic control input)

# waypoints = [...]
# current_waypoint = waypoints[0]
# next_waypoint = waypoints[1]

# state = state_0

# def check_waypoint_reached(state_list):
#     check whether the waypoint condition is met for any state in the state list


# while not final_waypoint_reached:
#     if check_waypoint_reached(state_list):
#         waypoint_index = i+1
#         current_waypoint = next_waypoint
#         next_waypoint = waypoints[i]
    
#     opti problem with warm start?




# """




# import casadi as ca
# import numpy as np
# from typing import List, Optional, Union
# from dataclasses import dataclass
# import os
# import sys

# BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASEPATH)
# sys.path.append(BASEPATH)

# from src.dynamics import Aircraft
# from collections import namedtuple
# from scipy.spatial.transform import Rotation as R
# from src.utils import TrajectoryConfiguration, load_model
# from matplotlib.pyplot import spy
# import json
# import matplotlib.pyplot as plt
# from liecasadi import Quaternion
# import h5py
# from scipy.interpolate import CubicSpline

# import torch

# BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
# NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
# DATAPATH = os.path.join(BASEPATH, 'data')
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
#                       ("mps" if torch.backends.mps.is_available() else "cpu"))
# sys.path.append(BASEPATH)
# from pathlib import Path
# from src.dynamics import AircraftOpts

# default_solver_options = {'ipopt': {'max_iter': 10000,
#                                     'tol': 1e-2,
#                                     'acceptable_tol': 1e-2,
#                                     'acceptable_obj_change_tol': 1e-2,
#                                     'hessian_approximation': 'limited-memory'
#                                     },
#                         'print_time': 10,
#                         # 'expand' : True
#                         }

# def cumulative_distances(waypoints:np.ndarray, VERBOSE:bool = False):
#     """
#     Given a set of waypoints, calculate the distance between each waypoint.

#     Parameters
#     ----------
#     waypoints : np.array
#         Array of waypoints. (d x n) where d is the dimension of the waypoints 
#         and n is the number of waypoints. The first waypoint is taken as the 
#         initial position.

#     Returns
#     -------
#     distance : np.array
#         Cumulative distance between waypoints.
    
#     """
#     differences = np.diff(waypoints, axis=0)
#     distances = np.linalg.norm(differences, axis=1)
#     distance = np.cumsum(distances)

#     if VERBOSE: 
#         print("Cumulative waypoint distances: ", distance)
#     return distance

# class ControlProblem:
#     def __init__(
#         self, 
#         opti:ca.Opti, 
#         aircraft:Aircraft,
#         trajectory_config:TrajectoryConfiguration,
#         num_control_nodes: int,
#         VERBOSE:bool = True

#         ):
#         self.opti = opti
#         self.aircraft = aircraft
#         self.state_dim = aircraft.num_states
#         self.control_dim = aircraft.num_controls

#         self.dynamics = aircraft.state_update

        

#         self.nodes = num_control_nodes
#         self.waypoints = trajectory_config.waypoints()
#         self.trajectory = trajectory_config
#         self.num_waypoints = self.waypoints.shape[0] - 1
#         self.distances = cumulative_distances(self.waypoints)
#         self.switch_var = np.array(
#             self.nodes * np.array(self.distances) 
#             / self.distances[-1], dtype = int
#             )
        
#         self.VERBOSE = VERBOSE
#         pass

#     @dataclass
#     class Node:
#         index:int
#         state:ca.MX
#         state_next:ca.MX
#         control:ca.MX
#         lam:ca.MX
#         lam_next:ca.MX
#         mu:ca.MX
#         nu:ca.MX

#     def setup_opti_vars(self, 
#                         scale_state = ca.vertcat(
#                             [1, 1, 1, 1],
#                             [1e3, 1e3, 1e3],
#                             [1e2, 1e2, 1e2],
#                             [1, 1, 1]
#                             ), 
#                         scale_control = ca.vertcat(
#                             5, 5, 5,
#                             [1e2, 1e2, 1e2],
#                             [1, 1, 1],
#                             [1e2, 1e2, 1e2]
#                             ), 
#                         scale_time = 1,
#                         ):
        
#         opti = self.opti
#         self.time = scale_time * opti.variable()
#         self.dt = self.time / self.nodes

        
#         state_list = []
#         control_list = []
#         lam_list = []
#         mu_list = []
#         nu_list = []

#         for i in range(self.nodes + 1):

#             state_list.append(ca.DM(scale_state) * 
#                               opti.variable(self.state_dim))
#             lam_list.append(opti.variable(self.num_waypoints))

#             if i < self.nodes:
#                 control_list.append(ca.DM(scale_control) *          
#                             opti.variable(self.control_dim))
#                 mu_list.append(opti.variable(self.num_waypoints))
#                 nu_list.append(opti.variable(self.num_waypoints))
                

#         self.state = ca.hcat(state_list)
#         self.control = ca.hcat(control_list)
#         self.lam = ca.hcat(lam_list)
#         self.mu = ca.hcat(mu_list)
#         self.nu = ca.hcat(nu_list)

#     def control_constraint(self, node:Node, fix_com:bool = True):
#         control_envelope = self.trajectory.control
#         opti = self.opti
#         com = self.trajectory.aircraft.aero_centre_offset

#         opti.subject_to(opti.bounded(control_envelope.lb[:6],
#                 node.control[:6], control_envelope.ub[:6]))
        
#         opti.subject_to(opti.bounded(np.zeros(node.control[9:].shape),
#                 node.control[9:], np.zeros(node.control[9:].shape)))
        
#         if fix_com:
#             opti.subject_to(node.control[6:9]==com)



#     def state_constraint(self, node:Node, dt:ca.MX):
        
#         state_envelope = self.trajectory.state
#         opti = self.opti
#         dynamics = self.dynamics

#         alpha = self.aircraft.alpha
#         beta = self.aircraft.beta
#         airspeed = self.aircraft.airspeed

#         opti.subject_to(opti.bounded(state_envelope.alpha.lb,
#             alpha(node.state, node.control), state_envelope.alpha.ub))

#         opti.subject_to(opti.bounded(state_envelope.beta.lb,
#             beta(node.state, node.control), state_envelope.beta.ub))

#         opti.subject_to(opti.bounded(state_envelope.airspeed.lb,
#             airspeed(node.state, node.control), state_envelope.airspeed.ub))
        
#         opti.subject_to(node.state_next == dynamics(node.state, node.control, dt))


#     def waypoint_constraint(self, node:Node):#, waypoint_node:int):
#         """
#         Waypoint constraint implementation from:
#         https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf
#         """
#         tolerance = self.trajectory.waypoints.tolerance
#         waypoint_indices = np.array(self.trajectory.waypoints.waypoint_indices)
#         num_waypoints = self.num_waypoints
#         waypoints = self.waypoints[1:, waypoint_indices]
#         opti = self.opti
        
#         # if node.index > self.switch_var[waypoint_node]:
#         #     waypoint_node += 1
        
#         for j in range(num_waypoints):
#             opti.subject_to(node.lam_next[j] - node.lam[j] + node.mu[j] == 0)
#             opti.subject_to(node.mu[j] >= 0)
#             if j < num_waypoints - 1:
#                 opti.subject_to(node.lam[j] - node.lam[j + 1] <= 0)

#             diff = node.state[4 + waypoint_indices] - waypoints[j, waypoint_indices]
#             opti.subject_to(opti.bounded(0, node.nu[j], tolerance**2))
#             opti.subject_to(node.mu[j] * (ca.dot(diff, diff) - node.nu[j]) == 0)

#         return None #waypoint_node

#     def loss(self, state:Optional[ca.MX] = None, control:Optional[ca.MX] = None, 
#              time:Optional[ca.MX] = None):
#         return time ** 2
    
#     def mhe_loss(self, state:Optional[ca.MX] = None, control:Optional[ca.MX] = None, 
#              time:Optional[ca.MX] = None):
#         return None
    
#     def setup_horizon(self):
#         opti = self.opti
#         trajectory = self.trajectory


#     def setup(self):
#         opti = self.opti
#         trajectory = self.trajectory

#         _, time_guess = self.state_guess(trajectory)
#         self.setup_opti_vars(scale_time=1/time_guess)
#         nodes = self.nodes
#         time = self.time
#         state = self.state
#         dt = self.dt
#         control = self.control
#         lam = self.lam
#         mu = self.mu
#         nu = self.nu

#         waypoint_info = trajectory.waypoints
#         num_waypoints = self.num_waypoints
#         waypoints = waypoint_info.waypoints
#         waypoint_indices = np.array(waypoint_info.waypoint_indices)
#         final_waypoint = waypoint_info.final_position[waypoint_indices]

        
#         opti.subject_to(time > 0)

#         if waypoint_info.initial_state is not None:
#             initial_state = waypoint_info.initial_state
#             opti.subject_to(state[4:, 0] == initial_state[4:])

#         opti.subject_to(ca.dot(state[:4, 0], state[:4, 0]) == 1)

#         opti.subject_to(lam[:, 0] == [1] * num_waypoints)

#         # waypoint_node = 0
#         for index in range(nodes):

#             node_data = self.Node(
#                 index=index,
#                 state_next = state[:, index + 1],
#                 state=state[:, index],
#                 control = control[:, index],
#                 lam=lam[:, index],
#                 lam_next=lam[:, index + 1],
#                 mu=mu[:, index],
#                 nu=nu[:, index]
#             )
                
#             self.state_constraint(node_data, dt)
            
#             self.control_constraint(node_data)
#             self.waypoint_constraint(node_data)#, waypoint_node)
#             # waypoint_node = self.waypoint_constraint(node_data, waypoint_node)
        
#         if self.VERBOSE:
#             print("Initial State: ", initial_state)
#             print("Waypoints: ", waypoints)
#             print("Waypoint Indices: ", waypoint_indices)
#             print("Final Waypoint: ", final_waypoint)
#             print("Predicted Switching Nodes: ", self.switch_var)

#         self.opti.subject_to(
#             self.state[4 + waypoint_indices, -1] ==  final_waypoint)
        
#         self.opti.subject_to(self.mu[:, -1] == [0] * self.num_waypoints)

#         self.initialise()

#         opti.minimize(self.loss(state = state, control = control, time = time))

#         if self.VERBOSE:
#             constraints = opti.g
#             print(f"Constraint 545: {constraints[576]}")

#     def plot_sparsity(self, ax:plt.axes):
#         jacobian = self.opti.debug.value(
#             ca.jacobian(self.opti.g, self.opti.x)
#             ).toarray()
        
#         ax.clear()
#         ax.spy(jacobian)
#         plt.draw()
#         plt.pause(0.01)

#     def plot_trajectory(self, ax:plt.axes):
#         state = self.opti.debug.value(self.state)
#         ax.clear()
#         ax.plot(state[4, :], state[5, :], state[6, :])
#         plt.draw()
#         plt.pause(0.01)

#     def save_progress(self, filepath, iteration):
#         if filepath is not None:
#             # save the state, control and time to a file
#             with h5py.File(filepath, "a") as h5file:
#                 for i, (X, U, time, lam, mu, nu) in enumerate(zip(self.sol_state_list[-10:], self.sol_control_list[-10:], self.final_times[-10:], self.lam_list[-10:], self.mu_list[-10:], self.nu_list[-10:])):
#                     grp = h5file.create_group(f'iteration_{iteration - 10 + i}')
#                     grp.create_dataset('state', data=X)
#                     grp.create_dataset('control', data=U)
#                     grp.create_dataset('time', data=time)
#                     grp.create_dataset('lam', data=lam)
#                     grp.create_dataset('mu', data=mu)
#                     grp.create_dataset('nu', data=nu)

#     def plot_convergence(self, ax:plt.axes, sol:ca.OptiSol):
#         ax.semilogy(sol.stats()['iterations']['inf_du'], label="Dual infeasibility")
#         ax.semilogy(sol.stats()['iterations']['inf_pr'], label="Primal infeasibility")

#         ax.set_xlabel('Iterations')
#         ax.set_ylabel('Infeasibility (log scale)')
#         ax.grid(True)
#         ax.legend()

#         plt.tight_layout()
#         plt.show(block = True)

#     def callback(self, axs:List[plt.axes], iteration:int, filepath:str):
#         if iteration % 10 == 5:
#             self.plot_sparsity(axs[0])
#             self.plot_trajectory(axs[1])

#         if filepath is not None:
#             self.sol_state_list.append(self.opti.debug.value(self.state))
#             self.sol_control_list.append(self.opti.debug.value(self.control))
#             self.final_times.append(self.opti.debug.value(self.time))
#             self.lam_list.append(self.opti.debug.value(self.lam))
#             self.mu_list.append(self.opti.debug.value(self.mu))
#             self.nu_list.append(self.opti.debug.value(self.nu))
#             if iteration % 10 == 0:
#                 self.save_progress(filepath, iteration)

#     def solve(self, 
#                 opts:dict = default_solver_options,
#                 warm_start:Union[ca.OptiSol, ca.Opti] = (None, None),
#                 filepath:str = None
#                 ):
        
#         self.sol_state_list = []
#         self.sol_control_list = []
#         self.final_times = []
#         self.lam_list = []
#         self.mu_list = []
#         self.nu_list = []

#         if filepath is not None:
#             if os.path.exists(filepath):
#                 os.remove(filepath)

#         fig = plt.figure(figsize=(10, 10))
#         ax = fig.add_subplot(211)
#         ax2 = fig.add_subplot(212, projection = '3d')
#         # TODO: investigate fig.add_subfigure for better plotting
#         self.opti.solver('ipopt', opts)
#         self.opti.callback(lambda i: self.callback([ax, ax2], i, filepath))
#         plt.show()

#         if warm_start != (None, None):
#             warm_sol, warm_opti = warm_start
#             self.opti.set_initial(warm_sol.value_variables())
#             # lam_g0 = warm_sol.value(warm_opti.lam_g)
#             # self.opti.set_initial(self.opti.lam_g, lam_g0)
#         sol = self.opti.solve()
        
#         return (sol, self.opti)



#     def initialise(self):
        
#         (lambda_guess, mu_guess, nu_guess) = self.waypoint_variable_guess()

#         x_guess, time_guess = self.state_guess(self.trajectory)

#         control_guess = np.zeros(self.control.shape)

#         control_guess[6:9, :] = np.repeat([self.trajectory.aircraft.aero_centre_offset], 
#                                           self.control.shape[1], axis = 0).T

#         if self.VERBOSE:
#             print("State Trajectory Guess: ", x_guess)


#         self.opti.set_initial(self.nu, nu_guess)
#         self.opti.set_initial(self.lam, lambda_guess)
#         self.opti.set_initial(self.mu, mu_guess)
#         self.opti.set_initial(self.state, x_guess)
#         self.opti.set_initial(self.time, time_guess)
#         self.opti.set_initial(self.control, control_guess)
    

#     def waypoint_variable_guess(self):

#         num_waypoints = self.num_waypoints

#         lambda_guess = np.zeros((num_waypoints, self.nodes + 1))
#         mu_guess = np.zeros((num_waypoints, self.nodes))
#         nu_guess = np.zeros((num_waypoints, self.nodes))

#         i_wp = 0
#         for i in range(1, self.nodes):
#             if i > self.switch_var[i_wp]:
#                 i_wp += 1

#             if ((i_wp == 0) and (i + 1 >= self.switch_var[0])) or i + 1 - self.switch_var[i_wp-1] >= self.switch_var[i_wp]:
#                 mu_guess[i_wp, i] = 1

#             for j in range(num_waypoints):
#                 if i + 1 >= self.switch_var[j]:
#                     lambda_guess[j, i] = 1

#         return (lambda_guess, mu_guess, nu_guess)
    
#     def smooth_trajectory(self, x_guess):
#             # Extract the points along the trajectory
#             x_vals = x_guess[4, :]  # x-coordinates
#             y_vals = x_guess[5, :]  # y-coordinates
#             z_vals = x_guess[6, :]  # z-coordinates

#             # Create a parameter t for the trajectory points
#             t = np.linspace(0, 1, len(x_vals))

#             # Fit cubic splines to the trajectory points
#             spline_x = CubicSpline(t, x_vals)
#             spline_y = CubicSpline(t, y_vals)
#             spline_z = CubicSpline(t, z_vals)

#             # Evaluate the splines at finer intervals for a smoother trajectory
#             t_fine = t#np.linspace(0, 1, len(x_vals) * 10)  # Increase resolution by 10x
#             x_smooth = spline_x(t_fine)
#             y_smooth = spline_y(t_fine)
#             z_smooth = spline_z(t_fine)

#             # Update x_guess with the smoothed values (optional, for visualization)
#             x_guess[4, :] = np.interp(np.linspace(0, len(x_vals)-1, len(x_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), x_smooth)
#             x_guess[5, :] = np.interp(np.linspace(0, len(y_vals)-1, len(y_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), y_smooth)
#             x_guess[6, :] = np.interp(np.linspace(0, len(z_vals)-1, len(z_vals)), np.linspace(0, len(t_fine)-1, len(t_fine)), z_smooth)
#             return x_guess


#     def state_guess(self, trajectory:TrajectoryConfiguration):
#         """
#         Initial guess for the state variables.
#         """
        

#         state_dim = self.aircraft.num_states
#         initial_pos = trajectory.waypoints.initial_position
#         velocity_guess = trajectory.waypoints.default_velocity
#         waypoints = self.waypoints[1:, :]
        
#         x_guess = np.zeros((state_dim, self.nodes + 1))
#         distance = self.distances
    
#         self.r_glide = 10
        

#         direction_guess = (waypoints[0, :] - initial_pos)
#         vel_guess = velocity_guess *  direction_guess / np.linalg.norm(direction_guess)

#         if self.VERBOSE:
#             print("Cumulative Waypoint Distances: ", distance)
#             print("Predicted Switching Nodes: ", self.switch_var)
#             print("Direction Guess: ", direction_guess)
#             print("Velocity Guess: ", vel_guess)
#             print("Initial Position: ", initial_pos)
#             print("Waypoints: ", waypoints)

#         x_guess[:3, 0] = initial_pos
#         x_guess[3:6, 0] = vel_guess

#         z_flip = R.from_euler('x', 180, degrees=True)

#         for i, waypoint in enumerate(waypoints):
#             if len(self.trajectory.waypoints.waypoint_indices) < 3:
#                     waypoint[2] += self.distances[i] / self.r_glide
#         i_wp = 0
#         for i in range(self.nodes):
#             # switch condition
#             if i > self.switch_var[i_wp]:
#                 i_wp += 1
                
#             if i_wp == 0:
#                 wp_last = initial_pos
#             else:
#                 wp_last = waypoints[i_wp-1, :]
#             wp_next = waypoints[i_wp, :]

#             if i_wp > 0:
#                 interpolation = (i - self.switch_var[i_wp-1]) / (self.switch_var[i_wp] - self.switch_var[i_wp-1])
#             else:
#                 interpolation = i / self.switch_var[0]

            

#             # extend position guess
#             pos_guess = (1 - interpolation) * wp_last + interpolation * wp_next

#             x_guess[4:7, i + 1] = np.reshape(pos_guess, (3,))
            

#             direction = (wp_next - wp_last) / ca.norm_2(wp_next - wp_last)
#             vel_guess = velocity_guess * direction
#             x_guess[7:10, i + 1] = np.reshape(velocity_guess * direction, (3,))

#             rotation, _ = R.align_vectors(np.array(direction).reshape(1, -1), [[1, 0, 0]])

#             # Check if the aircraft is moving in the opposite direction
#             if np.dot(direction.T, [1, 0, 0]) < 0:
#                 flip_y = R.from_euler('y', 180, degrees=True)
#                 rotation = rotation * flip_y

#             x_guess[:4, i + 1] = (rotation * z_flip).as_quat()

#         # x_guess = self.smooth_trajectory(x_guess)

#         time_guess = distance[-1] / velocity_guess
#         if self.VERBOSE:
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection = '3d')
#             ax.plot(x_guess[4, :], x_guess[5, :], x_guess[6, :])
#             plt.show(block = True)
        
        
#         return x_guess, time_guess
    

# def main():

#     model = load_model()
#     traj_dict = json.load(open('data/glider/problem_definition.json'))

#     trajectory_config = TrajectoryConfiguration(traj_dict)

#     aircraft_config = trajectory_config.aircraft

#     linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
#     model_path = Path(NETWORKPATH) / 'model-dynamics.pth'

#     opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)

#     aircraft = Aircraft(opts = opts)


#     opti = ca.Opti()

#     num_control_nodes = 40
#     # aircraft = Aircraft(traj_dict['aircraft'], model)#, LINEAR=True)
#     problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

#     problem.setup()
#     (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control.hdf5'))

#     _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
#     problem.plot_convergence(ax, sol)
    
#     # sol_traj = sol.value(problem.state)
#     opti = ca.Opti()
#     aircraft = Aircraft(traj_dict['aircraft'], model, LINEAR=False)
#     problem = ControlProblem(opti, aircraft, trajectory_config, num_control_nodes)

#     problem.setup()
#     # problem.opti.set_initial(problem.state, sol_traj)
#     (sol, opti) = problem.solve(filepath= os.path.join(BASEPATH, 'data', 'trajectories', 'traj_control_nn.hdf5'), warm_start=(sol, opti))

#     _, ax = plt.subplots(1, 1)  # 1 row, 1 column of subplots
#     problem.plot_convergence(ax, sol)

#     return sol

# if __name__ == "__main__":
#     main()



# MHE implementation using casadi 
# identification of damping coefficient b of mass spring damper system
# 21.03.2023 - Jonas Gentner

from scipy import linalg
from matplotlib import pyplot as plt
from casadi.tools import struct_symSX,struct_SX,entry
from casadi import DM,horzcat,Function,mtimes,vertcat,vec,fabs,nlpsol
import numpy as np
import random as rdn
#Make random numbers predictable
np.random.seed(0)
rdn.seed(3)

#N: horizon length
#dt: step width
#Nsimulation: Simulation Time
#R: initial R
#Q: initial Q
#sigma_x0: initial state vector

def init_MHE(N,dt,Nsimulation,R,Q,sigma_x0,initial_values,params):
    #%% initialize MHE
    m = params[0]
    k = params[1]
    #b = params[2]
    # the states
    states = struct_symSX(["dx1","dx2","b"]) # state vector
    Nstates = states.size # Number of states
    # Set up some aliases
    dx1,dx2,b = states[...]
    
    # the control inputs
    controls = struct_symSX(["u"]) # control vector
    Ncontrols = controls.size # Number of control inputs
    # Set up some aliases
    u = controls[...]
    
    # disturbances
    disturbances = struct_symSX(["w","v","z"]) # Process noise vector
    Ndisturbances = disturbances.size # Number of disturbances
    # Set up some aliases
    w,v,z = disturbances[...]
    
    # measurements
    measurements = struct_symSX(["dx1"]) # Measurement vector
    Nmeas = measurements.size # Number of measurements
    # Set up some aliases
    cu = measurements[...]
    
    # create Structure for the entire horizon
    # Structure that will be degrees of freedom for the optimizer
    shooting = struct_symSX([(entry("X",repeat=N,struct=states),entry("W",repeat=N-1,struct=disturbances))])
    # Structure that will be fixed parameters for the optimizer
    parameters = struct_symSX([(entry("U",repeat=N-1,struct=controls),entry("Y",repeat=N,struct=measurements),entry("S",shape=(Nstates,Nstates)),entry("x0",shape=(Nstates,1)))])
    S = parameters["S"]
    x0 = parameters["x0"]
    # define the ODE right hand side
    rhs = struct_SX(states)
    rhs["dx1"] = dx2
    rhs["dx2"] = -k/m*dx1-b/m*dx2
    rhs["b"] = 0

    f = Function('f', [states,controls,disturbances],[rhs])
    
    # build an integrator for this system: Runge Kutta 4 integrator
    k1 = f(states,controls,disturbances)
    k2 = f(states+dt/2.0*k1,controls,disturbances)
    k3 = f(states+dt/2.0*k2,controls,disturbances)
    k4 = f(states+dt*k3,controls,disturbances)
    
    states_1 = states+dt/6.0*(k1+2*k2+2*k3+k4)
    phi = Function('phi', [states,controls,disturbances],[states_1])
    PHI = phi.jacobian_old(0, 0)
    
    measure = struct_SX(measurements)
    measure["dx1"] = dx1

    
    # define the measurement system
    h = Function('h', [states],[measure])      #Kupfertemperatur wird gemessen (hier nur define)
    H = h.jacobian_old(0, 0)
    
    # create a holder for the estimated states and disturbances
    estimated_X= DM.zeros(Nstates,Nsimulation)
    estimated_W = DM.zeros(Ndisturbances,Nsimulation-1)
    
    # build the objective
    obj = 0
    # first the arrival cost
    obj += mtimes([(shooting["X",0]-parameters["x0"]).T,S,(shooting["X",0]-parameters["x0"])])
    #next the cost for the measurement noise
    for i in range(N):
      vm = h(shooting["X",i])-parameters["Y",i]
      obj += mtimes([vm.T,R,vm])
    #and also the cost for the process noise
    for i in range(N-1):
      obj += mtimes([shooting["W",i].T,Q,shooting["W",i]])
    
    # build the multiple shooting constraints
    g = []
    for i in range(N-1):
      g.append( shooting["X",i+1] - phi(shooting["X",i],parameters["U",i],shooting["W",i]) ) #prädizierte Zustände (phi(...)) müssen gleich X(i+1)  sein 
    
    # formulate the NLP
    nlp = {'x':shooting, 'p':parameters, 'f':obj, 'g':vertcat(*g)}
    
    #build the state constraints
    lbw = []
    ubw = []
    for i in range(N):      #how many?
        #    dx1  dx2  b    u   w  z  
        lbw+=[-1, -1,  0,   0,  0, 0]
        ubw+=[ 1,  1,  5,   5,  5, 5]
    
    for i in range(Nstates): #delete the last Nstates elements from constraint list
        lbw.pop(len(lbw)-1) 
        ubw.pop(len(ubw)-1)
        
    #the initial estimate and related covariance, which will be used for the arrival cost
    P = sigma_x0**2*DM.eye(Nstates)
    x0 = DM(initial_values) + sigma_x0*np.random.randn(Nstates,1)
    
    # create the solver
    opts = {"ipopt.print_level":0, "print_time":False, 'ipopt.max_iter':1000}
    #nlpsol = nlpsol("nlpsol", "ipopt", nlp, opts)
    
    ret =  {}
    for elem, name in [(shooting, 'shooting'),
                       (parameters, 'parameters'),
                       (f, 'f'),
                       (phi, 'phi'),
                       (PHI, 'PHI'),
                       (estimated_X, 'estimated_X'),
                       (estimated_W, 'estimated_W'),
                       (lbw, 'lbw'),
                       (ubw, 'ubw'),
                       (P, 'P'),
                       (x0, 'x0'),
                       (nlpsol("nlpsol", "ipopt", nlp, opts), 'nlpsol'),
                       (Nstates, 'Nstates'),
                       (h, 'h'),
                       (H, 'H'),
                       (Ndisturbances, 'Ndisturbances')]:
        ret[name]=elem
    return ret

def mass_spring_damper(x,t):
    m = 7.5 # mass
    k = 50 # spring coefficient
    b = 2.5 # damping coefficient
    dx1 = x[1]
    dx2 = -k/m*x[0]-b/m*x[1]
    return np.array([dx1,dx2])


def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    print(y0)
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y


# Settings of the filter
N = 10 # Horizon length MHE
dt = 0.1; # Time step
SimulationTime = 10 #sekunden

sigma_p = 0.005 # Standard deviation of the measurements
sigma_w = 1     # Standard deviation for the process noise
sigma_x0 = 1  # Standard deviation for the initial values


r = [40000]
R = np.diag(r)          #measurement noise matrix

sensor_noise = False

q = [0.001,0.001,0.001]
Q = np.diag(q)          #process noise matrix

Nsimulation = int(SimulationTime/dt) # Length of simulation
t = np.linspace(0,(Nsimulation-1)*dt,Nsimulation) # Time grid

initial_values = [0.3,0,1]
params = [7.5,50]

#%%init MHE
MHE = init_MHE(N,dt,Nsimulation,R,Q,sigma_x0,initial_values,params)
shooting = MHE['shooting']
parameters = MHE['parameters']
phi = MHE['phi']
PHI = MHE['PHI']
estimated_X = MHE['estimated_X']
estimated_W = MHE['estimated_W']
lbw = MHE['lbw']
ubw = MHE['ubw']
P = MHE['P']
x0 = MHE['x0']
nlpsol = MHE['nlpsol']
Nstates = MHE['Nstates']
h = MHE['h']
H = MHE['H']
Ndisturbances = MHE['Ndisturbances'] 

start_values = [0.3,0]
meas = [0]      #vector for measurement of dx1
meas2 = [0]     #vector for measurement of dx2 -> not used in MHE

dx1_res = [0]
dx2_res = [0]
b_res = [0]

for i in range(1,Nsimulation):
    print('Durchgang: ',i,'/',Nsimulation)
    # 1. simulate
    dt_rk = [0,0+dt]
    start_values = rungekutta4(mass_spring_damper,start_values,dt_rk)[1]
    print(start_values)
    # 2. measurement

    if sensor_noise:
        noise = rdn.uniform(0.01,-0.01)
        val = start_values[0] + noise

        meas.append(val)    #measurement of dx1
        meas2.append(start_values[1])   #measurement of dx2 -> not used in MHE
    else:
        meas.append(start_values[0])
        meas2.append(start_values[1])

    # 3. MHE
    if(i>=(N-1)):
        if(i==(N-1)):
            # for the first instance we run the filter, we need to initialize it.
            current_parameters = parameters(0)
            current_parameters["U",lambda x: horzcat(*x)] =  DM([l*2 for l in meas[0:N-1]])
            current_parameters["Y",lambda theta: horzcat(*theta)] = DM([meas]) #hier ersten gemessenen Horizont (erstes Messfenster)
            current_parameters["S"] = linalg.inv(P) # arrival cost is the inverse of the initial covariance
            current_parameters["x0"] = x0
            initialisation_state = shooting(0)
            Paramlist = [initial_values[1] for i in range(N)]
            Paramlist2 = [initial_values[2] for i in range(N)]
            initialisation_state["X",lambda x: horzcat(*x)] = DM([meas,Paramlist,Paramlist2]) #hier der erste simulierte Horizont unbekannte Zustände einfach als Startwert 
            res = nlpsol(p=current_parameters, x0=initialisation_state, lbg=0, ubg=0)
            
            # Get the solution
            solution = shooting(res["x"])
  
            #calculation of measurement cost
            b = 0
            for k in range(N):
                a = h(solution["X",k])-current_parameters["Y",k]
                b += (mtimes([a.T,R,a])).full()[0][0]
            #measurement_cost.append(b)
            
            #calculation of process cost
            for k in range(N-1):
                a += mtimes([solution["W",k].T,Q,solution["W",k]]).full()[0][0]
           # process_cost.append(a.full()[0][0])
            
            estimated_X[:,0:N] = solution["X",lambda x: horzcat(*x)]
            estimated_W[:,0:N-1] = solution["W",lambda x: horzcat(*x)]
        
        if(i>(N-1) and i<Nsimulation-(N-2)):
            # Now make a loop for the rest of the simulation
        
            # update the arrival cost, using linearisations around the estimate of MHE at the beginning of the horizon (according to the 'Smoothed EKF Update'):
            # first update the state and covariance with the measurement that will be deleted, and next propagate the state and covariance because of the shifting of the horizon
            print("step %d/%d (%s)" % (i-N, Nsimulation-N , nlpsol.stats()["return_status"]))
            H0 = H(solution["X",0])[0]
            K = mtimes([P,H0.T,linalg.inv(mtimes([H0,P,H0.T])+R)])
           # Gain_lst.append(K)
            P = mtimes((DM.eye(Nstates)-mtimes(K,H0)),P)
            h0 = h(solution["X",0])
            x0 = x0 + mtimes(K, current_parameters["Y",0]-h0-mtimes(H0,x0-solution["X",0]))
            x0 = phi(x0, current_parameters["U",0], solution["W",0])
            F = PHI(solution["X",0], current_parameters["U",0], solution["W",0])[0]
            P = mtimes([F,P,F.T]) + linalg.inv(Q)
            # Get the measurements and control inputs
            current_parameters["U",lambda x: horzcat(*x)] =  DM([l*2 for l in meas[0:N-1]])#simulated_U[i-N:i-1]
            current_parameters["Y",lambda x: horzcat(*x)] = DM([meas[i-N:i]])
            current_parameters["S"] = linalg.inv(P)
            current_parameters["x0"] = x0
            # Initialize the system with the shifted solution
            initialisation_state["W",lambda x: horzcat(*x),0:N-2] = estimated_W[:,i-N:i-2] # The shifted solution for the disturbances
            initialisation_state["W",N-2] = DM.zeros(Ndisturbances,1) # The last node for the disturbances is initialized with zeros
            initialisation_state["X",lambda x: horzcat(*x),0:N-1] = estimated_X[:,i-N:i-1] # The shifted solution for the state estimates
            # The last node for the state is initialized with a forward simulation
            phi0 = phi(initialisation_state["X",N-1], current_parameters["U",-1], initialisation_state["W",-1])
            initialisation_state["X",N-1] = phi0
            # And now initialize the solver and solve the problem
            res = nlpsol(p=current_parameters, x0=initialisation_state,lbx=lbw, ubx=ubw, lbg=0, ubg=0)
            solution = shooting(res["x"])
     
            #calculation of measurement cost
            b = 0
            for k in range(N):
                a = h(solution["X",k])-current_parameters["Y",k]
                b += (mtimes([a.T,R,a])).full()[0][0]
    
            
            #calculation of process cost
            for k in range(N-1):
                a += mtimes([solution["W",k].T,Q,solution["W",k]]).full()[0][0]
  
            
            # Now get the state estimate. Note that we are only interested in the last node of the horizon
            estimated_X[:,N-1+i-N] = solution["X",lambda x: horzcat(*x)][:,N-1]
            estimated_W[:,N-2+i-N] = solution["W",lambda x: horzcat(*x)][:,N-2]
            
            #get the solution
            dx1 = solution["X",lambda x: horzcat(*x)][:,N-1][0].full()[0][0]
            dx2 = solution["X",lambda x: horzcat(*x)][:,N-1][1].full()[0][0]
            b = solution["X",lambda x: horzcat(*x)][:,N-1][2].full()[0][0]

            dx1_res.append(dx1)
            dx2_res.append(dx2)
            b_res.append(b)
    else:
        dx1_res.append(0)
        dx2_res.append(0)
        b_res.append(0)

#print(len(meas))
#print(len(meas2))
b_real = [2.5 for i in range(Nsimulation)]
plt.plot(meas)
plt.plot(dx1_res)
plt.show()
plt.plot(b_res)
plt.plot(b_real)
plt.show()
print(b_res)