

import casadi as ca
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from l4casadi import L4CasADi
from l4casadi.naive import NaiveL4CasADiModule
from l4casadi.realtime import RealTimeL4CasADi
from liecasadi import Quaternion
from scipy.spatial.transform import Rotation as R
import torch
import json
import os
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
# from numba import jit
import h5py
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('AIrcraft')[0] + 'AIrcraft'
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)

from aircraft.surrogates.models import ScaledModel
from aircraft.utils.utils import load_model, TrajectoryConfiguration
from aircraft.plotting_minimal import TrajectoryPlotter
from dataclasses import dataclass

from aircraft.dynamics.dynamics import AircraftTrim, AircraftOpts

print(DEVICE)
    


if __name__ == '__main__':
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
    model_path = Path(NETWORKPATH) / 'model-dynamics.pth'
    poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'

    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)
    # opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)

    aircraft = AircraftTrim(opts = opts)

    perturbation = False
    
    trim_state_and_control = None
    if trim_state_and_control is not None:
        state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
        control = np.array(trim_state_and_control[aircraft.num_states:])
    else:

        x0 = np.zeros(3)
        v0 = ca.vertcat([50, 0, 0])
        # would be helpful to have a conversion here between actual pitch, roll and yaw angles and the Quaternion q0, so we can enter the angles in a sensible way.
        q0 = Quaternion(ca.vertcat(0, 0, 0, 1))
        # q0 = Quaternion(ca.vertcat(.259, 0, 0, 0.966))
        # q0 = ca.vertcat([0.215566, -0.568766, 0.255647, 0.751452])#q0.inverse()
        omega0 = np.array([0, 0, 0])

        state = ca.vertcat(x0, v0, q0, omega0)
        control = np.zeros(aircraft.num_controls)
        control[0] = +0
        control[1] = 0
        # control[6:9] = traj_dict['aircraft']['aero_centre_offset']

    dyn = aircraft.state_update
    dt = .1
    tf = 30
    state_list = np.zeros((aircraft.num_states, int(tf / dt)))
    # investigate stiffness:

    # Define f(state, control) (e.g., the dynamics function)
    f = aircraft.state_derivative(aircraft.state, aircraft.control)

    # Compute the Jacobian of f w.r.t state
    J = ca.jacobian(f, aircraft.state)

    # Create a CasADi function for numerical evaluation
    J_func = ca.Function('J', [aircraft.state, aircraft.control], [J])

    # Evaluate J numerically for a specific state and control
    J_val = J_func(state, control)

    # Compute eigenvalues using numpy
    eigvals = np.linalg.eigvals(np.array(J_val))

    print(eigvals)
    



    # dt_sym = ca.MX.sym('dt', 1)
    t = 0
    ele_pos = True
    ail_pos = True
    control_list = np.zeros((aircraft.num_controls, int(tf / dt)))
    for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
        if np.isnan(state[0]):
            print('Aircraft crashed')
            break
        else:
            state_list[:, i] = state.full().flatten()
            control_list[:, i] = control
            state = dyn(state, control, dt)
                    
            t += 1
    print(state)

    first = None
    # t -=10
    def save(filepath):
        with h5py.File(filepath, "a") as h5file:
            grp = h5file.create_group(f'iteration_0')
            grp.create_dataset('state', data=state_list[:, :t])
            grp.create_dataset('control', data=control_list[:, :t])
    
    
    filepath = os.path.join("data", "trajectories", "simulation.h5")
    if os.path.exists(filepath):
        os.remove(filepath)
    save(filepath)

    plotter = TrajectoryPlotter(aircraft)
    plotter.plot(filepath=filepath)
    plt.show(block = True)