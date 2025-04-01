import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

from liecasadi import Quaternion
from scipy.spatial.transform import Rotation as R
import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import h5py

from aircraft.config import BASEPATH, NETWORKPATH, DATAPATH, DEVICE
from aircraft.surrogates.models import ScaledModel
from aircraft.utils.utils import load_model, TrajectoryConfiguration, AircraftConfiguration, perturb_quaternion

from dataclasses import dataclass
from aircraft.dynamics.base import SixDOFOpts, SixDOF
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft


def main():
    from aircraft.plotting.plotting import TrajectoryPlotter

    mode = 1
    model = load_model()
    traj_dict = json.load(open('data/glider/problem_definition.json'))

    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    if mode == 0:
        model_path = Path(NETWORKPATH) / 'model-dynamics.pth'
        opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    elif mode == 1:
        poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
        opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)
    elif mode == 2:
        linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
        opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)

    perturbation = False
    
    trim_state_and_control = [0, 0, 0, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    # trim_state_and_control = [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]

    if trim_state_and_control is not None:
        state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
        control = np.zeros(aircraft.num_controls)
        control[:3] = trim_state_and_control[aircraft.num_states:-3]
        control[0] = 1
        control[1] = 0
        aircraft.com = np.array(trim_state_and_control[-3:])
    else:
        x0 = np.zeros(3)
        v0 = ca.vertcat([60, 0, 0])
        # would be helpful to have a conversion here between actual pitch, roll and yaw angles and the Quaternion q0, so we can enter the angles in a sensible way.
        q0 = Quaternion(ca.vertcat(0, 0, 0, 1))
        omega0 = np.array([0, 0, 0])
        state = ca.vertcat(x0, v0, q0, omega0)
        control = np.zeros(aircraft.num_controls)
        control[0] = 1
        control[1] = 5

    dyn = aircraft.state_update
    jacobian_elevator = ca.jacobian(aircraft.state_derivative(aircraft.state, aircraft.control), aircraft.control[1])
    jacobian_func = ca.Function('jacobian_func', [aircraft.state, aircraft.control], [jacobian_elevator])
    jacobian_elevator_val = jacobian_func(state, control)

    print("Jacobian of state derivatives w.r.t. elevator:")
    print(jacobian_elevator_val)
    dt = .01
    tf = 50
    state_list = np.zeros((aircraft.num_states, int(tf / dt)))
    t = 0
    ele_pos = True
    ail_pos = True
    control_list = np.zeros((aircraft.num_controls, int(tf / dt)))
    for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
        if np.isnan(state[0]):
            print('Aircraft crashed')
            break
        else:
            if aircraft.phi(state).full().flatten() > np.deg2rad(50):
                control[0] = -1
            elif aircraft.phi(state).full().flatten() < np.deg2rad(-50):
                control[0] = 1
            state_list[:, i] = state.full().flatten()
            control_list[:, i] = control
            state = dyn(state, control, dt)
                    
            t += 1

    first = None
    t -=10
    def save(filepath):
        with h5py.File(filepath, "a") as h5file:
            grp = h5file.create_group('iteration_0')
            grp.create_dataset('state', data=state_list[:, :t])
            grp.create_dataset('control', data=control_list[:, :t])
    
    
    filepath = os.path.join("data", "trajectories", "simulation.h5")
    if os.path.exists(filepath):
        os.remove(filepath)
    save(filepath)

    plotter = TrajectoryPlotter(aircraft)
    plotter.plot(filepath=filepath)
    plt.show(block = True)


if __name__ == "__main__":
    main()
