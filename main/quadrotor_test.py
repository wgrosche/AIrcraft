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
from aircraft.dynamics.quadrotor import Quadrotor

def main():
    from aircraft.plotting.plotting import TrajectoryPlotter

    quad = Quadrotor()

    x0 = np.zeros(3)
    v0 = ca.vertcat([10, 0, 0])
    # would be helpful to have a conversion here between actual pitch, roll and yaw angles and the Quaternion q0, so we can enter the angles in a sensible way.
    q0 = Quaternion(ca.vertcat(0, 0, 0, 1))

    omega0 = np.array([0, 0, 0])
    state = ca.vertcat(x0, v0, q0, omega0)
    control = np.zeros(4)

    # aircraft.STEPS = 100
    dyn = quad.state_update



    jacobian_elevator = ca.jacobian(quad.state_derivative(quad.state, quad.control), quad.control)
    jacobian_func = ca.Function('jacobian_func', [quad.state, quad.control], [jacobian_elevator])
    jacobian_elevator_val = jacobian_func(state, control)

    print("Jacobian of state derivatives w.r.t. elevator:")
    print(jacobian_elevator_val)
    dt = .01
    tf = 5
    state_list = np.zeros((quad.num_states, int(tf / dt)))
    t = 0
    control_list = np.zeros((quad.num_controls, int(tf / dt)))
    for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
        if np.isnan(state[0]):
            print('quad crashed')
            break
        else:

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

    plotter = TrajectoryPlotter(quad)
    plotter.plot(filepath=filepath)
    plt.show(block = True)


if __name__ == "__main__":
    main()
