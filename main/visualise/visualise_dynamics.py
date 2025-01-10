import json
import os
import numpy as np
import casadi as ca
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from matplotlib.widgets import Slider
from ipywidgets import interact
from IPython.display import display
from liecasadi import Quaternion
import sys

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('main')[0]
print(BASEPATH)
sys.path.append(BASEPATH)
DATAPATH = os.path.join(BASEPATH, 'data')

from src.dynamics_minimal import Aircraft, AircraftOpts, TrajectoryConfiguration
from src.utils import load_model
from src.plotting_minimal import TrajectoryPlotter



# Load required modules and configurations
model = load_model()
traj_dict = json.load(open(os.path.join(DATAPATH, 'glider/problem_definition.json')))
trajectory_config = TrajectoryConfiguration(traj_dict)
aircraft_config = trajectory_config.aircraft

poly_path = Path(os.path.join(BASEPATH, "main/fitted_models_casadi.pkl"))
opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)
aircraft = Aircraft(opts=opts)


model = load_model()
traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)

aircraft_config = trajectory_config.aircraft

NETWORKPATH = Path(BASEPATH) / 'data' / 'networks'

linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
model_path = NETWORKPATH / 'model-dynamics.pth'
poly_path = NETWORKPATH / 'fitted_models_casadi.pkl'

# opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)
# opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)

aircraft = Aircraft(opts = opts)

perturbation = False





# Function to run the simulation
def simulate_trajectory(tf, aileron, elevator, plotter):
    dt = 0.1
    num_steps = int(tf / dt)

    trim_state_and_control = [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]# [0, 0, 0, 60, 2.29589e-41, 0, 0, 9.40395e-38, -2.93874e-39, 1, 0, 1.46937e-39, 0, -5.73657e-45, 0, 0, 0.0134613, -7.8085e-09, 0.00436365]

    if trim_state_and_control is not None:
        state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
        control = np.array(trim_state_and_control[aircraft.num_states:-3])
        control[0] = +1
        control[1] = -0.5
        aircraft.com = np.array(trim_state_and_control[-3:])

    else:
        x0 = np.zeros(3)
        v0 = ca.vertcat([80, 0, 0])
        q0 = Quaternion(ca.vertcat(0, 0, 0, 1))
        omega0 = np.array([0, 0, 0])
        aircraft.com = np.array([0, 0, 0])

    # state = ca.vertcat(x0, v0, q0, omega0)
    # control = np.zeros(aircraft.num_controls)
    control[0] = aileron  # Aileron deflection
    control[1] = elevator  # Elevator deflection

    dyn = aircraft.state_update.expand()
    state_list = np.zeros((aircraft.num_states, num_steps))
    control_list = np.zeros((aircraft.num_controls, num_steps))

    for i in tqdm(range(num_steps), desc="Simulating Trajectory"):
        if np.isnan(state[0]):
            print("Aircraft crashed!")
            break
        elif np.linalg.norm(state[0:4]) > 1000:
            print("Aircraft crashed!")
            break
        else:
            state_list[:, i] = state.full().flatten()
            control_list[:, i] = control
            state = dyn(state, control, dt)

    # Save trajectory data
    filepath = os.path.join(DATAPATH, "trajectories", "simulation.h5")
    if os.path.exists(filepath):
        os.remove(filepath)

    with h5py.File(filepath, "a") as h5file:
        grp = h5file.create_group(f'iteration_0')
        grp.create_dataset('state', data=state_list[:, :i - 1])
        grp.create_dataset('control', data=control_list[:, :i - 1])

    # Plot the results
    
    plotter.plot(filepath=filepath)
    plt.show(block=True)

plotter = TrajectoryPlotter(aircraft, figsize=(10, 10))
ax_tf = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
ax_aileron = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_elevator = plt.axes([0.25, 0.0, 0.65, 0.03])

# Initial values
initial_tf = 5.0
initial_aileron = 0.0
initial_elevator = 0.0

slider_tf = Slider(ax_tf, 'Time Final (tf)', 1.0, 20.0, valinit=initial_tf, valstep=0.1)
slider_aileron = Slider(ax_aileron, 'Aileron', -5.0, 5.0, valinit=initial_aileron, valstep=0.1)
slider_elevator = Slider(ax_elevator, 'Elevator', -5.0, 5.0, valinit=initial_elevator, valstep=0.1)

# Update function for sliders
def update(val):
    tf = slider_tf.val
    aileron = slider_aileron.val
    elevator = slider_elevator.val
    simulate_trajectory(tf, aileron, elevator, plotter)



slider_tf.on_changed(update)
slider_aileron.on_changed(update)
slider_elevator.on_changed(update)

# Initial plot
simulate_trajectory(initial_tf, initial_aileron, initial_elevator, plotter)

plt.show()