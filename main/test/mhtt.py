import json
from aircraft.dynamics.aircraft import Aircraft, AircraftOpts
from pathlib import Path
from aircraft.config import NETWORKPATH, DATAPATH
from aircraft.utils.utils import TrajectoryConfiguration


from aircraft.control.initialisation import DubinsInitialiser, vis_traj_embed
from aircraft.control.moving_horizon import MHTT
from aircraft.control.aircraft import AircraftControl
from aircraft.control.base import SaveMixin
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from aircraft.plotting.plotting import TrajectoryData
from copy import deepcopy
from control import Controller as MinimumTimeController
from tqdm import tqdm
plt.ion()
class Controller(MHTT, AircraftControl):#, SaveMixin):
    def __init__(self, *, aircraft, track:DubinsInitialiser, num_nodes=200, dt=0.01, opts = {}, filepath:str = '', **kwargs):
        super().__init__(aircraft=aircraft, num_nodes=num_nodes, opts = opts, track = track, dt = dt)
        # if filepath:
        #     self.save_path = filepath
        # SaveMixin._init_saving(self, self.save_path, force_overwrite=False)

    def callback(self, iteration):
        return None



traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)




aircraft_config = trajectory_config.aircraft

poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'

opts = AircraftOpts(coeff_model_type='poly', coeff_model_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

aircraft = Aircraft(opts = opts)
trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
trajectory_config.waypoints.initial_state= trim_state_and_control[:aircraft.num_states]
state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
aircraft.com = np.array(trim_state_and_control[-3:])
dynamics = aircraft.state_update
progress = 0
dubins = DubinsInitialiser(trajectory_config)
dubins._build_track_functions()
controller_opts = {'time':'fixed', 'quaternion':'integration', 'integration':'explicit'}

mhtt = Controller(aircraft=aircraft, track = dubins, filepath = Path(DATAPATH) / 'trajectories' / 'mhtt_solution.h5', num_nodes=100, dt=0.01, opts = controller_opts)

pbar = tqdm(total=1.0, desc="Solving", unit="progress")
initial_state = state
guess = mhtt.initialise(initial_state, progress)
mhtt.setup(guess)
mhtt.plotter.waypoints = trajectory_config.waypoints.waypoints.T
mhtt.plotter.waypoints[2, :] *= -1

# vis_traj_embed(mhtt.plotter.axes.position, dubins.eval, dubins.eval_tangent, s_range=(0, 1), num_points=100, quiver_stride=10)

full_state = None
full_progress = None
full_control = None

overlap = 5

while progress < 1:
    sol = mhtt.solve()

    if not mhtt.success:
        print("Solution not found")
        break

    state = ca.DM(sol.value(mhtt.state))[:, :-overlap]
    new_progress = sol.value(mhtt.track_progress)[:-overlap]
    control = ca.DM(sol.value(mhtt.control))[:, :-overlap]

   
    if not isinstance(full_state, ca.DM):
        full_state = state[:, 1:]
        full_progress = new_progress
        full_control = control
    else:
        full_state = ca.hcat([full_state, state[:, 1:]])
        full_control = ca.hcat([full_control, control])
        full_progress = ca.vertcat(full_progress, new_progress)
    print(full_state.shape, full_control.shape, full_progress.shape)
    print("State: ", state.shape, " Progress: ", new_progress.shape, " Control: ", control.shape)
    # print("state: ", state, ", progress: ", new_progress)
    pbar.update(max(new_progress[-1] - progress, 0.0))  # Ensure non-negative
    progress = new_progress[-1]
    if new_progress[-1] >= 0.99:
        break
    initial_state = state[:, -1]

    guess = mhtt.initialise(initial_state, progress)
    mhtt.set_initial_from_array(guess)
    mhtt.update_parameters(guess)
    print(np.array(full_state).shape)
    # sol = mhtt.opti.solve()
    full_state_array = np.array(full_state)
    full_control_array = np.array(full_control)
    full_progress_array = np.array(full_progress)
    plot_data = TrajectoryData(
        state=full_state_array, 
        control=full_control_array,
        times=np.array([i * mhtt.dt for i in range(full_state_array.shape[1])]),
    )
    mhtt.plotter.plot(plot_data)
    plt.draw()
    mhtt.plotter.figure.canvas.start_event_loop(0.0002)
    plt.pause(0.001) 

    # mhtt.setup(guess)
    # sol = mhtt.solve()

    # # Extract new state and progress
    # state = ca.DM(sol.value(mhtt.state)[:, -1])
    # new_progress = sol.value(mhtt.track_progress[-1])

    # # Update progress bar
    

plt.show(block= True)
# traj_dict = json.load(open('data/glider/problem_definition.json'))

# trajectory_config = TrajectoryConfiguration(traj_dict)

# aircraft_config = trajectory_config.aircraft

# poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
# opts = AircraftOpts(coeff_model_type='poly', coeff_model_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

# aircraft = Aircraft(opts = opts)
# trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
# aircraft.com = np.array(trim_state_and_control[-3:])
# filepath = Path(DATAPATH) / 'trajectories' / 'basic_test.h5'

# # controller_opts = {'time':'fixed', 'quaternion':'', 'integration':'explicit'}
# controller_opts = {'time':'progress', 'quaternion':'integration', 'integration':'explicit'}
# controller = Controller(aircraft=aircraft, filepath=filepath, opts = controller_opts, num_nodes=300)
# guess = controller.initialise(ca.DM(trim_state_and_control[:aircraft.num_states]))
# controller.setup(guess)
# controller.logging = False

# sol = controller.solve()
# # print("Solution Status: ", sol.stats())
# final_state = controller.opti.debug.value(controller.state)[:, -1]
# final_control = controller.opti.debug.value(controller.control)[:, -1]
# final_time = controller.opti.debug.value(controller.times)[-1]
# print("Final State: ", final_state, " Final Control: ", final_control, " Final Forces: ", aircraft.forces_frd(final_state, final_control), " Final Time: ", final_time)
# plt.show(block = True)

# pbar.close()

# print("Running final minimum time problem...")


# final_mhtt = Controller(aircraft=aircraft, track=dubins, filepath=Path(DATAPATH) / 'trajectories' / 'mhtt_final_solution.h5', num_nodes=100, dt=0.01, opts=controller_opts)
