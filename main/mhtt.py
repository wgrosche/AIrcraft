import json
from aircraft.dynamics.aircraft import Aircraft, AircraftOpts
from pathlib import Path
from aircraft.config import NETWORKPATH, DATAPATH
from aircraft.utils.utils import TrajectoryConfiguration


from aircraft.control.initialisation import DubinsInitialiser
from aircraft.control.moving_horizon import MHTT
from aircraft.control.aircraft import AircraftControl
from aircraft.control.base import SaveMixin
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from aircraft.plotting.plotting import TrajectoryData

class Controller(MHTT, AircraftControl):#, SaveMixin):
    def __init__(self, *, aircraft, track:DubinsInitialiser, num_nodes=100, dt=0.1, opts = {}, filepath:str = '', **kwargs):
        super().__init__(aircraft=aircraft, num_nodes=num_nodes, opts = opts, track = track, dt = dt)
        # if filepath:
        #     self.save_path = filepath
        # SaveMixin._init_saving(self, self.save_path, force_overwrite=False)



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
progress = 1e-6
dubins = DubinsInitialiser(trajectory_config)
dubins._build_track_functions()

controller_opts = {'time':'fixed', 'quaternion':'integration', 'integration':'explicit'}

mhtt = Controller(aircraft=aircraft, track = dubins, filepath = Path(DATAPATH) / 'trajectories' / 'mhtt_solution.h5', num_nodes=100, dt=0.01, opts = controller_opts)

while progress < 1:
    print("initial state: ", state, ", progress: ", progress)
    initial_state = state
    
    # guess, progress = mhtt.initialise(initial_state, progress)
    guess = mhtt.initialise(initial_state, progress)
    print("Initialised guess: ", guess)
    plot_initial = TrajectoryData(
        state = guess[:aircraft.num_states, :], 
        control = guess[aircraft.num_states:aircraft.num_states+aircraft.num_controls, :], 
        times = np.array([i * mhtt.dt for i in range(mhtt.num_nodes + 1)]))
    
    # plt.show(block=True)
    mhtt.plotter.plot(plot_initial)
    mhtt.setup(guess)
    print("Setup guess: ", guess[-1, :])
    sol = mhtt.solve()
    print("Solution: ", sol)
    state = ca.DM(sol.value(mhtt.state)[:,-1])
    print(state)
    
    progress = sol.value(mhtt.track_progress)
    print(progress)
    # mhtt.plot(sol)
    break
    # plt.show(block=True)
    if sol.value(mhtt.track_progress[-1]) >= 1:
        # save the solution
        break
    state = ca.DM(sol.value(mhtt.state)[:,-1])
    print(state)
    
    progress = sol.value(mhtt.track_progress[-1])
    print(progress)

