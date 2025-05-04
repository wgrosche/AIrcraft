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

class Controller(MHTT, AircraftControl, SaveMixin):
    def __init__(self, *, aircraft, track, num_nodes=100, dt=0.1, track_length=1, opts = {}, filepath:str = '', **kwargs):
        super().__init__(aircraft=aircraft, num_nodes=num_nodes, opts = opts, track = track, dt = dt, track_length=track_length)
        if filepath:
            self.save_path = filepath
        SaveMixin._init_saving(self, self.save_path, force_overwrite=False)



traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)



aircraft_config = trajectory_config.aircraft

poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

aircraft = Aircraft(opts = opts)
trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]

state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
aircraft.com = np.array(trim_state_and_control[-3:])
aircraft.STEPS = 1
dynamics = aircraft.state_update
progress = 0
dubins = DubinsInitialiser(trajectory_config)
dubins.trajectory(1)
print("Trajectory: ", dubins.trajectory(1).full().flatten())
print("Length: :",dubins.length())
# dubins.visualise()
mhtt = Controller(aircraft=aircraft, track = dubins.trajectory, track_length = dubins.length(), filepath = Path(DATAPATH) / 'trajectories' / 'mhtt_solution.h5', num_nodes=100, dt=0.01, progress= False, implicit=False)

while progress < 1:
    print("initial state: ", state, ", progress: ", progress)
    initial_state = state
    
    guess, progress = mhtt.initialise(initial_state, progress)
    print("Initialisedd guess: ", guess)
    mhtt.setup(guess, progress)
    print("Setup guess: ")
    sol = mhtt.solve()
    print("Solution: ", sol)
    if sol.value(mhtt.progress[-1]) > 1:
        # save the solution
        break
    state = ca.DM(sol.value(mhtt.state)[:,90])
    print(state)
    
    progress = sol.value(mhtt.progress[90])
    print(progress)

