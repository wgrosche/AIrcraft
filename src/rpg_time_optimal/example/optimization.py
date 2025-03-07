import sys
import os
BASEPATH2 = os.path.abspath(__file__).split('rpg_time_optimal', 1)[0]+'rpg_time_optimal/'
print(BASEPATH2)
sys.path += [BASEPATH2 + 'src2']

from track import Track
from quad import Quad
from integrator import RungeKutta4
from planner import Planner
from trajectory import Trajectory
from plot import CallbackPlot

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('AIrcraft', 1)[0] + 'AIrcraft/'
sys.path.append(BASEPATH)
from src.dynamics_minimal import Aircraft, AircraftOpts
from pathlib import Path
from src.utils import TrajectoryConfiguration
import json
BASEPATH = Path(BASEPATH)
DATAPATH = BASEPATH / 'data'

traj_dict = json.load(open(os.path.join(DATAPATH,'glider/problem_definition.json')))

trajectory_config = TrajectoryConfiguration(traj_dict)

aircraft_config = trajectory_config.aircraft
poly_path = BASEPATH / "main/fitted_models_casadi.pkl"
opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)
aircraft = Aircraft(opts = opts)


track = Track(BASEPATH2 + "/tracks/track.yaml")
quad = Quad(BASEPATH2 + "/quads/quad.yaml")

cp = CallbackPlot(pos='xy', vel='xya', ori='xyzw', rate='xyz', inputs='u', prog='mn')

planner = Planner(aircraft, quad, track, RungeKutta4, {'tolerance': 1.0, 'nodes_per_gate': 30, 'vel_guess': 30.0})
planner.setup()
planner.set_iteration_callback(cp)
x = planner.solve()

traj = Trajectory(x, NPW=planner.NPW, wp=planner.wp)
traj.save(BASEPATH2 + '/example/result_cpc_format.csv', False)
traj.save(BASEPATH2 + '/example/result.csv', True)
