import sys
import os
# BASEPATH = os.path.abspath(__file__).split('AIrcraft', 1)[0]+'AIrcraft/'
# sys.path += [BASEPATH + 'src']

# from track import Track
# from quad import Quad
# from integrator import RungeKutta4

# from trajectory import Trajectory
# from plot import CallbackPlot

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('AIrcraft', 1)[0] + 'AIrcraft/'
sys.path.append(BASEPATH)
from src.dynamics_minimal import Aircraft, AircraftOpts
from src.planner import Planner
from src.utils import TrajectoryConfiguration
from src.plotting_minimal import TrajectoryPlotter, TrajectoryData
import casadi as ca

from pathlib import Path
import json
BASEPATH = Path(BASEPATH)
DATAPATH = BASEPATH / 'data'

traj_dict = json.load(open(os.path.join(DATAPATH,'glider/problem_definition.json')))

trajectory_config = TrajectoryConfiguration(traj_dict)

aircraft_config = trajectory_config.aircraft
poly_path = BASEPATH / "main/fitted_models_casadi.pkl"
opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)
aircraft = Aircraft(opts = opts)


# track = Track(BASEPATH2 + "/tracks/track.yaml")
# quad = Quad(BASEPATH2 + "/quads/quad.yaml")

plotter = TrajectoryPlotter(aircraft, trajectory_config)

class CallbackPlot(ca.Callback):
    def __init__(self, aircraft, trajectory_config):
        ca.Callback.__init__(self)
        self.plotter = TrajectoryPlotter(aircraft, trajectory_config)

    def set_size(self, nx, ng, NPW):
        self.nx = nx
        self.ng = ng
        self.NPW = NPW
        self.construct('CallbackPlot', self.opts)

    def set_wp(self, wp):
        self.wp = wp

    def get_n_in(self): return ca.nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i): return ca.nlpsol_out(i)
    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n=='f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity(0,0)
        
    def eval(self, arg):
        # Create dictionary
        darg = {}
        for (i,s) in enumerate(ca.nlpsol_out()): darg[s] = arg[i]

        X_opt = darg['x'].full().flatten()
        X_opt
        trajectory_data = TrajectoryData(state = None, control = None, time = None, lam = None, mu = None, nu = None, iteration = None)
        # self.plotter



cp = CallbackPlot(pos='xy', vel='xya', ori='xyzw', rate='xyz', inputs='u', prog='mn')

planner = Planner(aircraft, track, {'tolerance': 1.0, 'nodes_per_gate': 30, 'vel_guess': 80.0})
# planner.setup()
# planner.set_iteration_callback(cp)
x = planner.solve()

# traj = Trajectory(x, NPW=planner.NPW, wp=planner.wp)
# traj.save(BASEPATH2 + '/example/result_cpc_format.csv', False)
# traj.save(BASEPATH2 + '/example/result.csv', True)
