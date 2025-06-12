from aircraft.control.initialisation import DubinsInitialiser, visualize_trajectory
from aircraft.utils.utils import TrajectoryConfiguration
import json
import casadi as ca
from aircraft.plotting.plotting import TrajectoryPlotter

traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)

initialiser = DubinsInitialiser(trajectory_config)
initialiser._build_track_functions()
print(ca.norm_1(initialiser.eval_tangent(0)))
initialiser.visualize()
# initialiser.visualise()