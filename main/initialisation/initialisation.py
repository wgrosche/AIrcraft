from aircraft.control.initialisation import DubinsInitialiser, visualize_trajectory
from aircraft.utils.utils import TrajectoryConfiguration
import json
import casadi as ca
import numpy as np
from aircraft.plotting.plotting import TrajectoryPlotter

traj_dict = json.load(open('data/glider/problem_definition.json'))
traj_dict["waypoints"]["waypoints"] = np.array([[150.0, 10.0, -190.0], 
                        [0.0, 20.0, -180.0], 
                        [150.0, 10.0, -190.0]])
traj_dict["aircraft"]["r_min"] = 20.0


trajectory_config = TrajectoryConfiguration(traj_dict)
initialiser = DubinsInitialiser(trajectory_config)
initialiser._build_track_functions()
print(ca.norm_1(initialiser.eval_tangent(0)))
initialiser.visualize()
# initialiser.visualise()