from aircraft.control.initialisation import DubinsInitialiser
from aircraft.utils.utils import TrajectoryConfiguration
import json

from aircraft.plotting.plotting import TrajectoryPlotter

traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)

initaliser = DubinsInitialiser(trajectory_config)

initaliser.visualise()