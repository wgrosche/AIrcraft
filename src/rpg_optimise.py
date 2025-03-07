
import casadi as ca
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
import os
import sys

BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('AIrcraft', 1)[0] + 'AIrcraft/'

print(BASEPATH)
sys.path.append(BASEPATH)

from src.dynamics_minimal import Aircraft, AircraftOpts
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from src.utils import TrajectoryConfiguration, load_model
from matplotlib.pyplot import spy
import json
import matplotlib.pyplot as plt
from liecasadi import Quaternion
import h5py
from scipy.interpolate import CubicSpline
from src.plotting_minimal import TrajectoryPlotter, TrajectoryData

# from src.planner import Planner
from src.simple_planner import Planner
from src.plot import CallbackPlot

import threading
import torch

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
sys.path.append(BASEPATH)
from pathlib import Path

plt.ion()
# default_solver_options = {
#                         'print_time': 10
#                         }

default_solver_options = {'ipopt': {'max_iter': 10000,
                                    'tol': 1e-1,
                                    'acceptable_tol': 1e-1,
                                    'acceptable_obj_change_tol': 1e-1,
                                    'hessian_approximation': 'exact'
                                    },
                        'print_time': 10
                        }


if __name__ == '__main__':

    traj_dict = json.load(open('data/glider/problem_definition.json'))
    trajectory_config = TrajectoryConfiguration(traj_dict)

    aircraft_config = trajectory_config.aircraft

    # linear_path = Path(DATAPATH) / 'glider' / 'linearised.csv'
    # model_path = Path(NETWORKPATH) / 'model-dynamics.pth'

    poly_path = Path(NETWORKPATH) / "fitted_models_casadi.pkl"

    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)
    opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config)

    # opts = AircraftOpts(linear_path=linear_path, aircraft_config=aircraft_config)
    # opts = AircraftOpts(nn_model_path=model_path, aircraft_config=aircraft_config)

    aircraft = Aircraft(opts = opts)

    # [0, 0, 0, 30, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 
    aircraft.com = [0.0131991, -1.78875e-08, 0.00313384]
    planner = Planner(
        aircraft,
        options = {'tolerance': 20.0, 'nodes_per_gate': 40, 'vel_guess': 35.0, 'solver_type':'ipopt', 'solver_options' : default_solver_options}
        )
    
    cp = CallbackPlot(pos='xy', vel='xya', ori='xyzw', rate='xyz', inputs='u', prog='mn')
    planner.setup()
    planner.set_iteration_callback(cp)
    x = planner.solve()