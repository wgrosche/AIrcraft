from aircraft.control.aircraft import TrajectoryConfiguration
from pathlib import Path
from aircraft.config import NETWORKPATH
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
import numpy as np
from tqdm import tqdm
import casadi as ca
import json
from aircraft.control.quadrotor import QuadrotorControl
from aircraft.dynamics.quadrotor import Quadrotor

    

def main():
    """
    Minimal test of quadrotor control class
    """
    quad = Quadrotor()
    control_problem = QuadrotorControl(quad.state_update)

    control_problem.setup()
    control_problem.solve()
    


if __name__ =="__main__":
    main()