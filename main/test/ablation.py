"""
Study comparing convergence impact of:
 - Quaternion normalisation approach
 - Model type
 - Variable time implementation
"""

from aircraft.control.aircraft import TrajectoryConfiguration
from pathlib import Path
from aircraft.config import NETWORKPATH
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
import numpy as np
from tqdm import tqdm
import casadi as ca
import json
from aircraft.control.aircraft import AircraftControl, WaypointControl
from aircraft.control.base import SaveMixin#, VariableTimeMixin

from aircraft.control.variable_time import ProgressTimeMixin
from aircraft.config import DATAPATH
import matplotlib.pyplot as plt
from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData
from aircraft.dynamics.coefficient_models import COEFF_MODEL_REGISTRY
"""
Ablation table opts:

[
    {'time':'variable', 'quaternion':'', 'integration':'explicit'},
    {'time':'progress', 'quaternion':'', 'integration':'explicit'},
    {'time':'variable', 'quaternion':'', 'integration':'explicit'},
    {'time':'variable', 'quaternion':'', 'integration':'explicit'},
    {'time':'variable', 'quaternion':'', 'integration':'explicit'}
    },
]
"""

def dict_to_filename(opts):
    # Replace empty strings with a placeholder (e.g., 'none')
    safe_opts = {k: (v if v else 'none') for k, v in opts.items()}
    # Create key-value string pairs like time_fixed, model_nn, etc.
    return '_'.join(f"{k}_{safe_opts[k]}" for k in sorted(safe_opts))

opts = []
for time in ['fixed', 'progress', 'variable', 'adaptive']:
    for quaternion in ['constraint', 'baumgarte', 'integration', '']:
        for integration in ['explicit', 'implicit']:
            for model in ['linear', 'poly', 'nn', 'default']:
                for steps in ['1', '5', '10', '100']:
                    opts.append({'time':time, 'quaternion':quaternion, 'integration':integration, 'model':model, 'steps':steps})
model_path = {} 

traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)

aircraft_config = trajectory_config.aircraft

def run_test_case(opts):
    air_opts = AircraftOpts(coeff_model_type=opts.get('model'), 
                            coeff_model_path=model_path.get(opts.get('model')), 
                            aircraft_config=aircraft_config, 
                            physical_integration_substeps=opts.get('steps'))

    aircraft = Aircraft(opts = opts)
    filepath = Path(DATAPATH) / 'trajectories' / f'ablation_{dict_to_filename(opts)}.h5'
    controller = Controller(aircraft=aircraft, filepath=filepath, opts = opts)
    trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
    guess = controller.initialise(ca.DM(trim_state_and_control[:aircraft.num_states]))
    controller.setup(guess)
    sol = controller.solve()
    final_state = controller.opti.debug.value(controller.state)[:, -1]
    final_control = controller.opti.debug.value(controller.control)[:, -1]
    final_time = controller.opti.debug.value(controller.times)[-1]
    print("Final State: ", final_state, " Final Control: ", final_control, " Final Forces: ", aircraft.forces_frd(final_state, final_control), " Final Time: ", final_time)
    evaluate_sol(sol)

class Controller(AircraftControl, SaveMixin):#, ProgressTimeMixin):
    goal:np.ndarray
    def __init__(self, *, aircraft, num_nodes=300, dt=.01, opts = {}, filepath:str|Path = '', **kwargs):
        super().__init__(aircraft=aircraft, num_nodes=num_nodes, opts = opts, dt = dt, **kwargs)
        if filepath:
            self.save_path = filepath
        SaveMixin._init_saving(self, self.save_path, force_overwrite=True)
        

    def loss(self, nodes, time):
        """
        Try to scale everything to order 1
        """
        time_loss = self.times[-1]
        control_loss = ca.sumsqr(self.control[: 1:] / 10 - self.control[:, -1] / 10)
        indices = self.goal.shape[0]
        goal_loss = ca.sumsqr(self.state[-1, :indices] - self.goal)
        height_loss = ca.sumsqr(self.state[:, 2]/ 100) # we want to maximise negative height
        speed_loss = - ca.sumsqr(self.state[:, 2] / 100) # we want to maximise body frame x velocity
        return  time_loss + control_loss + goal_loss + height_loss + speed_loss
    
    def initialise(self, initial_state):
        """
        Initialize the optimization problem with initial state.
        """
        guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
        guess[13, :] = 1
        guess[14, :] = 1
        guess[:self.state_dim, 0] = initial_state.toarray().flatten()
        # dt_initial = self.dt
        # dt_initial = 0.01#2 / self.num_nodes
        # self.opti.set_initial(self.time, 2)
        # Propagate forward using nominal dynamics
        for i in range(self.num_nodes):
            guess[:self.state_dim, i + 1] = self.dynamics(
                guess[:self.state_dim, i],
                guess[self.state_dim:, i],
                self.dt 
                
            ).toarray().flatten()


        return guess
    

def main():
    for i, opt in enumerate(opts):
        print(f'Running test case {i} with opts: {opt}')
        run_test_case(opts = opt)
    
    
    
if __name__ =="__main__":
    main()