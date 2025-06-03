"""
Study comparing convergence impact of:
 - Quaternion normalisation approach
 - Model type
 - Variable time implementation
"""

# from aircraft.utils.utils import TrajectoryConfiguration
# from pathlib import Path
# from aircraft.config import NETWORKPATH
# from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
# import numpy as np
# from tqdm import tqdm
# import casadi as ca
# import json
# from aircraft.control.aircraft import AircraftControl, WaypointControl
# from aircraft.control.base import SaveMixin#, VariableTimeMixin

# from aircraft.control.variable_time import ProgressTimeMixin
# from aircraft.config import DATAPATH
# import matplotlib.pyplot as plt
# from aircraft.plotting.plotting import TrajectoryPlotter, TrajectoryData
# from aircraft.dynamics.coefficient_models import COEFF_MODEL_REGISTRY

from aircraft.utils.utils import TrajectoryConfiguration
from pathlib import Path
from aircraft.config import NETWORKPATH
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
import numpy as np
from tqdm import tqdm
import casadi as ca
import json
from typing import Any
from aircraft.control.aircraft import AircraftControl,  ControlNode
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
for time in ['fixed', 'progress']:#, 'variable']:#, 'adaptive']:
    for quaternion in ['constraint', 'baumgarte', 'integration', '']:
        for integration in ['explicit', 'implicit']:
            for model in ['default', 'poly']: #'linear', 'nn', 
                for steps in [1]:
                    opts.append({'time':time, 'quaternion':quaternion, 'integration':integration, 'model':model, 'steps':steps})

NETWORKPATH = Path(NETWORKPATH)
model_path:dict[str, Path] = {
    'linear': NETWORKPATH / 'linearised.csv',
    'poly': NETWORKPATH / 'fitted_models_casadi.pkl',
    'nn': NETWORKPATH / 'model-dynamics.pth',
    'default': Path(NETWORKPATH)
} 
traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)

aircraft_config = trajectory_config.aircraft

def evaluate_sol(sol, filename, title, controller):
    """
    Evaluate the solution and plot the trajectory.
    Save the optimization summary to a file.
    """
    state = sol.value(controller.state)
    control = sol.value(controller.control)
    times = sol.value(controller.times)

    stats = None
    if controller is not None:
        stats = controller.opti.stats()

    # Write summary to file
    with open(filename, 'a') as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write("=== Solution Summary ===\n")
        f.write(f"Final Time: {times[-1]}\n")
        f.write(f"Final State:\n{state[:,-1]}\n")
        f.write(f"Final Control:\n{control[:,-1]}\n")
        # f.write(f"Success: {success}\n")
        if stats is not None:
            f.write(f"Status: {stats.get('return_status')}\n")
            f.write(f"Iterations: {stats.get('iter_count')}\n")
            f.write(f"Final objective: {stats.get('f')}\n")
            f.write(f"Solver time (s): {stats.get('t_proc_total')}\n")
        f.write("\n" + "-"*40 + "\n")
            # f.write("\n" + metrics)


def run_test_case(opts, goal = [50, 0]): # first run was [50, 0]
    aircraft_config.aero_centre_offset = [0.0131991, -1.78875e-08, 0.00313384]
    air_opts = AircraftOpts(coeff_model_type=opts.get('model'), 
                            coeff_model_path=model_path.get(opts.get('model'), ''), 
                            aircraft_config=aircraft_config, 
                            physical_integration_substeps=opts.get('steps'))
    

    aircraft = Aircraft(opts = air_opts)
    goal_txt = f"{goal[0]}_{goal[1]}"
    filepath = Path(DATAPATH) / 'trajectories' / f'ablation_{dict_to_filename(opts)}_{goal_txt}.h5'

    try:
        controller = Controller(aircraft=aircraft, filepath=filepath, opts = opts, plotting = False)
        controller.goal = ca.DM(goal)
        trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
        guess = controller.initialise(ca.DM(trim_state_and_control[:aircraft.num_states]))
    except RuntimeError as e:
        print(f"Error initializing controller: {e}")
        return None
    
    # try:
    controller.setup(guess)
    # except RuntimeError as e:
    #     print(f"Error setting up the optimization problem: {e}")
    #     return None
    try:
        sol = controller.solve()
    except RuntimeError as e:
        print(f"Error solving the optimization problem: {e}")
        return None
    final_state = controller.opti.debug.value(controller.state)[:, -1]
    final_control = controller.opti.debug.value(controller.control)[:, -1]
    final_time = controller.opti.debug.value(controller.times)[-1]
    print("Final State: ", final_state, " Final Control: ", final_control, " Final Forces: ", aircraft.forces_frd(final_state, final_control), " Final Time: ", final_time)
    evaluate_sol(sol, Path(DATAPATH) / 'trajectories' / f'sol_summary_ipopt_changes_{goal_txt}.txt', title = dict_to_filename(opts), controller = controller)

class Controller(AircraftControl, SaveMixin):#, ProgressTimeMixin):
    goal:np.ndarray
    def __init__(self, *, aircraft, num_nodes=100, dt=.01, opts = {}, filepath:str|Path = '', **kwargs):
        super().__init__(aircraft=aircraft, num_nodes=num_nodes, opts = opts, dt = dt, **kwargs)
        if filepath:
            self.save_path = filepath
        SaveMixin._init_saving(self, self.save_path, force_overwrite=True)
        

    def loss(self, nodes, time):
        """
        Try to scale everything to order 1
        """
        
        time_loss = self.times[-1]
        control_loss = ca.sumsqr(self.control[:, 1:] / 10 - self.control[:, :-1] / 10)
        indices = self.goal.shape[0]
        goal_loss = 10 * ca.sumsqr(self.state[:indices, -1] - self.goal)
        height_loss = ca.sumsqr(self.state[2, :]/ 100) # we want to maximise negative height
        speed_loss = - ca.sumsqr(self.state[3, :] / 100) # we want to maximise body frame x velocity
        loss = goal_loss
        if not isinstance(self.times[-1], float):
            loss += time_loss
        return  loss# + time_loss + control_loss + height_loss + speed_loss
    
    def initialise(self, initial_state):
        """
        Initialize the optimization problem with initial state.
        """
        guess = np.zeros((self.state_dim + self.control_dim, self.num_nodes + 1))
        guess[13, :] = 0
        guess[14, :] = 0
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
    def callback(self, iteration: int):
        """
        """
        
        # print(state[:self.goal.shape[0], -1] - self.goal)
        # print(self.opti.debug.value(self.dt))
        if iteration % 100 == 0:
            state = self.opti.debug.value(self.state)
            print("Distance to goal: ", np.linalg.norm(state[:self.goal.shape[0], -1] - self.goal))
        # print("Final State: ", state[:, -1])
    

def main():
    first_run = False
    for i, opt in enumerate(opts):
        print(f'Running test case {i} with opts: {opt}')
        if first_run:
            first_run = False
            continue
        
        run_test_case(opts = opt)
    
    
    
if __name__ =="__main__":
    main()