import os
import torch
import numpy as np
rng = np.random.default_rng(42)
# Base path of the project
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]

# Data paths
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')
VISUPATH = os.path.join(BASEPATH, 'data', 'visualisation')
# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")



default_solver_options = {'ipopt': {'max_iter': 10000,
                                    'tol': 1e-2,
                                    'acceptable_tol': 1e-2,
                                    'acceptable_obj_change_tol': 1e-2,
                                    'hessian_approximation': 'limited-memory'
                                    },
                        'print_time': 10,
                        # 'expand' : True
                        }


control_dict = {'scale_state' : np.array(
                            [1, 1, 1, 1],
                            [1e3, 1e3, 1e3],
                            [1e2, 1e2, 1e2],
                            [1, 1, 1]
                            ),
                'scale_control' : np.array(
                            5, 5, 5,
                            [1e2, 1e2, 1e2],
                            [1, 1, 1],
                            [1e2, 1e2, 1e2]
                            )}