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


# default_solver_options = {'ipopt': {'max_iter': 10000,
#                                     'tol': 1e-2,
#                                     'acceptable_tol': 1e-2,
#                                     'acceptable_obj_change_tol': 1e-2,
#                                     'hessian_approximation': 'limited-memory'
#                                     },
#                         'print_time': 10,
#                         # 'expand' : True
#                         }

default_solver_options = {'ipopt': {'max_iter': 10000,
                                    'tol': 1e-6,
                                    'acceptable_tol': 1e-6,
                                    'acceptable_obj_change_tol': 1e-6,
                                    'hessian_approximation': 'exact', #'limited-memory',
                                    'linear_solver': 'mumps',
                                    # 'jacobian_approximation': 'exact',  # Use exact Jacobian as well
                                    # 'calc_lam_p': True,
                                    'mumps_mem_percent': 10000,      # Increase memory allocation percentage
                                    'mumps_pivtol': 1e-4,           # Pivot tolerance (can help with numerical stability)
                                    'mumps_pivtolmax': 1e-2,        # Maximum pivot tolerance
                                    'mumps_permuting_scaling': 7,   # Use a more robust scaling strategy
                                    'max_cpu_time': 1e4,             # Increase the maximum CPU time
                                    'print_level': 5,
                                    'theta_max_fact':100,
                                    'mu_strategy': 'adaptive',
                                    'mu_oracle': 'probing',
                                    'mu_init': 1e-2,
                                    'barrier_tol_factor': 0.1,  # Make barrier updates more aggressive
                                    },
                        'print_time': 10,
                        'expand': True # NOTE: Find way to set to true with interpolant

                        }


import casadi as ca
control_dict = {'scale_state' : ca.vertcat(
                            
                            [1e2, 1e2, 1e2],
                            [50, 50, 50],
                            [1, 1, 1, 1],
                            [np.pi, np.pi, np.pi]),
                "scale_control": ca.vertcat(10, 10, 0, 0, 0), "max_control_nodes" : 100}
                # 'scale_control' : ca.vertcat(
                #             5, 5, 5,
                #             [1e2, 1e2, 1e2],
                #             [1, 1, 1],
                #             [1e2, 1e2, 1e2]
                #             )}