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


