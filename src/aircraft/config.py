import os
import torch

# Base path of the project
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]

# Data paths
NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DATAPATH = os.path.join(BASEPATH, 'data')

# Device configuration
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    ("mps" if torch.backends.mps.is_available() else "cpu")
)
