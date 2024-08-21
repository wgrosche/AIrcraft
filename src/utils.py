import casadi as ca
import torch
import os
import sys

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
sys.path.append(BASEPATH)

from src.models import ScaledModel

NETWORKPATH = os.path.join(BASEPATH, 'data', 'networks')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def load_model(
        filepath:str = os.path.join(NETWORKPATH,'model-dynamics.pth'), 
        device = DEVICE
        ) -> ScaledModel:
    checkpoint = torch.load(filepath, map_location=device)

    scaler = (checkpoint['input_mean'], 
              checkpoint['input_std'], 
              checkpoint['output_mean'], 
              checkpoint['output_std'])
    
    model = ScaledModel(5, 6, scaler=scaler)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def u_from_aero(q_bar, alpha, beta):
    u_squared =((1 - ca.sin(beta)**2) * 2 * q_bar) / ((1 + ca.tan(alpha)**2) * 1.225)

    return ca.sqrt(u_squared)

def v_from_aero(q_bar, alpha, beta):
    v = ca.sin(beta) * ca.sqrt(q_bar * 2 / 1.225)

    return v

def w_from_aero(q_bar, alpha, beta):

    return u_from_aero(q_bar, alpha, beta) * ca.tan(alpha)

def aero_to_state(q_bar, alpha, beta):
    default_state = ca.vertcat(
        ca.DM([0., 0., 0., 1.]),  # Quaternion (orientation)
        ca.DM([0., 0., 0.]),      # Position
        ca.vertcat(               # Velocity
            u_from_aero(q_bar, alpha, beta),
            v_from_aero(q_bar, alpha, beta),
            w_from_aero(q_bar, alpha, beta)
        ),
        ca.DM([0., 0., 0.])       # Angular velocity
    )

    return default_state