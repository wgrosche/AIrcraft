"""
Linear Coefficients Generator for Aircraft Aerodynamic Model

This script generates linearized aerodynamic coefficients for a glider aircraft model.
It processes aerodynamic data to create linear regression models for various force
and moment coefficients (CX, CY, CZ, Cl, Cm, Cn).

The script:
1. Loads aerodynamic data from processed datasets
2. Fits linear regression models to each coefficient
3. Creates a special model for CX based on drag polar (CD0 + k*CL^2)
4. Exports the linearized coefficients to JSON and CSV files

The linearized models provide a simplified representation of the aircraft's
aerodynamic behavior that can be used in flight simulation or control system design.

Usage:
    python linear_coefficients.py

Configuration:
    - Set RETRAIN=True to retrain models, False to use existing models
    - Adjust SCALING, BATCH_SIZE, EPOCHS, etc. as needed
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.animation import FuncAnimation
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)

from aircraft.surrogates.models import ScaledModel, WeightedMSELoss
from aircraft.surrogates.dataloader import AeroDataset
from aircraft.plotting.plotting import create_grid

from scipy.optimize import minimize


DATAPATH = os.path.join(BASEPATH, 'data')
NETWORKPATH = os.path.join(DATAPATH, 'networks')
VISUPATH = os.path.join(DATAPATH, 'visualisation')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 2
LEARNING_RATE = 0.1
MOMENTUM = 0.7
SCALING = False
DATA_DIR = os.path.join(BASEPATH, 'data', 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

RETRAIN = False # Flag to retrain the model, otherwise just loads existing
# model for plotting

# Determine max number of workers for dataloader
num_workers = 0# if DEVICE == 'cpu' else 4

def prepare_datasets(scaling=False, input_features=None, output_features=None):
    if input_features is None:
        input_features = ['q','alpha','beta','ailerons','elevator']
    if output_features is None:
        output_features = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']
    
    scaler, output_scaler = (StandardScaler(), StandardScaler()) if scaling else (None, None)
    dataset = AeroDataset(DATA_DIR, input_features, output_features, scaler, output_scaler)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size]), scaler, output_scaler


def main():
    print(f'Using device: {DEVICE}')

    input_features = ['q','alpha','beta','aileron','elevator']
    output_features = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']

    (train_dataset, test_dataset), scaler, output_scaler = prepare_datasets(SCALING, input_features, output_features)

    print(train_dataset.dataset.data.head())

    dataset = train_dataset.dataset.data
    # Example: Constructing the feature matrix for CX
    X = np.column_stack([
        dataset['q'],
        dataset['alpha'],
        dataset['beta'],
        dataset['aileron'],
        dataset['elevator'],
    ])


    # Output vector (target)
    y_CY = dataset['CY']

    def model_CZ(dataset) -> LinearRegression:
        X = np.column_stack([
                dataset['q'],
                dataset['alpha'],
                dataset['beta'],
                dataset['aileron'],
                dataset['elevator'],
            ])
        CZ = dataset['CZ']

        reg_CZ = LinearRegression()
        reg_CZ.fit(X, CZ)

        return reg_CZ

    def model_CX(dataset):
        X = np.column_stack([
                dataset['q'],
                dataset['alpha'],
                dataset['beta'],
                dataset['aileron'],
                dataset['elevator'],
            ])
        
        CX = dataset['CX']
        CL_model = model_CZ(dataset)
        CL = CL_model.predict(X)

        def objective(k, CX, CL):
            return np.var(CX - k * CL ** 2)

        k_initial = 0.5

        result = minimize(objective, k_initial, args=(CX, CL))

        CD0 = np.mean(CX - result.x[0] * CL ** 2)

        return {'CD0' : CD0, 'k' : result.x[0]}

    def model_c_coeff(dataset, coeff):
        X = np.column_stack([
                dataset['q'],
                dataset['alpha'],
                dataset['beta'],
                dataset['aileron'],
                dataset['elevator'],
            ])
        data = dataset[coeff]

        reg = LinearRegression()
        reg.fit(X, data)

        return reg
    
    
    tabulated_coeffs = {}
    coeff_dframe = pd.DataFrame(np.zeros((len(output_features), len(input_features) + 1)), columns = input_features + ['intercept'])
    for i, feature in enumerate(output_features):
        reg = model_c_coeff(dataset, feature)

        tabulated_coeffs[feature] = {'coefs' : list(reg.coef_), 'intercept' : reg.intercept_}
        coeff_dframe.iloc[i, :-1] = list(reg.coef_)
        coeff_dframe.iloc[i, -1] = reg.intercept_
    tabulated_coeffs['CX'] = model_CX(dataset)
    print(tabulated_coeffs)
    with open(os.path.join(DATAPATH, 'glider', 'linearised.json'), 'w') as f:
        json.dump(tabulated_coeffs, f)
    coeff_dframe.to_csv(os.path.join(DATAPATH, 'glider', 'linearised.csv'), index=None)


if __name__ == '__main__':
    main()
