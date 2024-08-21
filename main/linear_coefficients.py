"""
Determines linear coefficients for the aerodynamic model by linear regression:

We neglect terms due to angle rates due to lacking dynamic data. 
No rudder deflection data is available, so its contribution is also neglected.

Cl = Cl_0 + Cl_alpha * alpha + Cl_beta * beta + Cl_delta_a * delta_a + Cl_delta_e * delta_e

NOTE: Alpha, beta in degrees

"""



import pandas as pd

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)

DATAPATH = os.path.join(BASEPATH, 'data')
CFDPATH = os.path.join(DATAPATH, 'cfd')
NETWORKPATH = os.path.join(DATAPATH, 'networks')
VISUPATH = os.path.join(DATAPATH, 'visualisation')

DATA_DIR = os.path.join(BASEPATH, 'data')

import numpy as np
from scipy.optimize import minimize

# def objective(params, alpha_data, CL_data):
#     # Extract parameters
#     stall_angle_positive, stall_angle_negative, max_stall_deviation, M, CL_alpha_temp, CL_0 = params
    
#     # Compute the lift coefficient using the provided function
#     predicted_CL = CL_alpha(alpha_data, stall_angle_positive, stall_angle_negative, max_stall_deviation, M, CL_alpha_temp, CL_0)
    
#     # Compute the sum of squared errors
#     error = np.sum((predicted_CL - CL_data) ** 2)
#     return error

# def CL_alpha(alpha, stall_angle_positive, stall_angle_negative, max_stall_deviation, M, CL_alpha_temp, CL_0):
#     alpha_clamped = np.maximum(stall_angle_negative - max_stall_deviation, 
#                         np.minimum(stall_angle_positive + max_stall_deviation, 
#                                 alpha))

#     sigmoid = (1 + np.exp(-M * (alpha_clamped - stall_angle_positive)) + \
#                     np.exp(M * (alpha_clamped - stall_angle_negative))) / \
#                 (1 + np.exp(-M * (alpha_clamped - stall_angle_positive))) \
#                 / (1 + np.exp(M * (alpha_clamped - stall_angle_negative)))
    
#     linear = (1.0 - sigmoid) * (CL_0 + CL_alpha_temp * alpha)  

#     flat_plate = sigmoid * (2 * np.sign(alpha) 
#                             * np.sin(alpha)**2 
#                             * np.cos(alpha))

#     CL_alpha = linear + flat_plate
#     return CL_alpha


def CL_alpha(alpha, coefficient_array):
    """
    Computes the lift coefficient of the aircraft as a function of the 
    angle of attack.

    Uses a thin airfoil model for the lift coefficient, 
    which includes a sigmoid function to model stall behaviour.

    CL_0: Zero angle of attack lift coefficient
    CL_alpha: Lift coefficient slope with respect to angle of attack
    M: Sigmoid function parameter

    """
    stall_angle_positive = np.deg2rad(15)
    stall_angle_negative = np.deg2rad(-15)
    max_stall_deviation = 0.3
    M = 15 # find good value to fit data
    CL_alpha_temp = coefficient_array[2, 0]#['alpha']['CL']
    CL_0 = coefficient_array[2, 4] # ['bias']['CL']

    alpha_clamped = np.maximum(stall_angle_negative - max_stall_deviation, 
                        np.minimum(stall_angle_positive + max_stall_deviation, 
                                alpha))

    
    #use casadi to compute the sigmoid function
    sigmoid = (1 + np.exp(-M * (alpha_clamped - stall_angle_positive)) + \
                    np.exp(M * (alpha_clamped - stall_angle_negative))) / \
                (1 + np.exp(-M * (alpha_clamped - stall_angle_positive))) \
                / (1 + np.exp(M * (alpha_clamped - stall_angle_negative)))
    
    print(sigmoid)
    # Lift is linear in alpha at small alpha 
    linear = (1.0 - sigmoid) * (CL_0 + CL_alpha_temp * alpha)  

    print(linear)
    # Flat plate function
    flat_plate = sigmoid * (2 * np.sign(alpha) 
                            * np.sin(alpha)**2 
                            * np.cos(alpha))

    CL_alpha = linear + flat_plate
    return CL_alpha

if __name__ == '__main__':

    data_real = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_real.csv'))
    data_sim = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_sim.csv'))

    inputs = ['alpha', 'beta', 'aileron', 'elevator']
    outputs = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']

    data = pd.concat([data_real, data_sim], axis=0)
    # data['alpha'] = np.deg2rad(data['alpha'])
    # data['beta'] = np.deg2rad(data['beta'])
    print(data.head())

    coeff_array = np.zeros((len(outputs), len(inputs) + 1))

    for output in outputs:
        print(f'Computing linear coefficients for {output}')

        reg = LinearRegression().fit(np.array(data[inputs].values), np.array(data[output].values).reshape(-1,1))
        coeff_array[outputs.index(output), :-1] = reg.coef_
        coeff_array[outputs.index(output), -1] = reg.intercept_[0]


    df = pd.DataFrame(coeff_array, columns=inputs + ['bias'], index=outputs)

    coeff_array[outputs.index('CD'), -1] = data['CD'].min()
    print(df)

    df.to_csv(os.path.join(DATA_DIR, 'glider', 'linear_coefficients.csv'))

    initial_guess = [np.deg2rad(20), np.deg2rad(-20), 0.7, 10, 0.1, 0.0]
    bounds = [(np.deg2rad(10), np.deg2rad(30)),  # stall_angle_positive bounds
            (np.deg2rad(-30), np.deg2rad(-10)),  # stall_angle_negative bounds
            (0.5, 1.0),  # max_stall_deviation bounds
            (1, 20),  # M bounds
            (0.01, 1.0),  # CL_alpha_temp bounds
            (-1.0, 1.0)]  # CL_0 bounds


    sample_data = data.sample(500)
    alpha_min = sample_data['alpha'].min()
    alpha_max = sample_data['alpha'].max()

    beta_min = sample_data['beta'].min()
    beta_max = sample_data['beta'].max()
    

    # plot hyperplanes for each output
    fig = plt.figure(figsize=(15, 10))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        output = outputs[i]
        x = np.linspace(alpha_min, alpha_max, 10)
        y = np.linspace(beta_min, beta_max, 10)
        X, Y = np.meshgrid(x, y)

        if i == 0:
            induced = (coeff_array[2, 0] * X)**2 / (np.pi * 0.3 * (2.0 ** 2 / 0.238))
            Z = induced + coeff_array[0, 1] * Y + coeff_array[0, 2] * 0 + coeff_array[0, 3] * 0 + coeff_array[0, 4]

        elif i == 2:
            Z = CL_alpha(X, coeff_array) + coeff_array[i, 1] * Y + coeff_array[i, 2] * 0 + coeff_array[i, 3] * 0 + coeff_array[i, 4]
        else:
            Z = coeff_array[i, 0] * X + coeff_array[i, 1] * Y + coeff_array[i, 2] * 0 + coeff_array[i, 3] * 0 + coeff_array[i, 4]
        ax.plot_surface(X, Y, Z)
        
        ax.scatter(sample_data['alpha'], sample_data['beta'], sample_data[output], label='real', color='red')
        ax.set_title(output)
        ax.set_xlabel(inputs[0])
        ax.set_ylabel(inputs[1])
    
    plt.tight_layout()
    # plt.savefig(os.path.join(VISUPATH, 'linear_coefficients.png'))
    plt.show()

    # # plot residuals
    # fig = plt.figure(figsize=(15, 10))
    # for i in range(6):
    #     ax = fig.add_subplot(2, 3, i+1)
    #     output = outputs[i]
    #     ax.scatter(data[output], coeff_array[i, 0] * data['alpha'] + coeff_array[i, 1] * data['beta'] + coeff_array[i, 2] * 0 + coeff_array[i, 3] * 0 + coeff_array[i, 4] - data[output])
    #     ax.set_title(output)
    #     ax.set_xlabel('Real Value')
    #     ax.set_ylabel('Residual')

    # plt.tight_layout()
    # # plt.savefig(os.path.join(VISUPATH, 'residuals.png'))
    # plt.show()

    # # plot correlation matrix
    # corr = data[inputs + outputs].corr()
    # plt.matshow(corr)
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    # plt.yticks(range(len(corr.columns)), corr.columns)
    # plt.colorbar()
    # plt.tight_layout()
    # # plt.savefig(os.path.join(VISUPATH, 'correlation_matrix.png'))

    # plt.show()






