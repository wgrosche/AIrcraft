import casadi as ca
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import pandas as pd
import os
import json
import sys
from torch.utils.data import random_split

from sklearn.preprocessing import StandardScaler

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]

sys.path.append(BASEPATH)

from aircraft.surrogates.dataloader import AeroDataset
from aircraft.plotting.plotting import create_grid

# # set plt to inline
# %matplotlib ipympl

DATAPATH = os.path.join(BASEPATH, 'data')
NETWORKPATH = os.path.join(DATAPATH, 'networks')
VISUPATH = os.path.join(DATAPATH, 'visualisation')

BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 2
LEARNING_RATE = 0.1
MOMENTUM = 0.7
SCALING = False
DATA_DIR = os.path.join(BASEPATH, 'data', 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

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

def create_casadi_function(output, fitted_models, input_features, used_features):
    coef = fitted_models[output]['coef']
    intercept = fitted_models[output]['intercept']
    poly = fitted_models[output]['poly']

    casadi_vars = {feat: ca.MX.sym(feat) for feat in input_features}
    casadi_input = ca.vertcat(*casadi_vars.values())

    # Generate polynomial terms with interactions
    terms = poly.get_feature_names_out(used_features)  # e.g., ['q', 'q^2', 'alpha beta']
    symbolic_expr = intercept
    for term, weight in zip(terms, coef):

        # if abs(weight) < 1e-10:  # Suppress terms below the threshold
        #     continue
        # Parse polynomial terms into CasADi expressions
        term_expr = 1
        for var_power in term.split(' '):  # Split terms like "alpha^2 beta"
            if '^' in var_power:
                var, power = var_power.split('^')
                term_expr *= casadi_vars[var] ** int(power)
            elif var_power in casadi_vars:
                term_expr *= casadi_vars[var_power]
        symbolic_expr += weight * term_expr

    # Create CasADi function
    casadi_func = ca.Function(output, [casadi_input], [symbolic_expr])

    return casadi_func

def create_interactive_aero_plot(data, casadi_functions, 
                                 input_features=['q', 'alpha', 'beta', 'aileron', 'elevator'], 
                                 output_features=['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']):
    fig = plt.figure(figsize=(18, 10))
    plt.subplots_adjust(bottom=0.2)
    
    # Create grid for input features (q, alpha, beta) and discretize aileron/elevator
    grid_alpha_beta = create_grid(data[input_features[:3]], 10)  # Grid for q, alpha, beta
    aileron_values = np.linspace(data['aileron'].min(), data['aileron'].max(), 10)
    elevator_values = np.linspace(data['elevator'].min(), data['elevator'].max(), 10)
    
    # Precompute aerodynamic outputs
    precomputed_outputs = {output: {} for output in output_features}

    for aileron in aileron_values:
        for elevator in elevator_values:
            temp_grid = grid_alpha_beta.copy()
            temp_grid['aileron'] = aileron
            temp_grid['elevator'] = elevator
            key = (aileron, elevator)
            
            for output in output_features:
                y_pred = casadi_functions[output](temp_grid.values.T).full().flatten()
                precomputed_outputs[output][key] = y_pred

    # Initialize 3D scatter plots
    scatter_plots = []
    training_data_scatter = []
    for i, output in enumerate(output_features):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        scatter = ax.scatter(grid_alpha_beta['alpha'], grid_alpha_beta['beta'], 
                             precomputed_outputs[output][(aileron_values[0], elevator_values[0])], 
                             marker='o', label='sim')
        scatter_plots.append(scatter)
        
        # Additional scatter for training data
        train_scatter = ax.scatter([], [], [], marker='x', color='red', label='training data')
        training_data_scatter.append(train_scatter)
        
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel(output)
        ax.legend()
    
    # Create sliders
    ax_aileron = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_elevator = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    s_aileron = Slider(ax_aileron, 'Aileron', data['aileron'].min(), data['aileron'].max(), valinit=0)
    s_elevator = Slider(ax_elevator, 'Elevator', data['elevator'].min(), data['elevator'].max(), valinit=0)
    
    # def update(val):
    #     # Find nearest precomputed grid values
    #     aileron_idx = np.argmin(np.abs(aileron_values - s_aileron.val))
    #     elevator_idx = np.argmin(np.abs(elevator_values - s_elevator.val))
    #     key = (aileron_values[aileron_idx], elevator_values[elevator_idx])
        
    #     for scatter, train_scatter, output in zip(scatter_plots, training_data_scatter, output_features):
    #         # Update precomputed outputs
    #         y_pred = precomputed_outputs[output][key]
    #         scatter._offsets3d = (grid_alpha_beta['alpha'], 
    #                               grid_alpha_beta['beta'], 
    #                               y_pred)
            
    #         # Update training data scatter points
    #         mask = (
    #             (np.abs(data['aileron'] - aileron_values[aileron_idx]) < 1e-0) &
    #             (np.abs(data['elevator'] - elevator_values[elevator_idx]) < 1e-0)
    #         )
    #         train_alpha = data.loc[mask, 'alpha']
    #         train_beta = data.loc[mask, 'beta']
    #         train_output = data.loc[mask, output]
            
    #         train_scatter._offsets3d = (train_alpha, train_beta, train_output)

    def update(val):
        # Find nearest precomputed grid values
        aileron_idx = np.argmin(np.abs(aileron_values - s_aileron.val))
        elevator_idx = np.argmin(np.abs(elevator_values - s_elevator.val))
        key = (aileron_values[aileron_idx], elevator_values[elevator_idx])
        
        for ax, scatter, train_scatter, output in zip(fig.axes, scatter_plots, training_data_scatter, output_features):
            # Update precomputed outputs
            y_pred = precomputed_outputs[output][key]
            scatter._offsets3d = (grid_alpha_beta['alpha'], 
                                grid_alpha_beta['beta'], 
                                y_pred)
            
            # Update training data scatter points
            mask = (
                (np.abs(data['aileron'] - aileron_values[aileron_idx]) < 1e-0) &
                (np.abs(data['elevator'] - elevator_values[elevator_idx]) < 1e-0)
            )
            train_alpha = data.loc[mask, 'alpha']
            train_beta = data.loc[mask, 'beta']
            train_output = data.loc[mask, output]
            
            train_scatter._offsets3d = (train_alpha, train_beta, train_output)
            
            # Adjust axis limits to fit new data
            ax.set_xlim(grid_alpha_beta['alpha'].min(), grid_alpha_beta['alpha'].max())
            ax.set_ylim(grid_alpha_beta['beta'].min(), grid_alpha_beta['beta'].max())
            ax.set_zlim(y_pred.min(), y_pred.max())
        
        fig.canvas.draw_idle()


    s_aileron.on_changed(update)
    s_elevator.on_changed(update)
    update(None)
    plt.show(block=True)




# def create_interactive_aero_plot(data, casadi_functions, 
#                                  input_features=['q', 'alpha', 'beta', 'aileron', 'elevator'], 
#                                  output_features=['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']):
#     fig = plt.figure(figsize=(18, 10))
#     plt.subplots_adjust(bottom=0.2)
    
#     # Create grid for input features (q, alpha, beta) and discretize aileron/elevator
#     grid_alpha_beta = create_grid(data[input_features[:3]], 10)  # Grid for q, alpha, beta
#     aileron_values = np.linspace(data['aileron'].min(), data['aileron'].max(), 10)
#     elevator_values = np.linspace(data['elevator'].min(), data['elevator'].max(), 10)
    
#     # Precompute aerodynamic outputs
#     precomputed_outputs = {output: {} for output in output_features}

#     for aileron in aileron_values:
#         for elevator in elevator_values:
#             temp_grid = grid_alpha_beta.copy()
#             temp_grid['aileron'] = aileron
#             temp_grid['elevator'] = elevator
#             key = (aileron, elevator)
            
#             for output in output_features:
#                 y_pred = casadi_functions[output](temp_grid.values.T).full().flatten()
#                 precomputed_outputs[output][key] = y_pred

#     # Initialize 3D scatter plots
#     scatter_plots = []
#     for i, output in enumerate(output_features):
#         ax = fig.add_subplot(2, 3, i + 1, projection='3d')
#         scatter = ax.scatter(data['alpha'], data['beta'], 
#                              data[output], marker='o', label='sim')
#         scatter_plots.append((scatter, output))
#         ax.set_xlabel('alpha')
#         ax.set_ylabel('beta')
#         ax.set_zlabel(output)
#         ax.legend()
    
#     # Create sliders
#     ax_aileron = plt.axes([0.2, 0.1, 0.6, 0.03])
#     ax_elevator = plt.axes([0.2, 0.05, 0.6, 0.03])
    
#     s_aileron = Slider(ax_aileron, 'Aileron', data['aileron'].min(), data['aileron'].max(), valinit=0)
#     s_elevator = Slider(ax_elevator, 'Elevator', data['elevator'].min(), data['elevator'].max(), valinit=0)
    
#     def update(val):
#         # Find nearest precomputed grid values
#         aileron_idx = np.argmin(np.abs(aileron_values - s_aileron.val))
#         elevator_idx = np.argmin(np.abs(elevator_values - s_elevator.val))
#         key = (aileron_values[aileron_idx], elevator_values[elevator_idx])
        
#         for scatter, output in scatter_plots:
#             y_pred = precomputed_outputs[output][key]
#             scatter._offsets3d = (grid_alpha_beta['alpha'], 
#                                   grid_alpha_beta['beta'], 
#                                   y_pred)
            
#         fig.canvas.draw_idle()

#     s_aileron.on_changed(update)
#     s_elevator.on_changed(update)
#     update(None)
#     plt.show(block=True)


def main():
    input_features = ['q','alpha','beta','aileron','elevator']
    output_features = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']
    used_features = ['alpha','beta','aileron','elevator']
    (train_dataset, test_dataset), scaler, output_scaler = prepare_datasets(SCALING, input_features, output_features)

    train_data = train_dataset.dataset.data

    poly_order = 3  # Polynomial order

    # Polynomial feature generation
    poly = PolynomialFeatures(degree=poly_order, include_bias=False)
    X = train_data[used_features].values
    X_poly = poly.fit_transform(X)

    # Fit multivariate polynomial models for each output
    fitted_models = {}
    for output in output_features:
        Y = train_data[output].values
        model = LinearRegression()
        model.fit(X_poly, Y)
        fitted_models[output] = {
            'model': model,
            'poly': poly,
            'coef': model.coef_,
            'intercept': model.intercept_,
        }

    # CasADi symbolic variables
    casadi_vars = {feat: ca.MX.sym(feat) for feat in input_features}
    casadi_input = ca.vertcat(*casadi_vars.values())

    # Create CasADi symbolic functions
    casadi_functions = {}
    for output in output_features:
        casadi_functions[output] = create_casadi_function(output, fitted_models, input_features, used_features)

    # Create interactive plot
    create_interactive_aero_plot(train_data, casadi_functions)


    # Save the fitted models and CasADi functions
    with open(os.path.join(NETWORKPATH, 'fitted_models_casadi.pkl'), 'wb') as file:
        pickle.dump({'fitted_models': fitted_models, 'casadi_functions': casadi_functions}, file)

if __name__ == "__main__":
    main()
