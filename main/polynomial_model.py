import casadi as ca
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# plt.ioff()
from matplotlib.animation import FuncAnimation
import json
import sys
from torch.utils.data import random_split

from sklearn.preprocessing import StandardScaler

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]

sys.path.append(BASEPATH)

from src.models import ScaledModel, WeightedMSELoss
from src.dataloader import AeroDataset
from src.plotting import create_grid

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

# Load the training data

def create_casadi_function(output, fitted_models, input_features):
    coef = fitted_models[output]['coef']
    intercept = fitted_models[output]['intercept']
    poly = fitted_models[output]['poly']

    casadi_vars = {feat: ca.MX.sym(feat) for feat in input_features}
    casadi_input = ca.vertcat(*casadi_vars.values())

    # Generate polynomial terms with interactions
    terms = poly.get_feature_names_out(input_features)  # e.g., ['q', 'q^2', 'alpha beta']
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

# def create_interactive_aero_plot(data, casadi_functions, 
#                                  input_features = ['q','alpha','beta','aileron','elevator'], 
#                                  output_features = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']):
#     fig = plt.figure(figsize=(18, 10))
#     plt.subplots_adjust(bottom=0.2)

#     x_plotting = create_grid(data[input_features], 10)
#     x_plotting['aileron'] = 0
#     x_plotting['elevator']
#     x_plotting.drop_duplicates(inplace=True)
    
    
#     scatter_plots = []
#     for i in range(6):
#         ax = fig.add_subplot(2, 3, i+1, projection='3d')
#         # ax.set_title(f"{data.columns[i + 6]}")
#         scatter = ax.scatter(data['alpha'], data['beta'], 
#                                 data[output_features[i]], marker='o', label='sim')
        
#         scatter_plots.append((scatter, i))
#         ax.set_xlabel('alpha')
#         ax.set_ylabel('beta')
#         ax.set_zlabel(output_features[i])
#         ax.legend()
    
#     ax_aileron = plt.axes([0.2, 0.1, 0.6, 0.03])
#     ax_elevator = plt.axes([0.2, 0.05, 0.6, 0.03])
    
#     s_aileron = Slider(ax_aileron, 'Aileron', data['aileron'].min(), data['aileron'].max(), 
#                     valinit=0, dragging=True)
#     s_elevator = Slider(ax_elevator, 'Elevator', data['elevator'].min(), data['elevator'].max(), 
#                     valinit=0, dragging=True)
    
    
#     def update(val):
#         x_plotting['aileron'] = s_aileron.val
#         x_plotting['elevator'] = s_elevator.val
        
#         for scatter, i in scatter_plots:
#             y_pred = casadi_functions[output_features[i]](x_plotting.values.T).full().flatten()
#             scatter._offsets3d = (x_plotting['alpha'], x_plotting['beta'], 
#                                 y_pred)
#         fig.canvas.draw_idle()

#     s_aileron.on_changed(update)
#     s_elevator.on_changed(update)
#     update(None)
#     plt.show(block=True)

from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def create_interactive_aero_plot(data, casadi_functions, 
                                 input_features=['q','alpha','beta','aileron','elevator'], 
                                 output_features=['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']):
    fig = plt.figure(figsize=(18, 10))
    plt.subplots_adjust(bottom=0.2)
    
    # Create grid for input features (alpha, beta) and discretize aileron/elevator
    grid_alpha_beta = create_grid(data[['q','alpha','beta']], 10)  # 10x10 grid for alpha/beta
    aileron_values = np.linspace(data['aileron'].min(), data['aileron'].max(), 10)  # 5 aileron steps
    elevator_values = np.linspace(data['elevator'].min(), data['elevator'].max(), 10)  # 5 elevator steps
    
    # Precompute grid points and CasADi outputs
    precomputed_inputs = []
    precomputed_outputs = {output: [] for output in output_features}
    
    # for aileron in aileron_values:
    #     for elevator in elevator_values:
    #         temp_grid = grid_alpha_beta.copy()
    #         temp_grid['aileron'] = aileron
    #         temp_grid['elevator'] = elevator
    #         temp_grid.drop_duplicates(inplace=True)
    #         # Store inputs
    #         precomputed_inputs.append(temp_grid.values)
    #         for output in output_features:
    #             y_pred = casadi_functions[output](temp_grid.values.T).full().flatten()
    #             print(y_pred)
    #             precomputed_outputs[output].append(y_pred)

    precomputed_outputs = {output: {} for output in output_features}

    for aileron in aileron_values:
        for elevator in elevator_values:
            temp_grid = grid_alpha_beta.copy()
            temp_grid['aileron'] = aileron
            temp_grid['elevator'] = elevator
            temp_grid.drop_duplicates(inplace=True)
            precomputed_inputs.append(temp_grid.values)
            
            # Use (aileron, elevator) as a key
            key = (aileron, elevator)
            
            for output in output_features:
                y_pred = casadi_functions[output](temp_grid.values.T).full().flatten()
                precomputed_outputs[output][key] = y_pred

    # print(len(precomputed_outputs['CX'][0]))
    
    # Flatten precomputed grids for quick access
    precomputed_inputs = np.vstack(precomputed_inputs)
    tree = cKDTree(precomputed_inputs[:, -2:])  # Aileron and elevator values
    
    precomputed_outputs = {output: np.vstack(values) 
                           for output, values in precomputed_outputs.items()}

    print(len(precomputed_outputs['CX'].flatten()))    
    # Initialize 3D scatter plots
    scatter_plots = []
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        scatter = ax.scatter(data['alpha'], data['beta'], 
                             data[output_features[i]], marker='o', label='sim')
        scatter_plots.append((scatter, i))
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel(output_features[i])
        ax.legend()
    
    # Create sliders
    ax_aileron = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_elevator = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    s_aileron = Slider(ax_aileron, 'Aileron', data['aileron'].min(), data['aileron'].max(), 
                       valinit=0, dragging=True)
    s_elevator = Slider(ax_elevator, 'Elevator', data['elevator'].min(), data['elevator'].max(), 
                        valinit=0, dragging=True)
    
    def update(val):
        # Query nearest precomputed values
        query = np.array([[s_aileron.val, s_elevator.val]])
        _, idx = tree.query(query)
        
        # Retrieve nearest aileron and elevator values
        nearest_aileron = aileron_values[idx // len(elevator_values)]
        nearest_elevator = elevator_values[idx % len(elevator_values)]
        key = (nearest_aileron, nearest_elevator)
        
        for scatter, i in scatter_plots:
            y_pred = precomputed_outputs[output_features[i]][key]
            scatter._offsets3d = (grid_alpha_beta['alpha'], 
                                grid_alpha_beta['beta'], 
                                y_pred)
        fig.canvas.draw_idle()

    
    # def update(val):
    #     # Query nearest precomputed values
    #     query = np.array([[s_aileron.val, s_elevator.val]])
    #     _, idx = tree.query(query)
        
    #     for scatter, i in scatter_plots:
    #         y_pred = precomputed_outputs[output_features[i]][idx].flatten()
    #         scatter._offsets3d = (grid_alpha_beta['alpha'], 
    #                               grid_alpha_beta['beta'], 
    #                               y_pred)
    #     fig.canvas.draw_idle()

    s_aileron.on_changed(update)
    s_elevator.on_changed(update)
    update(None)
    plt.show(block=True)

def main():
    input_features = ['q','alpha','beta','aileron','elevator']
    output_features = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']
    (train_dataset, test_dataset), scaler, output_scaler = prepare_datasets(SCALING, input_features, output_features)

    train_data = train_dataset.dataset.data

    poly_order = 3  # Polynomial order

    # Polynomial feature generation
    poly = PolynomialFeatures(degree=poly_order, include_bias=False)
    X = train_data[input_features].values
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
        casadi_functions[output] = create_casadi_function(output, fitted_models, input_features)

    # Create interactive plot
    create_interactive_aero_plot(train_data, casadi_functions)


    # Save the fitted models and CasADi functions
    with open('fitted_models_casadi.pkl', 'wb') as file:
        pickle.dump({'fitted_models': fitted_models, 'casadi_functions': casadi_functions}, file)

if __name__ == "__main__":
    main()

# Visualization for a specific output, e.g., 'Cl'

# from mpl_toolkits.mplot3d import Axes3D
# # plt.ion()
# # %matplotlib ipympl

# output_to_plot = 'Cl'
# Y_true = train_data[output_to_plot].values
# Y_fitted = casadi_functions[output_to_plot](X.T).full().flatten()

# # Extract true and fitted values
# Y_true = train_data[output_to_plot].values
# Y_fitted = casadi_functions[output_to_plot](X.T).full().flatten()

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter true and fitted values over alpha and beta
# ax.scatter(X[:, 1], X[:, 3], Y_true, label='True', color='b', alpha=0.7)
# ax.scatter(X[:, 1], X[:, 3], Y_fitted, label='Fitted', color='r', alpha=0.7)

# # Add labels and legend
# ax.set_xlabel('Alpha (X[:,1])')
# # ax.set_ylabel('Beta (X[:,2])')
# ax.set_ylabel('Aileron (X[:,3])')
# ax.set_zlabel(output_to_plot)
# ax.legend()

# plt.show(block=True)

# # create meshgrid for plotting
# inp_grid = pd.DataFrame(X, columns = ['q','alpha','beta','aileron','elevator'])
# x_plotting = create_grid(inp_grid, 10)
# plotting_features = [r"$C_X$", r"$C_Y$", r"$C_Z$", r"$C_L$", r"$C_M$", r"$C_N$"]
# x_plotting['aileron'] = 0
# x_plotting['elevator'] = 0
# x_plotting['q'] = 60
# x_plotting = x_plotting.drop_duplicates()
# # 3D plot for each output feature
# fig = plt.figure(figsize=(15, 10))
# for i in range(6):
#     output_to_plot = output_features[i]
#     f_y = casadi_functions[output_features[i]](x_plotting.values.T).full().flatten()
#     Y_true = train_data[output_to_plot].values
#     Y_fitted = casadi_functions[output_to_plot](X.T).full().flatten()
#     ax = fig.add_subplot(2, 3, i+1, projection='3d')
#     ax.scatter(X[:,1], X[:,2], Y_true, label='True', color='b')
#     ax.scatter(X[:,1], X[:,2], Y_fitted, label='Predicted', color='r')
#     ax.scatter(x_plotting.iloc[:,1], x_plotting.iloc[:,2], f_y, label = 'polymodel')

#     # ax.set_xlabel(input_features[1])
#     # ax.set_ylabel(input_features[2])
#     ax.set_xlabel(r"Angle of Attack ($\alpha$ [rad])")
#     ax.set_ylabel(r"Angle of Sideslip ($\beta$ [rad])")
#     ax.set_zlabel(plotting_features[i])
#     ax.set_title(plotting_features[i])
#     ax.legend()
# fig.suptitle("Predicted Aerodynamic Coefficients")
# fig.tight_layout()
# # plt.savefig(os.path.join(VISUPATH, 'predictions.png'))
# plt.show(block=True)


