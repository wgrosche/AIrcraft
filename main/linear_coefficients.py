# """
# Determines linear coefficients for the aerodynamic model by linear regression:

# We neglect terms due to angle rates due to lacking dynamic data. 
# No rudder deflection data is available, so its contribution is also neglected.

# Cl = Cl_0 + Cl_alpha * alpha + Cl_beta * beta + Cl_delta_a * delta_a + Cl_delta_e * delta_e

# NOTE: Alpha, beta in degrees

# """



# import pandas as pd

# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import json
# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.optimize import minimize
# from sklearn.linear_model import LinearRegression

# BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
# sys.path.append(BASEPATH)

# DATAPATH = os.path.join(BASEPATH, 'data')
# CFDPATH = os.path.join(DATAPATH, 'cfd')
# NETWORKPATH = os.path.join(DATAPATH, 'networks')
# VISUPATH = os.path.join(DATAPATH, 'visualisation')

# DATA_DIR = os.path.join(BASEPATH, 'data')

# import numpy as np
# from scipy.optimize import minimize

# # def objective(params, alpha_data, CL_data):
# #     # Extract parameters
# #     stall_angle_positive, stall_angle_negative, max_stall_deviation, M, CL_alpha_temp, CL_0 = params
    
# #     # Compute the lift coefficient using the provided function
# #     predicted_CL = CL_alpha(alpha_data, stall_angle_positive, stall_angle_negative, max_stall_deviation, M, CL_alpha_temp, CL_0)
    
# #     # Compute the sum of squared errors
# #     error = np.sum((predicted_CL - CL_data) ** 2)
# #     return error

# # def CL_alpha(alpha, stall_angle_positive, stall_angle_negative, max_stall_deviation, M, CL_alpha_temp, CL_0):
# #     alpha_clamped = np.maximum(stall_angle_negative - max_stall_deviation, 
# #                         np.minimum(stall_angle_positive + max_stall_deviation, 
# #                                 alpha))

# #     sigmoid = (1 + np.exp(-M * (alpha_clamped - stall_angle_positive)) + \
# #                     np.exp(M * (alpha_clamped - stall_angle_negative))) / \
# #                 (1 + np.exp(-M * (alpha_clamped - stall_angle_positive))) \
# #                 / (1 + np.exp(M * (alpha_clamped - stall_angle_negative)))
    
# #     linear = (1.0 - sigmoid) * (CL_0 + CL_alpha_temp * alpha)  

# #     flat_plate = sigmoid * (2 * np.sign(alpha) 
# #                             * np.sin(alpha)**2 
# #                             * np.cos(alpha))

# #     CL_alpha = linear + flat_plate
# #     return CL_alpha


# def CL_alpha(alpha, coefficient_array):
#     """
#     Computes the lift coefficient of the aircraft as a function of the 
#     angle of attack.

#     Uses a thin airfoil model for the lift coefficient, 
#     which includes a sigmoid function to model stall behaviour.

#     CL_0: Zero angle of attack lift coefficient
#     CL_alpha: Lift coefficient slope with respect to angle of attack
#     M: Sigmoid function parameter

#     """
#     stall_angle_positive = np.deg2rad(15)
#     stall_angle_negative = np.deg2rad(-15)
#     max_stall_deviation = 0.3
#     M = 15 # find good value to fit data
#     CL_alpha_temp = coefficient_array[2, 0]#['alpha']['CL']
#     CL_0 = coefficient_array[2, 4] # ['bias']['CL']

#     alpha_clamped = np.maximum(stall_angle_negative - max_stall_deviation, 
#                         np.minimum(stall_angle_positive + max_stall_deviation, 
#                                 alpha))

    
#     #use casadi to compute the sigmoid function
#     sigmoid = (1 + np.exp(-M * (alpha_clamped - stall_angle_positive)) + \
#                     np.exp(M * (alpha_clamped - stall_angle_negative))) / \
#                 (1 + np.exp(-M * (alpha_clamped - stall_angle_positive))) \
#                 / (1 + np.exp(M * (alpha_clamped - stall_angle_negative)))
    
#     print(sigmoid)
#     # Lift is linear in alpha at small alpha 
#     linear = (1.0 - sigmoid) * (CL_0 + CL_alpha_temp * alpha)  

#     print(linear)
#     # Flat plate function
#     flat_plate = sigmoid * (2 * np.sign(alpha) 
#                             * np.sin(alpha)**2 
#                             * np.cos(alpha))

#     CL_alpha = linear + flat_plate
#     return CL_alpha

# if __name__ == '__main__':

#     data_real = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_real.csv'))
#     data_sim = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'data_sim.csv'))

#     inputs = ['alpha', 'beta', 'aileron', 'elevator']
#     outputs = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']

#     data = pd.concat([data_real, data_sim], axis=0)
#     # data['alpha'] = np.deg2rad(data['alpha'])
#     # data['beta'] = np.deg2rad(data['beta'])
#     print(data.head())

#     coeff_array = np.zeros((len(outputs), len(inputs) + 1))

#     for output in outputs:
#         print(f'Computing linear coefficients for {output}')

#         reg = LinearRegression().fit(np.array(data[inputs].values), np.array(data[output].values).reshape(-1,1))
#         coeff_array[outputs.index(output), :-1] = reg.coef_
#         coeff_array[outputs.index(output), -1] = reg.intercept_[0]


#     df = pd.DataFrame(coeff_array, columns=inputs + ['bias'], index=outputs)

#     coeff_array[outputs.index('CD'), -1] = data['CD'].min()
#     print(df)

#     df.to_csv(os.path.join(DATA_DIR, 'glider', 'linear_coefficients.csv'))

#     initial_guess = [np.deg2rad(20), np.deg2rad(-20), 0.7, 10, 0.1, 0.0]
#     bounds = [(np.deg2rad(10), np.deg2rad(30)),  # stall_angle_positive bounds
#             (np.deg2rad(-30), np.deg2rad(-10)),  # stall_angle_negative bounds
#             (0.5, 1.0),  # max_stall_deviation bounds
#             (1, 20),  # M bounds
#             (0.01, 1.0),  # CL_alpha_temp bounds
#             (-1.0, 1.0)]  # CL_0 bounds


#     sample_data = data.sample(500)
#     alpha_min = sample_data['alpha'].min()
#     alpha_max = sample_data['alpha'].max()

#     beta_min = sample_data['beta'].min()
#     beta_max = sample_data['beta'].max()
    

#     # plot hyperplanes for each output
#     fig = plt.figure(figsize=(15, 10))
#     for i in range(6):
#         ax = fig.add_subplot(2, 3, i+1, projection='3d')
#         output = outputs[i]
#         x = np.linspace(alpha_min, alpha_max, 10)
#         y = np.linspace(beta_min, beta_max, 10)
#         X, Y = np.meshgrid(x, y)

#         if i == 0:
#             induced = (coeff_array[2, 0] * X)**2 / (np.pi * 0.3 * (2.0 ** 2 / 0.238))
#             Z = induced + coeff_array[0, 1] * Y + coeff_array[0, 2] * 0 + coeff_array[0, 3] * 0 + coeff_array[0, 4]

#         elif i == 2:
#             Z = CL_alpha(X, coeff_array) + coeff_array[i, 1] * Y + coeff_array[i, 2] * 0 + coeff_array[i, 3] * 0 + coeff_array[i, 4]
#         else:
#             Z = coeff_array[i, 0] * X + coeff_array[i, 1] * Y + coeff_array[i, 2] * 0 + coeff_array[i, 3] * 0 + coeff_array[i, 4]
#         ax.plot_surface(X, Y, Z)
        
#         ax.scatter(sample_data['alpha'], sample_data['beta'], sample_data[output], label='real', color='red')
#         ax.set_title(output)
#         ax.set_xlabel(inputs[0])
#         ax.set_ylabel(inputs[1])
    
#     plt.tight_layout()
#     # plt.savefig(os.path.join(VISUPATH, 'linear_coefficients.png'))
#     plt.show()

#     # # plot residuals
#     # fig = plt.figure(figsize=(15, 10))
#     # for i in range(6):
#     #     ax = fig.add_subplot(2, 3, i+1)
#     #     output = outputs[i]
#     #     ax.scatter(data[output], coeff_array[i, 0] * data['alpha'] + coeff_array[i, 1] * data['beta'] + coeff_array[i, 2] * 0 + coeff_array[i, 3] * 0 + coeff_array[i, 4] - data[output])
#     #     ax.set_title(output)
#     #     ax.set_xlabel('Real Value')
#     #     ax.set_ylabel('Residual')

#     # plt.tight_layout()
#     # # plt.savefig(os.path.join(VISUPATH, 'residuals.png'))
#     # plt.show()

#     # # plot correlation matrix
#     # corr = data[inputs + outputs].corr()
#     # plt.matshow(corr)
#     # plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
#     # plt.yticks(range(len(corr.columns)), corr.columns)
#     # plt.colorbar()
#     # plt.tight_layout()
#     # # plt.savefig(os.path.join(VISUPATH, 'correlation_matrix.png'))

#     # plt.show()






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

from src.models import ScaledModel, WeightedMSELoss
from src.dataloader import AeroDataset
from src.plotting import create_grid

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

        result = minimize(objective, k_initial, args=(CX, CL))#, bounds=[(0, None)])
        # print("k = ", result.x[0])
        # # CX -= result.x[0] * CL ** 2

        # plt.plot(dataset['alpha'], CX, 'x')
        # plt.plot(dataset['alpha'], result.x[0] * CL ** 2, 'x')
        # plt.show(block=True)

        # print("CX", CX)

        CD0 = np.mean(CX - result.x[0] * CL ** 2)

        return {'CD0' : CD0, 'k' : result.x[0]}

    def model_C_coeff(dataset, coeff):
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

        # print(reg.coef_)

        return reg
    
    # print(model_CX(dataset).coef_)
    
    tabulated_coeffs = {}
    coeff_dframe = pd.DataFrame(np.zeros((len(output_features), len(input_features) + 1)), columns = input_features + ['intercept'])
    for i, feature in enumerate(output_features):
        reg = model_C_coeff(dataset, feature)

        tabulated_coeffs[feature] = {'coefs' : list(reg.coef_), 'intercept' : reg.intercept_}
        coeff_dframe.iloc[i, :-1] = list(reg.coef_)
        coeff_dframe.iloc[i, -1] = reg.intercept_
    tabulated_coeffs['CX'] = model_CX(dataset)
    print(tabulated_coeffs)
    with open(os.path.join(DATAPATH, 'glider', 'linearised.json'), 'w') as f:
        json.dump(tabulated_coeffs, f)
    coeff_dframe.to_csv(os.path.join(DATAPATH, 'glider', 'linearised.csv'), index=None)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     # get input and output scales for normalization
#     inp_scaler = StandardScaler()
#     out_scaler = StandardScaler()

#     train_data = train_dataset.dataset.data
#     inp_scaler.fit(train_data[input_features].values)
#     out_scaler.fit(train_data[output_features].values)

#     scaler = (inp_scaler.mean_, inp_scaler.scale_, out_scaler.mean_, out_scaler.scale_)

#     model = ScaledModel(len(input_features), len(output_features), scaler).to(DEVICE)
#     criterion = WeightedMSELoss(out_scaler.scale_)
#     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
#     if RETRAIN:
#         best_loss = float('inf')
#         epochs_no_improve = 0
#         val_losses = []
#         train_losses = []
#         for epoch in range(EPOCHS):
#             train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            

#             if epoch % 3 == 0:
#                 val_loss = validate(model, test_loader, criterion)
#                 val_losses.append(val_loss)
#                 train_losses.append(train_loss)
#                 print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')

#                 if val_loss < best_loss:
#                     best_loss = val_loss
#                     epochs_no_improve = 0
#                     print(f'Saving model for epoch {epoch+1} with loss {val_loss:.4f}')
#                     print(f'Current training loss: {train_loss:.4f}')
#                     torch.save({
#                         'model_state_dict': model.state_dict(),
#                         'input_mean': model.input_mean,
#                         'input_std': model.input_std,
#                         'output_mean': model.output_mean,
#                         'output_std': model.output_std
#                     }, os.path.join(NETWORKPATH, 'model-dynamics.pth'))
#                 else:
#                     epochs_no_improve += 1
#                     if epochs_no_improve == PATIENCE:
#                         print(f'Early stopping at epoch {epoch+1}')
#                         break

#     else:
#         checkpoint = torch.load(os.path.join(NETWORKPATH, 'model-dynamics.pth'), map_location=DEVICE)
#         scaler = checkpoint['input_mean'], checkpoint['input_std'], checkpoint['output_mean'], checkpoint['output_std']
#         model = ScaledModel(5, 6, scaler=scaler).to(DEVICE)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()

#         val_losses = []
#         train_losses = []

#     # Prepare for plotting
#     inputs, targets, outputs = [], [], []
#     for inp, targ in test_loader:
#         inp, targ = inp.to(DEVICE), targ.to(DEVICE)
#         outp = model(inp).detach().cpu().numpy()
#         inp, targ = inp.detach().cpu().numpy(), targ.detach().cpu().numpy()
#         inputs.append(inp)
#         targets.append(targ)
#         outputs.append(outp)
#     inputs = np.concatenate(inputs, axis=0)
#     targets = np.concatenate(targets, axis=0)
#     outputs = np.concatenate(outputs, axis=0)

#     # create meshgrid for plotting
#     # inp_grid = pd.DataFrame(inputs, columns = ['q','alpha','beta','aileron','elevator'])
#     # x_plotting = create_grid(inp_grid, 10)
#     # x_plotting['aileron'] = 0
#     # x_plotting['elevator'] = 0
#     # x_plotting = x_plotting.drop_duplicates()
#     # x_plotting_inp = torch.tensor(x_plotting.astype('float32').values).to(DEVICE)
#     # y_plotting = model(x_plotting_inp).detach().cpu().numpy()
#     # # 3D plot for each output feature
#     # fig = plt.figure(figsize=(15, 10))
#     # for i in range(6):
#     #     ax = fig.add_subplot(2, 3, i+1, projection='3d')
#     #     ax.scatter(inputs[:,1], inputs[:,2], targets[:, i], label='True', color='b')
#     #     ax.scatter(inputs[:,1], inputs[:,2], outputs[:, i], label='Predicted', color='r')
#     #     ax.scatter(x_plotting.iloc[:,1], x_plotting.iloc[:,2], y_plotting[:,i])
#     #     ax.set_xlabel(input_features[1])
#     #     ax.set_ylabel(input_features[2])
#     #     ax.set_zlabel(output_features[i])
#     #     ax.legend()
#     # plt.savefig(os.path.join(VISUPATH, 'predictions.png'))
#     # plt.show(block=True)



#     # Assuming 'create_grid', 'model', 'DEVICE', and other variables are already defined

#     # Generate the input grid
#     inp_grid = pd.DataFrame(inputs, columns=['q', 'alpha', 'beta', 'aileron', 'elevator'])

#     # Create figure for plotting
#     fig = plt.figure(figsize=(15, 10))
#     axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

#     # Function to update the plot for each frame
#     def update(frame):
#         aileron_value = frame[0]
#         elevator_value = frame[1]

#         # Update aileron and elevator values in the grid
#         x_plotting = create_grid(inp_grid, 10)
#         x_plotting['aileron'] = aileron_value
#         x_plotting['elevator'] = elevator_value
#         x_plotting = x_plotting.drop_duplicates()
#         x_plotting_inp = torch.tensor(x_plotting.astype('float32').values).to(DEVICE)
        
#         # Predict outputs
#         y_plotting = model(x_plotting_inp).detach().cpu().numpy()
        
#         # Clear the axes and replot for each feature
#         for i, ax in enumerate(axes):
#             ax.cla()
#             ax.scatter(inputs[:, 1], inputs[:, 2], targets[:, i], label='True', color='b')
#             ax.scatter(inputs[:, 1], inputs[:, 2], outputs[:, i], label='Predicted', color='r')
#             ax.scatter(x_plotting.iloc[:, 1], x_plotting.iloc[:, 2], y_plotting[:, i], label='New Prediction', color='g')
#             ax.set_xlabel(input_features[1])
#             ax.set_ylabel(input_features[2])
#             ax.set_zlabel(output_features[i])
#             ax.legend()
        
#         fig.suptitle(f"Aileron: {aileron_value}, Elevator: {elevator_value}")

#     # Set up frames for the animation
#     aileron_values = np.linspace(-5, 5, num=20)
#     elevator_values = np.linspace(0, 0, num=1)
#     frames = [(aileron, elevator) for aileron in aileron_values for elevator in elevator_values]

#     # Create animation
#     ani = FuncAnimation(fig, update, frames=frames, repeat=False)

#     # Save the animation
#     ani.save(os.path.join(VISUPATH, 'predictions_animation.gif'), writer='imagemagick')

#     # Show the plot
#     plt.show()
    
#     # fig = plt.figure(figsize = (15, 10))
#     # plt.plot(train_losses, label='Training Loss')
#     # plt.plot(val_losses, label='Validation Loss')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Loss')
#     # plt.title('Loss over Training')
#     # plt.legend()
#     # plt.savefig(os.path.join(VISUPATH, 'loss.png'))
#     # plt.show(block=True)

#     # plot residuals
#     residuals = targets - outputs
#     # fig = plt.figure(figsize=(15, 10))
#     # for i in range(6):
#     #     ax = fig.add_subplot(2, 3, i+1)
#     #     ax.scatter(targets[:, i], residuals[:, i])
#     #     ax.set_title(output_features[i])
#     #     ax.set_xlabel('True Value')
#     #     ax.set_ylabel('Residual')
#     # plt.savefig(os.path.join(VISUPATH, 'residuals.png'))
#     # plt.show()

#     # plot target vs output
#     # fig = plt.figure(figsize=(15, 10))
#     # for i in range(6):
#     #     ax = fig.add_subplot(2, 3, i+1)
#     #     ax.scatter(targets[:, i], outputs[:, i])
#     #     ax.set_title(output_features[i])
#     #     ax.set_xlabel('True Value')
#     #     ax.set_ylabel('Predicted Value')
#     # plt.savefig(os.path.join(VISUPATH, 'target_vs_output.png'))
#     # plt.show()

#     # plot r2 scores
#     fig = plt.figure(figsize=(15, 10))
     
#     r2_scores = []
#     for i in range(6):
#         r2_scores.append(1 - np.sum(residuals[:, i] ** 2) / np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2))
#     plt.bar(output_features, r2_scores)
#     plt.xlabel('Output')
#     plt.ylabel('R2 Score')
#     plt.title('R2 Score for each output')
#     plt.savefig(os.path.join(VISUPATH, 'r2_scores.png'))
#     plt.show(block=True)


if __name__ == '__main__':
    main()
