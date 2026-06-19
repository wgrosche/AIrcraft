"""
main/train_nn_surrogate.py
------------------
Train and evaluate a neural network surrogate model for aircraft aerodynamics.
Includes data preparation, model training, evaluation, and visualization routines.
"""
# Standard library imports
import os
import sys
import json
import pickle

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Project imports
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)
from aircraft.surrogates.models import ScaledModel, WeightedMSELoss
from aircraft.surrogates.dataloader import AeroDataset
from aircraft.plotting.plotting import create_grid

# ----------------------
# Configuration
# ----------------------
DATAPATH = os.path.join(BASEPATH, 'data')
NETWORKPATH = os.path.join(DATAPATH, 'networks')
VISUPATH = os.path.join(DATAPATH, 'visualisation')
DATA_DIR = os.path.join(BASEPATH, 'data', 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 2
LEARNING_RATE = 0.1
MOMENTUM = 0.7
SCALING = False
RETRAIN = True  # Set True to retrain, False to load existing model
num_workers = 0  # Set >0 for GPU, 0 for CPU

# ----------------------
# Data Preparation
# ----------------------
def prepare_datasets(scaling=False, input_features=None, output_features=None):
    """
    Prepare train/test datasets and optional scalers.
    Args:
        scaling (bool): Whether to use StandardScaler for inputs/outputs.
        input_features (list): List of input feature names.
        output_features (list): List of output feature names.
    Returns:
        tuple: (train_dataset, test_dataset), input_scaler, output_scaler
    """
    if input_features is None:
        input_features = ['q', 'alpha', 'beta', 'ailerons', 'elevator']
    if output_features is None:
        output_features = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']
    scaler, output_scaler = (StandardScaler(), StandardScaler()) if scaling else (None, None)
    dataset = AeroDataset(DATA_DIR, input_features, output_features, scaler, output_scaler)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size]), scaler, output_scaler

# ----------------------
# Training and Validation
# ----------------------
def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one epoch.
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
    Returns:
        float: Average epoch loss
    """
    epoch_loss = 0.0
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(inputs, outputs, targets)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(train_loader)

def validate(model, test_loader, criterion):
    """
    Evaluate the model on the validation set.
    Args:
        model: PyTorch model
        test_loader: DataLoader for validation data
        criterion: Loss function
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(inputs, outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def plot_static_3d(inputs, targets, outputs, x_plotting, y_plotting, casadi_functions, output_features, plotting_features, visupath):
    """
    Plot static 3D scatter plots for each output feature comparing true, predicted, and polynomial model values.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 10))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(inputs[:, 1], inputs[:, 2], targets[:, i], label='True', color='b')
        ax.scatter(inputs[:, 1], inputs[:, 2], outputs[:, i], label='Predicted', color='r')
        ax.scatter(x_plotting.iloc[:, 1], x_plotting.iloc[:, 2], y_plotting[:, i])
        ax.scatter(x_plotting.iloc[:, 1], x_plotting.iloc[:, 2], casadi_functions[output_features[i]](x_plotting.values.T).full().flatten(), label='polymodel')
        ax.scatter(inputs[:, 1], inputs[:, 2], casadi_functions[output_features[i]](inputs.T).full().flatten(), label='polymodel_inputs')
        ax.set_xlabel(r"Angle of Attack ($\alpha$ [rad])")
        ax.set_ylabel(r"Angle of Sideslip ($\beta$ [rad])")
        ax.set_zlabel(plotting_features[i])
        ax.set_title(plotting_features[i])
        ax.legend()
    fig.suptitle("Predicted Aerodynamic Coefficients")
    fig.tight_layout()
    plt.savefig(os.path.join(visupath, 'predictions.png'))
    plt.show(block=True)

def plot_animated_3d(inputs, targets, model, create_grid, inp_grid, casadi_functions, output_features, visupath, device):
    """
    Create and save an animated 3D plot for varying aileron and elevator values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    fig = plt.figure(figsize=(15, 10))
    axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]
    def update(frame):
        aileron_value, elevator_value = frame
        x_plotting = create_grid(inp_grid, 10)
        x_plotting['aileron'] = aileron_value
        x_plotting['elevator'] = elevator_value
        x_plotting = x_plotting.drop_duplicates()
        x_plotting_inp = torch.tensor(x_plotting.astype('float32').values).to(device)
        y_plotting = model(x_plotting_inp).detach().cpu().numpy()
        mask = (inputs[:, 3] == aileron_value) & (inputs[:, 4] == elevator_value)
        plot_vals = np.where(mask[:, np.newaxis], targets, np.nan)
        for i, ax in enumerate(axes):
            ax.cla()
            ax.scatter(inputs[:, 1], inputs[:, 2], plot_vals[:, i], label='True', color='r')
            ax.scatter(x_plotting.iloc[:, 1], x_plotting.iloc[:, 2], y_plotting[:, i], label='New Prediction', color='b')
            ax.scatter(x_plotting.iloc[:, 1], x_plotting.iloc[:, 2], casadi_functions[output_features[i]](x_plotting.values.T).full().flatten(), label='polymodel')
            ax.scatter(inputs[:, 1], inputs[:, 2], casadi_functions[output_features[i]](inputs.T).full().flatten(), label='polymodel_inputs')
            ax.set_xlabel('alpha')
            ax.set_ylabel('beta')
            ax.set_zlabel(output_features[i])
            ax.legend()
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(-0.4, 0.4)
            ax.set_zlim(np.min(casadi_functions[output_features[i]](x_plotting.values.T).full().flatten()),
                        np.max(casadi_functions[output_features[i]](x_plotting.values.T).full().flatten()))
        fig.suptitle(f"Aileron: {aileron_value}, Elevator: {elevator_value}")
    aileron_values = np.linspace(-5, 5, num=10)
    elevator_values = np.linspace(-5, 5, num=10)
    frames = [(aileron, elevator) for aileron in aileron_values for elevator in elevator_values]
    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    ani.save(os.path.join(visupath, 'predictions_animation.gif'), writer='imagemagick')
    plt.show()

def plot_r2_bar(targets, outputs, output_features, visupath):
    """
    Plot and save a bar chart of R2 scores for each output feature.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    residuals = targets - outputs
    fig = plt.figure(figsize=(15, 10))
    r2_scores = [1 - np.sum(residuals[:, i] ** 2) / np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2) for i in range(6)]
    plt.bar(output_features, r2_scores)
    plt.xlabel('Output')
    plt.ylabel('R2 Score')
    plt.title('R2 Score for each output')
    plt.savefig(os.path.join(visupath, 'r2_scores.png'))
    plt.show(block=True)


# ----------------------
# Main Routine
# ----------------------
def main():
    """
    Main training, evaluation, and visualization routine.
    """
    print(f'Using device: {DEVICE}')
    input_features = ['q', 'alpha', 'beta', 'aileron', 'elevator']
    output_features = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']
    (train_dataset, test_dataset), scaler, output_scaler = prepare_datasets(SCALING, input_features, output_features)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Fit scalers for normalization
    inp_scaler = StandardScaler()
    out_scaler = StandardScaler()
    train_data = train_dataset.dataset.data
    inp_scaler.fit(train_data[input_features].values)
    out_scaler.fit(train_data[output_features].values)
    scaler = (inp_scaler.mean_, inp_scaler.scale_, out_scaler.mean_, out_scaler.scale_)

    model = ScaledModel(len(input_features), len(output_features), scaler).to(DEVICE)
    criterion = WeightedMSELoss(out_scaler.scale_)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    if RETRAIN:
        best_loss = float('inf')
        epochs_no_improve = 0
        val_losses = []
        train_losses = []
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            if epoch % 3 == 0:
                val_loss = validate(model, test_loader, criterion)
                val_losses.append(val_loss)
                train_losses.append(train_loss)
                print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    print(f'Saving model for epoch {epoch+1} with loss {val_loss:.4f}')
                    print(f'Current training loss: {train_loss:.4f}')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'input_mean': model.input_mean,
                        'input_std': model.input_std,
                        'output_mean': model.output_mean,
                        'output_std': model.output_std
                    }, os.path.join(NETWORKPATH, 'model-dynamics.pth'))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == PATIENCE:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
    else:
        checkpoint = torch.load(os.path.join(NETWORKPATH, 'model-dynamics.pth'), map_location=DEVICE, weights_only=True)
        scaler = checkpoint['input_mean'], checkpoint['input_std'], checkpoint['output_mean'], checkpoint['output_std']
        model = ScaledModel(5, 6, scaler=scaler).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        val_losses = []
        train_losses = []

    # Gather predictions and targets for evaluation
    inputs, targets, outputs = [], [], []
    for inp, targ in test_loader:
        inp, targ = inp.to(DEVICE), targ.to(DEVICE)
        outp = model(inp).detach().cpu().numpy()
        inp, targ = inp.detach().cpu().numpy(), targ.detach().cpu().numpy()
        inputs.append(inp)
        targets.append(targ)
        outputs.append(outp)
    inputs = np.concatenate(inputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)

    # Load casadi polynomial models for comparison
    with open(os.path.join(NETWORKPATH, 'fitted_models_casadi.pkl'), 'rb') as file:
        casadi_dict = pickle.load(file)
        casadi_functions = casadi_dict['casadi_functions']

    # Prepare meshgrid for plotting
    inp_grid = pd.DataFrame(inputs, columns=input_features)
    x_plotting = create_grid(inp_grid, 10)
    plotting_features = [r"$C_X$", r"$C_Y$", r"$C_Z$", r"$C_L$", r"$C_M$", r"$C_N$"]
    x_plotting['aileron'] = 0
    x_plotting['elevator'] = 0
    x_plotting['q'] = 60
    x_plotting = x_plotting.drop_duplicates()
    x_plotting_inp = torch.tensor(x_plotting.astype('float32').values).to(DEVICE)
    y_plotting = model(x_plotting_inp).detach().cpu().numpy()

    # Call plotting routines
    plot_static_3d(inputs, targets, outputs, x_plotting, y_plotting, casadi_functions, output_features, plotting_features, VISUPATH)
    plot_animated_3d(inputs, targets, model, create_grid, inp_grid, casadi_functions, output_features, VISUPATH, DEVICE)
    plot_r2_bar(targets, outputs, output_features, VISUPATH)

if __name__ == '__main__':
    main()
