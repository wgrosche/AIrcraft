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

BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)

from src.models import ScaledModel, WeightedMSELoss
from src.dataloader import AeroDataset
from src.plotting import create_grid



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

RETRAIN = True # Flag to retrain the model, otherwise just loads existing
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

def train_one_epoch(model, train_loader, criterion, optimizer):
    epoch_loss = 0.0
    model.train()
    printed = False
    for inputs, targets in train_loader:
        if not printed:
            # print(targets)
            printed = True
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(inputs, outputs, targets)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(train_loader)

def validate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(inputs, outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def main():
    print(f'Using device: {DEVICE}')
    input_features = ['q','alpha','beta','aileron','elevator']
    output_features = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']
    (train_dataset, test_dataset), scaler, output_scaler = prepare_datasets(SCALING, input_features, output_features)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # get input and output scales for normalization
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

    # Prepare for plotting
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

    # import seaborn as sns
    # import pandas as pd

    # inp_grid = pd.DataFrame(inputs, columns = ['q','alpha','beta','aileron','elevator'])
    # x_plotting = create_grid(inp_grid, 10)
    # x_plotting['aileron'] = 0
    # x_plotting['elevator'] = 0
    # x_plotting = x_plotting.drop_duplicates()
    # x_plotting_inp = torch.tensor(x_plotting.astype('float32').values).to(DEVICE)
    # y_plotting = model(x_plotting_inp).detach().cpu().numpy()

    # # Assume inputs, targets, and outputs are already available (e.g., as tensors or numpy arrays)
    # # `x_plotting` is the meshgrid for the input features (e.g., alpha, beta)
    # # `y_plotting` is the model's predicted outputs for the meshgrid
    # # `input_features` and `output_features` are the names of the respective variables

    # # 2D scatter plots for each output feature
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    # axes = axes.ravel()

    # for i, feature in enumerate(output_features):  # Loop through each output feature (e.g., CX, CY, ...)
    #     ax = axes[i]
        
    #     # Plot true data (targets)
    #     sns.scatterplot(x=inputs[:, 1], y=targets[:, i], ax=ax, label="True", color="blue", alpha=0.6)
        
    #     # Plot predicted data
    #     sns.scatterplot(x=inputs[:, 1], y=outputs[:, i], ax=ax, label="Predicted", color="orange", alpha=0.6)
        
    #     # Plot predictions on the meshgrid
    #     # sns.scatterplot(x=x_plotting.iloc[:, 1], y=y_plotting[:, i], ax=ax, label="Linearized Model", color="green", alpha=0.6)
        
    #     # Set labels and title
    #     ax.set_title(f"{feature} vs Alpha", fontsize=14)
    #     ax.set_xlabel(input_features[1], fontsize=12)  # E.g., 'alpha'
    #     ax.set_ylabel(feature, fontsize=12)  # E.g., 'CX', 'CY', ...
    #     ax.legend()

    # # Save the figure
    # plt.savefig("2D_output_comparison.png", format="png", dpi=300)

    # # Show the plots
    # plt.show()

    # return None


    # create meshgrid for plotting
    inp_grid = pd.DataFrame(inputs, columns = ['q','alpha','beta','aileron','elevator'])
    x_plotting = create_grid(inp_grid, 10)
    plotting_features = [r"$C_X$", r"$C_Y$", r"$C_Z$", r"$C_L$", r"$C_M$", r"$C_N$"]
    x_plotting['aileron'] = 0
    x_plotting['elevator'] = 0
    x_plotting = x_plotting.drop_duplicates()
    x_plotting_inp = torch.tensor(x_plotting.astype('float32').values).to(DEVICE)
    y_plotting = model(x_plotting_inp).detach().cpu().numpy()
    # 3D plot for each output feature
    fig = plt.figure(figsize=(15, 10))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(inputs[:,1], inputs[:,2], targets[:, i], label='True', color='b')
        ax.scatter(inputs[:,1], inputs[:,2], outputs[:, i], label='Predicted', color='r')
        ax.scatter(x_plotting.iloc[:,1], x_plotting.iloc[:,2], y_plotting[:,i])
        # ax.set_xlabel(input_features[1])
        # ax.set_ylabel(input_features[2])
        ax.set_xlabel(r"Angle of Attack ($\alpha$ [rad])")
        ax.set_ylabel(r"Angle of Sideslip ($\beta$ [rad])")
        ax.set_zlabel(plotting_features[i])
        ax.set_title(plotting_features[i])
        ax.legend()
    fig.suptitle("Predicted Aerodynamic Coefficients")
    fig.tight_layout()
    plt.savefig(os.path.join(VISUPATH, 'predictions.png'))
    plt.show(block=True)

    def linearised_model(filepath, inputs, feature):
        coeff_table = json.load(filepath)
        (q, alpha, beta, aileron, elevator) = inputs

        if feature =='CX':
            CZ = np.dot(np.array(coeff_table['CZ']['coefs']), inputs.T) + coeff_table['CZ']['intercept']

            return coeff_table['CX']['k'] * CZ ** 2 + coeff_table['CX']['CD0']
        return np.dot(np.array(coeff_table[feature]['coefs']), inputs.T) + coeff_table[feature]['intercept']


    # Assuming 'create_grid', 'model', 'DEVICE', and other variables are already defined

    # Generate the input grid
    inp_grid = pd.DataFrame(inputs, columns=['q', 'alpha', 'beta', 'aileron', 'elevator'])

    # Create figure for plotting
    fig = plt.figure(figsize=(15, 10))
    axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

    # Function to update the plot for each frame
    def update(frame):
        aileron_value = frame[0]
        elevator_value = frame[1]

        # Update aileron and elevator values in the grid
        x_plotting = create_grid(inp_grid, 10)
        x_plotting['aileron'] = aileron_value
        x_plotting['elevator'] = elevator_value
        x_plotting = x_plotting.drop_duplicates()
        x_plotting_inp = torch.tensor(x_plotting.astype('float32').values).to(DEVICE)
        
        # Predict outputs
        y_plotting = model(x_plotting_inp).detach().cpu().numpy()
        mask = (inputs[:, 3] == aileron_value) & (inputs[:, 4] == elevator_value)
        plot_vals = np.where(mask[:, np.newaxis], targets, np.nan)



        
        
        # Clear the axes and replot for each feature
        for i, ax in enumerate(axes):
            ax.cla()
            ax.scatter(inputs[:, 1], inputs[:, 2], plot_vals[:, i], label='True', color='r')
            # ax.scatter(inputs[:, 1], inputs[:, 2], outputs[:, i], label='Predicted', color='r')
            ax.scatter(x_plotting.iloc[:, 1], x_plotting.iloc[:, 2], y_plotting[:, i], label='New Prediction', color='b')
            ax.scatter(x_plotting.iloc[:, 1], x_plotting.iloc[:, 2], linearised_model(open('data/glider/linearised.json'), x_plotting, output_features[i]), label='Linear', color='g')
            
            ax.set_xlabel(input_features[1])
            ax.set_ylabel(input_features[2])
            ax.set_zlabel(output_features[i])
            ax.legend()
        
        fig.suptitle(f"Aileron: {aileron_value}, Elevator: {elevator_value}")

    # Set up frames for the animation
    
    aileron_values = np.linspace(-5, 5, num=10)
    elevator_values = np.linspace(0, 0, num=1)
    # aileron_values = np.linspace(-5, 5, num=10)
    # elevator_values = np.linspace(-5, 5, num=10)
    frames = [(aileron, elevator) for aileron in aileron_values for elevator in elevator_values]

    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, repeat=False)

    # Save the animation
    ani.save(os.path.join(VISUPATH, 'predictions_animation.gif'), writer='imagemagick')

    # Show the plot
    plt.show()
    
    # fig = plt.figure(figsize = (15, 10))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss over Training')
    # plt.legend()
    # plt.savefig(os.path.join(VISUPATH, 'loss.png'))
    # plt.show(block=True)

    # plot residuals
    residuals = targets - outputs
    # fig = plt.figure(figsize=(15, 10))
    # for i in range(6):
    #     ax = fig.add_subplot(2, 3, i+1)
    #     ax.scatter(targets[:, i], residuals[:, i])
    #     ax.set_title(output_features[i])
    #     ax.set_xlabel('True Value')
    #     ax.set_ylabel('Residual')
    # plt.savefig(os.path.join(VISUPATH, 'residuals.png'))
    # plt.show()

    # plot target vs output
    # fig = plt.figure(figsize=(15, 10))
    # for i in range(6):
    #     ax = fig.add_subplot(2, 3, i+1)
    #     ax.scatter(targets[:, i], outputs[:, i])
    #     ax.set_title(output_features[i])
    #     ax.set_xlabel('True Value')
    #     ax.set_ylabel('Predicted Value')
    # plt.savefig(os.path.join(VISUPATH, 'target_vs_output.png'))
    # plt.show()

    # plot r2 scores
    fig = plt.figure(figsize=(15, 10))
     
    r2_scores = []
    for i in range(6):
        r2_scores.append(1 - np.sum(residuals[:, i] ** 2) / np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2))
    plt.bar(output_features, r2_scores)
    plt.xlabel('Output')
    plt.ylabel('R2 Score')
    plt.title('R2 Score for each output')
    plt.savefig(os.path.join(VISUPATH, 'r2_scores.png'))
    plt.show(block=True)


if __name__ == '__main__':
    main()
