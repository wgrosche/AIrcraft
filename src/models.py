"""
Models for aerodynamic coefficient prediction

- ScaledModel: Fully connected neural network that automatically scales input and output data
- MiniModel: Smaller version of ScaledModel
- WeightedMSELoss: Weighted mean squared error loss function
- Net2D: Network for prediction of 2D airfoil dynamics
- ControlNet: Network for control prediction
- GPModel: Gaussian process model for aerodynamic coefficient prediction https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/ModelList_GP_Regression.html


"""


import torch
import math
import torch.nn as nn
__all__ = ['ScaledModel', 'MiniModel', 'WeightedMSELoss', 'Net2D']

try:
    import gpytorch

    class BaseGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(BaseGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    class MimoGPModel(gpytorch.models.IndependentModelList):
        def __init__(self, train_x, train_y, likelihood):
            for i in range(train_y.shape[1]):
                model = BaseGPModel(train_x, train_y[:, i], likelihood)
                if i == 0:
                    self.models = model
                    self.likelihood_list = model.likelihood
                else:
                    self.models = gpytorch.models.IndependentModelList(self.models, model)
                    self.likelihood_list = gpytorch.likelihoods.LikelihoodList(self.likelihood_list, model.likelihood)

        def forward(self, x):
            output = []
            for model in self.models:
                output.append(model(x))
            return torch.cat(output, dim=-1)
except:
    print("gpytorch not installed, gaussian process models not available")
    
                                                         


class MiniModel(nn.Module):
    """
    Network that automatically scales the input and output data
    """
    def __init__(self, input_size, output_size, scaler=None):
        """
        Inputs:
        - input_size: number of input features
        - output_size: number of output features
        - scaler: tuple of input mean, input std, output mean, output std
        """
        super(MiniModel, self).__init__()
        
        self.core_layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, output_size),
        )
        
        # Initialize buffers for scaling parameters if a scaler is provided
        if scaler is not None:
            input_mean, input_std, output_mean, output_std = scaler

            # Registering mean and std as buffers ensures they are moved with the model's device
            self.register_buffer('input_mean', torch.tensor(input_mean, dtype=torch.float32))
            self.register_buffer('input_std', torch.tensor(input_std, dtype=torch.float32))
            self.register_buffer('output_mean', torch.tensor(output_mean, dtype=torch.float32))
            self.register_buffer('output_std', torch.tensor(output_std, dtype=torch.float32))

    def forward(self, x):
        # Apply input scaling if mean and std buffers are present
        if hasattr(self, 'input_mean') and hasattr(self, 'input_std'):
            x = (x - self.input_mean) / self.input_std
        
        # Pass through the core model
        core_output = self.core_layers(x)
        
        # Apply inverse output scaling if mean and std buffers are present
        if hasattr(self, 'output_mean') and hasattr(self, 'output_std'):
            core_output = (core_output * self.output_std) + self.output_mean
        
        return core_output

class L4CasadiModel():
    def __init__(self):
        pass

class ScaledModel(nn.Module):
    """
    Network that automatically scales the input and output data
    """
    def __init__(self, input_size, output_size, scaler=None, device='cpu'):
        """
        Inputs:
        - input_size: number of input features
        - output_size: number of output features
        - scaler: tuple of input mean, input std, output mean, output std
        """
        super(ScaledModel, self).__init__()
        
        self.core_layers = nn.Sequential(
            nn.Linear(input_size, 16),
            # nn.ELU(),
            nn.Linear(16, 64),
            # nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, output_size),
        )
        
        # Initialize buffers for scaling parameters if a scaler is provided
        if scaler is not None:
            input_mean, input_std, output_mean, output_std = scaler

            if isinstance(input_mean, torch.Tensor):
                # Registering mean and std as buffers ensures they are moved with the model's device
                self.register_buffer('input_mean', input_mean)
                self.register_buffer('input_std', input_std)
                self.register_buffer('output_mean', output_mean)
                self.register_buffer('output_std', output_std)

            else:
                # Registering mean and std as buffers ensures they are moved with the model's device
                self.register_buffer('input_mean', torch.tensor(input_mean, dtype=torch.float32))
                self.register_buffer('input_std', torch.tensor(input_std, dtype=torch.float32))
                self.register_buffer('output_mean', torch.tensor(output_mean, dtype=torch.float32))
                self.register_buffer('output_std', torch.tensor(output_std, dtype=torch.float32))

    def forward(self, x):
        # Apply input scaling if mean and std buffers are present
        if hasattr(self, 'input_mean') and hasattr(self, 'input_std'):
            x = (x - self.input_mean) / self.input_std
        
        # Pass through the core model
        core_output = self.core_layers(x)
        
        # Apply inverse output scaling if mean and std buffers are present
        if hasattr(self, 'output_mean') and hasattr(self, 'output_std'):
            core_output = (core_output * self.output_std) + self.output_mean
        
        return core_output

class WeightedMSELoss(nn.Module):
    def __init__(self, target_std):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.Tensor(1 / target_std ** 2)

    def forward(self, inputs, model_output,  targets):
        # Ensure weights are on the same device as inputs
        weights = self.weights.to(model_output.device)
        # Calculate weighted MSE
        return torch.mean(weights * (model_output - targets) ** 2)
    
import torch
import torch.nn as nn

class WeightedMSELossConstraint(nn.Module):
    def __init__(self, target_std):
        super(WeightedMSELossConstraint, self).__init__()
        # Weights based on the inverse of variance
        self.weights = torch.Tensor(1 / target_std ** 2)

    def forward(self, inputs, model_output, targets):
        # Assuming full_inputs contains [qbar, alpha, beta, aileron, elevator] at indices 1 to 4
        # inputs, controls = full_inputs[:, 0], full_inputs[:, 1:]  # Separate out parameters with 0 constraint

        # Ensure weights are on the same device as inputs
        weights = self.weights.to(model_output.device)

        # Normal weighted MSE calculation
        mse_loss = torch.mean(weights * (model_output - targets) ** 2)

        # Check if control inputs (alpha, beta, aileron, elevator) are all zero
        zero_condition = torch.all(inputs[:, 2:] == 0, dim=1)
        zero_condition = zero_condition.unsqueeze(1)  # Make it (batch_size, 1) to match outputs

        # Conditionally apply zero-output constraint
        zero_output = torch.zeros_like(model_output)
        zero_output[:, 1] = model_output[:, 1] 
        zero_output[:, 1] = model_output[:, 3]
        zero_output[:, 1] = model_output[:, 5]
        zero_loss = torch.where(zero_condition, 100 * model_output, torch.zeros_like(model_output))
        zero_loss = torch.mean(zero_loss)  # Mean over all samples

        # Combine the losses, you might want to weight the zero_loss differently
        combined_loss = mse_loss + zero_loss

        return combined_loss



class Net2D(nn.Module):
    """
    Network for prediction of 2D airfoil dynamics

    Inputs:
        - Shape parameters: nD vector
        - Angle of attack: scalar
        - Airspeed: scalar
    
    Outputs:
        - Lift coefficient: scalar
        - Drag coefficient: scalar
    """
    def __init__(self):
        super(Net2D, self).__init__()
        self.norm_in = nn.BatchNorm1d(5, affine = False)
        self.fc1 = nn.Linear(5, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 6)
        self.norm_out = nn.BatchNorm1d(6, affine = False)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x
    

class ControlNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ControlNet, self).__init__()
        self.norm_in = nn.BatchNorm1d(5, affine = False)
        self.fc1 = nn.Linear(5, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 6)
        self.norm_out = nn.BatchNorm1d(6, affine = False)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x