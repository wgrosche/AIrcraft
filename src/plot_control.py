""" 
Loads the results of the control.py file from pickle and plots them.

We visualise the final trajectory and controls as well as the convergence
behaviour of the NLP and the sparsity of the jacobian.



For convergence plot details: https://web.casadi.org/blog/nlp-scaling/
For sparsity details: get_jac_sparsity(*args) https://web.casadi.org/python-api/#callback
"""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASEPATH)
sys.path.append(BASEPATH)

from src.plotting import plot_trajectory
from src.dynamics import Aircraft


def plot_trajectory(ax:Axes3D, state:np.ndarray):
    step = state.shape[0] // 10

    # Precompute quaternion directions
    body_x_axis = R.from_quat(state[:4, :].T).apply(np.array([1, 0, 0]))
    body_z_axis = R.from_quat(state[:4, :].T).apply(np.array([0, 0, 1]))
    
    # 3D Trajectory Plot
    ax.plot(state[4, :], state[5, :], state[6, :])
    
    ax.quiver(state[4, ::step], state[5, ::step], state[6, ::step], 
               state[7, ::step], state[8, ::step], state[9, ::step], 
               color='g', length=0.1, label='Velocity')
    
    ax.quiver(state[4, ::step], state[5, ::step], state[6, ::step], 
               body_x_axis[0, ::step], body_x_axis[1, ::step], 
               body_x_axis[2, ::step], color='g', length=0.1, label='x-axis')
    
    ax.quiver(state[4, ::step], state[5, ::step], state[6, ::step], 
               body_z_axis[0, ::step], body_z_axis[1, ::step], 
               body_z_axis[2, ::step], color='g', length=0.1, label='z-axis')
    
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(state[4, :].min(), state[4, :].max())
    ax.set_zlim(state[6, :].min(), state[6, :].max())
    ax.grid(True)
    ax.set_title('Trajectory (NED)')

def plot_orientation(ax:plt.Axes, state:np.ndarray):
    ax.plot(state[0, :], label='x')
    ax.plot(state[1, :], label='y')
    ax.plot(state[2, :], label='z')
    ax.plot(state[3, :], label='w')
    ax.legend()
    ax.grid(True)
    ax.set_title('Orientation')

def plot_state(fig, state, control, aircraft):
    ax = fig.add_suplot(1, 1, 1)
    plot_trajectory(ax, state)
    
    
    
def plot_control(fig, control):

    ax_flaps = fig.add_suplot()
    plot_control_surfaces(ax_flaps, control)

    ax_wind = fig.add_subplot()

def plot_control_surfaces(ax, control):
    ax.plot(control[0,:], 'r', label='aileron')
    ax.plot(control[1,:], 'g', label='eleator')
    ax.plot(control[2,:], 'b', label='rudder')

def plot_aerodynamics(fig, state, control, aircraft):
    pass

def plot_coefficients(fig:plt.Figure, 
                      state:np.ndarray, 
                      control:np.ndarray, 
                      aircraft:Aircraft
                      ):
    coefficients = aircraft._coefficients(state, control)
    titles = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']
    for i in range(6):
        ax = fig.add_subplot(3, 6, 13 + i)
        ax.plot(coefficients[i, :])
        ax.set_title(titles[i] + ' (FRD)')
        ax.grid(True)


def plot_sparsity(fig, opti):
    pass

def plot_convergence(fig, opti):
    pass

def main():
    pass

if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt

def plot(fig, data):
    # Sample data based on 'windtunnel' condition
    data_wt = data.where(data['windtunnel'] == True).sample(frac=.1)
    data_fs = data.where(data['windtunnel'] == False)

    # Loop over the 6 subplots
    for i in range(6):
        # Desired subplot position (2 rows, 3 columns, subplot index i+1)
        desired_position = (2, 3, i+1)
        
        # Check if the subplot already exists
        existing_ax = None
        for ax in fig.axes:
            # Check if this ax corresponds to the desired subplot
            if ax.get_geometry() == desired_position and ax.name == '3d':
                existing_ax = ax
                break

        # Use the existing ax or create a new one if it doesn't exist
        if existing_ax is None:
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
        else:
            ax = existing_ax

        # Plotting on the ax
        ax.scatter(data_wt['alpha'], data_wt['beta'], data_wt.iloc[:, i+6], 
                   marker='o', label='windtunnel')
        ax.scatter(data_fs['alpha'], data_fs['beta'], data_fs.iloc[:, i+6], 
                   marker='o', label='freestream')
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel(data.columns[i+6])
        ax.legend()

# Example usage
# fig = plt.figure()
# plot(fig, data)
# plt.show()
