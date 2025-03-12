import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
from typing import Optional, List, Dict, Union
import pandas as pd
import numpy as np
import h5py
import json
import casadi as ca
from dataclasses import dataclass

import sys
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('main')[0]
sys.path.append(BASEPATH)

from aircraft.utils.utils import load_model
from aircraft.config import DEVICE, NETWORKPATH, DATAPATH, VISUPATH
from aircraft.dynamics.dynamics import Aircraft, AircraftOpts

__all__ = ['create_grid', 'TrajectoryPlotter']

def create_grid(data:pd.DataFrame, num_points:Optional[int] = 10) -> pd.DataFrame:
    """
    Creates a meshgrid to plot with.
    The ranges are taken from the data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the data.
    num_points : int, optional
        Number of points in each dimension of the meshgrid. The default is 10.

    Returns
    -------
    X : pd.DataFrame
        Dataframe with the meshgrid data.

    """

    ingrid = [np.linspace(data[key].min() - 1*data[key].std(), data[key].max() + 1*data[key].std(), num_points) for key in data.keys()]

    meshgrid = np.meshgrid(*ingrid)

    meshgrid = pd.DataFrame(np.array(meshgrid).reshape(len(data.keys()), -1).T, columns = data.keys())

    return meshgrid


@dataclass
class TrajectoryData:
    state: np.ndarray = None
    control: np.ndarray = None
    time: np.ndarray = None
    lam: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    nu: Optional[np.ndarray] = None
    iteration:Optional[int] = None

    def load(self, filepath:str, iteration:int):
        with h5py.File(filepath, "r") as h5file:
            try:
                # Access the group for the current iteration
                grp = h5file[f'iteration_{iteration}']
                self.state = grp.get('state')[:] if 'state' in grp else None
                self.control = grp.get('control')[:] if 'control' in grp else None
                self.time = grp.get('time')[:] if 'time' in grp else None
                self.lam = grp.get('lam')[:] if 'lam' in grp else None
                self.mu = grp.get('mu')[:] if 'mu' in grp else None
                self.nu = grp.get('nu')[:] if 'nu' in grp else None
                self.iteration = iteration
            except KeyError:
                # If the group for the iteration doesn't exist, return None for all
                self.state = None
                self.control = None
                self.time = None
                self.lam = None
                self.mu = None
                self.nu = None
                self.iteration = None

        return self

class PlotAxes:
    def __init__(self, fig:plt.Figure):
        
        self.position = fig.add_subplot(3, 4, 1, projection='3d')
        self.angles = fig.add_subplot(3, 4, 2)
        self.velocity = fig.add_subplot(3, 4, 3)
        self.rates = fig.add_subplot(3, 4, 4)
        self.controls = fig.add_subplot(3, 4, 5)
        self.thrust = fig.add_subplot(3, 4, 6)
        self.convergence = fig.add_subplot(3, 4, 7)
        self.forces = fig.add_subplot(3, 4, 8)
        self.progress = fig.add_subplot(3, 4, 9)
        self.moments = fig.add_subplot(3, 4, 10)


    def __call__(self):
        return [self.position, 
                self.angles, 
                self.velocity, 
                self.rates, 
                self.controls, 
                self.thrust, 
                self.forces, 
                self.progress, 
                self.convergence]
    
    def clear(self):
        for ax in self():
            ax.clear()


class TrajectoryPlotter:
    def __init__(self, aircraft, figsize=(15, 15)):
        self.aircraft = aircraft
        self.figure = plt.figure(figsize=figsize)
        self.axes = PlotAxes(self.figure)
        self.lines = {}

    def load_last_iteration(self, filepath:str) -> Union[TrajectoryData, None]:
        """
        Load state, control, time, lam, mu, nu from the last iteration in the HDF5 file.
        Return None for datasets that do not exist.
        
        Parameters:
        - filepath: Path to the HDF5 file.
        
        Returns:
        - A dictionary containing the state, control, time, lam, mu, and nu from the last iteration, or None for missing datasets.
        """
        with h5py.File(filepath, "r") as h5file:
            iteration_keys = list(h5file.keys())
            if not iteration_keys:
                # Return None if there are no iterations in the file
                return None
            
            # Sort keys to ensure correct numeric ordering
            iteration_keys.sort(key=lambda x: int(x.split('_')[-1]))

            # Access the last iteration group
            last_iteration_key = iteration_keys[-1]
            last_iteration = int(last_iteration_key.split('_')[-1])

            trajectory_data = TrajectoryData().load(filepath, last_iteration)

        return trajectory_data


    
    def orientation(self, quaternion:np.ndarray):
        """
        Transforms axes into the frame defined by quaternion:4xN
        """
        # Convert quaternion from x,y,z,w to w,x,y,z format
        quaternion = np.roll(quaternion, 1, axis=0)
        
        # Ensure the quaternion is normalized
        quaternion = quaternion / np.linalg.norm(quaternion, axis=0)
        
        rotation = R.from_quat(quaternion.T)
        
        # Create orthogonal basis vectors
        x_axis = rotation.apply(np.array([1, 0, 0]))
        y_axis = rotation.apply(np.array([0, 1, 0]))
        z_axis = rotation.apply(np.array([0, 0, -1]))
        
        return (x_axis, y_axis, z_axis)
    
    def plot_position(self, trajectory_data: TrajectoryData, 
                  waypoints: Optional[np.ndarray] = None):
        """
        Plots the trajectory position and orientation axes in a 3D NED frame.
        """
        # I think these are wrong (inherited); corrected lines below
        # position = trajectory_data.state[4:7, :]
        # quaternion = trajectory_data.state[:4, :]
        position = trajectory_data.state[0:3, :]
        quaternion = trajectory_data.state[6:10, :]
        ax = self.axes.position
        
        # Negate Z-axis for NED convention
        position[2, :] *= -1
        
        # Compute orientation axes
        x_axis, y_axis, z_axis = self.orientation(quaternion)
        
        # Step size for quiver plotting
        step = position.shape[1] // 15 if position.shape[1] >= 15 else 1
        
        # Plot waypoints
        if isinstance(waypoints, np.ndarray):
            if not hasattr(self, '_waypoints_line'):
                self._waypoints_line, = ax.plot(waypoints[0, :], waypoints[1, :], waypoints[2, :], 
                                                marker='o', linestyle='--', label='Waypoints')
            else:
                self._waypoints_line.set_data_3d(waypoints[0, :], waypoints[1, :], waypoints[2, :])
        
        # Plot trajectory
        if not hasattr(self, '_trajectory_line'):
            self._trajectory_line, = ax.plot(position[0, :], position[1, :], position[2, :], 
                                            label='Trajectory')
        else:
            self._trajectory_line.set_data_3d(position[0, :], position[1, :], position[2, :])
        
        # Plot or update orientation quivers
        if not hasattr(self, '_quivers'):
            self._quivers = {
                'x': ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step],
                            x_axis[::step, 0], x_axis[::step, 1], x_axis[::step, 2],
                            color='r', length=1, label='Forward', normalize=True),
                'y': ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step],
                            y_axis[::step, 0], y_axis[::step, 1], y_axis[::step, 2],
                            color='g', length=1, label='Right', normalize=True),
                'z': ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step],
                            z_axis[::step, 0], z_axis[::step, 1], z_axis[::step, 2],
                            color='b', length=1, label='Down', normalize=True),
            }
        # In your plot_position method, modify the else block for quivers:
        else:
            for axis, data in zip(['x', 'y', 'z'], [x_axis, y_axis, z_axis]):
                starts = np.array([position[0, ::step], position[1, ::step], position[2, ::step]]).T
                ends = starts + data[::step] * 10  # Scale factor of 10 matches your original length
                segments = np.stack([starts, ends], axis=1)
                self._quivers[axis].set_segments(segments)

        # update axis limits
        ax.set_xlim(np.min(position[0, :]), np.max(position[0, :]))
        ax.set_ylim(np.min(position[1, :]), np.max(position[1, :]))
        ax.set_zlim(np.min(position[2, :]), np.max(position[2, :]))
        # Customize plot appearance
        ax.set_xlabel('North')
        ax.set_ylabel('East')
        ax.set_zlabel('Up')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Trajectory (NED)')

    def _update_or_create_line(self, ax, attr_name, y_data, label):
        """
        Updates an existing line if available; otherwise, creates a new one.
        """
        if not hasattr(self, attr_name):
            line, = ax.plot(y_data, label=label)
            setattr(self, attr_name, line)
        else:
            line = getattr(self, attr_name)
            line.set_ydata(y_data)
            line.set_xdata(np.arange(len(y_data)))

    def plot_angles(self, trajectory_data: TrajectoryData):
        """
        Plots aerodynamic and Euler angles, updating existing lines if available.
        """
        ax = self.axes.angles
        state = trajectory_data.state
        control = trajectory_data.control

        print("state",state.shape, "control",control.shape)

        # Compute aerodynamic and Euler angles
        alpha = np.rad2deg(self.aircraft.alpha(state, control).full().flatten())
        beta = np.rad2deg(self.aircraft.beta(state, control).full().flatten())
        phi = np.rad2deg(self.aircraft.phi(state).full().flatten())
        theta = np.rad2deg(self.aircraft.theta(state).full().flatten())
        psi = np.rad2deg(self.aircraft.psi(state).full().flatten())

        self._update_or_create_line(ax, '_alpha_line', alpha, r'$\alpha$ (attack)')
        self._update_or_create_line(ax, '_beta_line', beta, r'$\beta$ (sideslip)')
        self._update_or_create_line(ax, '_phi_line', phi, r'$\phi$ (roll)')
        self._update_or_create_line(ax, '_theta_line', theta, r'$\theta$ (pitch)')
        self._update_or_create_line(ax, '_psi_line', psi, r'$\psi$ (yaw)')

        # Customize plot appearance
        ax.relim()  # Recompute axes limits
        ax.autoscale_view()
        ax.legend()
        ax.grid(True)
        ax.set_title('Aerodynamic and Euler Angles')

    def plot_velocity(self, trajectory_data:TrajectoryData):
        ax = self.axes.velocity
        state = trajectory_data.state
        control = trajectory_data.control

        v_frd_rel = self.aircraft.v_frd_rel(state, control).full()

        self._update_or_create_line(ax, '_u_line', v_frd_rel[0, :], 'u')
        self._update_or_create_line(ax, '_v_line', v_frd_rel[1, :], 'v')
        self._update_or_create_line(ax, '_w_line', v_frd_rel[2, :], 'w')

        ax.legend()
        ax.grid(True)
        ax.set_title('Velocity (Body Frame)')

    def plot_rates(self, trajectory_data:TrajectoryData):
        state = trajectory_data.state
        ax = self.axes.rates
        phi_dot = self.aircraft.phi_dot(state).full().flatten()
        theta_dot = self.aircraft.theta_dot(state).full().flatten()
        psi_dot = self.aircraft.psi_dot(state).full().flatten()

        self._update_or_create_line(ax, '_phi_dot_line', phi_dot, r'$\dot{\phi}$')
        self._update_or_create_line(ax, '_theta_dot_line', theta_dot, r'$\dot{\theta}$')
        self._update_or_create_line(ax, '_psi_dot_line', psi_dot, r'$\dot{\psi}$')

        ax.legend()
        ax.grid(True)
        ax.set_title('Angular Rates')

    def plot_forces(self, trajectory_data:TrajectoryData):
        state = trajectory_data.state
        control = trajectory_data.control
        ax = self.axes.forces
        forces_frd = self.aircraft.forces_frd(state, control).full()

        self._update_or_create_line(ax, '_Fx_line', forces_frd[0, :], 'Fx')
        self._update_or_create_line(ax, '_Fy_line', forces_frd[1, :], 'Fy')
        self._update_or_create_line(ax, '_Fz_line', forces_frd[2, :], 'Fz')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aero Forces (FRD)')

    def plot_moments(self, trajectory_data:TrajectoryData):
        ax = self.axes.moments
        state = trajectory_data.state
        control = trajectory_data.control
        moments_frd = self.aircraft.moments_frd(state, control).full()

        moments_aero_frd = self.aircraft.moments_aero_frd(state, control).full()

        self._update_or_create_line(ax, '_Mx_line', moments_frd[0, :], 'Mx')
        self._update_or_create_line(ax, '_My_line', moments_frd[1, :], 'My')
        self._update_or_create_line(ax, '_Mz_line', moments_frd[2, :], 'Mz')

        self._update_or_create_line(ax, '_Mx_aero_line', moments_aero_frd[0, :], 'Mx_aero')
        self._update_or_create_line(ax, '_My_aero_line', moments_aero_frd[1, :], 'My_aero')
        self._update_or_create_line(ax, '_Mz_aero_line', moments_aero_frd[2, :], 'Mz_aero')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aero Moments (FRD)')

    def plot_state(self, trajectory_data:TrajectoryData):
        state = trajectory_data.state
        control = trajectory_data.control

        if state is not None:
            if control is not None:
                self.plot_moments(trajectory_data)

                self.plot_forces(trajectory_data)
            else:
                trajectory_data.control = np.zeros((3, state.shape[1]))
            
            self.plot_position(trajectory_data)

            self.plot_angles(trajectory_data)
            self.plot_velocity(trajectory_data)
            self.plot_rates(trajectory_data)

            
        
    def plot_deflections(self, trajectory_data:TrajectoryData):
        ax = self.axes.controls
        control = trajectory_data.control
        self._update_or_create_line(ax, '_delta_a_line', control[0, :], r'$\delta_a$')
        self._update_or_create_line(ax, '_delta_e_line', control[1, :], r'$\delta_e$')
        # self._update_or_create_line(ax, '_delta_r_line', control[2, :], r'$\delta_r$')
        ax.legend()
        ax.grid(True)
        ax.set_title('Control Surface Deflctions')

    def plot_thrust(self, trajectory_data:TrajectoryData):
        state= trajectory_data.state
        # control = trajectory_data.control
        # if control.shape[0] < 6:
        #     return
        ax = self.axes.thrust
        self._update_or_create_line(ax, '_T_x_line', state[0, :], r'$T_x$')
        self._update_or_create_line(ax, '_T_y_line', state[1, :], r'$T_y$')
        self._update_or_create_line(ax, '_T_z_line', state[2, :], r'$T_z$')
        ax.legend()
        ax.grid(True)
        ax.set_title('Thrust (N)')

    def plot_progress_variables(self, trajectory_data:TrajectoryData):
        ax = self.axes.progress
        lam = trajectory_data.lam
        mu = trajectory_data.mu
        nu = trajectory_data.nu

        print('lam: ', lam, 'mu: ', mu, 'nu: ', nu)
       
        if lam is not None and mu is not None and nu is not None:
            # check that lam is 2dim
            if len(lam.shape) == 2:
                for waypoint in range(len(lam)):
                    self._update_or_create_line(ax, f'_lam_line_{waypoint}', lam[waypoint], r'$\lambda$')
                    self._update_or_create_line(ax, f'_mu_line_{waypoint}', mu[waypoint], r'$\mu$')
                    self._update_or_create_line(ax, f'_nu_line_{waypoint}', nu[waypoint], r'$\nu$')
            else:
                self._update_or_create_line(ax, '_lam_line', lam, r'$\lambda$')
                self._update_or_create_line(ax, '_mu_line', mu, r'$\mu$')
                self._update_or_create_line(ax, '_nu_line', nu, r'$\nu$')

            ax.legend()
            ax.grid(True)
            ax.set_title('Progress Variables')

    def plot_control(self, trajectory_data:TrajectoryData):
        control = trajectory_data.control
        if control is not None:
            self.plot_deflections(trajectory_data)
            self.plot_thrust(trajectory_data)
        
    def plot(self, trajectory_data:TrajectoryData = None, filepath:str = None, iteration:Optional[int] = None):
        if isinstance(filepath, str):
            if iteration is None:
                trajectory_data = self.load_last_iteration(filepath)
            else:
                trajectory_data = TrajectoryData().load(filepath, iteration)

        assert isinstance(trajectory_data, TrajectoryData), "trajectory_data must be a TrajectoryData object or a filepath"

        # self.axes.clear()
        self.figure.suptitle(f"Final Time: {trajectory_data.time}")
        self.plot_state(trajectory_data)
        self.plot_control(trajectory_data)
        self.plot_progress_variables(trajectory_data)

    def save(self, iteration:Optional[int] = None, save_path = os.path.join(VISUPATH, 'trajectory_image.png')):
        if iteration is not None:
            save_path = str(save_path).replace(".png", f"iter_{iteration}.png")
        self.figure.savefig(save_path)

    def animation(self, iterations:List[int], save:bool = False, save_path:Optional[str] = os.path.join(VISUPATH, 'trajectory_animation.gif')):

        def update(i):
            self.plot(i)
        
        ani = animation.FuncAnimation(self.figure, update, frames=iterations, repeat=False)
        if save_path and save:
            ani.save(save_path, writer='imagemagick')
        return ani

    def show(self):
        self.figure.show()


def main():
    aircraft_params = json.load(open(os.path.join(BASEPATH, 'data', 'glider', 'glider_fs.json')))
    model = load_model()
    
    aircraft = Aircraft(aircraft_params, model, STEPS=100, LINEAR=True)
    plotter = TrajectoryPlotter(aircraft)

    plotter.plot(iteration = -1)
    plotter.show()
    plotter.save(iteration = -1)

    plotter.animation([i for i in range(50)], save = True)

if __name__ == '__main__':
    main()