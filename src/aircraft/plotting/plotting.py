import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
from typing import Optional, List, Dict, Union, cast
import pandas as pd
from pandas import DataFrame
import numpy as np
import h5py
import json
import casadi as ca
from dataclasses import dataclass
from aircraft.dynamics.base import SixDOF
from aircraft.config import VISUPATH
from matplotlib.figure import Figure

__all__ = ['create_grid', 'TrajectoryPlotter', 'TrajectoryData', 'plot_convergence']

def plot_convergence(ax:Axes, sol:ca.OptiSol):
    ax.semilogy(sol.stats()['iterations']['inf_du'], label="Dual infeasibility")
    ax.semilogy(sol.stats()['iterations']['inf_pr'], label="Primal infeasibility")

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Infeasibility (log scale)')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show(block = True)

def create_grid(data:DataFrame, num_points:Optional[int] = 10) -> DataFrame:
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

    ingrid = [np.linspace(start = data[key].min() - 1*data[key].std(), stop = data[key].max() + 1*data[key].std(), num = num_points) for key in data.keys()]

    meshgrid = np.meshgrid(*ingrid)

    meshgrid = pd.DataFrame(np.array(meshgrid).reshape(len(data.keys()), -1).T, columns = data.keys())

    return meshgrid


@dataclass
class TrajectoryData:
    state: Optional[np.ndarray] = None
    control: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    lam: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    nu: Optional[np.ndarray] = None
    iteration:Optional[int] = None

    def load(self, filepath:str, iteration:int):
        with h5py.File(filepath, "r") as h5file:
            try:
                # Access the group for the current iteration
                grp = h5file[f'iteration_{iteration}']
                # assert isinstance(grp, dict)
                self.state = grp.get('state')[:] if 'state' in grp else None
                self.control = grp.get('control')[:] if 'control' in grp else None
                self.times = grp.get('times')[:] if 'times' in grp else None
                self.lam = grp.get('lam')[:] if 'lam' in grp else None
                self.mu = grp.get('mu')[:] if 'mu' in grp else None
                self.nu = grp.get('nu')[:] if 'nu' in grp else None
                self.iteration = iteration
            except KeyError:
                # If the group for the iteration doesn't exist, return None for all
                self.state = None
                self.control = None
                self.times = None
                self.lam = None
                self.mu = None
                self.nu = None
                self.iteration = None

        return self


class PlotAxes:
    def __init__(self, fig:Figure):
        
        
        self.position:Axes3D = cast(Axes3D, fig.add_subplot(3, 4, 1, projection='3d'))  # cast to Axes3D
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
                self.convergence,
                self.moments]
    
    def clear(self):
        for ax in self():
            ax.clear()


class TrajectoryPlotter:
    def __init__(self, six_dof:SixDOF, figsize=(15, 15)):
        self.six_dof = six_dof
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
        # quaternion = np.roll(quaternion, 1, axis=0)
        
        # Ensure the quaternion is normalized
        quaternion = quaternion / np.linalg.norm(quaternion, axis=0)
        
        rotation = R.from_quat(quaternion.T).inv()
        
        # Create orthogonal basis vectors
        x_axis = rotation.apply(np.array([1, 0, 0]))
        y_axis = rotation.apply(np.array([0, 1, 0]))
        z_axis = rotation.apply(np.array([0, 0, 1]))
        
        return (x_axis, y_axis, z_axis)
    
    def plot_position(self, trajectory_data: TrajectoryData, 
                  waypoints: Optional[np.ndarray] = None) -> None:
        """
        Plots the trajectory position and orientation axes in a 3D NED frame.
        """
        if trajectory_data.state is None:
            return None
        # I think these are wrong (inherited); corrected lines below
        # position = trajectory_data.state[4:7, :]
        # quaternion = trajectory_data.state[:4, :]

        position = trajectory_data.state[0:3, :]
        quaternion = trajectory_data.state[6:10, :]
        ax:Axes3D = self.axes.position
        
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
                            color='r', length=3, label='Forward', normalize=True),
                'y': ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step],
                            y_axis[::step, 0], y_axis[::step, 1], y_axis[::step, 2],
                            color='g', length=3, label='Right', normalize=True),
                'z': ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step],
                            z_axis[::step, 0], z_axis[::step, 1], z_axis[::step, 2],
                            color='b', length=3, label='Down', normalize=True),
            }
        else:
            for axis, data in zip(['x', 'y', 'z'], [x_axis, y_axis, z_axis]):
                starts = np.array([position[0, ::step], position[1, ::step], position[2, ::step]]).T
                ends = starts + data[::step] * 3
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

    def _update_or_create_line(self, ax, attr_name, y_data:np.ndarray, x_data:Optional[np.ndarray] = None, label:Optional[str] = None, drawstyle = None):
        """
        Updates an existing line if available; otherwise, creates a new one.
        """
        if not hasattr(self, attr_name):
            if drawstyle:
                if x_data is not None:
                    line, = ax.plot(x_data, y_data, drawstyle=drawstyle, label=label)
                else:
                    line, = ax.plot(np.arange(len(y_data)), y_data, drawstyle=drawstyle, label=label)

            else:
                if x_data is not None:
                    line, = ax.plot(x_data, y_data, label=label)
                else:
                    line, = ax.plot(y_data, label=label)
            setattr(self, attr_name, line)
        else:
            line = getattr(self, attr_name)
            line.set_ydata(y_data)
            if x_data is not None:
                line.set_xdata(x_data)
            else:
                line.set_xdata(np.arange(len(y_data)))

    def plot_angles(self, trajectory_data: TrajectoryData):
        """
        Plots aerodynamic and Euler angles, updating existing lines if available.
        """
        ax = self.axes.angles
        state = trajectory_data.state
        control = trajectory_data.control
        times = trajectory_data.times
        print("state",state.shape, "control",control.shape)

        # Compute aerodynamic and Euler angles
        alpha = np.rad2deg(self.six_dof.alpha(state, control).full().flatten())
        beta = np.rad2deg(self.six_dof.beta(state, control).full().flatten())
        phi = np.rad2deg(self.six_dof.phi(state).full().flatten())
        theta = np.rad2deg(self.six_dof.theta(state).full().flatten())
        psi = np.rad2deg(self.six_dof.psi(state).full().flatten())

        self._update_or_create_line(ax, '_alpha_line', alpha, x_data = times, label = r'$\alpha$ (attack)')
        self._update_or_create_line(ax, '_beta_line', beta, x_data = times, label = r'$\beta$ (sideslip)')
        self._update_or_create_line(ax, '_phi_line', phi, x_data = times, label = r'$\phi$ (roll)')
        self._update_or_create_line(ax, '_theta_line', theta, x_data = times, label = r'$\theta$ (pitch)')
        self._update_or_create_line(ax, '_psi_line', psi, x_data = times, label = r'$\psi$ (yaw)')

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
        times = trajectory_data.times
        v_frd_rel = self.six_dof.v_frd_rel(state, control).full()

        self._update_or_create_line(ax, '_u_line', v_frd_rel[0, :], x_data = times, label =  'u')
        self._update_or_create_line(ax, '_v_line', v_frd_rel[1, :], x_data = times, label =  'v')
        self._update_or_create_line(ax, '_w_line', v_frd_rel[2, :], x_data = times, label =  'w')

        ax.legend()
        ax.grid(True)
        ax.set_title('Velocity (Body Frame)')

    def plot_rates(self, trajectory_data:TrajectoryData):
        state = trajectory_data.state
        times = trajectory_data.times
        ax = self.axes.rates
        phi_dot = self.six_dof.phi_dot(state).full().flatten()
        theta_dot = self.six_dof.theta_dot(state).full().flatten()
        psi_dot = self.six_dof.psi_dot(state).full().flatten()

        self._update_or_create_line(ax, '_phi_dot_line', phi_dot, x_data = times, label =  r'$\dot{\phi}$')
        self._update_or_create_line(ax, '_theta_dot_line', theta_dot, x_data = times, label =  r'$\dot{\theta}$')
        self._update_or_create_line(ax, '_psi_dot_line', psi_dot, x_data = times, label =  r'$\dot{\psi}$')

        ax.legend()
        ax.grid(True)
        ax.set_title('Angular Rates')

    def plot_forces(self, trajectory_data:TrajectoryData):
        state = trajectory_data.state
        control = trajectory_data.control
        times = trajectory_data.times
        ax = self.axes.forces
        forces_frd = self.six_dof.forces_frd(state, control).full()

        self._update_or_create_line(ax, '_Fx_line', forces_frd[0, :], x_data = times, label =  'Fx')
        self._update_or_create_line(ax, '_Fy_line', forces_frd[1, :], x_data = times, label =  'Fy')
        self._update_or_create_line(ax, '_Fz_line', forces_frd[2, :], x_data = times, label =  'Fz')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aero Forces (FRD)')

    def plot_moments(self, trajectory_data:TrajectoryData):
        ax = self.axes.moments
        state = trajectory_data.state
        control = trajectory_data.control
        times = trajectory_data.times
        moments_frd = self.six_dof.moments_frd(state, control).full()

        moments_aero_frd = self.six_dof.moments_aero_frd(state, control).full()

        self._update_or_create_line(ax, '_Mx_line', moments_frd[0, :], x_data = times, label =  'Mx')
        self._update_or_create_line(ax, '_My_line', moments_frd[1, :], x_data = times, label =  'My')
        self._update_or_create_line(ax, '_Mz_line', moments_frd[2, :], x_data = times, label =  'Mz')

        self._update_or_create_line(ax, '_Mx_aero_line', moments_aero_frd[0, :], x_data = times, label =  'Mx_aero')
        self._update_or_create_line(ax, '_My_aero_line', moments_aero_frd[1, :], x_data = times, label =  'My_aero')
        self._update_or_create_line(ax, '_Mz_aero_line', moments_aero_frd[2, :], x_data = times, label =  'Mz_aero')
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
        times = trajectory_data.times
        self._update_or_create_line(ax, '_delta_a_line', control[0, :], x_data = times, label = r'$\delta_a$', drawstyle='steps-post')
        self._update_or_create_line(ax, '_delta_e_line', control[1, :], x_data = times, label = r'$\delta_e$', drawstyle='steps-post')
        self._update_or_create_line(ax, '_delta_r_line', control[2, :], x_data = times, label = r'$\delta_r$', drawstyle='steps-post')

        ax.legend()
        ax.grid(True)
        ax.set_title('Control Surface Deflections')

    def plot_thrust(self, trajectory_data:TrajectoryData):
        state = trajectory_data.state
        times = trajectory_data.times
        # control = trajectory_data.control
        # if control.shape[0] < 6:
        #     return
        ax = self.axes.thrust
        self._update_or_create_line(ax, '_T_x_line', state[0, :], x_data = times, label = r'$T_x$')
        self._update_or_create_line(ax, '_T_y_line', state[1, :], x_data = times, label = r'$T_y$')
        self._update_or_create_line(ax, '_T_z_line', state[2, :], x_data = times, label = r'$T_z$')
        ax.legend()
        ax.grid(True)
        ax.set_title('Thrust (N)')

    def plot_progress_variables(self, trajectory_data:TrajectoryData):
        ax = self.axes.progress
        lam = trajectory_data.lam
        times = trajectory_data.times
        mu = trajectory_data.mu
        nu = trajectory_data.nu

        print('lam: ', lam, 'mu: ', mu, 'nu: ', nu)
       
        if lam is not None and mu is not None and nu is not None:
            # check that lam is 2dim
            if len(lam.shape) == 2:
                for waypoint in range(len(lam)):
                    self._update_or_create_line(ax, f'_lam_line_{waypoint}', lam[waypoint], x_data = times, label =  r'$\lambda$')
                    self._update_or_create_line(ax, f'_mu_line_{waypoint}', mu[waypoint], x_data = times, label =  r'$\mu$')
                    self._update_or_create_line(ax, f'_nu_line_{waypoint}', nu[waypoint], x_data = times, label =  r'$\nu$')
            else:
                self._update_or_create_line(ax, '_lam_line', lam, x_data = times, label = r'$\lambda$')
                self._update_or_create_line(ax, '_mu_line', mu, x_data = times, label = r'$\mu$')
                self._update_or_create_line(ax, '_nu_line', nu, x_data = times, label = r'$\nu$')

            ax.legend()
            ax.grid(True)
            ax.set_title('Progress Variables')

    def plot_control(self, trajectory_data:TrajectoryData):
        control = trajectory_data.control
        if control is not None:
            self.plot_deflections(trajectory_data)
            self.plot_thrust(trajectory_data)

    def plot_quat_norm(self, trajectory_data:TrajectoryData):
        quat = trajectory_data.state[6:10,:]
        times = trajectory_data.times
        ax = self.axes.convergence
        self._update_or_create_line(ax, '_quat_norm_line', np.linalg.norm(quat, axis = 0), x_data = times, label = 'norm')

        ax.legend()
        ax.grid(True)
        # ax.yaxis.get_offset_text().set_x(-0.1)  # adjust X offset
        # ax.yaxis.get_offset_text().set_y(-0.1)  # or adjust Y manually
        ax.set_title('Quaternion Norm')
        
    def plot(self, trajectory_data:TrajectoryData = None, filepath:str = None, iteration:Optional[int] = None):
        if isinstance(filepath, str):
            if iteration is None:
                trajectory_data = self.load_last_iteration(filepath)
            else:
                trajectory_data = TrajectoryData().load(filepath, iteration)

        assert isinstance(trajectory_data, TrajectoryData), "trajectory_data must be a TrajectoryData object or a filepath"

        # self.axes.clear()
        self.figure.suptitle(f"Final Time: {trajectory_data.times[-1]}")
        self.plot_state(trajectory_data)
        self.plot_control(trajectory_data)
        self.plot_progress_variables(trajectory_data)
        self.plot_quat_norm(trajectory_data)



        # Auto-adjust all axes limits
        for ax in self.axes():
            if ax != self.axes.position:  # Skip 3D position plot as it's handled separately
                ax.relim()  # Recompute the data limits
                ax.autoscale_view()  # Autoscale the view based on the data limits
                ax.grid(True)  # Ensure grid is visible

    def save(self, iteration:Optional[int] = None, save_path = os.path.join(VISUPATH, 'trajectory_image.png')):
        if iteration is not None:
            save_path = str(save_path).replace(".png", f"iter_{iteration}.png")
        self.figure.savefig(save_path)

    def animation(self, iterations:List[int], save:bool = False, save_path:Optional[str] = os.path.join(VISUPATH, 'trajectory_animation.gif')):

        def update(i:TrajectoryData) -> None:
            self.plot(i)
        
        ani = animation.FuncAnimation(self.figure, update, frames=iterations, repeat=False)
        if save_path and save:
            ani.save(save_path, writer='imagemagick')
        return ani

    def show(self):
        self.figure.show()
