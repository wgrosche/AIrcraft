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

from utils import load_model
DATAPATH = os.path.join(BASEPATH, 'data')
NETWORKPATH = os.path.join(DATAPATH, 'networks')
VISUPATH = os.path.join(DATAPATH, 'visualisation')

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

class TrajectoryPlotter:

    @dataclass
    class PlotAxes:
        position:Axes
        angles:Axes
        velocity:Axes
        rates:Axes
        controls:Axes
        thrust:Axes
        convergence:Axes
        forces:Axes
        progress:Axes
        moments:Axes

        @property
        def axes(self):
            return [self.position, self.angles, self.velocity, self.rates, 
                    self.controls, self.thrust, self.forces, self.progress, self.convergence]

    def __init__(self, filepath:str, aircraft):
        self.filepath = filepath
        self.aircraft = aircraft
        self.figure = plt.figure()
        position = self.figure.add_subplot(3, 4, 1, projection='3d')
        axs=[]
        for i in range(11):
            axs.append(self.figure.add_subplot(3, 4, i+2))

        self.axes = self.PlotAxes(position=position, angles=axs[0], 
                                  velocity=axs[1], rates=axs[2], forces=axs[3], 
                                  controls=axs[4], thrust=axs[5], 
                                  convergence=axs[6], progress = axs[7], moments = axs[8])

    def read_trajectory(self, filepath, iteration):
        """
        Read state, control, time, lam, mu, nu from the HDF5 file for a given iteration.
        Return None for datasets that do not exist.
        
        Parameters:
        - filepath: Path to the HDF5 file.
        - iteration: Iteration number to read.
        
        Returns:
        - Tuple of state, control, time, lam, mu, nu (or None for missing entries)
        """
        with h5py.File(filepath, "r") as h5file:
            try:
                # Access the group for the current iteration
                grp = h5file[f'iteration_{iteration}']
                state = grp.get('state')[:] if 'state' in grp else None
                control = grp.get('control')[:] if 'control' in grp else None
                time = grp.get('time')[:] if 'time' in grp else None
                lam = grp.get('lam')[:] if 'lam' in grp else None
                mu = grp.get('mu')[:] if 'mu' in grp else None
                nu = grp.get('nu')[:] if 'nu' in grp else None
            except KeyError:
                # If the group for the iteration doesn't exist, return None for all
                return None, None, None, None, None, None

        return state, control, time, lam, mu, nu

    def load_last_iteration(filepath):
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
            last_grp = h5file[last_iteration_key]

            # Extract datasets from the last group, return None if the dataset doesn't exist
            last_state = last_grp.get('state')[:] if 'state' in last_grp else None
            last_control = last_grp.get('control')[:] if 'control' in last_grp else None
            last_time = last_grp.get('time')[:] if 'time' in last_grp else None
            lam = last_grp.get('lam')[:] if 'lam' in last_grp else None
            mu = last_grp.get('mu')[:] if 'mu' in last_grp else None
            nu = last_grp.get('nu')[:] if 'nu' in last_grp else None

        return {
            'iteration': last_iteration_key,
            'state': last_state,
            'control': last_control,
            'time': last_time,
            'lam': lam,
            'mu': mu,
            'nu': nu
        }



    def derive_angles(self, aircraft, state:np.ndarray, control:np.ndarray):
        """
        
        """
        alpha = aircraft._alpha(state, control).full().flatten()
        beta = aircraft._beta(state, control).full().flatten()

        q_frd_ned = state[:4, :]
        omega_frd_frd = state[-3:, :]
        print("q: ", q_frd_ned.shape)
        print("omega: ", omega_frd_frd.shape)
        euler = aircraft.compute_euler_and_body_rates(q_frd_ned, omega_frd_frd)

        return (euler, alpha, beta)

    
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
        z_axis = rotation.apply(np.array([0, 0, 1]))
        
        return (x_axis, y_axis, z_axis)
    
    def plot_position(self, position:Optional[np.ndarray] = None, 
                      quaternion:Optional[np.ndarray] = None, 
                      waypoints:Optional[Union[np.ndarray, np.ndarray]] = None, 
                      ax:Optional[Axes3D] = None):
        if not isinstance(position, np.ndarray):
            position = self.state[4:7,:]
        if not isinstance(quaternion, np.ndarray):
            quaternion = self.state[:4, :]
        if not isinstance(ax, Axes3D):
            ax = self.axes.position
        
        step = position.shape[1] // 50

        position[2, :] *= -1
        (x_axis, y_axis, z_axis) = self.orientation(quaternion)

        if isinstance(waypoints, np.ndarray):
            ax.plot(waypoints[0, :], waypoints[1, :], waypoints[2, :], )

        # plot trajectory
        ax.plot(position[0, :], position[1, :], position[2, :])

        ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step], 
                x_axis[::step, 0], x_axis[::step, 1], x_axis[::step, 2], 
                color='r', length=10, label = 'forward', normalize =True)
        ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step], 
                y_axis[::step, 0], y_axis[::step, 1], y_axis[::step, 2], 
                color='g', length=10, label = 'right', normalize =True)
        ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step], 
                z_axis[::step, 0], z_axis[::step, 1], z_axis[::step, 2], 
                color='b', length=10, label = 'down', normalize =True)
        
        ax.set_xlabel('North')
        ax.set_ylabel('East')
        ax.set_zlabel('Down')
        # ax.set_xlim(position[0, :].min(), position[0, :].max())
        # ax.set_zlim(position[2, :].max(), position[2, :].min())
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Trajectory (NED)')

    def plot_angles(self, euler, alpha, beta, ax:Optional[Axes]=None):
        if not isinstance(ax, Axes):
            ax = self.axes.angles
        phis = euler['phi']
        print("phi: ", type(phis))
        thetas = euler['theta']
        psis = euler['psi']

        ax.plot(np.rad2deg(alpha), label=r'$\alpha$ (attack)')
        ax.plot(np.rad2deg(beta), label=r'$\beta$ (sideslip)')

        ax.plot(np.rad2deg(phis), label=r'$\phi$ (roll)')
        ax.plot(np.rad2deg(thetas), label=r'$\theta$ (pitch)')
        # ax.plot(np.rad2deg(psis), label=r'$\psi$ (yaw)')
        

        # ax.axhline(10, color='r')
        # ax.axhline(-10, color='r')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aerodynamic and Euler Angles')

    def plot_velocity(self, state, control, ax:Optional[Axes] = None):
        if not isinstance(ax, Axes):
            ax = self.axes.velocity

        v_frd_rel = self.aircraft._v_frd_rel(state, control).full()

        ax.plot(v_frd_rel[0, :], label='u')
        ax.plot(v_frd_rel[1, :], label='v')
        ax.plot(v_frd_rel[2, :], label='w')

        ax.legend()
        ax.grid(True)
        ax.set_title('Velocity (Body Frame)')

    def plot_rates(self, euler, ax:Optional[Axes] = None):
        if not isinstance(ax, Axes):
            ax = self.axes.rates

        # ca.MX.eval_mx()
        phis = euler['phi_dot']
        thetas = euler['theta_dot']
        psis = euler['psi_dot']

        ax.plot(phis, label=r'$\dot{\phi}$')
        ax.plot(thetas, label=r'$\dot{\theta}$')
        # ax.plot(psis, label=r'$\dot{\psi}$')
        ax.legend()
        ax.grid(True)
        ax.set_title('Angular Rates')

    def plot_forces(self, state, control, ax:Optional[Axes] = None):
        if not isinstance(ax, Axes):
            ax = self.axes.forces
        forces_frd = self.aircraft._forces_frd(state, control).full()

        ax.plot(forces_frd[0, :], label='Fx')
        ax.plot(forces_frd[1, :], label='Fy')
        ax.plot(forces_frd[2, :], label='Fz')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aero Forces (FRD)')

    def plot_moments(self, state, control, ax:Optional[Axes] = None):
        if not isinstance(ax, Axes):
            ax = self.axes.moments
        moments_frd = self.aircraft._moments_frd(state, control).full()

        ax.plot(moments_frd[0, :], label='Mx')
        ax.plot(moments_frd[1, :], label='My')
        ax.plot(moments_frd[2, :], label='Mz')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aero Moments (FRD)')

    def plot_state(self, iteration:int):
        if iteration == -1:
            result = self.load_last_iteration(self.filepath)
            state, control = result['state'], result['control']
        else:
            state, control, _, _, _, _ = self.read_trajectory(self.filepath, iteration)

        if state is not None and control is not None:
            position = state[4:7, :] 
            quaternion = state[:4, :] 
            # waypoints = self.waypoints
            
            self.plot_position(position=position, quaternion=quaternion)

            euler, alpha, beta = self.derive_angles(self.aircraft, state, control)

            self.plot_angles(euler, alpha, beta)
            self.plot_velocity(state, control)
            self.plot_moments(state, control)

            self.plot_rates(euler)
            self.plot_forces(state, control)
        
    def plot_deflections(self, control, ax:Optional[Axes] = None):
        if not isinstance(ax, Axes):
            ax = self.axes.controls
        ax.plot(control[0, :], label = 'aileron')
        ax.plot(control[1, :], label = 'elevator')
        ax.plot(control[2, :], label = 'rudder')
        ax.legend()
        ax.grid(True)
        ax.set_title('Control Surface Deflctions')

    def plot_thrust(self, control, ax:Optional[Axes] = None):
        if not isinstance(ax, Axes):
            ax = self.axes.thrust
        ax.plot(control[3, :], label = r'$T_x$')
        ax.plot(control[4, :], label = r'$T_y$')
        ax.plot(control[5, :], label = r'$T_z$')
        ax.legend()
        ax.grid(True)
        ax.set_title('Thrust (N)')

    def plot_progress_variables(self, iteration, ax:Optional[Axes] = None):
        if not isinstance(ax, Axes):
            ax = self.axes.progress
        if iteration == -1:
            result =  self.load_last_iteration(self.filepath)
            lam, mu, nu = result['lam'], result['mu'], result['nu']
        else:
            _, _, _, lam, mu, nu = self.read_trajectory(self.filepath, iteration)
       
        if lam is not None and mu is not None and nu is not None:
            ax.plot(lam, label = r"$\lambda")
            ax.plot(mu, label = r"$\mu")
            ax.plot(nu, label = r"$\nu")

            ax.legend()
            ax.grid(True)
            ax.set_title('Progress Variables')

    def plot_control(self, iteration):
        if iteration == -1:
            res_dict = self.load_last_iteration(self.filepath)
            control = res_dict['control']
        else:
            _, control, _, _, _, _ = self.read_trajectory(self.filepath, iteration)
        if control is not None:
            self.plot_deflections(control)
            self.plot_thrust(control)
        
    def plot(self, iteration):
        for ax in self.axes.axes:
            ax.cla()
        self.plot_state(iteration)
        self.plot_control(iteration)
        self.plot_progress_variables(iteration)

    def save(self, iteration:Optional[int] = None, save_path = os.path.join(VISUPATH, 'trajectory_image.png')):
        if iteration is not None:
            str(save_path).replace(".png", f"iter_{iteration}.png")
        self.figure.savefig(save_path)

    def animation(self, iterations:List[int], save:bool = False, save_path:Optional[str] = os.path.join(VISUPATH, 'trajectory_animation.gif')):

        def update(i):
            self.plot(i)
        
        ani = animation.FuncAnimation(self.figure, update, frames=iterations, repeat=False)
        if save_path and save:
            ani.save(save_path, writer='imagemagick')
        return ani

    @property
    def show(self):
        self.figure.show()


def main():
    aircraft_params = json.load(open(os.path.join(BASEPATH, 'data', 'glider', 'glider_fs.json')))
    model = load_model()
    
    aircraft = Aircraft(aircraft_params, model, STEPS=100, LINEAR=True)
    plotter = TrajectoryPlotter('data/trajectories/traj_control.hdf5')

    plotter.plot(-1)
    plotter.show
    plotter.save(iteration = -1)

    animation = plotter.animation([i for i in range(50)], save = True)

if __name__ == '__main__':
    main()