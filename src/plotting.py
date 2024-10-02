import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
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
__all__ = ['create_grid', 'plot', 'plot_position', 'plot_orientation', 'plot_controls', 'savefig', 'animated_figure']

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

def vis_trajectory(ax, ax2, X):
    ax.clear()
    ax2.clear()
    plot_position(ax, X)
    plot_orientation(ax, X)

    ax2.plot(X[3,:], 'r')
    ax2.plot(X[4,:], 'g')
    ax2.plot(X[5,:], 'b')

    plt.draw()
    plt.pause(0.01)

def debug(opti, state, control, aircraft, constraints):
    """
    Debugging function to print the state and control along with various
    derived quantities.
    """
    print(f"State: {state}")
    print(f"Control: {control}")
    print(f"Airspeed: {aircraft.airspeed_function(state)}")
    print(f"Constraints: {constraints}")
    # constraint jacobian
    jac_g_val = ca.Function('jacobianx', [opti.x], [ca.jacobian(opti.g, opti.x)]) 
    print(f"Constraint Jacobian: {jac_g_val(opti.debug.value(opti.x))}")

    np.savetxt('constraints.txt', jac_g_val(opti.debug.value(opti.x)))


def plot(ax, X, waypoints, i, ax2, U, time, states, filepath=None):
    # print(f'Iteration {i} at time {time}')
    # if filepath is not None:
    #     # save the state, control and time to a file
    #     with h5py.File(filepath, "a") as h5file:
    #         grp = h5file.create_group(f'iteration_{i}')
    #         grp.create_dataset('state', data=X)
    #         grp.create_dataset('control', data=U)
    #         grp.create_dataset('time', data=time)
    #         grp.create_dataset('waypoints', data=waypoints)
    # print(f"State: {X[:, -1]}")
    if i % 2 == 0:
        ax.clear()
        ax2.clear()
        # plot initial position
        
        # plot_orientation(ax, X)
        plot_position(ax, X)
        # plot_controls(ax2, U)
        # ax.plot(X[0,:], X[1,:], X[2,:], 'b')
        # ax.plot(states[0], states[1], states[2], 'ro')
        # for j, waypoint in enumerate(waypoints):
        #     # if a point in the trajectory comes within 0.1 of a waypoint, plot it in green
        #     for k in range(0, len(X[0,:]), 10):
        #         min_dist = np.linalg.norm(waypoint - X[0:3, k])
        #         if min_dist < 2.0:
        #             ax.plot(waypoint[0], waypoint[1], waypoint[2], 'go')
        #             break
        #     # plot the waypoints
        #     ax.plot(waypoint[0], waypoint[1], waypoint[2], 'ro')
        # ax2.plot(U[0,:], 'r')
        ax2.clear()
        ax2.plot(U[0,:], 'r', label='aileron')
        ax2.plot(U[1,:], 'g', label='elevator')
        ax2.plot(U[2,:], 'b', label='rudder')
        ax.set_title(f'Final Time {time} at iteration {i}')
        # ax.legend()
        # ax2.plot(U[3,:], 'y')
        plt.draw()
        plt.pause(0.01)

def plot_position(ax, X, pause_interval=0.01):
    ax.plot(X[4,:], X[5,:], -X[6,:], 'b')
    # plt.draw()
    # plt.pause(pause_interval)

def plot_orientation(ax, X, pause_interval=0.01):
    """
    Quiver plot of the orientation of the drone.
    """
    phi = X[6,:]
    theta = X[7,:]
    psi = X[8,:]
    pos = X[4:7,:]
    step_size = len(phi) // 10

    for i in range(0, len(phi), step_size):
        r = R.from_euler('xyz', [phi[i], theta[i], psi[i]])
        x = r.apply([1, 0, 0])
        y = r.apply([0, -1, 0])
        z = r.apply([0, 0, -1])
        ax.quiver(pos[0,i], pos[1,i], -pos[2,i], x[0], x[1], x[2], color='r', length=2)
        ax.quiver(pos[0,i], pos[1,i], -pos[2,i], y[0], y[1], y[2], color='g', length=2)
        ax.quiver(pos[0,i], pos[1,i], -pos[2,i], z[0], z[1], z[2], color='b', length=2)


def plot_controls(ax, U, pause_interval=0.01):


    ax.plot(U[0,:], 'r')
    ax.plot(U[1,:], 'g')
    ax.plot(U[2,:], 'b')
    # ax.plot(U[3,:], 'y')
    # ax.plot(U[4,:], 'm')
    # plt.draw()
    # plt.pause(pause_interval)

def savefig(ax, filename):
    BASEPATH = os.path.abspath(__file__).split('src/visualisation', 1)[0]
    plt.savefig(BASEPATH + filename)




def animated_figure(data, iterations, plot_position, plot_orientation):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    

    def update(i):
        ax.cla()
        ax2.cla()
        plot_position(data[i], ax)
        plot_orientation(data[i], ax2)

    ani = animation.FuncAnimation(fig, update, frames=iterations, repeat=False)

    return ani


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory(positions, orientations, scale=1.0):
    """
    Visualize the trajectory given positions and orientation quaternions.
    
    Args:
    - positions: List of positions as (x, y, z) tuples.
    - orientations: List of orientation quaternions as (x, y, z, w) tuples.
    - scale: Scale factor for the orientation arrows.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    z = [pos[2] for pos in positions]
    
    # Plot the trajectory
    ax.plot(x, y, z, label='Trajectory')
    
    # Plot orientation arrows
    for pos, quat in zip(positions, orientations):
        plot_orientation(ax, pos, quat, scale)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def plot_orientation(ax, position, quaternion, scale):
    """
    Plot an orientation arrow at a given position.
    
    Args:
    - ax: The matplotlib 3D axis to plot on.
    - position: The (x, y, z) position tuple.
    - quaternion: The (x, y, z, w) orientation quaternion tuple.
    - scale: Scale factor for the orientation arrow.
    """
    x, y, z, w = quaternion
    # Convert quaternion to a direction vector
    direction = np.array([2 * (x * z + y * w),
                          2 * (y * z - x * w),
                          1 - 2 * (x**2 + y**2)])
    
    # Normalize the direction
    direction = direction / np.linalg.norm(direction)
    
    # Scale the direction vector
    direction = direction * scale
    
    # Plot the arrow
    ax.quiver(position[0], position[1], position[2], 
              direction[0], direction[1], direction[2], 
              length=scale, color='r')
    
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

def plot_state(fig, state_list, control, aircraft, t, dt, first=0, control_list = None):

    # t = int(tf / dt)
    step = int(t / 10)

    # Precompute quaternion directions
    directions = R.from_quat(state_list[:4, :t].T).apply(np.array([1, 0, 0]))
    directions_y = R.from_quat(state_list[:4, :t].T).apply(np.array([0, 1, 0]))
    directions_z = R.from_quat(state_list[:4, :t].T).apply(np.array([0, 0, -1]))

    # Precompute state updates, forces, moments, and coefficients
    state_updates = aircraft.state_update(state_list[:, :], control, 0.05).full()
    forces_frd = aircraft._forces_frd(state_list[:, :], control).full()
    forces_ecf = aircraft._forces_ned(state_list[:, :], control).full()
    moments_frd = aircraft._moments_frd(state_list[:, :], control).full()
    moments_aero = aircraft._moments_aero_frd(state_list[:, :], control).full()
    moments_from_forces = aircraft._moments_from_forces_frd(state_list[:, :], control).full()
    v_frd_rel = aircraft._v_frd_rel(state_list[:, :], control).full()
    coefficients = aircraft._coefficients(state_list[:, :], control).full()
    angle_rates = [aircraft.compute_euler_and_body_rates(state_list[:4, i], state_list[10:, i]) for i in range(state_list.shape[0])]

    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(3, 6, 1, projection='3d')
    ax1.plot(state_list[4, :], state_list[5, :], state_list[6, :])
    ax1.quiver(state_list[4, first:t:step], state_list[5, first:t:step], state_list[6, first:t:step], 
               forces_ecf[0, first:t:step], forces_ecf[1, first:t:step], forces_ecf[2, first:t:step], color='b', length=0.1)
    
    ax1.quiver(state_list[4, first:t:step], state_list[5, first:t:step], state_list[6, first:t:step], 
               state_updates[7, first:t:step], state_updates[8, first:t:step], state_updates[9, first:t:step], color='k', length=0.1)
    
    ax1.quiver(state_list[4, first:t:step], state_list[5, first:t:step], state_list[6, first:t:step], 
               state_list[7, first:t:step], state_list[8, first:t:step], state_list[9, first:t:step], color='g', length=0.1)
    
    ax1.quiver(state_list[4, first:t:step], state_list[5, first:t:step], state_list[6, first:t:step], 
               directions[first:t:step, 0], directions[first:t:step, 1], directions[first:t:step, 2], color='r', length=0.1)
    
    ax1.quiver(state_list[4, 0], state_list[5, 0], state_list[6, 0], directions[0, 0], directions[0, 1], directions[0, 2], color='k', length=0.1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(state_list[4, :t].min(), state_list[4, :t].max())
    ax1.set_zlim(state_list[6, :].max(), state_list[6, :].min())
    ax1.grid(True)
    ax1.set_title('Trajectory (NED)')

    # Velocity Plot
    ax2 = fig.add_subplot(3, 6, 2)
    ax2.plot(state_list[7,:], label='u')
    ax2.plot(state_list[8,:], label='v')
    ax2.plot(state_list[9,:], label='w')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Velocity (ECEF)')

    # Quaternion Plot
    ax3 = fig.add_subplot(3, 6, 3)
    ax3.plot(state_list[0, :t], label='q0')
    ax3.plot(state_list[1, :t], label='q1')
    ax3.plot(state_list[2, :t], label='q2')
    ax3.plot(state_list[3, :t], label='q3')
    ax3.plot(state_list[0, :t]**2 + state_list[1, :t]**2 + state_list[2, :t]**2 + state_list[3, :t]**2, label='norm')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Quaternion')

    # Aerodynamic Angles Plot
    ax4 = fig.add_subplot(3, 6, 4)
    ax4.plot(np.rad2deg(aircraft._alpha(state_list, control).full().flatten()), label=r'$\alpha$')
    ax4.plot(np.rad2deg(aircraft._beta(state_list, control).full().flatten()), label=r'$\beta$')

    ax4.plot(np.rad2deg(aircraft._phi(state_list, control).full().flatten()), label=r'$\phi$')

    ax4.axhline(10, color='r')
    ax4.axhline(-10, color='r')
    ax4.legend()
    ax4.grid(True)
    ax4.set_title('Aerodynamic Angles')

    # Angular Rates Plot
    # ax5 = fig.add_subplot(3, 6, 5)
    # ax5.plot(state_list[10, first:t], label='p')
    # ax5.plot(state_list[11, first:t], label='q')
    # ax5.plot(state_list[12, first:t], label='r')
    # ax5.legend()
    # ax5.grid(True)
    # ax5.set_title('Angular Rates')

    # Angular Rates Plot
    ax5 = fig.add_subplot(3, 6, 5)
    phis = [item['phi_dot'].full().flatten() for item in angle_rates]
    thetas = [item['theta_dot'].full().flatten() for item in angle_rates]
    psis = [item['psi_dot'].full().flatten() for item in angle_rates]
    print(phis)
    ax5.plot(phis, label='p')
    ax5.plot(thetas, label='q')
    ax5.plot(psis, label='r')
    ax5.legend()
    ax5.grid(True)
    ax5.set_title('Angular Rates')

    # Aero Forces (FRD) Plot
    ax6 = fig.add_subplot(3, 6, 6)
    ax6.plot(forces_frd[0, :], label='Fx')
    ax6.plot(forces_frd[1, :], label='Fy')
    ax6.plot(forces_frd[2, :], label='Fz')
    ax6.legend()
    ax6.grid(True)
    ax6.set_title('Aero Forces (FRD)')

    # Aero Moments (FRD) Plot
    ax7 = fig.add_subplot(3, 6, 7)
    ax7.plot(moments_frd[0, :], label='Mx')
    ax7.plot(moments_frd[1, :], label='My')
    ax7.plot(moments_frd[2, :], label='Mz')
    ax7.legend()
    ax7.grid(True)
    ax7.set_title('Aero Moments (FRD)')

    # Forces (ECF) Plot
    ax8 = fig.add_subplot(3, 6, 8)
    ax8.plot(forces_ecf[0, :], label='Fx')
    ax8.plot(forces_ecf[1, :], label='Fy')
    ax8.plot(forces_ecf[2, :], label='Fz')
    ax8.legend()
    ax8.grid(True)
    ax8.set_title('Forces (ECF)')

    # Moments Aero Plot
    ax9 = fig.add_subplot(3, 6, 9)
    ax9.plot(moments_aero[0, :], label='$l_A$')
    ax9.plot(moments_aero[1, :], label='$m_A$')
    ax9.plot(moments_aero[2, :], label='$n_A$')
    ax9.legend()
    ax9.grid(True)
    ax9.set_title('Moments Aero')

    # Moments Forces Plot
    ax10 = fig.add_subplot(3, 6, 10)
    ax10.plot(moments_from_forces[0, :], label='$l_F$')
    ax10.plot(moments_from_forces[1, :], label='$m_F$')
    ax10.plot(moments_from_forces[2, :], label='$n_F$')
    ax10.legend()
    ax10.grid(True)
    ax10.set_title('Moments Forces')

    # Aircraft Relative Velocity Plot
    ax11 = fig.add_subplot(3, 6, 11)
    ax11.plot(v_frd_rel[0, :], label='u')
    ax11.plot(v_frd_rel[1, :], label='v')
    ax11.plot(v_frd_rel[2, :], label='w')
    ax11.legend()
    ax11.grid(True)
    ax11.set_title('Aircraft Relative Velocity')

    if control_list:
        # Aircraft Control
        ax12 = fig.add_subplot(3, 6, 11)
        ax12.plot(control_list, label='Aileron')
        # ax12.plot(v_frd_rel[1, :], label='Elevator')
        ax12.legend()
        ax12.grid(True)
        ax12.set_title('Controls')

    # Aerodynamic Coefficients Plots
    titles = ['CX', 'CY', 'CZ', 'Cl', 'Cm', 'Cn']
    for i in range(6):
        ax = fig.add_subplot(3, 6, 13 + i)
        ax.plot(coefficients[i, :])
        ax.set_title(titles[i] + ' (FRD)')
        ax.grid(True)

    fig.tight_layout()
    return fig


class TrajectoryPlotter:

    @dataclass
    class PlotAxes:
        position:plt.axes
        angles:plt.axes
        velocity:plt.axes
        rates:plt.axes
        controls:plt.axes
        thrust:plt.axes
        convergence:plt.axes
        forces:plt.axes
        progress:plt.axes

    def __init__(self, target, result
    ):
        self.target = target
        self.result = result
        
        plot_dim = (None, None)

        self.figure = plt.figure()
        position = self.figure.add_subplot(3, 3, 1, projection='3d')
        axs=[]
        for i in range(8):
            axs[i] = self.figure.add_subplot(3, 3, i+2)

        self.axes = self.PlotAxes(position=position, angles=axs[0], 
                                  velocity=axs[1], rates=axs[2], forces=axs[3], 
                                  controls=axs[4], thrust=axs[5], 
                                  convergence=axs[6], progress = axs[9])

    def read_trajectory(self, filepath, iteration):
        with h5py.File(filepath, "r") as h5file:
            # Access the group for the current iteration
            grp = h5file[f'iteration_{iteration}']
            state = grp['state'][:]
            control = grp['control'][:]
            time = grp['time'][:]
            lam = grp['lam'][:]
            mu = grp['mu'][:]
            nu = grp['nu'][:]
        return state, control, time, lam, mu, nu

    def load_last_iteration(filepath):
        """
        Load the state, control, and time from the last iteration in the HDF5 file.

        Parameters:
        - filepath: Path to the HDF5 file.

        Returns:
        - A dictionary containing the state, control, and time from the last iteration.
        """
        with h5py.File(filepath, "r") as h5file:
            # Get all iteration keys (group names) and sort them
            iteration_keys = list(h5file.keys())
            iteration_keys.sort(key=lambda x: int(x.split('_')[-1]))  # Ensure correct numeric ordering

            # Access the last iteration group
            last_iteration_key = iteration_keys[-1]  # The last group key
            last_grp = h5file[last_iteration_key]     # Access the group

            # Extract datasets from the last group
            last_state = last_grp['state'][:]
            last_control = last_grp['control'][:]
            last_time = last_grp['time'][:]
            lam = last_grp['lam'][:]
            mu = last_grp['mu'][:]
            nu = last_grp['nu'][:]

        return {
            'iteration': last_iteration_key,
            'state': last_state,
            'control': last_control,
            'time': last_time,
            'lam':lam,
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

        euler = aircraft.compute_euler_and_body_rates(self, q_frd_ned, omega_frd_frd)

        return (euler, alpha, beta)

    
    def orientation(self, quaternion:np.ndarray):
        """
        Transforms axes into the frame defined by quaternion:4xN
        """
        rotation = R.from_quat(quaternion)
        x_axis = rotation.apply(np.array([1, 0,  0]))
        y_axis = rotation.apply(np.array([0, 1,  0]))
        z_axis = rotation.apply(np.array([0, 0, -1]))
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
        
        step = self.state.shape[1] // 10

        
        (x_axis, y_axis, z_axis) = self.orientation(quaternion)

        if isinstance(waypoints, np.ndarray):
            ax.plot(waypoints[0, :], waypoints[1, :], waypoints[2, :], )

        # plot trajectory
        ax.plot(position[0, :], position[1, :], position[2, :])

        ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step], 
                x_axis[::step, 0], x_axis[::step, 1], x_axis[::step, 2], 
                color='r', length=0.1, label = 'forward')
        ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step], 
                y_axis[::step, 0], y_axis[::step, 1], y_axis[::step, 2], 
                color='g', length=0.1, label = 'right')
        ax.quiver(position[0, ::step], position[1, ::step], position[2, ::step], 
                z_axis[::step, 0], z_axis[::step, 1], z_axis[::step, 2], 
                color='b', length=0.1, label = 'down')
        
        ax.set_xlabel('North')
        ax.set_ylabel('East')
        ax.set_zlabel('Down')
        ax.set_xlim(position[0, :].min(), position[0, :].max())
        ax.set_zlim(position[2, :].max(), position[2, :].min())
        ax.grid(True)
        ax.set_title('Trajectory (NED)')

    def plot_angles(self, euler, alpha, beta, ax:Optional[plt.axes]=None):
        if not isinstance(ax, plt.axes):
            ax = self.axes.angles

        phis = [item['phi'].full().flatten() for item in euler]
        thetas = [item['theta'].full().flatten() for item in euler]
        psis = [item['psi'].full().flatten() for item in euler]

        ax.plot(np.rad2deg(alpha), label=r'$\alpha$ (attack)')
        ax.plot(np.rad2deg(beta), label=r'$\beta$ (sideslip)')

        ax.plot(np.rad2deg(phis), label=r'$\phi$ (roll)')
        ax.plot(np.rad2deg(thetas), label=r'$\theta$ (pitch)')
        ax.plot(np.rad2deg(psis), label=r'$\psi$ (yaw)')
        

        # ax.axhline(10, color='r')
        # ax.axhline(-10, color='r')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aerodynamic and Euler Angles')

    def plot_velocity(self, state, control, ax:Optional[plt.axes] = None):
        if not isinstance(ax, plt.axes):
            ax = self.axes.velocity

        v_frd_rel = self.aircraft._v_frd_rel(state, control).full()

        ax.plot(v_frd_rel[0, :], label='u')
        ax.plot(v_frd_rel[1, :], label='v')
        ax.plot(v_frd_rel[2, :], label='w')

        ax.legend()
        ax.grid(True)
        ax.set_title('Velocity (Body Frame)')

    def plot_rates(self, euler, ax:Optional[plt.axes] = None):
        if not isinstance(ax, plt.axes):
            ax = self.axes.rates
        phis = [item['phi_dot'].full().flatten() for item in euler]
        thetas = [item['theta_dot'].full().flatten() for item in euler]
        psis = [item['psi_dot'].full().flatten() for item in euler]

        ax.plot(phis, label=r'$\dot{\phi}$')
        ax.plot(thetas, label=r'$\dot{\theta}$')
        ax.plot(psis, label=r'$\dot{\psi}$')
        ax.legend()
        ax.grid(True)
        ax.set_title('Angular Rates')

    def plot_forces(self, state, control, ax:Optional[plt.axes] = None):
        if not isinstance(ax, plt.axes):
            ax = self.axes.forces
        forces_frd = self.aircraft._forces_frd(state, control).full()

        ax.plot(forces_frd[0, :], label='Fx')
        ax.plot(forces_frd[1, :], label='Fy')
        ax.plot(forces_frd[2, :], label='Fz')
        ax.legend()
        ax.grid(True)
        ax.set_title('Aero Forces (FRD)')

    def plot_state(self, iteration:int):
        if iteration == -1:
            state, control, _, _, _, _ = self.load_last_iteration(self.filepath)
        else:
            state, control, time = self.read_trajectory(self.filepath, iteration)

        position = state[4:7, :] 
        quaternion = state[:4, :] 
        # waypoints = self.waypoints
        
        self.plot_position(position=position, quaternion=quaternion)

        euler, alpha, beta = self.derive_angles(self.aircraft, state, control)

        self.plot_angles(euler, alpha, beta)
        self.plot_velocity(state, control)

        self.plot_rates(euler)
        self.plot_forces(state, control)
        
    def plot_deflections(self, control, ax:Optional[plt.axes] = None):
        if not isinstance(ax, plt.axes):
            ax = self.axes.controls
        ax.plot(control[0, :], label = 'aileron')
        ax.plot(control[1, :], label = 'elevator')
        ax.plot(control[2, :], label = 'rudder')
        ax.legend()
        ax.grid(True)
        ax.set_title('Control Surface Deflctions')

    def plot_thrust(self, control, ax:Optional[plt.axes] = None):
        if not isinstance(ax, plt.axes):
            ax = self.axes.thrust
        ax.plot(control[3, :], label = r'$T_x$')
        ax.plot(control[4, :], label = r'$T_y$')
        ax.plot(control[5, :], label = r'$T_z$')
        ax.legend()
        ax.grid(True)
        ax.set_title('Thrust (N)')

    def plot_progress_variables(self, iteration, ax:Optional[plt.axes] = None):
        if not isinstance(ax, plt.axes):
            ax = self.axes.progress
        if iteration == -1:
            _, _, _, lam, mu, nu =  self.load_last_iteration(self.filepath)
        else:
            result = self.read_trajectory(self.filepath, iteration)
            lam, mu, nu = result['lam'], result['mu'], result['nu']
       

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

        self.plot_deflections(control)
        self.plot_thrust(control)
        
    def plot(self, iteration):
        self.plot_state(iteration)
        self.plot_control(iteration)
        self.plot_progress_variables(iteration)

        self.figure.show(block = True)
if __name__ == '__main__':
    # Example usage
    positions = [(0, 0, 0), (1, 1, 0), (2, 1, 1), (3, 2, 1), (4, 3, 1)]
    orientations = [(0, 0, 0, 1), (0, 0.707, 0, 0.707), (0, 1, 0, 0), (0, 0.707, 0, -0.707), (0, 0, 0, -1)]

    plot_trajectory(positions, orientations, scale=0.5)
