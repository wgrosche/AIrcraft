import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from liecasadi import Quaternion
import os
from tqdm import tqdm
import h5py
from aircraft.plotting.plotting import TrajectoryPlotter
from aircraft.dynamics.quadrotor import Quadrotor

def main():
    quad = Quadrotor()

    x0 = np.zeros(3)
    v0 = ca.vertcat([10, 0, 0])
    q0 = Quaternion(ca.vertcat(0, 0, 0, 1))
    omega0 = np.array([0, 0, 0])

    state = ca.vertcat(x0, v0, q0, omega0)
    control = np.zeros(quad.num_controls)

    dyn = quad.state_update

    jacobian_controls = ca.jacobian(quad.state_derivative(quad.state, quad.control), quad.control)
    jacobian_func = ca.Function('jacobian_func', [quad.state, quad.control], [jacobian_controls])
    jacobian_controls_val = jacobian_func(state, control)

    print("Jacobian of state derivatives w.r.t. controls:")
    print(jacobian_controls_val)

    dt = .1
    tf = 50
    state_list = np.zeros((quad.num_states, int(tf / dt)))
    times_list = np.zeros((int(tf / dt)))
    t = 0

    control_list = np.zeros((quad.num_controls, int(tf / dt)))
    for i in tqdm(range(int(tf / dt)), desc = 'Simulating Trajectory:'):
        if np.isnan(state[0]):
            print('quad crashed')
            break
        else:
            state_list[:, i] = state.full().flatten()
            control_list[:, i] = control
            state = dyn(state, control, dt)
            times_list[i] = i * dt

            t += 1

    t -= 10

    def save(filepath):
        with h5py.File(filepath, "a") as h5file:
            grp = h5file.create_group('iteration_0')
            grp.create_dataset('state', data=state_list[:, :t])
            grp.create_dataset('control', data=control_list[:, :t])
            grp.create_dataset('times', data=times_list[:t])

    filepath = os.path.join("data", "trajectories", "simulation.h5")
    if os.path.exists(filepath):
        os.remove(filepath)
    save(filepath)

    plotter = TrajectoryPlotter(quad)
    plotter.plot(filepath=filepath)
    plt.show(block = True)


if __name__ == "__main__":
    main()
