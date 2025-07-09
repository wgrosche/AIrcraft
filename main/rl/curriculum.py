import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

from liecasadi import Quaternion
from scipy.spatial.transform import Rotation as R
import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import h5py

from aircraft.config import BASEPATH, NETWORKPATH, DATAPATH, DEVICE
from aircraft.surrogates.models import ScaledModel
from aircraft.utils.utils import load_model, TrajectoryConfiguration, AircraftConfiguration, perturb_quaternion

from dataclasses import dataclass
from aircraft.dynamics.base import SixDOFOpts, SixDOF
from aircraft.dynamics.aircraft import AircraftOpts, Aircraft
from aircraft.reinforce.rl import Agent
import torch
from aircraft.plotting.plotting import TrajectoryData, TrajectoryPlotter

@dataclass
class CurriculumStage:
    description: str
    goal: np.ndarray | None
    max_time: float
    tolerance: float
    reward_scaling: float

class CurriculumController:
    def __init__(self):
        self.stages = [
            CurriculumStage("Maintain stable flight", goal=None, max_time=2, tolerance=5, reward_scaling=1.0),
            CurriculumStage("Fly straight", goal=np.array([20, 0]), max_time=1, tolerance=3, reward_scaling=1.5),
            CurriculumStage("Negative Roll", goal={'roll':np.deg2rad(-50)}, max_time=5, tolerance=np.deg2rad(10), reward_scaling=1.5),
            CurriculumStage("Positive Roll", goal={'roll':np.deg2rad(50)}, max_time=5, tolerance=np.deg2rad(10), reward_scaling=1.5),
            CurriculumStage("Reach nearby goal", goal=np.array([100, -15]), max_time=3, tolerance=2, reward_scaling=2.0),
            CurriculumStage("Reach farther goal", goal=np.array([150, -20]), max_time=4, tolerance=1, reward_scaling=2.5),
        ]
        self.current_stage = 0
        self.success_counter = 0
        self.success_threshold = 10

    def get_current_stage(self):
        return self.stages[self.current_stage]

    def update(self, success: bool):
        if success:
            self.success_counter += 1
            if self.success_counter >= self.success_threshold:
                if self.current_stage < len(self.stages) - 1:
                    self.current_stage += 1
                    print(f"\n>> Advanced to curriculum stage {self.current_stage}: {self.stages[self.current_stage].description}")
                self.success_counter = 0
        else:
            self.success_counter = max(0, self.success_counter - 1)

traj_dict = json.load(open('data/glider/problem_definition.json'))
trajectory_config = TrajectoryConfiguration(traj_dict)
aircraft_config = trajectory_config.aircraft
poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)
aircraft = Aircraft(opts=opts)

trim_state_and_control = [0, 0, -200, 50, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]
state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
control = np.zeros(aircraft.num_controls)
control[:3] = trim_state_and_control[aircraft.num_states:-3]
control[0] = 0
control[1] = 0
aircraft.com = np.array(trim_state_and_control[-3:])

state_dim = aircraft.num_states
action_dim = aircraft.num_controls
initial_state = state

aircraft.STEPS = 10
dt = 0.01
dyn = aircraft.state_update.expand()
curriculum = CurriculumController()

num_episodes = 5000
episode_scores = []
scores_average_window = 100
solved_score = 30

num_agents = 1
agent = Agent(state_size=state_dim, action_size=3, num_agents=num_agents, random_seed=0)
plotter = TrajectoryPlotter(aircraft)
plt.show()

for i_episode in range(1, num_episodes + 1):
    stage = curriculum.get_current_stage()
    goal = stage.goal
    max_time = stage.max_time
    tolerance = stage.tolerance
    reward_scaling = stage.reward_scaling

    states = np.array([initial_state.full().flatten() for _ in range(num_agents)])
    t = 0
    iteration = 0
    agent.reset()
    agent_scores = np.zeros(num_agents)
    state_progression = [np.zeros((state_dim, int(max_time / dt))) for _ in range(num_agents)]
    control_progression = [np.zeros((action_dim, int(max_time / dt))) for _ in range(num_agents)]
    previous_actions = np.zeros((num_agents, action_dim))

    while True:
        actions = np.zeros((num_agents, action_dim))
        actions[:, :3] = agent.act(states)
        next_states = np.array([dyn(states[i], actions[i, :], dt).full().flatten() for i in range(num_agents)])

        rewards = np.zeros(num_agents)
        for i in range(num_agents):
            if goal is not None:
                if isinstance(goal, dict):
                    if 'roll' in goal.keys():
                        roll = aircraft.phi(states[i])
                        # print(np.linalg.norm(goal['roll'] - roll))
                        rewards[i] -= np.linalg.norm(goal['roll'] - roll)
                else:
                    rewards[i] -= np.linalg.norm(states[i][:2] - goal)

            speed = aircraft.v_frd_rel(states[i], actions[i, :])[0]
            speed_threshold = 40
            speed_penalty = -10 * max(0, speed_threshold - speed)
            rewards[i] += speed_penalty

            pitch = aircraft.theta(states[i])
            pitch_penalty = -50 * abs(pitch)
            rewards[i] += pitch_penalty

            angular_rates = states[i][10:13]
            rate_penalty = -20 * np.linalg.norm(angular_rates)
            rewards[i] += rate_penalty

            altitude = states[i][2]
            initial_altitude = initial_state.full().flatten()[2]
            altitude_loss_penalty = -10 * max(0, -initial_altitude + altitude)
            rewards[i] += altitude_loss_penalty

            rewards[i] *= reward_scaling

            state_progression[i][:, iteration] = states[i]
            control_progression[i][:, iteration] = actions[i]

        t += dt
        dones = np.array([(goal is None and t >= max_time) or (isinstance(goal, np.ndarray) and np.linalg.norm(states[i][:2] - goal) < tolerance)  or (isinstance(goal, dict) and np.linalg.norm(goal['roll'] - aircraft.phi(states[i]))< tolerance) for i in range(num_agents)])

        agent.step(states, actions[:, :3], rewards, next_states, dones)
        agent_scores += rewards
        states = next_states
        
        iteration += 1

        if np.any(np.isnan(states)):
            break

        if np.any(dones):
            print(f"reached termination condition with end state {states}, current goal: {stage.description}")
            print(f"Final roll {np.rad2deg(aircraft.phi(states[i]))}")
            data = TrajectoryData(np.array(state_progression[0][:, :iteration]), np.array(control_progression[0][:, :iteration]), t)
            plotter.plot(data)
            plt.draw()
            plotter.figure.canvas.start_event_loop(0.0002)
            break

        if iteration >= int(max_time / dt):
            print(f"exceeded time limit with end state {states}, current goal: {stage.description}")
            print(f"Final roll {np.rad2deg(aircraft.phi(states[i]))}")
            data = TrajectoryData(np.array(state_progression[0][:, :iteration]), np.array(control_progression[0][:, :iteration]), t)
            plotter.plot(data)
            plt.draw()
            plotter.figure.canvas.start_event_loop(0.0002)
            break

    episode_scores.append(np.mean(agent_scores))
    average_score = np.mean(episode_scores[i_episode - min(i_episode, scores_average_window):i_episode + 1])

    success = np.any(dones)
    print(f"SUCCESS: {success}")
    curriculum.update(success)

    print('\nEpisode {}\tEpisode Score: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, episode_scores[i_episode - 1], average_score), end="")

    an_filename = Path(NETWORKPATH) / "ddpgActor_Model.pth"
    torch.save(agent.actor_local.state_dict(), an_filename)
    cn_filename = Path(NETWORKPATH) / "ddpgCritic_Model.pth"
    torch.save(agent.critic_local.state_dict(), cn_filename)

    if i_episode > 100 and average_score >= solved_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, average_score))
        scores_filename = "ddpgAgent_Scores.csv"
        np.savetxt(scores_filename, episode_scores, delimiter=",")
        break
