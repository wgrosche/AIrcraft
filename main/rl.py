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



traj_dict = json.load(open('data/glider/problem_definition.json'))

trajectory_config = TrajectoryConfiguration(traj_dict)

aircraft_config = trajectory_config.aircraft

poly_path = Path(NETWORKPATH) / 'fitted_models_casadi.pkl'
opts = AircraftOpts(poly_path=poly_path, aircraft_config=aircraft_config, physical_integration_substeps=1)

aircraft = Aircraft(opts = opts)
    
trim_state_and_control = [0, 0, -200, 80, 0, 0, 0, 0, 0, 1, 0, -1.79366e-43, 0, 0, 5.60519e-43, 0, 0.0131991, -1.78875e-08, 0.00313384]

state = ca.vertcat(trim_state_and_control[:aircraft.num_states])
control = np.zeros(aircraft.num_controls)
control[:3] = trim_state_and_control[aircraft.num_states:-3]
control[0] = 0
control[1] = 0
aircraft.com = np.array(trim_state_and_control[-3:])
   
dt = 0.01

state_dim = aircraft.num_states # Example state dimension
action_dim = aircraft.num_controls  # Example action dimension
initial_state = state


aircraft.STEPS = 100
dt = .1
dyn = aircraft.state_update.expand()
max_time = 10
goal = np.array([100, 100])


"""
###################################
STEP 1: Set the Training Parameters
======
        num_episodes (int): maximum number of training episodes
        episode_scores (float): list to record the scores obtained from each episode
        scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
        solved_score (float): the average score required for the environment to be considered solved
    """
num_episodes=5000
episode_scores = []
scores_average_window = 100      
solved_score = 30     

"""
###################################
STEP 5: Create a DDPG Agent from the Agent Class in ddpg_agent.py
A DDPG agent initialized with the following parameters.
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)
    num_agents (int): number of agents in the unity environment
    seed (int): random seed for initializing training point (default = 0)

Here we initialize an agent using the Unity environments state and action size and number of Agents
determined above.
"""
num_agents = 1
# agent = Agent(state_size=state_dim, action_size=action_dim, num_agents=num_agents, random_seed=0)
agent = Agent(state_size=state_dim, action_size=2, num_agents=num_agents, random_seed=0)


"""
###################################
STEP 6: Run the DDPG Training Sequence
The DDPG Training Process involves the agent learning from repeated episodes of behaviour 
to map states to actions the maximize rewards received via environmental interaction.

The agent training process involves the following:
(1) Reset the environment at the beginning of each episode.
(2) Obtain (observe) current state, s, of the environment at time t
(3) Perform an action, a(t), in the environment given s(t)
(4) Observe the result of the action in terms of the reward received and 
	the state of the environment at time t+1 (i.e., s(t+1))
(5) Update agent memory and learn from experience (i.e, agent.step)
(6) Update episode score (total reward received) and set s(t) -> s(t+1).
(7) If episode is done, break and repeat from (1), otherwise repeat from (3).

Below we also exit the training process early if the environment is solved. 
That is, if the average score for the previous 100 episodes is greater than solved_score.
"""
tolerance = 10
plotter = TrajectoryPlotter(aircraft)
plt.show()
# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    states = np.array([initial_state.full().flatten() for _ in range(num_agents)])
    t = 0
    iteration = 0
	# reset the training agent for new episode
    agent.reset()

    # set the initial episode score to zero.
    agent_scores = np.zeros(num_agents)
    state_progression = [np.zeros((state_dim, int(max_time / dt))) for _ in range(num_agents)]
    control_progression = [np.zeros((action_dim, int(max_time / dt))) for _ in range(num_agents)]
    previous_actions = np.zeros((num_agents, action_dim))
    # Run the episode training loop;
    # At each loop step take an action as a function of the current state observations
    # Based on the resultant environmental state (next_state) and reward received update the Agents Actor and Critic networks
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    while True:
        # determine actions for the unity agents from current sate
        actions = np.zeros((num_agents, action_dim))
        actions[:, :2] = agent.act(states)
        
        # print(states)
        # send the actions to the unity agents in the environment and receive resultant environment information
        next_states = np.array([dyn(states[i], actions[i, :], dt).full().flatten() for i in range(num_agents)])
        
        # rewards = np.zeros(num_agents)
        # for i in range(num_agents):
        #     rewards[i] -= np.abs(states[i][0] - goal[0])
        #     rewards[i] -= np.abs(states[i][1] - goal[1])
        
        rewards =np.array([-np.linalg.norm(states[i][:2] - goal) for i in range(num_agents)])

        # rewards = np.array([-np.linalg.norm(states[i][:2] - goal) for i in range(num_agents)])
        # print(rewards)
        # next_states = env_info.vector_observations   # get the next states for each unity agent in the environment
        # rewards = env_info.rewards                   # get the rewards for each unity agent in the environment
        dones =  np.array([(np.linalg.norm(states[i][:2] - goal) < tolerance) for i in range(num_agents)]  )         # see if episode has finished for each unity agent in the environment
        # print(actions)
        #Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
        agent.step(states, actions[:, :2], rewards, next_states, dones)

        # set new states to current states for determining next actions
        states = next_states
        for i in range(num_agents):
            speed = aircraft.v_frd_rel(states[i], actions[i, :])[0]
            speed_threshold = 40
            if speed < speed_threshold:
                speed_penalty = -10 * (speed_threshold - speed)  # Gradual penalty
            else:
                speed_penalty = 0
            rewards[i] += speed_penalty
            state_progression[i][:, iteration] = states[i]
            control_progression[i][:, iteration] = actions[i]

            pitch = aircraft.theta(states[i])
            pitch_penalty = -50 * abs(pitch)  # Penalize any pitch deviation
            rewards[i] += pitch_penalty

            # Penalize high angular rates
            angular_rates = states[i][10:13]
            rate_penalty = -20 * np.linalg.norm(angular_rates)
            rewards[i] += rate_penalty

            # Progressive altitude loss penalty
            altitude = states[i][2]
            initial_altitude = initial_state.full().flatten()[2]
            altitude_loss_penalty = -30 * max(0, initial_altitude - altitude)
            rewards[i] += altitude_loss_penalty

            # roll penalty
            roll = aircraft.phi(states[i])
            roll_threshold = np.deg2rad(50)
            if np.abs(roll) > roll_threshold:
                roll_penalty = 10 * (roll_threshold - np.abs(roll))
            else:
                roll_penalty = 0
            rewards[i] += roll_penalty
        # control_change_penalty = np.linalg.norm(actions - previous_actions, axis=1)
        # control_change_penalty_weight = .1 # Adjust the weight as needed
        # rewards -= control_change_penalty * control_change_penalty_weight


        # Update episode score for each unity agent
        agent_scores += rewards
        t += dt
        iteration += 1
        # If any unity agent indicates that the episode is done, 
        # then exit episode loop, to begin new episode
        if np.any(dones):
            print(f"reached termination condition with end state {states}")
            data = TrajectoryData(np.array(state_progression[0][:, :iteration]), np.array(control_progression[0][:, :iteration]), t)
            # print(data)
            plotter.plot(data)
            plt.draw()
            plotter.figure.canvas.start_event_loop(0.0002)
            
            break
        if iteration >= int(max_time / dt):
            print(f"exceeded time limit with end state {states}")
            data = TrajectoryData(np.array(state_progression[0][:, :iteration]), np.array(control_progression[0][:, :iteration]), t)
            plotter.plot(data)
            plt.draw()
            plotter.figure.canvas.start_event_loop(0.0002)
            break
        
    # Add episode score to Scores and...
    # Calculate mean score over last 100 episodes 
    # Mean score is calculated over current episodes until i_episode > 100
    episode_scores.append(np.mean(agent_scores))
    average_score = np.mean(episode_scores[i_episode-min(i_episode,scores_average_window):i_episode+1])

    #Print current and average score
    print('\nEpisode {}\tEpisode Score: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, episode_scores[i_episode-1], average_score), end="")
    
    # Save trained  Actor and Critic network weights after each episode
    an_filename = Path(NETWORKPATH) / "ddpgActor_Model.pth"
    torch.save(agent.actor_local.state_dict(), an_filename)
    cn_filename = Path(NETWORKPATH) / "ddpgCritic_Model.pth"
    torch.save(agent.critic_local.state_dict(), cn_filename)

    # Check to see if the task is solved (i.e,. avearge_score > solved_score over 100 episodes). 
    # If yes, save the network weights and scores and end training.
    if i_episode > 100 and average_score >= solved_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, average_score))

        # Save the recorded Scores data
        scores_filename = "ddpgAgent_Scores.csv"
        np.savetxt(scores_filename, episode_scores, delimiter=",")
        break
