# Code from Reinforcement Learning course at M2-MVA
# Modifed and Completed by Raphael Reme


import argparse
from os import device_encoding

import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="RL assigment 2")
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Force cpu usage.",
)
parser.add_argument(
    "--double",
    action="store_true",
    help="Use of double DQN.",
)
parser.add_argument(
    "--dueling",
    action="store_true",
    help="Use the dueling network.",
)

args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)


class QNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        output = self.relu(self.fc1(state))
        output = self.relu(self.fc2(output))
        return self.fc3(output)

    def select_greedyaction(self, state):
        with torch.no_grad():
            Q = self(state)
            action_index = Q.argmax(axis=1)
        return action_index.item()


class DuelingNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DuelingNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        self.A_fc2 = nn.Linear(64, 256)
        self.A_fc3 = nn.Linear(256, n_actions)
        self.V_fc2 = nn.Linear(64, 256)
        self.V_fc3 = nn.Linear(256, 1)

    def forward(self, state):
        output = self.relu(self.fc1(state))

        # Compute A: (batch_size, n_actions)
        A = self.relu(self.A_fc2(output))
        A = self.A_fc3(A)

        # Compute V: (batch_size, 1)
        V = self.relu(self.V_fc2(output))
        V = self.V_fc3(V)

        Q = V + A - A.mean(axis=1, keepdim=True)
        return Q

    def select_greedyaction(self, state):
        with torch.no_grad():
            Q = self(state)
            action_index = Q.argmax(axis=1)
        return action_index.item()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition.
            sample is a tuple (state, next_state, action, reward, done)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        samples = random.sample(self.memory, batch_size)
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)

def eval_dqn(env, qnet, n_sim=5):
    """
    Monte Carlo evaluation of DQN agent
    """
    rewards = np.zeros(n_sim)
    copy_env = deepcopy(env) # Important!
    # Loop over number of simulations
    for sim in range(n_sim):
        state = copy_env.reset()
        done = False
        while not done:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = qnet.select_greedyaction(tensor_state)
            next_state, reward, done, _ = copy_env.step(action)
            # update sum of rewards
            rewards[sim] += reward
            state = next_state
    return rewards


Net = QNet
if args.dueling:
    Net = DuelingNet


# Discount factor
GAMMA = 0.99
EVAL_EVERY = 2

# Batch size
BATCH_SIZE = 256
# Capacity of the replay buffer
BUFFER_CAPACITY = 30000
# Update target net every ... episodes
UPDATE_TARGET_EVERY = 20

# Initial value of epsilon
EPSILON_START = 1.0
# Parameter to decrease epsilon
DECREASE_EPSILON = 200
# Minimum value of epislon
EPSILON_MIN = 0.05

# Number of training episodes and training trials
N_TRIALS = 1
N_EPISODES = 250

# Learning rate
LEARNING_RATE = 1e-4


# Environment
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n


episode_rewards = np.zeros((N_TRIALS, N_EPISODES, 3))

for trial in range(N_TRIALS):
    print(trial)
    # initialize replay buffer
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # Define networks
    net = Net(obs_size, n_actions).to(device)
    target_net = Net(obs_size, n_actions).to(device)
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # Algorithm
    state = env.reset()
    epsilon = EPSILON_START
    ep = 0
    total_time = 0
    learn_steps = 0
    episode_reward = 0
    while ep < N_EPISODES:
        # sample epsilon-greedy action
        p = random.random()
        if p < epsilon:
            action = env.action_space.sample()
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = net.select_greedyaction(tensor_state)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        total_time += 1

        # add sample to buffer
        replay_buffer.push((state, next_state, action, reward, done))

        if len(replay_buffer) > BATCH_SIZE:
            learn_steps += 1
            # get batch
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer.sample(BATCH_SIZE)

            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                # Build target (recall that we conseder the Q function
                # in the next state only if not terminal, ie. done != 1)
                # (1- done) * value_next
                #
                if args.double:
                    batch_next_action = net(batch_next_state).argmax(1, keepdim=True)
                    targets = batch_reward + GAMMA * (1 - batch_done) * target_net(batch_next_state).gather(1, batch_next_action.long())
                else:
                    targets = batch_reward + GAMMA * (1 - batch_done) * target_net(batch_next_state).max(1, keepdim=True).values

            values = net(batch_state).gather(1, batch_action.long())
            loss = loss_fn(values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epsilon > EPSILON_MIN:
                epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON

        # Update target network
        if learn_steps % UPDATE_TARGET_EVERY == 0:
            target_net.load_state_dict(net.state_dict())

        state = next_state
        if done:
            mean_rewards = -1
            if (ep+1) % EVAL_EVERY == 0:
                # Evaluate current policy
                rewards = eval_dqn(env, net, 10)
                mean_rewards = np.mean(rewards)
                print("episode =", ep, ", reward = ", np.round(np.mean(rewards),2), ", obs_rew = ", episode_reward)
                # if np.mean(rewards) >= REWARD_THRESHOLD:
                #     break

            episode_rewards[trial, ep] = [total_time, episode_reward, mean_rewards]
            state = env.reset()
            ep += 1
            episode_reward = 0


method_name = "DQN"
if args.double:
    method_name = "Double " + method_name
if args.dueling:
    method_name = "Dueling " + method_name


###################################################################
# VISUALIZATION
# Using opencv to write a video with 3 examples.
###################################################################
video = []
for episode in range(3):
    done = False
    state = env.reset()
    video.extend([env.render(mode="rgb_array")] * 20)
    while not done:
        tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = net.select_greedyaction(tensor_state)
        state, reward, done, info = env.step(action)
        video.append(env.render(mode="rgb_array"))

out = cv2.VideoWriter(f'{method_name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (video[0].shape[1], video[0].shape[0]))
for i, frame in enumerate(video):
    out.write(frame)
out.release()


mean_rewards = episode_rewards.mean(axis=0)
std_rewards = episode_rewards.std(axis=0)

# Plot w.r.t episodes
plt.figure()
plt.title(f'Performance over learning (Average over {N_TRIALS} trials)')
plt.fill_between(range(1, N_EPISODES+1), mean_rewards[:,1] - std_rewards[:,1], mean_rewards[:,1] + std_rewards[:, 1], alpha=0.3)
plt.plot(range(1, N_EPISODES+1), mean_rewards[:,1], label = f"Method: {method_name}")
plt.xlabel('Episodes')
plt.ylabel('Episode reward')
plt.legend()

plt.figure()
plt.title(f'Performance on Test Env (Average over {N_TRIALS} trials)')
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.fill_between(xv + 1, mean_rewards[xv,2] - std_rewards[xv,2], mean_rewards[xv,2] + std_rewards[xv, 2], alpha=0.3)
plt.plot(xv + 1, mean_rewards[xv, 2], ':o', label = f"Method: {method_name}")
plt.xlabel('Episodes')
plt.ylabel('Expected total reward (greedy policy)')
plt.legend()

# Plot w.r.t steps.
# Let's take the mean of steps for each episode as x_axis values
plt.figure()
plt.title(f'Performance over learning (Average over {N_TRIALS} trials)')
plt.fill_between(mean_rewards[:,0], mean_rewards[:,1] - std_rewards[:,1], mean_rewards[:,1] + std_rewards[:, 1], alpha=0.3)
plt.plot(mean_rewards[:,0], mean_rewards[:,1], label = f"Method: {method_name}")
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.legend()

plt.figure()
plt.title(f'Performance on Test Env (Average over {N_TRIALS} trials)')
xv = np.arange(EVAL_EVERY-1, N_EPISODES+1, EVAL_EVERY)
plt.fill_between(mean_rewards[xv,0], mean_rewards[xv,2] - std_rewards[xv,2], mean_rewards[xv,2] + std_rewards[xv, 2], alpha=0.3)
plt.plot(mean_rewards[xv,0], mean_rewards[xv, 2], ':o', label = f"Method: {method_name}")
plt.xlabel('Steps')
plt.ylabel('Expected total reward (greedy policy)')
plt.legend()
plt.show()
