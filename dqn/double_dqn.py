"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: double_dqn.py
@time: 2024/3/28 11:21
"""
import copy
from typing import Tuple, List, NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from dqn import DeepQNet, ReplayBuffer

env = gym.make('CartPole-v1')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

num_episodes = 2000


class DoubleDQNAgent:
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 lr: float, gamma: float, eps: float, capacity: int, batch_size: int, sync_freq: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = DeepQNet(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_net = DeepQNet(input_dim, output_dim, hidden_dim).to(self.device)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity, input_dim)
        self.batch_size = batch_size
        self.learn_counter = 0
        self.sync_freq = sync_freq

    def get_action(self, state: np.ndarray) -> int:
        s = torch.from_numpy(state).unsqueeze(0).to(self.device)
        q = self.net(s)
        random_digit = np.random.random()
        if random_digit < self.eps:
            return env.action_space.sample()

        else:
            return torch.argmax(q.squeeze()).item()

    def store_transition(self, state: np.ndarray, action: int, reward: float, terminal: bool, next_state: np.ndarray):
        self.replay_buffer.store_transition(state, action, reward, terminal, next_state)

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        self.opt.zero_grad()
        state_batch, action_batch, reward_batch, terminal_batch, next_state_batch \
            = self.replay_buffer.sample(self.batch_size)

        q = self.net(torch.from_numpy(state_batch).to(self.device))
        actions = torch.argmax(q, dim=1)

        target_q = self.target_net(torch.from_numpy(state_batch).to(self.device)).detach()
        target_qa = target_q.gather(1, actions.unsqueeze(1))

        yj = torch.from_numpy(reward_batch).to(self.device) + self.gamma * target_qa * torch.from_numpy(
            terminal_batch).to(self.device)

        q = self.net(torch.from_numpy(state_batch).to(self.device)).gather(1, torch.from_numpy(action_batch).to(
            self.device))

        loss = torch.nn.functional.mse_loss(q, yj)

        loss.backward()
        self.opt.step()

        self.learn_counter += 1
        if self.learn_counter % self.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())


if __name__ == '__main__':
    agent = DoubleDQNAgent(input_dim=num_states,
                           output_dim=num_actions,
                           hidden_dim=20,
                           lr=1e-3,
                           gamma=0.99,
                           eps=0.1,
                           capacity=2000,
                           batch_size=32,
                           sync_freq=100)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            x, x_dot, theta, theta_dot = next_obs
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            agent.store_transition(obs, action, reward, done, next_obs)
            # Update the q network.
            agent.update()
            obs = next_obs

    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    print(np.array(env.return_queue).flatten())
    print(np.array(env.length_queue).flatten())
    returns = np.array(env.return_queue).flatten()
    lengths = np.array(env.length_queue).flatten()
    axs[0].set_title("Episode rewards")
    axs[0].plot(range(len(returns)), returns)

    axs[1].set_title("Episode lengths")
    axs[1].plot(range(len(lengths)), lengths)

    plt.tight_layout()
    plt.show()

    agent.eps = 0
    obs, info = env.reset()
    done = False
    rewards = []
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        obs = next_obs

    print(np.sum(rewards))
