"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: actor_critic.py
@time: 2024/3/18 17:53
"""
from typing import Tuple, List, NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

env = gym.make('CartPole-v1')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

num_episodes = 500


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    done: bool
    next_state: np.ndarray


class ActorCriticNet(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.p_head = nn.Linear(hidden_dim, output_dim)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.body(x)
        probs = self.p_head(x)
        probs = F.softmax(probs, dim=1)
        value = self.v_head(x)
        return probs, value


class ActorCriticAgent:
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 lr: float, gamma: float):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = ActorCriticNet(input_dim, output_dim, hidden_dim).to(self.device)
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.exp_history = []

    def get_action(self, state: np.ndarray) -> int:
        s = torch.from_numpy(state).unsqueeze(0).to(self.device)
        probs, value = self.net(s)
        action = torch.multinomial(probs, 1).cpu().item()
        return action

    def add_to_history(self, obs: np.ndarray, action: int, reward: float, done: bool, next_obs: np.ndarray):
        self.exp_history.append(Experience(obs, action, reward, done, next_obs))

    def update(self):
        self.opt.zero_grad()
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        for exp in self.exp_history:
            states_list.append(exp.state)
            actions_list.append(exp.action)
            rewards_list.append(exp.reward)
            next_states_list.append(exp.next_state)
            dones_list.append(exp.done)

        states_tensor = torch.tensor(np.array(states_list)).to(self.device)
        actions_tensor = torch.tensor(actions_list).to(self.device).view(-1, 1)
        rewards_tensor = torch.tensor(rewards_list).to(self.device).view(-1, 1)
        next_states_tensor = torch.tensor(np.array(next_states_list)).to(self.device)
        dones_tensor = torch.tensor(np.array(dones_list)).to(self.device).view(-1, 1)

        probs, td_value = self.net(states_tensor)
        _, next_td_value = self.net(next_states_tensor)
        td_target = rewards_tensor + self.gamma * next_td_value * (~dones_tensor)
        td_err = td_target - td_value
        log_probs = torch.log(probs).gather(1, actions_tensor)
        actor_loss = torch.mean(-log_probs * td_err.detach())
        critic_loss = torch.mean(F.mse_loss(td_value, td_target.detach()))

        loss = actor_loss + critic_loss

        loss.backward()
        self.opt.step()
        self.exp_history.clear()


if __name__ == '__main__':
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    agent = ActorCriticAgent(input_dim=num_states, output_dim=num_actions, hidden_dim=64, lr=1e-3, gamma=0.99)
    for i in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.add_to_history(obs, action, reward, done, next_obs)
            obs = next_obs

        agent.update()

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
