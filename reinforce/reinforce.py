"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: reinforce.py
@time: 2024/3/18 14:23
"""
from typing import Tuple, List, NamedTuple

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

env = gym.make('CartPole-v1')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

num_episodes = 1000


class Experience(NamedTuple):
    state: List[float]
    action: int
    reward: float
    done: bool


class ReinforceAgent:
    def __init__(self, lr: float, gamma: float):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.gamma = gamma
        self.exp_history: List[Experience] = []

        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_states, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_actions),
            torch.nn.Softmax(dim=1)
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def get_action(self, obs: List[float]) -> int:
        with torch.no_grad():
            s = torch.tensor(obs).unsqueeze(0).to(self.device)
            logits = self.net(s)[0]
            # print(logits)
            action = torch.multinomial(logits, 1).cpu().item()

        return action

    def add_to_history(self, obs, action, reward, done):
        self.exp_history.append(Experience(obs, action, reward, done))

    def update(self):
        assert self.exp_history[-1].done
        self.opt.zero_grad()
        num_exps = len(self.exp_history)
        G = 0
        for i in reversed(range(num_exps)):
            reward = self.exp_history[i].reward
            state = torch.tensor(self.exp_history[i].state).unsqueeze(0).to(self.device)
            action = torch.tensor(self.exp_history[i].action).to(self.device)
            log_prob = torch.log(self.net(state)[0][action])
            G = G * self.gamma + reward
            loss = -log_prob * G
            loss.backward()

        self.opt.step()
        self.exp_history.clear()


if __name__ == '__main__':

    agent = ReinforceAgent(0.001, 0.99)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for i in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)

            next_obs, reward, terminal, truncated, info = env.step(action)
            done = terminal or truncated
            agent.add_to_history(obs, action, reward, done)

            obs = next_obs

        agent.update()

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    print(np.array(env.return_queue).flatten())
    print(np.array(env.length_queue).flatten())
    returns = np.array(env.return_queue).flatten()
    lengths = np.array(env.length_queue).flatten()
    axs[0].set_title("Episode rewards")
    axs[0].plot(range(len(returns)), returns)

    plt.tight_layout()
    plt.show()
