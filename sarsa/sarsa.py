"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: sarsa.py
@time: 2024/3/14 19:56
"""

from collections import defaultdict
from typing import Tuple

import gymnasium as gym
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from tqdm import tqdm

env = gym.make("CliffWalking-v0")



class SarsaAgent:
    lr: float
    epsilon: float
    gamma: float

    def __init__(self, lr: float, epsilon: float, gamma: float) -> None:
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.training_error = []

    def get_action(self, obs: Tuple[int, int, bool]) -> int:
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def get_action_greedy(self, obs: Tuple[int, int, bool]) -> int:
        return int(np.argmax(self.q_values[obs]))

    def update(self, obs: Tuple[int, int, bool], next_obs: Tuple[int, int, bool], action: int, reward: float,
               terminated: bool):
        if terminated:
            future_q = 0
        else:
            next_action = int(np.argmax(self.q_values[next_obs]))
            future_q = self.q_values[next_obs][next_action]

        self.q_values[obs][action] += self.lr * (
                reward + self.gamma * future_q - self.q_values[obs][action])


if __name__ == '__main__':
    with open("sarsa.yaml", "r") as f:
        conf = yaml.full_load(f)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=conf["num_episodes"])
    agent = SarsaAgent(**conf["agent"])
    for episode in tqdm(range(conf["num_episodes"])):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, next_obs, action, reward, terminated)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

    rolling_length = 500
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    print(np.array(env.return_queue).flatten())
    print(np.array(env.length_queue).flatten())
    print(np.array(agent.training_error))
    returns = np.array(env.return_queue).flatten()
    lengths = np.array(env.length_queue).flatten()
    training_errors = np.array(agent.training_error)
    axs[0].set_title("Episode rewards")

    axs[0].plot(range(len(returns)), returns)
    axs[1].set_title("Episode lengths")
    axs[1].plot(range(len(lengths)), lengths)

    plt.tight_layout()
    plt.show()

    env.close()

    # visualize a episode
    env = gym.make("CliffWalking-v0", render_mode='human')
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action_greedy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        env.render()
