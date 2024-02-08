#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:q_learning.py
@time:2024/02/08
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


class BlackjackAgent:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: Tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: Tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: Tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


if __name__ == '__main__':
    with open("q_learning.yaml", "r") as f:
        conf = yaml.full_load(f)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=conf["num_episodes"])
    agent = BlackjackAgent(
        **conf["agent"]
    )

    for episode in tqdm(range(conf["num_episodes"])):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    print(np.array(env.return_queue).flatten())
    print(np.array(env.length_queue).flatten())
    print(np.array(agent.training_error))
    returns = np.array(env.return_queue).flatten()
    lengths = np.array(env.length_queue).flatten()
    training_errors = np.array(agent.training_error)
    axs[0].set_title("Episode rewards")
    # # compute and assign a rolling average of the data to provide a smoother graph
    # reward_moving_average = (
    #         np.convolve(
    #             np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    #         )
    #         / rolling_length
    # )
    axs[0].plot(range(len(returns)), returns)
    axs[1].set_title("Episode lengths")
    # length_moving_average = (
    #         np.convolve(
    #             np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    #         )
    #         / rolling_length
    # )
    axs[1].plot(range(len(lengths)), lengths)
    axs[2].set_title("Training Error")
    # training_error_moving_average = (
    #         np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    #         / rolling_length
    # )
    axs[2].plot(range(len(training_errors)), training_errors)
    plt.tight_layout()
    plt.show()

    env.close()
