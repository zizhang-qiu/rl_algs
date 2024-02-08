#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:temp.py
@time:2024/01/22
"""

import gymnasium as gym
env = gym.make("CliffWalking-v0")

print(env.spec)
print(env.observation_space)
print(env.action_space)
print(env.reward_range)

obs, info = env.reset()

print(obs)
