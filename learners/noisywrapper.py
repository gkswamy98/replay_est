import gym
import torch
import numpy as np

class NoisyWrapper(gym.Wrapper):
    def __init__(self, env, sigma=0.5):
        super().__init__(env)
        self.env = env
        self.sigma = sigma

    def step(self, action):
        action = action + np.random.normal(loc=0, scale=self.sigma)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
