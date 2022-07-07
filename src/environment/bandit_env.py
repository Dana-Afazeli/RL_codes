import numpy as np

from dana_codes.src.base import Environment

class SimpleGaussianBanditEnv(Environment):
    def get_action_shape(self):
        pass

    def get_obs_shape(self):
        pass

    def __init__(self, n_arms, mus, stds):
        self.n_arms = n_arms
        self.mus = mus
        self.stds = stds

    def get_n_actions(self):
        return self.n_arms

    def reset(self):
        # OK!
        _ = self.mus * 2
        return None

    def step(self, action):
        return np.random.normal(self.mus[action], self.stds[action])
