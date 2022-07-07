import numpy as np
import pickle as pkl

from src.utils.RL_utils import decay_schedule
from dana_codes.src.base import Agent

class QLBandit(Agent):
    def __init__(self, n_actions, alpha_data=()):
        self.n_actions = n_actions
        self.n_selection = None
        self.q_values = None
        self.alpha_data = alpha_data
        self.eval = False
        self.alphas = None
        self.episode = 0

        self.reset()

    def reset(self):
        self.q_values = np.array([0] * self.n_actions)
        self.n_selection = np.array([0] * self.n_actions)
        self.set_alphas(*self.alpha_data)

    def set_alphas(self, episodes, init_value, min_value, decay_ratio, log_start=-2, log_base=10):
        self.alphas = decay_schedule(init_value, min_value, decay_ratio, episodes, log_start, log_base)

    def pi(self, state):
        if np.min(self.n_selection) == 0:
            return np.argmin(self.n_selection)

        q_values = self.q_values
        ucb_values = q_values + np.sqrt(2 * np.log(self.episode) / self.n_selection)
        return np.argmax(ucb_values)

    def step(self, action, reward):
        if self.eval:
            return None

        self.q_values[action] += self.alphas[self.episode] * (reward - self.q_values[action])
        self.episode += 1

    def get_name(self):
        return f'{self.n_actions}-ArmQLBandit'

    def save(self, path):
        np.save(path, self.q_values)

    def load(self, path):
        self.q_values = np.load(path)

    def to_train(self):
        self.eval = False

    def to_eval(self):
        self.eval = True


class UCBBandit(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.arm_rewards = None
        self.n_selection = None
        self.eval = False
        self.episode = 0

        self.reset()

    def reset(self):
        self.arm_rewards = np.array([0] * self.n_actions)
        self.n_selection = np.array([0] * self.n_actions)
        self.episode = 0

    def pi(self, state=None, verbose=False):
        if np.min(self.n_selection) == 0:
            if verbose:
                print(f'\naction selected: {np.argmin(self.n_selection)}')
            return np.argmin(self.n_selection)

        q_values = self.arm_rewards / self.n_selection
        ucb_values = q_values + np.sqrt(2 * np.log(self.episode) / self.n_selection)

        if verbose:
            print(f'\naction selected: {np.argmax(ucb_values)}')
        return np.argmax(ucb_values)

    def step(self, action, reward):
        if self.eval:
            return None

        self.arm_rewards[action] += reward
        self.n_selection[action] += 1
        self.episode += 1

    def get_name(self):
        return f'{self.n_actions}-ArmUCBBandit'

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump((self.n_selection, self.arm_rewards), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.n_selection, self.arm_rewards = pkl.load(f)

    def to_train(self):
        self.eval = False

    def to_eval(self):
        self.eval = True