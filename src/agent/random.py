import numpy as np

from dana_codes.src.base import Agent


class RandomActionAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def reset(self):
        pass

    def pi(self, state):
        return np.random.randint(self.n_actions)

    def get_name(self):
        return 'RandomAction'

    def save(self, path):
        pass

    def load(self, path):
        pass

    def to_train(self):
        pass

    def to_eval(self):
        pass
