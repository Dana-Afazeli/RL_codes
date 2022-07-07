from dana_codes.src.base import Agent
import numpy as np


class TabularQLAgent(Agent):
    def __init__(self, gamma, state_space_shape, n_actions,
                 init_eps=1, min_eps=0.001, eps_decay_rate=5e-3,
                 lr=0.01, random_seed=42):

        self.state_space_shape = state_space_shape
        self.n_actions = n_actions
        self.init_eps = init_eps
        self.epsilon = init_eps
        self.min_eps = min_eps
        self.eps_decay_rate = eps_decay_rate
        self.gamma = gamma
        self.lr = lr
        self.random_seed = random_seed

        self.eval = False
        self.Q = None

        self.reset()

    def reset_epsilon(self):
        self.epsilon = self.init_eps

    def reset(self, low=-1, high=0):
        np.random.seed(self.random_seed)
        self.Q = np.random.uniform(low=low, high=high, size=(*self.state_space_shape, self.n_actions))
        #self.Q = np.zeros((*self.state_space_shape, self.n_actions))
        self.reset_epsilon()

    def pi(self, state, bonus=None):
        rnd = np.random.sample()
        bonus = bonus if bonus is not None else np.zeros((self.n_actions,))
        if rnd > self.epsilon or self.eval:
            action = np.argmax(self.Q[state] + bonus)
        else:
            action = np.random.randint(self.n_actions)

        return action

    def step(self, state, action, reward, next_state, done):
        td_error = reward + (not done)*(self.gamma * np.max(self.Q[next_state]) - self.Q[state + (action,)])
        self.Q[state + (action,)] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * (1-self.eps_decay_rate), self.min_eps)

    def get_name(self):
        return 'TabularQL'

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)

    def to_train(self):
        self.eval = False

    def to_eval(self):
        self.eval = True
