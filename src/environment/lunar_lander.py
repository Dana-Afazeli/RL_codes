import gym

from dana_codes.src.base import Environment

class LunarLanderEnv(Environment):
    def __init__(self):
        self.env = gym.make('LunarLander-v2')

    def get_n_actions(self):
        return self.env.action_space.n

    def get_action_shape(self):
        return self.env.action_space.shape

    def get_obs_shape(self):
        return self.env.observation_space.shape

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()