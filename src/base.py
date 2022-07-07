from abc import ABC, abstractmethod
from dana_codes.src.utils.RL_utils import generate_trajectory

import numpy as np
import pandas as pd
import random
import time

class Agent(ABC):
    """
    This is the base class for all agents. It provides basic properties of an agent.
    """

    @abstractmethod
    def reset(self):
        """
        resets any learning progress in agent. There can be other methods for resetting different aspects
            of an Agent i.e. a method to reset epsilon to initial epsilon in an Epsilon Greedy agent.
            This method in particular is a high level method that encapsulates all those functionalities.
        :return: None (probably)
        """
        pass

    @abstractmethod
    def pi(self, state, bonus=None):
        """
        the current policy of the agent. This method does as one expects: inputs a state and returns an action
            based on the current learning state.
        :param state: MDP state
        :param bonus: np.array with shape of (n_actions,). adds a bonus to each action's utility. used for
            situations where one wants to artificially change the Q values
        :return: action (a number from 0 to n_actions in case of discrete actions
            and a vector of proper size in case of continuous action space
        """
        pass

    @abstractmethod
    def get_name(self):
        """
        return the name of the instantiated agent. for example DQNAgent or RandomAgent.
        :return: a str that is the name of the agent
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        save an image of the current state of the model
        :param path: str. where to save to
        :return: None
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        take a guess
        :param path: str. where to load from.
        :return: None. loading is inplace.
        """
        pass

    @abstractmethod
    def to_train(self):
        """
        puts agent in training state. The main use is in deep models,
            but can be useful in other agents as well
        :return: None
        """
        pass

    @abstractmethod
    def to_eval(self):
        """
        puts agent in evaluation state. The main use is in deep models,
            but can be useful in other agents as well
        :return: None
        """
        pass


class Environment(ABC):
    """
    This class provides an abstraction over hand-made and gym-based environments.
    """
    def __init__(self, max_steps=-1):
        self.max_steps = None
        self.set_max_steps(max_steps)

    @abstractmethod
    def get_n_actions(self):
        """
        returns the number of actions in case of discrete action space.
        :return: int or None. number of actions or None if action space is continuous.
        """
        pass

    @abstractmethod
    def get_action_shape(self):
        """
        returns the action vector shape in case of continuous action space.
        :return: np.array or None. action vector shape or None if action space is discrete.
        """
        pass

    @abstractmethod
    def get_obs_shape(self):
        """
        returns the state or observation vector shape.
        :return: np.array. state vector shape
        """
        pass

    @abstractmethod
    def reset(self):
        """
        resets the environment to the initial state. It is supposed to have the same functionality as
            gym.env.reset()
        :return: np.array or None. initial state of the episode or None if environment is a bandit env.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        steps in the environment with the action given. It is supposed to have the same functionality as
            gym.env.step() but it has max_steps specified by self.max_steps.
        :param action: int or np.array. action to take in the current state.
        :return:  (reward, next_state, done)
        """
        pass

    def set_max_steps(self, max_steps=-1):
        """
        set the maximum number of steps in an episode. if -1, episodes will elongate as long as possible.
        :param max_steps: int. maximum number of steps. should be either positive or -1
        :return: None
        """
        self.max_steps = max_steps

    def get_max_steps(self):
        return self.max_steps


class Dataset:
    """
    This class is used to facilitate creation and maintenance of datasets in the project.
        Data is stored and transferred as pd.DataFrame.
        Data in the dataframe is basically a collection of transitions. The columns of dataframe are:
            state, action, reward, next_state, done, episode, step

        Episode and step are used to identify the number of episode and the step that the transition is from.
        Basically the data is usually gathered by creating rollouts, these two are roll out numbers and
        step number in that rollout. Note that some datasets may not have valid values for these two columns;
        for example replay buffer datasets have sample of transitions in different states of learning.
    """

    def __init__(self, random_seed=None, data=None, name='Dataset'):
        """
        init method!
        :param random_seed: int. used for sampling from Dataset. if not provided it is set from time.time().
        :param data: pd.DataFrame.
        """
        self.data = data
        self.name = None
        self.set_dataset_name(name)

        if random_seed is None:
            random_seed = int(time.time())

        self.random_seed = random_seed

    @staticmethod
    def label_trajectories(trajectories):
        """
        labels the trajectories with episode and step data. Only use if trajectories are created
        by creating online rollouts from an agent
        :param trajectories: list(tuple). List of trajectories to be labeled.
        :return: list(tuple). Labeled list of trajectories.
        """
        return [[(*trj[j], i+1, j+1) for j in range(len(trj))] for i, trj in enumerate(trajectories)]

    @staticmethod
    def aggregate_dataset(exploratory_agent, env, dataset_size):
        """
        this is an util function to create a dataset using a static agent.
        :param exploratory_agent: Agent. The agent to create trajectories with.
        :param env: Environment. The environment to create trajectories with.
        :param dataset_size: int. The size of the desired dataset.
        :return: Dataset. returns a Dataset object that contains the trajectories created.
        """
        trajectories, _ = generate_trajectory(exploratory_agent, env, dataset_size)

        labeled_trajectories = Dataset.label_trajectories(trajectories)
        aggr_trajectories = np.concatenate(labeled_trajectories, axis=0)

        data = pd.DataFrame(aggr_trajectories,
                            columns=['state', 'action', 'reward', 'next_state', 'done', 'episode', 'step'])
        return Dataset(data=data, random_seed=random.randint(0, 1000))

    def load(self, path):
        """
        loads the dataset at the path.
        :param path: str. The path to the dataset.
        :return: None
        """
        self.data = pd.read_csv(path, index_col=False,
                                converters={
                                    'state': eval,
                                    'next_state': eval,
                                })

    def save(self, path):
        """
        saves the current contents of dataset in the specified path. It should be a folder
        :param path: str. The path to save the dataset at.
        :return: None
        """
        self.data.to_csv(path + self.get_dataset_name() + '.csv',
                         index=False)

    def set_dataset_name(self, name):
        """
        used to set the name of the current dataset.
        :param name: str. The name to be set.
        :return: None
        """
        self.name = name

    def get_dataset_name(self):
        """
        used to get the name of the current dataset.
        :return: str. dataset name.
        """
        return self.name

    def sample(self, batch_size, random_seed=None):
        """
        used to sample transitions from the current dataset.
        :param batch_size: int.
        :param random_seed: int.
        :return: pd.DataFrame. the sampled transitions.
        """
        if random_seed is None:
            random_seed = self.random_seed

        sample_size = min(batch_size, len(self.data))
        return self.data.sample(n=sample_size, random_state=random_seed)
