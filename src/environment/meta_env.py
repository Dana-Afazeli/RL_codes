from dana_codes.src.base import Environment, Dataset

class MountainCarMetaEnvSimple(Environment):
    def get_action_shape(self):
        pass

    def get_obs_shape(self):
        pass

    def __init__(self, agent, dataset_names, batch_size, train_episodes=10, eval_episodes=10):
        self.agent = agent
        self.datasets = []
        self.dataset_names = dataset_names
        self.batch_size = batch_size
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes

        self.init_datasets()

    def reset(self):
        return self.agent.env.reset()

    def init_datasets(self):
        for dataset in self.dataset_names:
            df = Dataset()
            df.load(dataset)
            self.datasets.append(
                df
            )

    def get_n_actions(self):
        return len(self.datasets)

    def step(self, action, render_result=False, reset_agent=True, train_episodes=None, eval_episodes=None):
        if train_episodes is None:
            train_episodes = self.train_episodes
        if eval_episodes is None:
            eval_episodes = self.eval_episodes

        batch = self.datasets[action].sample(self.batch_size)

        if reset_agent:
            self.agent.reset_table()

        self.agent.train_offline(batch, train_episodes)

        return self.agent.eval(num_evals=eval_episodes, render=render_result)


class MetaEnvDiscreteSequential(Environment):
    def __init__(self, agent, dataset_names, batch_size, render=False):
        self.agent = agent
        self.datasets = []
        self.dataset_names = dataset_names
        self.batch_size = batch_size
        self.render = render

        self.init_datasets()

    def get_action_shape(self):
        return ()

    # TODO
    def get_obs_shape(self):
        pass

    def reset(self):
        return self.agent.env.reset()

    def init_datasets(self):
        for dataset in self.dataset_names:
            df = Dataset()
            df.load(dataset)
            self.datasets.append(
                df
            )

    def get_n_actions(self):
        return len(self.datasets)

    def step(self, action):
        batch = self.datasets[action].sample(self.batch_size)

        self.agent.train_offline(batch)

        return self.agent.eval(render=self.render)
