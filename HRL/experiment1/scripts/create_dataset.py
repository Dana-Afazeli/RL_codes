from dana_codes.src.base import Dataset
from dana_codes.src.agent.random import RandomActionAgent
from dana_codes.src.agent.ql_tabular import TabularQLAgent
from dana_codes.src.environment.mountain_car import MountainCarEnv

DATASET_SIZE = 10_000

env = MountainCarEnv()

expert_agent = TabularQLAgent(env, exploratory=True)
expert_agent.load('../agent/EpsilonGreedy-15k.npy')

random_agent = RandomActionAgent(env)

expert_dataset = Dataset.aggregate_dataset(expert_agent, DATASET_SIZE)
random_dataset = Dataset.aggregate_dataset(random_agent, DATASET_SIZE)

expert_dataset.save('../datasets/')
random_dataset.save('../datasets/')