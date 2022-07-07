from dana_codes.src import MountainCarEnv
from dana_codes.src import EpsilonGreedyAgent

INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO = 0.8, 0.2, 0.3
INIT_EPS, MIN_EPS, EPS_DECAY_RATIO = 1.0, 0.01, 0.2

env = MountainCarEnv()
agent = EpsilonGreedyAgent(env,
                           (INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO),
                           (INIT_EPS, MIN_EPS, EPS_DECAY_RATIO))

rewards_10k = agent.train_Q(10000, show_every=500)
#agent.save('../agent/EpsilonGreedy-1k.npy')
rewards_15k = agent.train_Q(2000, show_every=500)
#agent.save('../agent/EpsilonGreedy-3k.npy')