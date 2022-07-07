from dana_codes.src import UCBBandit, OfflineQLAgent
from dana_codes.src import MountainCarMetaEnvSimple, MountainCarEnv
import numpy as np
from tqdm import tqdm

INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO = 0.8, 0.2, 0.3
BATCH_SIZE = 10_000
EPISODES = 10
TRAIN_EPISODES, EVAL_EPISODES = 100, 10
TRIALS = 1
dataset_names = ['../datasets/agent=EpsilonGreedy-size=159376-step=6-time=2022-04-15 19:21:41.865366.csv',
                '../datasets/agent=RandomAction-size=200000-step=6-time=2022-04-15 19:21:42.640256.csv']

env = MountainCarEnv()
agent = OfflineQLAgent(env, (INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO))

meta_env = MountainCarMetaEnvSimple(agent, dataset_names, BATCH_SIZE,
                                    train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES)
meta_agent_count = UCBBandit(meta_env, (INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO), mode='count')
meta_agent_qvalue = UCBBandit(meta_env, (INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO), mode='qvalue')


success = 0
failed = []
succeeded = []
for tr in tqdm(range(TRIALS)):
    meta_agent_qvalue.train_Q(EPISODES, verbose=True)
    q_values = meta_agent_qvalue.q_values
    if np.argmax(q_values) == 0:
        success += 1
        succeeded.append(q_values)
    else:
        failed.append(q_values)

print('\nsuccess rate with qvalue: {:3.2f}'.format(100*success/TRIALS))
print(succeeded)
print('-'*80)
for failed_q in failed:
    print(failed_q)

success = 0
failed = []
succeeded = []
for tr in tqdm(range(TRIALS)):
    meta_agent_count.train_Q(EPISODES, verbose=True)
    q_values = meta_agent_count.arm_rewards/meta_agent_count.n_selection
    if np.argmax(q_values) == 0:
        success += 1
        succeeded.append(q_values)
    else:
        failed.append(q_values)

print('\nsuccess rate with count: {:3.2f}'.format(100 * success / TRIALS))
print(succeeded)
print('-'*80)
for failed_q in failed:
    print(failed_q)

