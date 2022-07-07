from dana_codes.src import SimpleGaussianBanditEnv
from dana_codes.src import UCBBandit
import numpy as np
import time
from tqdm import tqdm
from collections import Counter

modes = ['qvalue', 'count']
EPISODES = 100
N_ARMS = 2


def bandit_test(mode, verbose=True):
    MUS = np.random.normal(10, 10, N_ARMS)
    STDS = np.random.uniform(1, 10, N_ARMS)

    env = SimpleGaussianBanditEnv(N_ARMS, MUS, STDS)
    agent = UCBBandit(env, (0.1, 0.1, 0.9), mode=mode)

    transitions = agent.train_Q(EPISODES, verbose=False)
    actions, rewards = zip(*transitions)
    action_counts = Counter(actions)
    action_percentages = np.round(100 * np.array([v for _, v in action_counts.items()])/len(actions),2)

    learned = agent.q_values if mode == 'qvalue' else (agent.arm_rewards / agent.n_selection)
    if verbose:
        print(f'learned {learned} actual: {env.mus} {env.stds} {action_percentages} {np.max(env.mus) * EPISODES - sum(rewards)}')
    return learned, env.mus


TRIALS = 1000
FAIL = 5

success = 0
failed_attempts_qvalue = []
for i in tqdm(range(TRIALS)):
    q, mu = bandit_test(mode='qvalue', verbose=False)
    if np.argmax(q) == np.argmax(mu):
        success += 1
    else:
        failed_attempts_qvalue.append((q, mu))

print(f'success rate with qvalue: {round(100 * success / TRIALS)}%')
for i in range(FAIL):
    print(failed_attempts_qvalue[i])

time.sleep(0.1)

success = 0
failed_attempts_count = []
for i in tqdm(range(TRIALS)):
    q, mu = bandit_test(mode='count', verbose=False)
    if np.argmax(q) == np.argmax(mu):
        success += 1
    else:
        failed_attempts_count.append((q, mu))

print(f'success rate with count: {round(100 * success / TRIALS)}%')
for i in range(FAIL):
    print(failed_attempts_count[i])