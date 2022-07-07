import gym
from src.utils.RL_utils import *

env = gym.make('MountainCar-v0')
GAMMA = 0.9
INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO = 0.8, 0.2, 0.3
INIT_EPS, MIN_EPS, EPS_DECAY_RATIO = 1.0, 0.1, 0.09
EPISODES = 10000
MAX_STEPS = 300
DISCRETE_ObS_SIZE = [20] * len(env.observation_space.high)
SHOW_EVERY = 500

env._max_episode_steps = MAX_STEPS
alphas = decay_schedule(INIT_ALPHA, MIN_ALPHA, ALPHA_DECAY_RATIO,
                        EPISODES)
eps = decay_schedule(INIT_EPS, MIN_EPS, EPS_DECAY_RATIO,
                        EPISODES)

discrete_obs_step = (env.observation_space.high - env.observation_space.low) / DISCRETE_ObS_SIZE


def discretize(obs):
    d_obs = (obs - env.observation_space.low) / discrete_obs_step
    return tuple(d_obs.astype(int))


num_episode = 0
Q = np.zeros(shape=([x+1 for x in DISCRETE_ObS_SIZE] + [env.action_space.n]))

rewards = []

while num_episode < EPISODES:
    done = False
    state = discretize(env.reset())
    render = num_episode % SHOW_EVERY == 0
    if num_episode % (SHOW_EVERY//10) == 0:
        print(f'im alive at {num_episode}')

    acc_reward = 0
    while not done:
        action = epsilon_greedy_tabular(Q, init_eps, state)[0]
        next_state, reward, done, _ = env.step(action)
        acc_reward += reward
        if render:
            env.render()

        next_state_dis = discretize(next_state)

        if not done:
            TD_error = reward + GAMMA * np.max(Q[next_state_dis]) - Q[state + (action,)]
            Q[state + (action,)] += alphas[num_episode] * TD_error
        elif next_state[0] >= env.goal_position:
            print(f'we made it on {num_episode}')
            Q[state + (action,)] = 0

        state = next_state_dis
    if init_eps > min_eps:
        init_eps *= 0.95

    rewards.append(acc_reward)
    num_episode += 1
env.close()

window = 50
processed_reward = [(min(r := rewards[50*i: 50*(i+1)]), max(r), np.average(np.array(r))) for i in range(EPISODES//50)]
plt.plot(processed_reward)
plt.show()
