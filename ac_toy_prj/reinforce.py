from dana_codes.src.environment.lunar_lander import LunarLanderEnv
from dana_codes.src.environment.mountain_car import MountainCarEnv, MountainCar3DEnv
from dana_codes.src.agent.deep import ReinforceAgent
from dana_codes.src.utils.RL_utils import plot_curve
from tqdm import tqdm
import os

LR = 0.01
GAMMA = 0.99
EPISODES = 10001
STAT_EVERY = 200
RENDER_EVERY = 100

def train_model():
    for episode in tqdm(range(EPISODES)):
        state = env.reset()
        step = 0
        done = False
        render = not episode % RENDER_EVERY

        while not done:
            action = agent.pi(state)
            next_state, reward, done, _ = env.step(action)

            modified_reward = reward + 20 * abs(next_state[1])
            agent.store_transition(modified_reward)

            if render:
                env.render()

            state = next_state
            step += 1

        loss = agent.train_episode()

        if episode % STAT_EVERY == 0 and episode > 0:
            plot_file_prefix = plot_folder_name + 'episode_' + str(episode) + '_'
            x = [i + 1 for i in range(episode)]
            plot_curve(x, cum_rewards, 'Cumulative Rewards', 'Episode', 'Reward',
                       plot_file_prefix + 'reward.png', color='#c22719', window_size=20)
            plot_curve(x, losses, 'Actor Loss', 'Episode', 'Loss',
                       plot_file_prefix + 'loss.png', color='#8fdb7f')

            agent.save(model_folder_name + 'episode_' + str(episode) + '_model')

        cum_rewards.append(sum(agent.episode_rewards))
        losses.append(loss)
        agent.reset_episode()


if __name__ == '__main__':
    env = MountainCarEnv()
    state_shape = env.get_obs_shape()
    n_actions = env.get_n_actions()

    layers = [f'fc:{state_shape[0]},64', 'relu', f'fc:64,{n_actions}', 'sftmx']

    agent = ReinforceAgent(lr=LR, gamma=GAMMA, layers=layers,
                     state_shape=state_shape, n_actions=n_actions)

    cum_rewards, steps, losses = [], [], []

    plot_folder_name = 'mc_RS_reinforce_plots/'
    model_folder_name = 'mc_RS_reinforce_models/'

    for p in [plot_folder_name, model_folder_name]:
        if not os.path.exists(p):
            os.makedirs(p)
            print('created directory', p)

    train_model()