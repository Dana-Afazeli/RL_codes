from dana_codes.src.agent.deep import DQNAgent
from dana_codes.src.agent.deep import CuriosityDrivenDQNAgent

from src.environment.mountain_car import MountainCarEnv
from src.environment.lunar_lander import LunarLanderEnv
from dana_codes.src.utils.RL_utils import plot_curve
from tqdm import tqdm
import os

Q_LR = 0.001
T_LR = 0.0001
GAMMA = 0.99
EPISODES = 10001
UPDATE_TARGET_EVERY = 4
STAT_EVERY = 200
RENDER_EVERY = 100
BATCH_SIZE = 64
ER_SIZE = 10_000
EPS_DECAY_EXP = 0.005
EPS_DECAY_LIN = 1e-3
TAU = 1e-3
EPS_MIN = 1e-2


def train_curiosity_model(episodes=EPISODES):
    print(agent.Q_eval)
    print(agent.T_model)
    for g in agent.Q_eval.optimizer.param_groups:
        g['lr'] = Q_LR

    for g in agent.T_model.optimizer.param_groups:
        g['lr'] = T_LR

    for episode in tqdm(range(episodes)):
        step = 0
        cum_reward = 0
        cum_intrinsic_reward = 0
        Q_loss = 0
        T_loss = 0
        done = False
        state = env.reset()
        render = not episode % RENDER_EVERY

        while not done:
            action = agent.pi(state)
            next_state, reward, done, _ = env.step(action)
            int_reward = agent.get_intrinsic_reward(state, action, next_state)
            cum_reward += reward
            cum_intrinsic_reward += int_reward

            agent.store_transition(state, action, reward + int_reward, next_state, done)
            if render:
                env.render()

            state = next_state
            step += 1

            if step % UPDATE_TARGET_EVERY == 0:
                q_loss = agent.train_Q()
                Q_loss += 0 if q_loss is None else q_loss

                t_loss = agent.train_T()
                T_loss += 0 if t_loss is None else t_loss
                agent.soft_update_target()

        if episode % STAT_EVERY == 0 and episode > 0:
            plot_file_prefix = plot_folder_name + 'episode_' + str(episode) + '_'
            x = [i + 1 for i in range(episode)]
            plot_curve(x, cum_rewards, 'Cumulative Rewards', 'Episode', 'Reward',
                       plot_file_prefix + 'reward.png', color='#c22719', window_size=20)
            plot_curve(x, losses, 'DQN Loss', 'Episode', 'Loss',
                       plot_file_prefix + 'loss.png', color='#8fdb7f')
            plot_curve(x, int_rewards, 'Intrinsic Rewards', 'Episode', 'Int Reward',
                       plot_file_prefix + 'int_reward.png', color='#abb849', window_size=20)
            plot_curve(x, T_losses, 'Transition Model Loss', 'Episode', 'T Loss',
                       plot_file_prefix + 't_loss.png', color='#b849a2')
            plot_curve(x, eps_history, 'Epsilon Decay', 'Episode', 'Epsilon',
                       plot_file_prefix + 'epsilon.png', color='#4d70d1')

            agent.save(model_folder_name + 'episode_' + str(episode) + '_model')

        steps.append(step)
        eps_history.append(agent.epsilon)
        agent.decay_epsilon()
        cum_rewards.append(cum_reward)
        losses.append(Q_loss)

        int_rewards.append(cum_intrinsic_reward)
        T_losses.append(T_loss)

        if episode % 100 == 0 and episode > 0:
            print('\n' + 'episode: %3d. avg step-reward-loss-int_reward-int_loss in last 100 ' % episode +
                  '%6.2f | %6.2f | %6.4f | %6.2f | %6.4f'
                  % (sum(steps[-100:]) / 100, sum(cum_rewards[-100:]) / 100, sum(losses[-100:]) / 100,
                     sum(int_rewards[-100:]) / 100, sum(T_losses[-100:]) / 100))


def train_model(lr=Q_LR, episodes=EPISODES):
    print(normal_agent.Q_eval)
    for g in normal_agent.Q_eval.optimizer.param_groups:
        g['lr'] = lr

    for episode in tqdm(range(episodes)):
        step = 0
        cum_reward = 0
        losss = 0
        done = False
        state = env.reset()
        render = not episode % RENDER_EVERY

        while not done:
            action = normal_agent.pi(state)
            next_state, reward, done, _ = env.step(action)

            cum_reward += reward
            normal_agent.store_transition(state, action, reward, next_state, done)
            if render:
                env.render()

            state = next_state
            step += 1

            if step % UPDATE_TARGET_EVERY == 0:
                lows = normal_agent.train()
                losss += 0 if lows is None else lows
                normal_agent.soft_update_target()

        if episode % STAT_EVERY == 0 and episode > 0:
            plot_file_prefix = plot_folder_name + 'episode_' + str(episode) + '_'
            x = [i + 1 for i in range(episode)]
            plot_curve(x, cum_rewards, 'Cumulative Rewards', 'Episode', 'Reward',
                       plot_file_prefix + 'reward.png', color='#c22719', window_size=20)
            plot_curve(x, losses, 'DQN Loss', 'Episode', 'Loss',
                       plot_file_prefix + 'loss.png', color='#8fdb7f')
            plot_curve(x, eps_history, 'Epsilon Decay', 'Episode', 'Epsilon',
                       plot_file_prefix + 'epsilon.png', color='#4d70d1')

            normal_agent.save(model_folder_name + 'episode_' + str(episode) + '_model')

        steps.append(step)
        eps_history.append(normal_agent.epsilon)
        normal_agent.decay_epsilon()
        cum_rewards.append(cum_reward)
        losses.append(losss)

        if episode % 100 == 0 and episode > 0:
            print('\n' + 'episode: %3d. avg step-reward-loss in last 100 %6.2f | %6.2f | %6.4f'
                  % (episode, sum(steps[-100:]) / 100, sum(cum_rewards[-100:]) / 100, sum(losses[-100:]) / 100))


if __name__ == '__main__':
    env = MountainCarEnv(continuous=False)
    state_shape = env.get_obs_shape()
    print(state_shape)
    n_actions = env.get_n_actions()

    layers = [f'fc:{state_shape[0]},64', 'relu', 'fc:64,64', 'relu', f'fc:64,{n_actions}']
    t_layers = [f'fc:{state_shape[0] + n_actions},64', 'relu',
                'fc:64,64', 'relu', 'fc:64,64', 'relu', f'fc:64,{state_shape[0]}']

    normal_agent = DQNAgent(lr=Q_LR, gamma=GAMMA, layers=layers,
                     state_shape=state_shape, n_actions=n_actions,
                     eps_decay_rate=EPS_DECAY_EXP,
                     er_size=ER_SIZE, batch_size=BATCH_SIZE, min_eps=EPS_MIN,
                     tau=TAU)

    agent = CuriosityDrivenDQNAgent(q_lr=Q_LR, t_lr=T_LR, gamma=GAMMA, q_layers=layers,
                                    t_layers=t_layers, curiosity_scaling_factor=100,
                     state_shape=state_shape, n_actions=n_actions,
                     eps_decay_rate=EPS_DECAY_EXP,
                     er_size=ER_SIZE, batch_size=BATCH_SIZE, min_eps=EPS_MIN,
                     tau=TAU)

    cum_rewards, eps_history, steps, losses = [], [], [], []
    int_rewards, T_losses = [], []

    plot_folder_name = 'mc_normal_plots/'
    model_folder_name = 'mc_normal_models/'

    for p in [plot_folder_name, model_folder_name]:
        if not os.path.exists(p):
            os.makedirs(p)
            print('created directory', p)

    train_model()
