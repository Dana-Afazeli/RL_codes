from dana_codes.src import LunarLanderEnv
from dqn_toy_prj.dqn import DQNAgent
from src.utils import plot_curve
from tqdm import tqdm
import os

LR = 0.01
GAMMA = 0.99
EPISODES = 10001
UPDATE_TARGET_EVERY = 100
RENDER_EVERY = 100
BATCH_SIZE = 256
ER_SIZE = 500_000
EPS_DECAY_EXP = 0.001
EPS_DECAY_LIN = 1e-3
EPS_MIN = 1e-3
HIDDEN_LAYERS = [100]

def train_model(lr=LR, episodes=EPISODES):
    for g in agent.Q_eval.optimizer.param_groups:
        g['lr'] = lr

    for episode in tqdm(range(episodes)):
        step = 0
        cum_reward = 0
        losss = 0
        done = False
        state = env.reset()
        render = not episode % RENDER_EVERY

        while not done:
            action = agent.pi(state)
            next_state, reward, done, _ = env.step(action)

            cum_reward += reward
            agent.store_transition(state, action, reward, next_state, done)
            if render:
                env.render()

            l = agent.train_Q()
            losss += 0 if l is None else l
            state = next_state
            step += 1

        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target()

        if episode % 500 == 0 and episode > 0:
            plot_file_prefix = plot_folder_name + 'episode_' + str(episode) + '_'
            x = [i + 1 for i in range(episode)]
            plot_curve(x, cum_rewards, 'Cumulative Rewards', 'Episode', 'Reward',
                       plot_file_prefix + 'reward.png', color='#c22719', window_size=20)
            plot_curve(x, losses, 'DQN Loss', 'Episode', 'Loss',
                       plot_file_prefix + 'loss.png', color='#8fdb7f')
            plot_curve(x, eps_history, 'Epsilon Decay', 'Episode', 'Epsilon',
                       plot_file_prefix + 'epsilon.png', color='#4d70d1')

            agent.save(model_folder_name + 'episode_' + str(episode) + '_model')

        steps.append(step)
        eps_history.append(agent.epsilon)
        agent.decay_epsilon(linear=False)
        cum_rewards.append(cum_reward)
        losses.append(losss)

        if episode % 100 == 0 and episode > 0:
            print('\nepisode: %3d. avg step-reward-loss in last 100 %6.2f | %6.2f | %6.4f'
                  % (episode, sum(steps[-100:]) / 100, sum(cum_rewards[-100:]) / 100, sum(losses[-100:])/100))


if __name__ == '__main__':
    env = LunarLanderEnv()
    agent = DQNAgent(env, LR, GAMMA, hidden_layers=HIDDEN_LAYERS,
                     eps_dec_exp=EPS_DECAY_EXP, eps_dec_lin=EPS_DECAY_LIN,
                     er_size=ER_SIZE, batch_size=BATCH_SIZE, min_eps=EPS_MIN)
    print(agent.Q_eval)
    cum_rewards, eps_history, steps, losses = [], [], [], []

    plot_folder_name = 'lunar_lander_plots/'
    model_folder_name = 'lunar_lander_models/'

    for p in [plot_folder_name, model_folder_name]:
        if not os.path.exists(p):
            os.makedirs(p)
            print('created directory', p)

    train_model()

