import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dana_codes.src.base import Agent
import torch.optim as optim
import re

# class ExperienceReplay(object):
#     def __init__(self, state_shape, er_size=100_000, ):
#         self.state_shape = state_shape
#         self.er_max_size = er_size
#         self.er_cur_size = None
#         self.state_er = None
#         self.next_state_er = None
#         self.reward_er = None
#         self.done_er = None
#         self.action_er = None
#
#         self.init_er()
#
#     def init_er(self):
#         self.state_er = np.zeros(shape=(self.er_size, *self.state_shape), dtype=np.float32)
#         self.next_state_er = np.zeros(shape=(self.er_size, *self.state_shape), dtype=np.float32)
#         self.reward_er = np.zeros(shape=(self.er_size,), dtype=np.float32)
#         self.action_er = np.zeros(shape=(self.er_size,), dtype=np.int32)
#         self.done_er = np.zeros(shape=(self.er_size,), dtype=bool)
#         self.er_cur_size = 0
#
#     def store_transition(self, state, action, reward, next_state, done):
#         if self.er_cur_size < self.er_max_size:
#         self.state_er[ptr] = state
#         self.action_er[ptr] = action
#         self.reward_er[ptr] = reward
#         self.next_state_er[ptr] = next_state
#         self.done_er[ptr] = done
#
#         self.er_ptr += 1

class GeneralNN(nn.Module):
    def __init__(self, lr, layers, loss_fn=nn.MSELoss()):
        super(GeneralNN, self).__init__()
        self.lr = lr
        self.layers = layers

        processed_layers = GeneralNN.process_layers(self.layers)

        self.layers = nn.Sequential(*processed_layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss_fn

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def process_layers(layers):
        processed_layers = []
        for i in range(len(layers)):
            layer = GeneralNN.get_layer(layers[i])
            processed_layers.append(layer)
        return processed_layers

    @staticmethod
    def get_layer(layer):
        if layer == 'relu':
            return nn.ReLU()
        elif layer == 'sftmx':
            return nn.Softmax(dim=1)
        elif (match := re.match(r'fc:(\d+),(\d+)', layer)) is not None:
            return nn.Linear(int(match.group(1)), int(match.group(2)))
        elif (match := re.match(r'drop:(0.\d+)', layer)) is not None:
            return nn.Dropout(p=float(match.group(1)))

class EpisodicACAgent(Agent):

    def __init__(self, lr, gamma, layers, state_shape, n_actions):
        self.lr = lr
        self.gamma = gamma
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.layers = layers

        self.episode_pure_rewards = []
        self.episode_action_probs = []
        self.episode_state_values = []

        self.nn = self.ActorCriticNN(self.layers, self.n_actions, self.lr)
        self.reset()

    class ActorCriticNN(nn.Module):
        def __init__(self, shared_layers, n_actions, lr):
            super().__init__()
            self.lr = lr
            self.shared_layers = shared_layers
            self.n_actions = n_actions

            self.shared = None
            self.actor = None
            self.critic = None
            self.optimizer = None
            self.reset()

        def forward(self, state):
            if not torch.is_tensor(state):
                state = torch.tensor(state)
            features = self.shared(state)
            probs = F.softmax(self.actor(features), dim=0)
            value = self.critic(features)

            return probs, value

        def reset(self):
            self.shared = nn.Sequential(*GeneralNN.process_layers(self.shared_layers))
            self.actor = nn.Linear(self.shared[-2].out_features, self.n_actions)
            self.critic = nn.Linear(self.shared[-2].out_features, 1)
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def reset_episode(self):
        self.episode_pure_rewards = []
        self.episode_action_probs = []
        self.episode_state_values  = []

    def reset(self):
        self.nn.reset()
        self.reset_episode()

    def pi(self, state):
        s = torch.from_numpy(state).float()
        probs, value = self.nn(s)

        m = Categorical(probs)
        action = m.sample()

        self.episode_action_probs.append(m.log_prob(action))
        self.episode_state_values.append(value)

        return action.item()

    def store_transition(self, reward):
        self.episode_pure_rewards.append(reward)

    def train_episode(self, normalize=False):
        returns = []
        R = 0
        for r in self.episode_pure_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = returns - returns.mean()
        if normalize:
            returns = returns / (returns.std() + 1e-8)

        state_values = torch.tensor(self.episode_state_values)
        critic_loss = F.mse_loss(state_values, returns)

        actor_loss = 0
        for lp, r, sv in zip(self.episode_action_probs, returns, state_values):
            actor_loss += -lp*(r - sv)

        loss = critic_loss + actor_loss
        self.nn.optimizer.zero_grad()
        loss.backward()
        self.nn.optimizer.step()
        return actor_loss.item(), critic_loss.item()

    def get_name(self):
        return 'EpisodicACAgent'

    def save(self, path):
        torch.save(self.nn.state_dict(), path)

    def load(self, path):
        self.nn.load_state_dict(torch.load(path))

    def to_train(self):
        self.nn.train()

    def to_eval(self):
        self.nn.eval()


class ReinforceAgent(Agent):

    def __init__(self, lr, gamma, layers, state_shape, n_actions):
        self.lr = lr
        self.gamma = gamma
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.layers = layers

        self.actor = GeneralNN(lr, self.layers)

        self.episode_rewards = []
        self.episode_actions = []

    def reset_episode(self):
        self.episode_rewards = []
        self.episode_actions = []

    def reset(self):
        self.actor = GeneralNN(self.lr, self.layers)

    def pi(self, state):
        s = torch.from_numpy(state).float()
        probs = self.actor(s.unsqueeze(0))

        m = Categorical(probs)
        action = m.sample()
        self.episode_actions.append((action, m.log_prob(action)))

        return action.item()

    def store_transition(self, reward):
        self.episode_rewards.append(reward)

    def train_episode(self, normalize=False):
        returns = []
        R = 0
        for r in self.episode_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = returns - returns.mean()
        if normalize:
            returns = returns / (returns.std() + 1e-8)

        loss = 0
        _, log_probs = zip(*self.episode_actions)
        for lp, r in zip(log_probs, returns):
            loss += -lp*r

        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        return loss.item()

    def get_name(self):
        return 'Reinforce Agent'

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path))

    def to_train(self):
        self.actor.train()

    def to_eval(self):
        self.actor.eval()


class DQNAgent(Agent):

    def __init__(self, lr, gamma, layers, state_shape, n_actions,
                 init_eps=1, min_eps=0.01, eps_decay_rate=5e-3,
                 er_size=10000, batch_size=256, tau=1e-3):
        self.lr = lr
        self.gamma = gamma
        self.init_eps = init_eps
        self.min_eps = min_eps
        self.eps_decay_rate = eps_decay_rate
        self.epsilon = init_eps
        self.er_size = er_size
        self.batch_size = batch_size
        self.tau = tau

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.layers = layers

        self.Q_eval = GeneralNN(lr, self.layers)
        self.Q_target = GeneralNN(lr, self.layers)
        self.update_target()

        self.er_ptr = None
        self.done_er = None
        self.action_er = None
        self.next_state_er = None
        self.reward_er = None
        self.state_er = None
        self.init_er()

    def pi(self, state):
        self.to_eval()
        if np.random.sample() > self.epsilon:
            state_t = torch.tensor(np.array([state])).to(self.Q_eval.device).float()
            q_values = self.Q_eval(state_t)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.n_actions)
        self.to_train()
        return action

    def init_er(self):
        self.state_er = np.zeros(shape=(self.er_size, *self.state_shape), dtype=np.float32)
        self.next_state_er = np.zeros(shape=(self.er_size, *self.state_shape), dtype=np.float32)
        self.reward_er = np.zeros(shape=(self.er_size,), dtype=np.float32)
        self.action_er = np.zeros(shape=(self.er_size,), dtype=np.int32)
        self.done_er = np.zeros(shape=(self.er_size,), dtype=bool)
        self.er_ptr = 0

    def reset_epsilon(self):
        self.epsilon = self.init_eps

    def reset(self):
        self.reset_epsilon()
        self.Q_eval = GeneralNN(self.lr, self.layers)
        self.Q_target = GeneralNN(self.lr, self.layers)
        self.update_target()
        self.init_er()

    def get_name(self):
        return 'DQNAgent'

    def store_transition(self, state, action, reward, next_state, done):
        ptr = self.er_ptr % self.er_size
        self.state_er[ptr] = state
        self.action_er[ptr] = action
        self.reward_er[ptr] = reward
        self.next_state_er[ptr] = next_state
        self.done_er[ptr] = done

        self.er_ptr += 1

    def soft_update_target(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_target(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def decay_epsilon(self):
        self.epsilon = self.epsilon * (1-self.eps_decay_rate) if self.epsilon > self.min_eps else self.min_eps

    def get_batch(self, normalized=True):
        er_curr_size = min(self.er_ptr, self.er_size)
        if normalized:
            probs = F.softmax(torch.tensor(self.reward_er[:er_curr_size]), dim=0)
            return np.random.choice(a=np.arange(er_curr_size, dtype=np.int32), size=self.batch_size, p=np.array(probs))
        else:
            return np.random.choice(er_curr_size, self.batch_size, replace=False)

    def train_double(self, normalize_batch=True):
        if self.er_ptr < self.batch_size:
            return

        self.Q_target.eval()

        batch = self.get_batch(normalized=normalize_batch)
        state_batch = torch.tensor(self.state_er[batch]).to(self.Q_eval.device)
        action_batch = self.action_er[batch]
        reward_batch = torch.tensor(self.reward_er[batch]).to(self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_er[batch]).to(self.Q_eval.device)
        done_batch = torch.tensor(self.done_er[batch]).to(self.Q_eval.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_pred = self.Q_eval(state_batch)[batch_index, action_batch]
        actions = self.Q_eval(next_state_batch).max(1)[1].unsqueeze(1).detach()
        q_next = self.Q_target(next_state_batch).gather(1, actions).squeeze().detach()
        y = reward_batch + q_next * self.gamma * (~done_batch)

        loss = self.Q_eval.loss(q_pred, y)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()
        return loss.item()

    def train(self, clip_grad=False, normalize_batch=True):
        if self.er_ptr < self.batch_size:
            return

        self.Q_target.eval()

        batch = self.get_batch(normalized=normalize_batch)
        state_batch = torch.tensor(self.state_er[batch]).to(self.Q_eval.device)
        action_batch = self.action_er[batch]
        reward_batch = torch.tensor(self.reward_er[batch]).to(self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_er[batch]).to(self.Q_eval.device)
        done_batch = torch.tensor(self.done_er[batch]).to(self.Q_eval.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_pred = self.Q_eval(state_batch)[batch_index, action_batch]
        q_next = torch.max(self.Q_target(next_state_batch), dim=1)[0]
        q_next[done_batch] = 0
        y = reward_batch + self.gamma * q_next

        loss = self.Q_eval.loss(q_pred, y)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()

        if clip_grad:
            for param in self.Q_eval.parameters():
                param.grad.data.clamp_(-1, 1)

        self.Q_eval.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.Q_eval.state_dict(), path)

    def load(self, path):
        self.Q_eval.load_state_dict(torch.load(path))
        self.Q_target.load_state_dict(torch.load(path))

    def to_train(self):
        self.Q_eval.train()

    def to_eval(self):
        self.Q_eval.eval()


class CuriosityDrivenDQNAgent(Agent):

    def __init__(self, q_lr, t_lr, gamma, q_layers, state_shape, n_actions, t_layers,
                 curiosity_scaling_factor=0.1,
                 init_eps=1, min_eps=0.01, eps_decay_rate=5e-3,
                 er_size=10000, batch_size=256, tau=1e-3):
        self.q_lr = q_lr
        self.t_lr = t_lr
        self.gamma = gamma
        self.init_eps = init_eps
        self.min_eps = min_eps
        self.eps_decay_rate = eps_decay_rate
        self.epsilon = init_eps
        self.er_size = er_size
        self.q_layers = q_layers
        self.t_layers = t_layers
        self.batch_size = batch_size
        self.tau = tau

        self.state_shape = state_shape
        self.n_actions = n_actions

        self.Q_eval = GeneralNN(q_lr, self.q_layers)
        self.Q_target = GeneralNN(q_lr, self.q_layers)
        self.update_target()

        self.T_model = GeneralNN(t_lr, self.t_layers)
        self.curiosity_factor = None
        self.set_curiosity_factor(curiosity_scaling_factor)

        self.er_ptr = None
        self.done_er = None
        self.action_er = None
        self.next_state_er = None
        self.reward_er = None
        self.state_er = None
        self.init_er()

    def pi(self, state):
        self.to_eval()
        if np.random.sample() > self.epsilon:
            state_t = torch.tensor(np.array([state])).to(self.Q_eval.device)
            q_values = self.Q_eval(state_t)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.n_actions)
        self.to_train()
        return action

    def init_er(self):
        self.state_er = np.zeros(shape=(self.er_size, *self.state_shape), dtype=np.float32)
        self.next_state_er = np.zeros(shape=(self.er_size, *self.state_shape), dtype=np.float32)
        self.reward_er = np.zeros(shape=(self.er_size,), dtype=np.float32)
        self.action_er = np.zeros(shape=(self.er_size,), dtype=np.int32)
        self.done_er = np.zeros(shape=(self.er_size,), dtype=bool)
        self.er_ptr = 0

    def reset_epsilon(self):
        self.epsilon = self.init_eps

    def reset(self):
        self.reset_epsilon()
        self.Q_eval = GeneralNN(self.q_lr, self.q_layers)
        self.Q_target = GeneralNN(self.q_lr, self.q_layers)
        self.T_model = GeneralNN(self.t_lr, self.t_layers)
        self.update_target()
        self.init_er()

    def get_name(self):
        return 'CuriosityDrivenDQNAgent'

    def store_transition(self, state, action, reward, next_state, done):
        ptr = self.er_ptr % self.er_size
        self.state_er[ptr] = state
        self.action_er[ptr] = action
        self.reward_er[ptr] = reward
        self.next_state_er[ptr] = next_state
        self.done_er[ptr] = done

        self.er_ptr += 1

    def soft_update_target(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_target(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def decay_epsilon(self):
        self.epsilon = self.epsilon * (1 - self.eps_decay_rate) if self.epsilon > self.min_eps else self.min_eps

    def train_double_Q(self):
        if self.er_ptr < self.batch_size:
            return

        self.Q_target.eval()

        er_curr_size = min(self.er_ptr, self.er_size)
        batch = np.random.choice(er_curr_size, self.batch_size, replace=False)
        state_batch = torch.tensor(self.state_er[batch]).to(self.Q_eval.device)
        action_batch = self.action_er[batch]
        reward_batch = torch.tensor(self.reward_er[batch]).to(self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_er[batch]).to(self.Q_eval.device)
        done_batch = torch.tensor(self.done_er[batch]).to(self.Q_eval.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_pred = self.Q_eval(state_batch)[batch_index, action_batch]
        actions = self.Q_eval(next_state_batch).max(1)[1].unsqueeze(1).detach()
        q_next = self.Q_target(next_state_batch).gather(1, actions).squeeze().detach()
        y = reward_batch + q_next * self.gamma * (~done_batch)

        loss = self.Q_eval.loss(q_pred, y)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()
        return loss.item()

    def train_Q(self):
        if self.er_ptr < self.batch_size:
            return

        self.Q_target.eval()

        er_curr_size = min(self.er_ptr, self.er_size)
        batch = np.random.choice(er_curr_size, self.batch_size, replace=False)
        state_batch = torch.tensor(self.state_er[batch]).to(self.Q_eval.device)
        action_batch = self.action_er[batch]
        reward_batch = torch.tensor(self.reward_er[batch]).to(self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_er[batch]).to(self.Q_eval.device)
        done_batch = torch.tensor(self.done_er[batch]).to(self.Q_eval.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_pred = self.Q_eval(state_batch)[batch_index, action_batch]
        q_next = torch.max(self.Q_target(next_state_batch), dim=1)[0]
        q_next[done_batch] = 0
        y = reward_batch + self.gamma * q_next

        loss = self.Q_eval.loss(q_pred, y)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()
        return loss.item()

    def get_intrinsic_reward(self, state, action, next_state):
        self.to_eval()
        one_hot_action = F.one_hot(torch.tensor(np.array([action])), num_classes=self.n_actions)
        state_t = torch.tensor(np.array([state]))
        state_action = torch.cat((state_t, one_hot_action), dim=1).to(self.Q_eval.device)
        next_state_hat = self.T_model(state_action).detach()

        loss = F.mse_loss(next_state_hat.squeeze(), torch.tensor(np.array(next_state)))
        self.to_train()
        return loss.item()*self.curiosity_factor

    def train_T(self):
        if self.er_ptr < self.batch_size:
            return

        er_curr_size = min(self.er_ptr, self.er_size)
        batch = np.random.choice(er_curr_size, self.batch_size, replace=False)
        state_batch = torch.tensor(self.state_er[batch]).to(self.Q_eval.device)
        action_batch = torch.tensor(self.action_er[batch]).to(self.Q_eval.device)
        one_hot_action = F.one_hot(action_batch.to(torch.int64), num_classes=self.n_actions)
        next_state_batch = torch.tensor(self.next_state_er[batch]).to(self.Q_eval.device)

        state_pred = self.T_model(torch.cat((state_batch, one_hot_action), dim=1))

        loss = self.T_model.loss(state_pred, next_state_batch)
        self.T_model.optimizer.zero_grad()
        loss.backward()
        self.T_model.optimizer.step()
        return loss.item()

    def set_curiosity_factor(self, factor):
        self.curiosity_factor = factor

    def to_train(self):
        self.Q_eval.train()
        self.T_model.train()

    def to_eval(self):
        self.Q_eval.eval()
        self.T_model.eval()

    def save(self, path):
        torch.save(self.Q_eval.state_dict(), path)

    def load(self, path):
        self.Q_eval.load_state_dict(torch.load(path))
        self.Q_target.load_state_dict(torch.load(path))