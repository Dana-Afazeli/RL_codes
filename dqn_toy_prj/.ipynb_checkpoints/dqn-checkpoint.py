import torch
import numpy as np
import torch.nn as nn
from dana_codes.src import Agent
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, lr, input_dim, hidden_dims, n_actions):
        super(DQN, self).__init__()
        self.lr = lr
        self.input_dim = input_dim
        self.n_actions = n_actions

        all_dims = [*input_dim] + hidden_dims + [n_actions]
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                #layers.append(nn.Dropout(p=0.2))
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)


class DQNAgent(Agent):
    def __init__(self, env, lr, gamma, hidden_layers, init_eps=1, min_eps=0.01, eps_dec_lin=1e-3, eps_dec_exp=5e-3,
                 er_size=10000, batch_size=256):
        super(DQNAgent, self).__init__(env)
        self.lr = lr
        self.gamma = gamma
        self.init_eps = init_eps
        self.min_eps = min_eps
        self.eps_dec_lin = eps_dec_lin
        self.eps_dec_exp = eps_dec_exp
        self.epsilon = init_eps
        self.er_size = er_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size

        self.state_shape = self.env.get_obs_shape()
        self.n_actions = self.env.get_n_actions()
        self.Q_eval = DQN(lr, self.state_shape, self.hidden_layers, self.n_actions)
        self.Q_target = DQN(lr, self.state_shape, self.hidden_layers, self.n_actions)
        self.update_target()

        self.er_ptr = None
        self.done_er = None
        self.action_er = None
        self.next_state_er = None
        self.reward_er = None
        self.state_er = None
        self.init_er()

    def pi(self, state, episode=None):
        if np.random.sample() > self.epsilon:
            state_t = torch.tensor(np.array([state])).to(self.Q_eval.device)
            q_values = self.Q_eval(state_t)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.env.get_n_actions())

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
        self.Q_eval = DQN(self.lr, self.state_shape, self.hidden_layers, self.n_actions)
        self.Q_target = DQN(self.lr, self.state_shape, self.hidden_layers, self.n_actions)
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

    def update_target(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def decay_epsilon(self, linear=False):
        if linear:
            self.epsilon = (self.epsilon - self.eps_dec_lin) if self.epsilon > self.min_eps else self.min_eps
        else:
            self.epsilon = self.epsilon * (1-self.eps_dec_exp) if self.epsilon > self.min_eps else self.min_eps

    def train(self):
        if self.er_ptr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
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

        loss = self.Q_eval.loss(y, q_pred)
        loss.backward()
        self.Q_eval.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.Q_eval.state_dict(), path)

    def load(self, path):
        self.Q_eval.load_state_dict(torch.load(path + '_eval'))
        self.Q_target.load_state_dict(torch.load(path + '_target'))
