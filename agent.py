## Import packages
import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer


## Soft update
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


## Agent
class DQNAgent():
    def __init__(self, gamma, epsilon, lr, ln, n_actions, input_dims, mem_size, batch_size, n_neuron=8, eps_min=0.01,
                 eps_dec=5e-7, replace=1000, algo=None, target_tau=1e-2, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.ln = ln
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.n_neuron = n_neuron
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.algo = algo
        self.tau = target_tau
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims, ln=self.ln,
                                   n_neuron=self.n_neuron, name=self.algo + '_q_eval', chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims, ln=self.ln,
                                   n_neuron=self.n_neuron, name=self.algo + '_q_next', chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        self.q_eval.eval()
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        self.q_eval.train()
        return action

    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def sample_memory(self):
        state, action, reward, new_state = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        soft_update(self.q_eval, self.q_next, self.tau)

        states, actions, rewards, states_ = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1