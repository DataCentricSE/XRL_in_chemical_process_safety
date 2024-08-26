## Import packages
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim


## Deep Q-network

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, ln, n_neuron, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.ln = ln
        self.init = init

        if self.ln == 1:
            self.fc1 = nn.Linear(input_dims, n_neuron)
            self.fc3 = nn.Linear(n_neuron, n_actions)

            self.bn1 = nn.LayerNorm(n_neuron)


        else:
            self.fc1 = nn.Linear(input_dims, n_neuron)
            self.fc2 = nn.Linear(n_neuron, n_neuron)
            self.fc3 = nn.Linear(n_neuron, n_actions)

            self.bn1 = nn.LayerNorm(n_neuron)
            self.bn2 = nn.LayerNorm(n_neuron)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if self.ln == 1:
            x = self.fc1(state)
            x = F.relu(self.bn1(x))
            x = self.fc3(x)
        else:
            x = self.fc1(state)
            x = F.relu(self.bn1(x))
            x = self.fc2(x)
            x = F.relu(self.bn2(x))
            x = self.fc3(x)

        return x

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))
