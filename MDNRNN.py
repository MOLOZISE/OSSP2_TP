"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MDNRNN(nn.Module):
    def __init__(self, z_size=32, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lstm = nn.LSTM(z_size, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians * z_size)
        self.fc2 = nn.Linear(n_hidden, n_gaussians * z_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians * z_size)

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, log_sigma = self.fc1(y), self.fc2(y), self.fc3(y)

        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        log_sigma = log_sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)
        pi = F.softmax(pi, 2)
        sigma = torch.exp(log_sigma)
        return pi, mu, sigma

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.n_hidden)).to(self.device)  # Hidden State
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.n_hidden)).to(self.device)  # Internal Process States
        y, (h, c) = self.lstm(x, (h_0, c_0))
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)

