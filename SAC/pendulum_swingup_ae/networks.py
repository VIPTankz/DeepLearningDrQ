import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class Encoder(nn.Module):
    """ Convolutional encoder for image-based observations"""

    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Linear(39200, self.feature_dim)
        self.head2 = nn.LayerNorm(self.feature_dim)

    def forward_convs(self, obs):
        obs = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            obs = torch.relu(self.convs[i](obs))

        h = obs.flatten()

        return h

    def forward(self, obs, detach=False):
        h = self.forward_convs(obs)
        if detach:
            h.detach()
        out = self.head(h)
        out = self.head2(out)

        if not self.output_logits:
            out = torch.tanh(out)
        return out

    def copy_conv_weights_from(self, source):
        for i in range(self.num_layers):
            source.convs[i].weight = self.convs[i].weight
            source.convs[i].bias = self.convs[i].bias


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, feature_dim, action_dim, actor_lr):
        super(PolicyNetwork, self).__init__()

        self.encoder = Encoder(state_dim, feature_dim)

        self.fc_1 = nn.Linear(feature_dim, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_mu = nn.Linear(1024, action_dim)
        self.fc_std = nn.Linear(1024, action_dim)

        self.lr = actor_lr

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = 2
        self.min_action = -2
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, detach_encoder=False):
        x = self.encoder(x, detach=detach_encoder)

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, feature_dim, action_dim, critic_lr):
        super(QNetwork, self).__init__()

        self.encoder = Encoder(state_dim, feature_dim)

        self.fc_1 = nn.Linear(feature_dim, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, action_dim)

        self.lr = critic_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        x = self.encoder(x)
        cat = torch.cat([x, a], dim=-1)
        q = F.relu(self.fc_1(cat))
        q = F.relu(self.fc_2(q))
        q = self.fc_3(q)
        return q
