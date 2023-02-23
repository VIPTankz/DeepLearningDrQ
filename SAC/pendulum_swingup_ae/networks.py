from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_filters=32):
        super().__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_filters = num_filters

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.fc = nn.Linear(num_filters * 35 * 35, feature_dim[0])
        self.ln = nn.LayerNorm(feature_dim[0])

    def forward(self, obs, detach=False):
        for conv_layer in self.convs:
            obs = F.leaky_relu(conv_layer(obs))
        obs = obs.view(obs.size(0), -1)

        if detach:
            obs = obs.detach()

        obs = self.fc(obs)
        obs = self.ln(obs)
        obs = torch.tanh(obs)
        return obs

    def copy_weights_from(self, source):
        for i in range(4):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, feature_dim, action_shape, actor_lr,
                 log_std_min, log_std_max, max_action, min_action):
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)

        self.fc_1 = nn.Linear(feature_dim[0], 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_mu = nn.Linear(1024, action_shape[0])
        self.fc_std = nn.Linear(1024, action_shape[0])

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action
        self.min_action = min_action
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0


    def forward(self, obs, detach_encoder=False):
        assert len(obs.shape) == 4, 'Requires (Batch, Channel, Height, Width)'
        obs = self.encoder(obs, detach=detach_encoder)
        obs = F.leaky_relu(self.fc_1(obs))
        obs = F.leaky_relu(self.fc_2(obs))
        mu = self.fc_mu(obs)
        log_std = self.fc_std(obs)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)
        action = self.action_scale * y + self.action_bias

        log_prob = dist.log_prob(x)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6), dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, obs_shape, feature_dim, action_shape, critic_lr):
        super().__init__()
        self.encoder = Encoder(obs_shape, feature_dim)
        self.fc_1 = nn.Linear(feature_dim[0] + action_shape[0], 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, action_shape[0])


    def forward(self, obs, action):
        obs = self.encoder(obs)
        cat = torch.cat([obs, action], dim=-1)
        q = F.leaky_relu(self.fc_1(cat))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_3(q)
        return q

