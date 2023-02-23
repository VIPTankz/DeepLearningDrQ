import numpy as np
import torch
from collections import deque
import random
import pickle
from torchvision import transforms


class ReplayBuffer:
    def __init__(self, buffer_limit, obs_shape, action_shape, device):
        self.buffer = deque(maxlen=buffer_limit)
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def put(self, transition):
        o, a, r, o_, d = transition
        o = o * 255.
        o_ = o_ * 255.
        self.buffer.append((o.type(torch.uint8), a, r, o_.type(torch.uint8), d))

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        o_batch = torch.empty(((n,) + self.obs_shape), dtype=torch.float)
        a_batch = torch.empty((n,) + self.action_shape, dtype=torch.float)
        r_batch = torch.empty((n, 1), dtype=torch.float)
        o_batch_ = torch.empty((n,) + self.obs_shape, dtype=torch.float)
        d_batch = torch.empty((n, 1), dtype=torch.float)

        for i, transition in enumerate(mini_batch):
            o, a, r, o_, d = transition
            o = o / 255.
            o_ = o_ / 255.
            o = o.type(torch.float32)
            o_ = o.type(torch.float32)
            o_batch[i] = o.clone().detach()
            a_batch[i] = torch.tensor(a, dtype=torch.float)
            r_batch[i] = torch.tensor(r, dtype=torch.float)
            o_batch_[i] = o_.clone().detach()
            d_batch[i] = 0.0 if d else 1.0

        return o_batch.to(self.device), a_batch.to(self.device), r_batch.to(self.device), \
            o_batch_.to(self.device), d_batch.to(self.device)

    def size(self):
        return len(self.buffer)

    def save(self, path):
        pickle.dump(self.buffer, open(path + '/replaybuffer.pkl', 'wb'))

    def load(self, path):
        self.buffer = pickle.load(open(path + 'replaybuffer.pkl', 'rb'))
