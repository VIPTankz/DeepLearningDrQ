import numpy as np
import torch
from collections import deque
import random
import pickle


class ReplayBuffer():
    def __init__(self, buffer_limit, state_dim, action_dim, Device):

    def put(self, state, ):

class ReplayBuffer():
    def __init__(self, obs_shape, action_shape, buffer_limit, stack_n_frames=3, device):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_limit = buffer_limit
        self.stack_n_frames = stack_n_frames

        self.device = device

        self.obs_store = np.empty((buffer_limit, *obs_shape), dtype=np.uint8)
        self.obs_store_ = np.empty((buffer_limit, *obs_shape), dtype=np.uint8)
        self.action_store = np.empty((buffer_limit, *action_shape), dtype=np.float32)
        self.reward_store = np.empty((buffer_limit, 1), dtype=np.float32)
        self.done_store = np.empty((buffer_limit, 1), dtype=np.float32)

        self.dqueue

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.buffer_limit if self.full else self.idx

    def add(self, obs, action, reward, obs_, done):
        np.copyto(self.obs_store[self.idx], obs)
        np.copyto(self.obs_store_[self.idx], obs_)
        np.copyto(self.action_store[self.idx], action)
        np.copyto(self.reward_store[self.idx], reward)
        np.copyto(self.done_store[self.idx], done)

    def stack_frames(self, idx):
        obs = torch.stack([torch.as_tensor(self.obs_store[i]) for i in range(idx-self.stack_n_frames+1, idx+1)], dim=2).float()
        obs_ = torch.stack([torch.as_tensor(self.obs_store_[i]) for i in range(idx-self.stack_n_frames+1, idx+1)], dim=2).float()
        action = torch.as_tensor(self.action_store[idx])
        reward = torch.as_tensor(self.reward_store[idx])
        done = torch.as_tensor(self.done_store[idx])
        return obs, action, reward, obs_, done

    def sample(self, batch_size):
        idxs = np.random.randint(3, self.buffer_limit if self.full else self.idx, size=batch_size)
        obses = self.obs_store[idxs]
        obses_ = self.obs_store_[idxs]
        actions = self.action_store[idxs]
        rewards = self.reward_store[idxs]
        dones = self.done_store[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        obses_ = torch.as_tensor(obses_, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        return obses, actions, rewards, obses_, dones



    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_batch = torch.empty((n, self.obs_shap), dtype=torch.float)
        a_batch = torch.empty((n, self.action_dim), dtype=torch.float)
        r_batch = torch.empty((n, 1), dtype=torch.float)
        s_next_batch = torch.empty((n, self.state_dim), dtype=torch.float)
        d_batch = torch.empty((n, 1), dtype=torch.float)

        for i, transition in enumerate(mini_batch):
            s, a, r, s_, d = transition
            s_batch[i] = torch.tensor(s, dtype=torch.float)
            a_batch[i] = torch.tensor(a, dtype=torch.float)
            r_batch[i] = torch.tensor(r, dtype=torch.float)
            s_next_batch[i] = torch.tensor(s_, dtype=torch.float)
            d_batch[i] = 0.0 if d else 1.0

        return s_batch.to(self.dev), a_batch.to(self.dev), r_batch.to(self.dev), s_next_batch.to(self.dev), d_batch.to(
            self.dev)

    def size(self):
        return len(self.buffer)

    def save(self, path):
        pickle.dump(self.buffer, open(path + '/replaybuffer.pkl', 'wb'))

    def load(self, path):
        self.buffer = pickle.load(open(path + 'replaybuffer.pkl', 'rb'))
