from dm_control import suite
from dm_control.suite.wrappers import pixels
from torchvision import transforms
import torch
import numpy as np


class Environment:
    def __init__(self):
        self.env = pixels.Wrapper(suite.load("cartpole", "swingup"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((84,84))
        ])
        self.state_buffer = []

    def step(self, action):
        timestep = self.env.step(action)
        obs = self.transforms(timestep.observation["pixels"].astype(float))
        self.state_buffer.append(obs)
        if len(self.state_buffer) < 3:
            z = torch.zeros(3, 84, 84)
            obs = torch.vstack(self.state_buffer + [z]).float()/255.
            timestep.observation["pixels"] = obs
            return timestep
        elif len(self.state_buffer) == 3:
            timestep.observation["pixels"] = torch.vstack(self.state_buffer).float()/255.
            return timestep
        else:
            self.state_buffer.remove(self.state_buffer[0])
            timestep.observation["pixels"] = torch.vstack(self.state_buffer).float()/255.
            return timestep

    def reset(self):
        self.state_buffer = []
        stack = torch.zeros(6, 84, 84)
        timestep = self.env.reset()
        obs = self.transforms(timestep.observation["pixels"].astype(float))
        timestep.observation["pixels"] = torch.vstack([obs, stack]).float()/255.
        self.state_buffer.append(obs)
        return timestep

