from dm_control import suite
from dm_control.suite.wrappers import pixels
from torchvision import transforms
import torch
import matplotlib.pyplot as plt


class Environment:
    def __init__(self):
        self.env = pixels.Wrapper(suite.load("cartpole", "swingup"))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((180, 180)),
            transforms.Resize((84, 84)),
            transforms.Grayscale(num_output_channels=1)
        ])

    def step(self, action, action_repeat=6):
        reward = 0
        observations = torch.zeros(3, 84, 84)
        timestep = self.env.step(action)

        for c in range(3):
            observations[c, :, :] = self.transforms(timestep.observation["pixels"].astype(float))[0]

        for i in range(action_repeat):
            timestep = self.env.step(action)
            if timestep.last():
                return observations, reward, timestep.last(), {}

            timestep = self.env.step(action)
            reward += timestep.reward

            if i == 2:
                for c in range(1, 3):
                    observations[1, :, :] = self.transforms(timestep.observation["pixels"].astype(float))[0]
            elif i == 4:
                observations[2, :, :] = self.transforms(timestep.observation["pixels"].astype(float))[0]
        return observations, reward, timestep.last(), {}

    def reset(self):
        timestep = self.env.reset()
        obs = self.transforms(timestep.observation["pixels"].astype(float))
        obs = torch.vstack([obs, obs, obs]).float() / 255.
        return obs, {}
