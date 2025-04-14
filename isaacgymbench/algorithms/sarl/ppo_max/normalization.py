import numpy as np
import torch

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, num_envs, shape, device='cpu'):  # shape:the dimension of input data
        self.device = device
        self.n = 0
        self.mean = torch.zeros(num_envs, shape, device=self.device)
        self.S = torch.zeros(num_envs, shape, device=self.device)
        self.std = torch.sqrt(self.S)
        self.old_mean = torch.zeros(num_envs, shape, device=self.device)
       

    def update(self, x):
        # x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean.copy_(x)
            self.std.copy_(x)
        else:
            # old_mean = self.mean.copy()
            self.old_mean.copy_(self.mean)
            self.mean = self.old_mean + (x - self.old_mean) / self.n
            self.S = self.S + (x - self.old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, num_envs, shape, device='cpu'):
        self.device = device
        self.running_ms = RunningMeanStd(num_envs=num_envs, shape=shape, device=self.device)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)

        # print(x.is_cuda, self.running_ms.mean.is_cuda, self.running_ms.std.is_cuda)

        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, num_envs, shape, gamma, device='cpu'):
        self.num_envs = num_envs
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.device = device
        self.running_ms = RunningMeanStd(num_envs=self.num_envs, shape=self.shape, device=self.device)
        self.R = torch.zeros(self.num_envs, self.shape, device=self.device)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = torch.zeros(self.num_envs, self.shape, device=self.device)
