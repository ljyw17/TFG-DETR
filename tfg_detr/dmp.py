import torch
import torch.nn as nn


class ICN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, u):
        mean = u.mean(dim=(2, 3), keepdim=True)
        var = u.var(dim=(2, 3), keepdim=True, unbiased=False)
        normalized = (u - mean) / (var + 1e-5).sqrt()
        return self.gamma * normalized + self.beta


class DMP(nn.Module):
    def __init__(self, in_channels, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.icn = ICN(in_channels)
        self.branches = nn.ModuleList(
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        )
        self.branch_weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, u):
        u_bar = self.icn(u)
        weights = self.branch_weights
        weights = weights / (weights.sum() + 1e-6)
        v = sum(w * branch(u_bar)
                for w, branch in zip(weights, self.branches))
        return self.tail(v)
