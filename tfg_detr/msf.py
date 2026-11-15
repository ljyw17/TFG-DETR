import torch
import torch.nn as nn


class DSC(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        k = kernel_size
        self.horizontal = nn.Conv2d(channels, channels, kernel_size=(1, k),
                                    padding=(0, k // 2), groups=channels)
        self.vertical = nn.Conv2d(channels, channels, kernel_size=(k, 1),
                                  padding=(k // 2, 0), groups=channels)

    def forward(self, u):
        return torch.cat([self.horizontal(u), self.vertical(u)], dim=1)


class CA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, z):
        b, c, _, _ = z.shape
        avg = z.mean(dim=(2, 3))
        mx = z.amax(dim=(2, 3))
        weight = torch.sigmoid(self.fc(avg) + self.fc(mx)).view(b, c, 1, 1)
        return z * weight


class MSF(nn.Module):
    def __init__(self, channels, expansion=4, dsc_kernel=5):
        super().__init__()
        hidden = channels * expansion

        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

        self.dsc = DSC(channels, kernel_size=dsc_kernel)
        self.ca = CA(2 * channels)

        self.mlp = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, u):
        s_g = torch.sigmoid(self.gate(u)) * u
        s_c = self.context(u)
        s_d = self.ca(self.dsc(u))
        return self.mlp(torch.cat([s_g, s_c, s_d], dim=1))
