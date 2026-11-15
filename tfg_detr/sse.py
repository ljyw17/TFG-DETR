import torch
import torch.nn as nn
import torch.nn.functional as F

from .msf import MSF
from .dmp import DMP


class SSE(nn.Module):
    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.branch_a = nn.Conv2d(low_channels, low_channels, kernel_size=1)
        self.branch_b = nn.Conv2d(low_channels, low_channels, kernel_size=1)
        self.msf = MSF(low_channels)
        self.dmp_enhance = DMP(high_channels)
        self.dmp_suppress = DMP(high_channels)
        self.fuse = nn.Conv2d(2 * low_channels, low_channels, kernel_size=3,
                              padding=1)
        self.norm = nn.BatchNorm2d(low_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_low, f_high):
        if f_high.shape[-2:] != f_low.shape[-2:]:
            f_high = F.interpolate(f_high, size=f_low.shape[-2:], mode="bilinear",
                                   align_corners=False)

        e = self.dmp_enhance(f_high)
        m = self.dmp_suppress(f_high)

        f_u = self.branch_a(f_low) + self.msf(self.branch_b(f_low))

        fused = torch.cat([f_u * e, f_u * m], dim=1)
        return self.relu(self.norm(self.fuse(fused))) + f_low
