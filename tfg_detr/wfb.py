import torch.nn as nn

from .wfc import WFC


def _projection(in_ch, out_ch, stride=1):
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_ch),
    )


def _residual_mapping(in_ch, out_ch, stride, levels):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
                  bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        WFC(out_ch, levels=levels),
        nn.BatchNorm2d(out_ch),
    )


class WFB(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, levels=2):
        super().__init__()
        self.refine = _residual_mapping(in_channels, out_channels, 1, levels)
        self.compress = _residual_mapping(out_channels, out_channels,
                                          2 if downsample else 1, levels)
        self.refine_shortcut = _projection(in_channels, out_channels)
        self.compress_shortcut = _projection(out_channels, out_channels,
                                             stride=2 if downsample else 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.refine(x) + self.refine_shortcut(x))
        return self.relu(self.compress(x) + self.compress_shortcut(x))
