import torch
import torch.nn as nn

from .wavelet import dwt_haar, idwt_haar, dwt_split, dwt_merge


class WFC(nn.Module):
    def __init__(self, channels, levels=2):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.channels = channels
        self.levels = levels

        self.depthwise = nn.ModuleList(
            nn.Conv2d(4 * channels, 4 * channels, kernel_size=5, padding=2,
                      groups=4 * channels)
            for _ in range(levels)
        )
        self.level_scale = nn.ParameterList(
            nn.Parameter(torch.ones(4 * channels)) for _ in range(levels)
        )
        self.proj = nn.Conv2d(channels * levels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if h % (2 ** self.levels) or w % (2 ** self.levels):
            raise ValueError(
                f"WFC requires H and W divisible by 2**levels "
                f"(got {h}x{w} with levels={self.levels}); pad the input."
            )

        z_levels = []
        ll = x
        for lvl in range(self.levels):
            sub = dwt_haar(ll)
            z = self.depthwise[lvl](sub) * self.level_scale[lvl].view(1, -1, 1, 1)
            z_levels.append(z)
            ll = z[:, 0::4]

        outputs = []
        ll_next = torch.zeros_like(z_levels[-1][:, 0::4])
        for lvl in reversed(range(self.levels)):
            z_ll, z_lh, z_hl, z_hh = dwt_split(z_levels[lvl])
            ll_hat = idwt_haar(dwt_merge(z_ll + ll_next, z_lh, z_hl, z_hh), c)
            outputs.append(ll_hat)
            ll_next = ll_hat

        return self.proj(torch.cat(outputs, dim=1))
