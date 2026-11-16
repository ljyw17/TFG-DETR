import torch
import torch.nn as nn


class DFSA(nn.Module):
    def __init__(self, channels, bands=3, reduction=16):
        super().__init__()
        if bands < 1:
            raise ValueError("bands must be >= 1")
        self.channels = channels
        self.bands = bands

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, bands),
        )
        self.spatial_gate = nn.Conv2d(2 * channels, 1, kernel_size=7,
                                      padding=3)
        self.sigmoid = nn.Sigmoid()
        self._mask_cache = {}

    def _radial_masks(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key in self._mask_cache:
            return self._mask_cache[key]

        hh = torch.fft.fftfreq(h, d=1.0 / h, device=device)
        ww = torch.fft.fftfreq(w, d=1.0 / w, device=device)
        rho_y = torch.where(hh < h / 2, hh, hh - h).abs()
        rho_x = torch.where(ww < w / 2, ww, ww - w).abs()
        rho = (rho_y[:, None] ** 2 + rho_x[None, :] ** 2).sqrt()

        r_max = rho.max()
        bounds = [r_max * k / self.bands for k in range(1, self.bands)]
        masks = []
        lower = torch.zeros_like(rho)
        for k in range(self.bands):
            upper = bounds[k] if k < self.bands - 1 else r_max * 2.0
            masks.append(((rho >= lower) & (rho < upper)).to(dtype))
            lower = bounds[k]
        masks = torch.stack(masks)
        self._mask_cache[key] = masks
        return masks

    def forward(self, f):
        b, c, h, w = f.shape

        z = torch.fft.fftshift(torch.fft.fft2(f, dim=(-2, -1)), dim=(-2, -1))

        masks = self._radial_masks(h, w, z.device, torch.float32)
        z_bands = z.unsqueeze(1) * masks.unsqueeze(0).unsqueeze(2)

        pi = torch.softmax(self.gate(f), dim=1)

        z_tilde = (z_bands * pi.view(b, self.bands, 1, 1, 1)).sum(dim=1)
        f_tilde = torch.fft.ifft2(
            torch.fft.ifftshift(z_tilde, dim=(-2, -1)), dim=(-2, -1)
        ).real

        avg = f_tilde.mean(dim=1, keepdim=True)
        mx = f_tilde.amax(dim=1, keepdim=True)
        gate_map = self.sigmoid(self.spatial_gate(torch.cat([avg, mx], dim=1)))
        return f_tilde + f * gate_map
