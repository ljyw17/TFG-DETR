import torch
import torch.nn.functional as F

_HAAR_KERNELS = torch.tensor([
    [[ 0.5,  0.5],
     [ 0.5,  0.5]],
    [[ 0.5,  0.5],
     [-0.5, -0.5]],
    [[ 0.5, -0.5],
     [ 0.5, -0.5]],
    [[ 0.5, -0.5],
     [-0.5,  0.5]],
])


def dwt_haar(x):
    c = x.shape[1]
    kernels = _HAAR_KERNELS.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    return F.conv2d(x, kernels[:, None], stride=2, groups=c)


def idwt_haar(z, channels):
    c = channels
    kernels = _HAAR_KERNELS.to(device=z.device, dtype=z.dtype).repeat(c, 1, 1, 1)
    return F.conv_transpose2d(z, kernels[:, None], stride=2, groups=c)


def dwt_split(z):
    return z[:, 0::4], z[:, 1::4], z[:, 2::4], z[:, 3::4]


def dwt_merge(ll, lh, hl, hh):
    return torch.stack([ll, lh, hl, hh], dim=2).reshape(
        ll.shape[0], ll.shape[1] * 4, ll.shape[2], ll.shape[3]
    )
