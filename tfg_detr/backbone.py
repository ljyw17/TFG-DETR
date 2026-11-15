import torch.nn as nn

from .wfb import WFB


class TFGDetrBackbone(nn.Module):
    def __init__(self, out_channels=(128, 256, 512), levels=2, in_channels=3):
        super().__init__()
        c3, c4, c5 = out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c3 // 2, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(c3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3 // 2, c3 // 2, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(c3 // 2),
            nn.ReLU(inplace=True),
        )
        self.wfb3 = WFB(c3 // 2, c3, downsample=True, levels=levels)
        self.wfb4 = WFB(c3, c4, downsample=True, levels=levels)
        self.wfb5 = WFB(c4, c5, downsample=True, levels=levels)
        self.out_channels = out_channels
        self.strides = (8, 16, 32)

    def forward(self, images):
        x = self.stem(images)
        s3 = self.wfb3(x)
        s4 = self.wfb4(s3)
        s5 = self.wfb5(s4)
        return [s3, s4, s5]
