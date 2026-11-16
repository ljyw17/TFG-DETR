import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from .wfb import WFB


class TFGDetrBackbone(nn.Module):
    def __init__(self, out_channels=(256, 512, 1024), levels=2,
                 pretrained=True, freeze_stem=True):
        super().__init__()
        c3, c4, c5 = out_channels
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet50(weights=weights)
        self.stage_proj = nn.ModuleList([
            nn.Conv2d(512, c3, kernel_size=1),
            nn.Conv2d(1024, c4, kernel_size=1),
            nn.Conv2d(2048, c5, kernel_size=1),
        ])
        self.wfb3 = WFB(c3, c3, downsample=False, levels=levels)
        self.wfb4 = WFB(c4, c4, downsample=False, levels=levels)
        self.wfb5 = WFB(c5, c5, downsample=False, levels=levels)
        if freeze_stem:
            for p in self.resnet.conv1.parameters():
                p.requires_grad_(False)
            for p in self.resnet.bn1.parameters():
                p.requires_grad_(False)
            for p in self.resnet.layer1.parameters():
                p.requires_grad_(False)
        self.out_channels = out_channels
        self.strides = (8, 16, 32)

    def forward(self, images):
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x3 = self.resnet.layer2(x)
        x4 = self.resnet.layer3(x3)
        x5 = self.resnet.layer4(x4)
        s3 = self.wfb3(self.stage_proj[0](x3))
        s4 = self.wfb4(self.stage_proj[1](x4))
        s5 = self.wfb5(self.stage_proj[2](x5))
        return [s3, s4, s5]
