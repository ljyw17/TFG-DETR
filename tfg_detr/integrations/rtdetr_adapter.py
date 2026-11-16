import torch.nn as nn

from .backbone import TFGDetrBackbone
from .dfsa import DFSA
from .sse import SSE


class TFGDetrFeatureExtractor(nn.Module):
    def __init__(self, aifi, ccff, backbone_channels=(256, 512, 1024),
                 bands=3, levels=2, pretrained=True, freeze_stem=True,
                 p5_channels=None):
        super().__init__()
        self.backbone = TFGDetrBackbone(backbone_channels, levels=levels,
                                        pretrained=pretrained,
                                        freeze_stem=freeze_stem)
        c3, c4, _ = backbone_channels
        high_channels = p5_channels if p5_channels is not None else backbone_channels[2]
        self.dfsa_s3 = DFSA(c3, bands=bands)
        self.dfsa_s4 = DFSA(c4, bands=bands)
        self.aifi = aifi
        self.ccff = ccff
        self.sse = SSE(c3, high_channels)

    def forward(self, images):
        s3, s4, s5 = self.backbone(images)
        s5 = self.aifi(s5)
        s3 = self.dfsa_s3(s3)
        s4 = self.dfsa_s4(s4)
        p3, p4, p5 = self.ccff([s3, s4, s5])
        p3 = self.sse(p3, p5)
        return [p3, p4, p5]
