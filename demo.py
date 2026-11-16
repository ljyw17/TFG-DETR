import torch

from tfg_detr import (
    DFSA,
    DMP,
    MSF,
    WFB,
    WFC,
    SSE,
    TFGDetrBackbone,
    dwt_haar,
    idwt_haar,
)
from tfg_detr.integrations.rtdetr_adapter import TFGDetrFeatureExtractor


def check_haar_round_trip():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 32, 32)
    z = dwt_haar(x)
    assert z.shape == (2, 32, 16, 16), f"unexpected DWT shape: {tuple(z.shape)}"
    recon = idwt_haar(z, x.shape[1])
    err = (recon - x).abs().max().item()
    assert err < 1e-5, f"Haar round-trip error too large: {err:.2e}"
    print(f"[1/3] Haar round-trip OK (max abs error = {err:.2e})")


def check_standalone_modules():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 40, 40)
    wfc = WFC(channels=64, levels=2)
    y = wfc(x)
    assert y.shape == x.shape
    print(f"[2/3] WFC     : {tuple(x.shape)} -> {tuple(y.shape)}")

    wfb_same = WFB(in_channels=64, out_channels=64, downsample=False, levels=2)
    y = wfb_same(x)
    assert y.shape == x.shape
    print(f"      WFB same: {tuple(x.shape)} -> {tuple(y.shape)}")

    wfb_down = WFB(in_channels=64, out_channels=128, downsample=True, levels=2)
    y = wfb_down(x)
    assert y.shape == (2, 128, 20, 20)
    print(f"      WFB down: {tuple(x.shape)} -> {tuple(y.shape)}")

    dfsa = DFSA(channels=64, bands=3)
    y = dfsa(x)
    assert y.shape == x.shape
    print(f"      DFSA    : {tuple(x.shape)} -> {tuple(y.shape)}")

    msf = MSF(channels=64, expansion=4, dsc_kernel=5)
    u = torch.randn(2, 64, 40, 40)
    y = msf(u)
    assert y.shape == u.shape
    print(f"      MSF     : {tuple(u.shape)} -> {tuple(y.shape)}")

    dmp = DMP(in_channels=64, kernel_sizes=(3, 5, 7))
    y = dmp(x)
    assert y.shape == (2, 1, 40, 40)
    print(f"      DMP     : {tuple(x.shape)} -> {tuple(y.shape)}")
    print("[2/3] standalone modules OK")


def check_full_pipeline():
    torch.manual_seed(0)
    backbone = TFGDetrBackbone(out_channels=(256, 512, 1024), levels=2,
                               pretrained=False)
    s3, s4, s5 = backbone(torch.randn(2, 3, 640, 640))
    assert s3.shape == (2, 256, 80, 80)
    assert s4.shape == (2, 512, 40, 40)
    assert s5.shape == (2, 1024, 20, 20)
    print(f"[3/3] backbone : S3 {tuple(s3.shape)}, S4 {tuple(s4.shape)}, S5 {tuple(s5.shape)}")

    class _AIFI(torch.nn.Module):
        def forward(self, x):
            return x

    class _CCFF(torch.nn.Module):
        def forward(self, s3, s4, s5):
            return s3, s4, s5

    extractor = TFGDetrFeatureExtractor(
        aifi=_AIFI(), ccff=_CCFF(), backbone_channels=(256, 512, 1024),
        bands=3, levels=2, pretrained=False
    )
    p3, p4, p5 = extractor(torch.randn(2, 3, 640, 640))
    assert p3.shape == s3.shape and p4.shape == s4.shape and p5.shape == s5.shape
    print(f"[3/3] adapter  : P3 {tuple(p3.shape)}, P4 {tuple(p4.shape)}, P5 {tuple(p5.shape)}")
    print("[3/3] full pipeline OK")


if __name__ == "__main__":
    check_haar_round_trip()
    check_standalone_modules()
    check_full_pipeline()
    print("\nAll smoke tests passed.")
