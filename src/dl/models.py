"""
Two model options, pick via config.yaml `dl.architecture`:

  - srcnn : simple 3-layer CNN, good first baseline, trains fast
  - edsr  : deeper residual network, standard strong choice for SR tasks,
            adapted here to take N-band satellite input (not just RGB)
"""
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EDSR(nn.Module):
    """Since tiles are pre-upsampled to a fixed size in preprocess.py, this
    network refines detail at fixed resolution rather than using pixel-shuffle
    upsampling layers -- simplest correct fit for the tiling scheme above.
    Swap in nn.PixelShuffle blocks if you switch to native LR-sized inputs."""

    def __init__(self, in_channels: int = 4, num_features: int = 64, num_res_blocks: int = 16):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_res_blocks)])
        self.body_tail = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.tail = nn.Conv2d(num_features, in_channels, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body_tail(self.body(feat))
        feat = feat + res
        return self.tail(feat) + x  # global residual: predict the *refinement*


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["dl"]["architecture"]
    if arch == "srcnn":
        return SRCNN(in_channels=cfg["dl"]["in_channels"])
    elif arch == "edsr":
        return EDSR(
            in_channels=cfg["dl"]["in_channels"],
            num_features=cfg["dl"]["num_features"],
            num_res_blocks=cfg["dl"]["num_res_blocks"],
        )
    raise ValueError(f"Unknown architecture: {arch}")
