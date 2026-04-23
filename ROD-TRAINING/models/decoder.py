"""
U-Net decoder with skip connections for change detection.

Takes temporally-aggregated feature maps from the spatio-temporal
encoder and produces per-pixel logits.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv3x3 + BN + ReLU x2."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    U-Net decoder with skip connections from temporal encoder.

    Input:  List of 4 temporally-aggregated feature maps:
        f1: (B, 256, H/4, W/4)
        f2: (B, 512, H/8, W/8)
        f3: (B, 1024, H/16, W/16)
        f4: (B, 2048, H/32, W/32)

    Output: (B, num_classes, H, W) logits.

    Args:
        encoder_channels: Channel dims from encoder at each scale.
        num_classes: Output classes (default 2: healthy, mortality).
    """

    def __init__(
        self,
        encoder_channels: list[int] | None = None,
        num_classes: int = 2,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [256, 512, 1024, 2048]

        c1, c2, c3, c4 = encoder_channels

        # Decoder blocks: upsample → cat(skip) → conv
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        # Final upsample to original resolution
        self.up0 = nn.ConvTranspose2d(c1, 64, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(64, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, encoder_feats: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_feats: [f1, f2, f3, f4] from SpatioTemporalEncoder.

        Returns:
            logits: (B, num_classes, H, W).
        """
        f1, f2, f3, f4 = encoder_feats

        d3 = self.dec3(torch.cat([self.up3(f4), f3], dim=1))  # (B, 1024, H/8, W/8)
        d2 = self.dec2(torch.cat([self.up2(d3), f2], dim=1))  # (B, 512, H/4, W/4)
        d1 = self.dec1(torch.cat([self.up1(d2), f1], dim=1))  # (B, 256, H/2, W/2)
        d0 = self.dec0(self.up0(d1))                           # (B, 64, H, W)

        return self.head(d0)
