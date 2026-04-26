"""
U-Net decoder with skip connections for change detection.

Takes temporally-aggregated feature maps from the spatio-temporal
encoder and produces per-pixel logits.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv3x3 + BN + ReLU x2."""

    def __init__(self, ch_in: int, ch_out: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

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
        dropout: float = 0.0,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [256, 512, 1024, 2048]

        c1, c2, c3, c4 = encoder_channels

        # Decoder blocks: upsample → cat(skip) → conv
        # f4 at H/32 → d3 at H/16 → d2 at H/8 → d1 at H/4 → d0 at H/2 → final at H
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1, dropout=dropout)

        # H/4 → H/2
        self.up0 = nn.ConvTranspose2d(c1, 64, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(64, 64, dropout=dropout)

        # H/2 → H (final upsample to original resolution)
        self.up_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_final = ConvBlock(32, 32, dropout=dropout)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, encoder_feats: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_feats: [f1, f2, f3, f4] from SpatioTemporalEncoder.

        Returns:
            logits: (B, num_classes, H, W).
        """
        f1, f2, f3, f4 = encoder_feats

        d3 = self.dec3(torch.cat([self.up3(f4), f3], dim=1))  # (B, 1024, H/16, W/16)
        d2 = self.dec2(torch.cat([self.up2(d3), f2], dim=1))  # (B, 512, H/8, W/8)
        d1 = self.dec1(torch.cat([self.up1(d2), f1], dim=1))  # (B, 256, H/4, W/4)
        d0 = self.dec0(self.up0(d1))                           # (B, 64, H/2, W/2)
        df = self.dec_final(self.up_final(d0))                  # (B, 32, H, W)

        return self.head(df)
