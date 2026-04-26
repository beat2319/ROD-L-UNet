"""
S1ChangeDetector: complete Sentinel-1 change detection model.

ResNet-50 (spatial) → ConvLSTM (temporal) → U-Net (decoder).

Input:  (B, T, C, H, W) + optional DoY per timestep
Output: (B, num_classes, H, W) logits
"""

import torch
import torch.nn as nn

from .temporal_encoder import SpatioTemporalEncoder
from .temporal_encoding import SinusoidalDoYEncoding
from .decoder import UNetDecoder


class S1ChangeDetector(nn.Module):
    """
    Full Sentinel-1 change detection model.

    Args:
        checkpoint_path: Path to SSL4EO-S12 MoCo-v2 ResNet-50 weights.
        num_classes: Output classes (default 2: healthy, mortality).
        input_channels: SAR channels per timestep (2=VV/VH, 3=VV/VH/RVI).
        temporal_encoding_dim: DoY encoding dimension (0 disables).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        num_classes: int = 2,
        input_channels: int = 2,
        temporal_encoding_dim: int = 32,
        temporal_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()

        self.use_temporal_encoding = temporal_encoding_dim > 0

        self.encoder = SpatioTemporalEncoder(
            checkpoint_path=checkpoint_path,
            input_channels=input_channels,
            temporal_encoding_dim=temporal_encoding_dim if self.use_temporal_encoding else 0,
            temporal_dropout=temporal_dropout,
        )

        if self.use_temporal_encoding:
            self.temporal_encoding = SinusoidalDoYEncoding(
                encoding_dim=temporal_encoding_dim,
            )

        self.decoder = UNetDecoder(num_classes=num_classes, dropout=decoder_dropout)

    def forward(
        self,
        x: torch.Tensor,
        doy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) SAR time series.
            doy: (B, T) day-of-year per timestep, or None.

        Returns:
            logits: (B, num_classes, H, W) = (B, 2, 256, 256).
        """
        temporal_emb = None
        if self.use_temporal_encoding and doy is not None:
            if doy.ndim != 2 or doy.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"Expected doy with shape (B, T) matching x, got {tuple(doy.shape)} "
                    f"for input {tuple(x.shape)}"
                )
            temporal_emb = self.temporal_encoding(doy)

        encoder_feats = self.encoder(x, temporal_emb=temporal_emb)
        return self.decoder(encoder_feats)
