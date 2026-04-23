"""
ConvLSTM temporal encoder for spatio-temporal change detection.

Applies ConvLSTM across the temporal dimension at each spatial scale
to aggregate temporal context into per-scale feature maps.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell operating on 2D feature maps.

    All four gates (input, forget, output, cell) computed via one
    concatenated Conv2d for efficiency.

    Args:
        input_dim: Number of channels in the input feature map.
        hidden_dim: Number of channels in the hidden/cell state.
        kernel_size: Convolution kernel size (default 3).
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim, H, W) input features.
            h: (B, hidden_dim, H, W) hidden state.
            c: (B, hidden_dim, H, W) cell state.

        Returns:
            (h_next, c_next) each (B, hidden_dim, H, W).
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMStack(nn.Module):
    """
    Processes a temporal sequence of spatial feature maps through a ConvLSTM.

    Args:
        input_dim: Channels in the input per timestep.
        hidden_dim: Channels in the ConvLSTM hidden state.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, B, C_in, H, W) temporal sequence of spatial features.

        Returns:
            h_final: (B, hidden_dim, H, W) hidden state from last timestep.
            h_all: (T, B, hidden_dim, H, W) hidden states from all timesteps.
        """
        T, B, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)

        h_all = []
        for t in range(T):
            h, c = self.cell(x[t], h, c)
            h_all.append(h)

        h_all = torch.stack(h_all)  # (T, B, hidden_dim, H, W)
        return h, h_all


class SpatioTemporalEncoder(nn.Module):
    """
    ResNet-50 per date → ConvLSTM across dates at each spatial scale.

    Input:  (B, T, C, H, W) = (B, 6, 3, 256, 256)
    Output: List of 4 temporally-aggregated feature maps.

    If temporal_encoding_dim > 0, expects a doy tensor to produce
    per-timestep embeddings that get concatenated to spatial features
    before the ConvLSTM.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        temporal_encoding_dim: int = 0,
    ):
        super().__init__()
        self.spatial = ResNet50Encoder(checkpoint_path)

        # ResNet output channels at each scale
        spatial_channels = [256, 512, 1024, 2048]
        self.temporal_encoding_dim = temporal_encoding_dim

        # ConvLSTM at each spatial scale
        self.lstms = nn.ModuleList()
        for ch in spatial_channels:
            input_dim = ch + temporal_encoding_dim
            self.lstms.append(ConvLSTMStack(input_dim, ch))

    def forward(
        self,
        x: torch.Tensor,
        temporal_emb: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) = (B, 6, 3, 256, 256)
            temporal_emb: (T, encoding_dim) per-timestep temporal embeddings,
                          or None if no temporal encoding.

        Returns:
            List of 4 tensors, each (B, C_i, H_i, W_i),
            temporally aggregated via ConvLSTM.
        """
        B, T, C, H, W = x.shape

        # Process all dates through ResNet in one pass
        x_flat = x.reshape(B * T, C, H, W)
        spatial_feats = self.spatial(x_flat)  # List of 4: each (B*T, C_i, H_i, W_i)

        # Reshape each scale to (T, B, C_i, H_i, W_i) and run ConvLSTM
        temporal_feats = []
        for feat, lstm in zip(spatial_feats, self.lstms):
            C_i = feat.shape[1]
            H_i, W_i = feat.shape[2], feat.shape[3]

            feat_t = feat.reshape(B, T, C_i, H_i, W_i)
            feat_t = feat_t.permute(1, 0, 2, 3, 4)  # (T, B, C_i, H_i, W_i)

            # Inject temporal encoding if provided
            if temporal_emb is not None:
                # temporal_emb: (T, encoding_dim) → (T, B, encoding_dim, H_i, W_i)
                t_emb = temporal_emb.unsqueeze(1).unsqueeze(3).unsqueeze(4)
                t_emb = t_emb.expand(-1, B, -1, H_i, W_i)
                feat_t = torch.cat([feat_t, t_emb], dim=2)  # (T, B, C_i + enc_dim, H_i, W_i)

            h_final, _ = lstm(feat_t)  # (B, C_i, H_i, W_i)
            temporal_feats.append(h_final)

        return temporal_feats


# Lazy import to avoid circular dependency
from .spatial_encoder import ResNet50Encoder
