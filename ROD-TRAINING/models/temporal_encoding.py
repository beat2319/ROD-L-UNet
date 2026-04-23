"""
Sinusoidal day-of-year temporal encoding for Sentinel-1 time series.

Encodes acquisition timing using multi-frequency sinusoidal functions
so the model can represent seasonality and irregular revisit intervals.
"""

import math

import torch
import torch.nn as nn


class SinusoidalDoYEncoding(nn.Module):
    """
    Sinusoidal day-of-year encoding for temporal awareness.

    Uses multi-frequency sinusoidal encoding with period = 365.2425
    (mean tropical year) for correct leap-year handling.

    Args:
        encoding_dim: Output embedding dimension (default 64).
        max_period: Base period for sinusoidal encoding (default 365.2425).
    """

    def __init__(self, encoding_dim: int = 64, max_period: float = 365.2425):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.max_period = max_period

        self.num_freq = encoding_dim // 2
        raw_dim = 2 * self.num_freq

        self.projection = nn.Sequential(
            nn.Linear(raw_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim),
        )

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            doy: (T,) day-of-year integers, range [0, 365].

        Returns:
            (T, encoding_dim) temporal embeddings.
        """
        freqs = torch.arange(self.num_freq, device=doy.device, dtype=torch.float32)
        freqs = (2.0 * math.pi) / (self.max_period ** (freqs / self.num_freq))

        # (T, 1) * (1, num_freq) -> (T, num_freq)
        angles = doy.unsqueeze(1).float() * freqs.unsqueeze(0)
        raw = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return self.projection(raw)
