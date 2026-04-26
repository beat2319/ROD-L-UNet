"""
ResNet-50 spatial feature extractor for Sentinel-1 SAR input.

Loads SSL4EO-S12 MoCo-v2 pretrained weights.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Encoder(nn.Module):
    """
    Pretrained ResNet-50 feature extractor for SAR input.

    Input:  (B, C, H, W)  -- single-date SAR in dB
    Output: List of 4 feature maps at progressively lower resolutions:
        [0]: (B, 256, H/4, W/4)   -- from layer1
        [1]: (B, 512, H/8, W/8)   -- from layer2
        [2]: (B, 1024, H/16, W/16) -- from layer3
        [3]: (B, 2048, H/32, W/32) -- from layer4
    """

    def __init__(self, checkpoint_path: str | None = None, input_channels: int = 2):
        super().__init__()

        backbone = models.resnet50(weights=None)

        backbone.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Handle MoCo wrapped state dicts
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Strip MoCo prefixes (module., encoder_q.)
            cleaned = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "").replace("encoder_q.", "")
                cleaned[k] = v

            # Load any overlapping pretrained input channels directly.
            conv1_weight = cleaned.pop("conv1.weight", None)
            backbone.load_state_dict(cleaned, strict=False)

            if conv1_weight is not None:
                with torch.no_grad():
                    in_ch = min(conv1_weight.shape[1], input_channels)
                    backbone.conv1.weight.data[:, :in_ch] = conv1_weight[:, :in_ch]

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) SAR input for one date.

        Returns:
            List of 4 feature maps [c1, c2, c3, c4].
        """
        c0 = self.stem(x)      # (B, 64, H/4, W/4)
        c1 = self.layer1(c0)   # (B, 256, H/4, W/4)
        c2 = self.layer2(c1)   # (B, 512, H/8, W/8)
        c3 = self.layer3(c2)   # (B, 1024, H/16, W/16)
        c4 = self.layer4(c3)   # (B, 2048, H/32, W/32)

        return [c1, c2, c3, c4]
