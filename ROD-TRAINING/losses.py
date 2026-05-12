"""
Segmentation losses for change detection.

Provides standalone cross-entropy (with label smoothing) and focal loss.
All components respect ignore_index=255 (nodata pixels excluded).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothedCrossEntropyLoss(nn.Module):
    """
    Multi-class cross-entropy with class weights, label smoothing,
    and ignore_index support.

    Args:
        alpha: Per-class weights (e.g., [1.0, 2.0]).
        label_smoothing: Label smoothing factor (default 0.05).
        ignore_index: Class label to ignore (default 255).
    """

    def __init__(
        self,
        alpha: list[float] | None = None,
        label_smoothing: float = 0.05,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes, H, W) raw predictions.
            targets: (B, H, W) long tensor class labels.

        Returns:
            Scalar cross-entropy loss.
        """
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
        else:
            alpha = None

        valid = targets != self.ignore_index
        if int(valid.sum()) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(
            logits,
            targets,
            weight=alpha,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        return loss[valid].mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss with class weights and ignore_index support.

    Args:
        alpha: Per-class weights (e.g., [1.0, 2.0]).
        gamma: Focusing parameter (default 2.0).
        ignore_index: Class label to ignore (default 255).
    """

    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes, H, W) raw predictions.
            targets: (B, H, W) long tensor class labels.

        Returns:
            Scalar focal loss.
        """
        valid = targets != self.ignore_index
        if int(valid.sum()) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits.permute(0, 2, 3, 1)[valid]
        targets_valid = targets[valid]

        log_probs = F.log_softmax(logits_valid, dim=1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha[targets_valid]

        return loss.mean()

