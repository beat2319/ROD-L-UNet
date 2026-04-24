"""
Combined Cross-Entropy + Dice losses for change detection.

Both components respect ignore_index=255 (nodata pixels excluded).
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


class DiceLoss(nn.Module):
    """
    Soft Dice loss with class weighting and ignore support.

    Args:
        num_classes: Number of classes (default 2).
        class_weights: Per-class weights for Dice averaging.
        ignore_index: Class label to ignore (default 255).
        smooth: Smoothing constant (default 1.0).
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes, H, W) raw predictions.
            targets: (B, H, W) long tensor class labels.

        Returns:
            Scalar Dice loss (1 - weighted_dice).
        """
        valid = targets != self.ignore_index
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0

        probs = F.softmax(logits, dim=1)

        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        total_dice = 0.0
        weight_sum = 0.0

        for c in range(self.num_classes):
            # One-hot for class c, masked to valid pixels
            target_c = (targets_clamped == c).float()
            pred_c = probs[:, c]

            # Only compute on valid pixels
            target_c = target_c * valid.float()
            pred_c = pred_c * valid.float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            w = 1.0
            if weights is not None:
                w = weights[c].item()

            total_dice += w * dice
            weight_sum += w

        if weight_sum == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return 1.0 - total_dice / weight_sum


class CombinedCrossEntropyDiceLoss(nn.Module):
    """
    Combined cross-entropy + Dice loss for change detection.

    loss = ce_weight * CrossEntropy + dice_weight * DiceLoss

    Args:
        num_classes: Number of output classes (default 2).
        alpha: Per-class weights for both CE and Dice (default [1.0, 2.0]).
        label_smoothing: Label smoothing factor for CE (default 0.05).
        ce_weight: Weight for CE component (default 1.0).
        dice_weight: Weight for Dice component (default 0.5).
        ignore_index: Class label to ignore (default 255).
    """

    def __init__(
        self,
        num_classes: int = 2,
        alpha: list[float] | None = None,
        label_smoothing: float = 0.05,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int = 255,
    ):
        super().__init__()
        if alpha is None:
            alpha = [1.0, 2.0]

        self.cross_entropy = SmoothedCrossEntropyLoss(
            alpha=alpha,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(
            num_classes=num_classes,
            class_weights=alpha,
            ignore_index=ignore_index,
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes, H, W).
            targets: (B, H, W) long tensor.

        Returns:
            Scalar combined loss.
        """
        ce_loss = self.cross_entropy(logits, targets)
        d_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * d_loss
