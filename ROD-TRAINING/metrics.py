"""
Change detection evaluation metrics.

Computes IoU, F1, precision, recall per class, plus overall accuracy
and a collapse detection signal (max prediction probability).
"""

import torch


class ChangeDetectionMetrics:
    """
    Computes per-epoch metrics for change detection.

    All metrics exclude pixels where target == ignore_index.

    Args:
        num_classes: Number of valid classes (default 2).
        ignore_index: Class label to ignore (default 255).
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = [[0] * self.num_classes for _ in range(self.num_classes)]
        self.total_valid = 0
        self.max_pred_prob = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits:  (B, num_classes, H, W) raw model output.
            targets: (B, H, W) long tensor ground truth.
        """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # (B, H, W)

            valid = targets != self.ignore_index
            valid_count = int(valid.sum())
            if valid_count == 0:
                return

            preds_flat = preds[valid].to(torch.int64)
            targets_flat = targets[valid].to(torch.int64)
            in_range = (
                (targets_flat >= 0)
                & (targets_flat < self.num_classes)
                & (preds_flat >= 0)
                & (preds_flat < self.num_classes)
            )

            if bool(in_range.any()):
                encoded = (
                    targets_flat[in_range] * self.num_classes
                    + preds_flat[in_range]
                )
                counts = torch.bincount(
                    encoded,
                    minlength=self.num_classes * self.num_classes,
                ).reshape(self.num_classes, self.num_classes)

                counts_list = counts.cpu().tolist()
                for target_class in range(self.num_classes):
                    for pred_class in range(self.num_classes):
                        self.confusion[target_class][pred_class] += int(
                            counts_list[target_class][pred_class],
                        )

            self.total_valid += valid_count

            # Track max predicted probability for collapse detection
            max_prob = probs.max(dim=1).values[valid].max().item()
            self.max_pred_prob = max(self.max_pred_prob, max_prob)

    def compute(self) -> dict:
        """
        Returns:
            Dict with IoU, F1, precision, recall per class + mean,
            overall accuracy, and max_pred_prob.
        """
        ious = []
        f1s = []
        precisions = []
        recalls = []
        per_class_accs = []

        for c in range(self.num_classes):
            tp = self.confusion[c][c]
            fp = sum(self.confusion[j][c] for j in range(self.num_classes)) - tp
            fn = sum(self.confusion[c][j] for j in range(self.num_classes)) - tp

            # IoU
            union = tp + fp + fn
            iou = tp / union if union > 0 else 0.0
            ious.append(iou)

            # Precision / Recall
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

            # F1
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)

            # Per-class accuracy
            class_total = sum(self.confusion[c])
            per_class_accs.append(tp / class_total if class_total > 0 else 0.0)

        total_correct = sum(self.confusion[c][c] for c in range(self.num_classes))
        accuracy = total_correct / self.total_valid if self.total_valid > 0 else 0.0

        result = {
            "accuracy": accuracy,
            "mean_iou": sum(ious) / len(ious),
            "mean_f1": sum(f1s) / len(f1s),
            "max_pred_prob": self.max_pred_prob,
        }

        for c in range(self.num_classes):
            result[f"iou_class_{c}"] = ious[c]
            result[f"f1_class_{c}"] = f1s[c]
            result[f"precision_class_{c}"] = precisions[c]
            result[f"recall_class_{c}"] = recalls[c]

        return result
