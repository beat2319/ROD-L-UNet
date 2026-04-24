"""
Training script for Sentinel-1 change detection model.

Usage:
    python train.py --manifest manifest.csv --checkpoint B2_rn50_moco_0099.pth
    python train.py --manifest manifest.csv --epochs 50 --batch_size 4
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import S1ChangeDataset
from losses import CombinedCrossEntropyDiceLoss
from metrics import ChangeDetectionMetrics
from models import S1ChangeDetector


def make_balanced_sampler(df):
    """Create a WeightedRandomSampler for ~50/50 positive/negative balance."""
    labels = (df["label"] == "positive").values.astype(int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> dict:
    model.train()
    metrics = ChangeDetectionMetrics(num_classes=2, ignore_index=255)
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (chip, mask, doy, valid) in enumerate(loader):
        chip = chip.to(device)
        mask = mask.to(device)
        doy = doy.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(chip, doy)
            loss = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        metrics.update(logits.detach(), mask.detach())

        if batch_idx % 20 == 0:
            print(
                f"  [{batch_idx}/{len(loader)}] loss={loss.item():.4f}",
                flush=True,
            )

    avg_loss = total_loss / max(num_batches, 1)
    result = metrics.compute()
    result["loss"] = avg_loss
    return result


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    metrics = ChangeDetectionMetrics(num_classes=2, ignore_index=255)
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for chip, mask, doy, valid in loader:
            chip = chip.to(device)
            mask = mask.to(device)
            doy = doy.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(chip, doy)
                loss = criterion(logits, mask)

            total_loss += loss.item()
            num_batches += 1
            metrics.update(logits, mask)

    avg_loss = total_loss / max(num_batches, 1)
    result = metrics.compute()
    result["loss"] = avg_loss
    return result


def main():
    parser = argparse.ArgumentParser(description="Train S1 change detection model")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--freeze-epochs", type=int, default=10,
                        help="Epochs to freeze encoder layers 1-2")
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--temporal-lr", type=float, default=5e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 2.0],
                        help="Class weights for loss [healthy, mortality]")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Label smoothing for cross-entropy")
    parser.add_argument("--ce-weight", type=float, default=1.0,
                        help="Weight for cross-entropy component")
    parser.add_argument("--dice-weight", type=float, default=0.5,
                        help="Weight for Dice component")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifest
    df = pd.read_csv(args.manifest)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Datasets
    train_dataset = S1ChangeDataset(train_df, augment=True)
    val_dataset = S1ChangeDataset(val_df, augment=False)

    # Balanced sampler for training
    sampler = make_balanced_sampler(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = S1ChangeDetector(
        checkpoint_path=args.checkpoint,
        num_classes=2,
        temporal_encoding_dim=64,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    criterion = CombinedCrossEntropyDiceLoss(
        num_classes=2,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        ignore_index=255,
    )

    # Optimizer with discriminative learning rates
    encoder_spatial_params = []
    encoder_spatial_low_params = []
    temporal_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if "encoder.spatial.stem" in name or "encoder.spatial.layer1" in name or "encoder.spatial.layer2" in name:
            encoder_spatial_low_params.append(param)
        elif "encoder.spatial" in name:
            encoder_spatial_params.append(param)
        elif "encoder.lstms" in name:
            temporal_params.append(param)
        elif "temporal_encoding" in name:
            temporal_params.append(param)
        else:
            decoder_params.append(param)

    encoder_spatial_mid_lr = (args.encoder_lr + args.temporal_lr) / 2
    optimizer = optim.AdamW([
        {"params": encoder_spatial_low_params, "lr": args.encoder_lr},
        {"params": encoder_spatial_params, "lr": encoder_spatial_mid_lr},
        {"params": temporal_params, "lr": args.temporal_lr},
        {"params": decoder_params, "lr": args.decoder_lr},
    ], weight_decay=1e-2)

    # Decay from the configured starting rates instead of ramping upward.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0.0,
    )

    scaler = torch.amp.GradScaler("cuda")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = save_dir / "metrics.csv"
    metrics_writer = csv.writer(open(metrics_csv, "w", newline=""))
    metric_keys = [
        "epoch", "phase", "loss", "accuracy", "mean_iou", "mean_f1",
        "iou_class_0", "iou_class_1", "f1_class_0", "f1_class_1",
        "precision_class_1", "recall_class_1", "max_pred_prob", "lr",
    ]
    metrics_writer.writerow(metric_keys)

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        # Encoder freezing schedule
        if epoch <= args.freeze_epochs:
            for param in encoder_spatial_low_params:
                param.requires_grad = False
        else:
            for param in encoder_spatial_low_params:
                param.requires_grad = True

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        current_lr = optimizer.param_groups[-1]["lr"]
        t0 = time.time()
        train_result = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch,
        )
        train_time = time.time() - t0

        # Validate
        val_result = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Print results
        print(f"\nTrain | loss={train_result['loss']:.4f} "
              f"iou={train_result['mean_iou']:.4f} "
              f"iou_mort={train_result['iou_class_1']:.4f} "
              f"f1_mort={train_result['f1_class_1']:.4f} "
              f"max_prob={train_result['max_pred_prob']:.3f}")
        print(f"Val   | loss={val_result['loss']:.4f} "
              f"iou={val_result['mean_iou']:.4f} "
              f"iou_mort={val_result['iou_class_1']:.4f} "
              f"f1_mort={val_result['f1_class_1']:.4f} "
              f"prec={val_result['precision_class_1']:.4f} "
              f"rec={val_result['recall_class_1']:.4f} "
              f"max_prob={val_result['max_pred_prob']:.3f}")
        print(f"Time: {train_time:.1f}s | LR: {current_lr:.2e}")

        # Collapse warning
        if val_result["max_pred_prob"] < 0.5 and epoch >= 3:
            print("WARNING: Model may have collapsed (max_pred_prob < 0.5). "
                  "Consider increasing alpha.")

        # Log metrics
        for phase, result in [("train", train_result), ("val", val_result)]:
            row = [
                epoch, phase, result["loss"], result["accuracy"],
                result["mean_iou"], result["mean_f1"],
                result["iou_class_0"], result["iou_class_1"],
                result["f1_class_0"], result["f1_class_1"],
                result["precision_class_1"], result["recall_class_1"],
                result["max_pred_prob"], current_lr,
            ]
            metrics_writer.writerow(row)

        # Save best model by mortality IoU
        if val_result["iou_class_1"] > best_iou:
            best_iou = val_result["iou_class_1"]
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  New best mortality IoU: {best_iou:.4f}")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_dir / f"model_epoch{epoch}.pt")

    print(f"\nTraining complete. Best mortality IoU: {best_iou:.4f}")
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
