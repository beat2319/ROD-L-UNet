"""
S1ChangeDataset: dataset loader for Sentinel-1 change detection chips.

Handles temporal reversal (chips saved newest-first, model needs oldest-first),
NaN filling, per-channel normalization, DoY computation, and augmentation.
"""

import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset


class S1ChangeDataset(Dataset):
    """
    Sentinel-1 temporal chip dataset for change detection.

    Each sample returns:
        chip:  (T, C, H, W) float32 — normalized, NaN filled, oldest→newest
        mask:  (H, W) int64 — values {0, 1, 255}
        doy:   (T,) int64 — day-of-year per timestep (oldest→newest)
        valid: (H, W) bool — True where mask!=255 AND no NaN in any channel/timestep

    Args:
        manifest_df: DataFrame with columns chip_path, mask_path, acquisition_dates, split.
        stats: Dict with per-channel mean/std keys (vv_mean, vv_std, etc.).
        augment: Whether to apply data augmentation.
        sentinel1_repeat_days: Sentinel-1 repeat cycle (default 12 days) for DoY fallback.
    """

    # Channel ordering in the chip: [VV_dB, VH_dB, RVI]
    CHANNEL_STATS_KEYS = [
        ("vv", "vv_mean", "vv_std"),
        ("vh", "vh_mean", "vh_std"),
        ("rvi", "rvi_mean", "rvi_std"),
    ]

    def __init__(
        self,
        manifest_df,
        stats: dict,
        augment: bool = False,
        sentinel1_repeat_days: int = 12,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.augment = augment
        self.repeat_days = sentinel1_repeat_days

        # Per-channel normalization: (mean, std) for each of 3 channels
        self.channel_stats = []
        for _, mean_key, std_key in self.CHANNEL_STATS_KEYS:
            self.channel_stats.append((stats[mean_key], stats[std_key]))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        row = self.df.iloc[idx]

        # --- Load chip ---
        chip = np.load(row["chip_path"])  # (6, 3, 256, 256) — index 0=newest

        # Reverse temporal order: index 0=oldest, index 5=newest
        chip = chip[::-1].copy()

        # --- NaN handling ---
        # Build NaN mask across all timesteps and channels before filling
        nan_mask = np.isnan(chip).any(axis=(0, 1))  # (H, W) bool
        chip = np.nan_to_num(chip, nan=0.0)

        # --- Normalize per-channel ---
        for c, (mean, std) in enumerate(self.channel_stats):
            chip[:, c] = (chip[:, c] - mean) / std

        # --- Load mask ---
        mask = np.load(row["mask_path"])  # (256, 256) uint8
        mask = mask.astype(np.int64)

        # --- Valid pixel mask ---
        valid = (mask != 255) & (~nan_mask)

        # --- Compute DoY ---
        doy = self._compute_doy(row)

        # --- Augmentation ---
        if self.augment:
            chip, mask, valid = self._augment(chip, mask, valid)

        return (
            torch.from_numpy(chip).float(),
            torch.from_numpy(mask).long(),
            torch.tensor(doy, dtype=torch.long),
            torch.from_numpy(valid),
        )

    def _compute_doy(self, row) -> list[int]:
        """
        Compute day-of-year for each timestep.

        Uses acquisition_dates from manifest if available (JSON array of date strings).
        Falls back to 12-day offset approximation from the label date.
        """
        dates_str = row.get("acquisition_dates", None)

        if dates_str and isinstance(dates_str, str) and dates_str.startswith("["):
            dates = json.loads(dates_str)
            # Dates are in chip-native order (newest first).
            # After chip reversal (oldest first), sort ascending.
            dates_sorted = sorted(dates)
            return [datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates_sorted]

        # Fallback: approximate from label date using repeat cycle
        label_date_str = row.get("date", None)
        if label_date_str:
            target_doy = datetime.strptime(str(label_date_str), "%Y-%m-%d").timetuple().tm_yday
        else:
            target_doy = 180  # mid-year default

        # 6 timesteps: oldest = target - 5*repeat, ..., newest = target
        doy = [target_doy - i * self.repeat_days for i in range(5, -1, -1)]
        return doy

    def _augment(
        self,
        chip: np.ndarray,
        mask: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random augmentation: H-flip, V-flip, 90-degree rotation."""
        # Random horizontal flip
        if np.random.random() < 0.5:
            chip = chip[:, :, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
            valid = valid[:, ::-1].copy()

        # Random vertical flip
        if np.random.random() < 0.5:
            chip = chip[:, :, ::-1, :].copy()
            mask = mask[::-1, :].copy()
            valid = valid[::-1, :].copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            chip = np.rot90(chip, k=k, axes=(2, 3)).copy()
            mask = np.rot90(mask, k=k).copy()
            valid = np.rot90(valid, k=k).copy()

        return chip, mask, valid
