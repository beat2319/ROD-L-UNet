"""
S1ChangeDataset: dataset loader for Sentinel-1 change detection chips.

Handles temporal reversal (chips saved newest-first, model needs oldest-first),
NaN filling, DoY computation, nodata masking, and augmentation.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset


class S1ChangeDataset(Dataset):
    """
    Sentinel-1 temporal chip dataset for change detection.

    Each sample returns:
        chip:  (T, C, H, W) float32 — VV_dB and VH_dB only, NaN filled, oldest→newest
        mask:  (H, W) int64 — values {0, 1, 255} (SAR nodata merged into 255)
        doy:   (T,) int64 — day-of-year per timestep (oldest→newest)
        valid: (H, W) bool — diagnostic mask, True where data is valid

    Args:
        manifest_df: DataFrame with columns chip_path, mask_path, acquisition_dates, split.
        augment: Whether to apply data augmentation.
        sentinel1_repeat_days: Sentinel-1 repeat cycle (default 12 days) for DoY fallback.
        input_channels: 2 keeps VV/VH; 3 keeps VV/VH/RVI for legacy checkpoints.
    """

    def __init__(
        self,
        manifest_df,
        augment: bool = False,
        sentinel1_repeat_days: int = 12,
        input_channels: int = 2,
    ):
        if input_channels not in (2, 3):
            raise ValueError("input_channels must be 2 (VV/VH) or 3 (VV/VH/RVI).")

        self.df = manifest_df.reset_index(drop=True)
        self.augment = augment
        self.repeat_days = sentinel1_repeat_days
        self.input_channels = input_channels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        row = self.df.iloc[idx]

        # --- Load chip ---
        chip = np.load(row["chip_path"])  # (6, 3, 256, 256) — index 0=newest

        # Reverse temporal order: index 0=oldest, index 5=newest
        chip = chip[::-1].copy()

        # Keep VV/VH by default; legacy RVI checkpoints use all three channels.
        chip = chip[:, :self.input_channels]

        # --- NaN handling ---
        # Build NaN mask across all timesteps and channels before filling
        nan_mask = np.isnan(chip).any(axis=(0, 1))  # (H, W) bool
        chip = np.nan_to_num(chip, nan=0.0)

        # --- Load mask ---
        mask = np.load(row["mask_path"])  # (256, 256) uint8
        mask = mask.astype(np.int64)

        # --- Merge SAR nodata into mask as ignore_index=255 ---
        # This lets losses and metrics handle all nodata via their existing ignore_index logic
        valid = (mask != 255) & (~nan_mask)
        mask[~valid] = 255

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
            target_date = datetime.strptime(str(label_date_str), "%Y-%m-%d")
        else:
            # Use a fixed mid-year anchor so the fallback remains stable.
            target_date = datetime(2001, 6, 29)

        # 6 timesteps: oldest = target - 5*repeat, ..., newest = target
        fallback_dates = [
            target_date - timedelta(days=i * self.repeat_days)
            for i in range(5, -1, -1)
        ]
        return [d.timetuple().tm_yday for d in fallback_dates]

    def _augment(
        self,
        chip: np.ndarray,
        mask: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply full augmentation to every training patch.
        """
        # --- 1. Random crop + resize ---
        chip, mask, valid = self._random_crop(chip, mask, valid, scale_range=(0.8, 1.0))

        # --- 2. Random horizontal flip ---
        if np.random.random() < 0.5:
            chip = chip[:, :, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
            valid = valid[:, ::-1].copy()

        # --- 3. Random vertical flip ---
        if np.random.random() < 0.5:
            chip = chip[:, :, ::-1, :].copy()
            mask = mask[::-1, :].copy()
            valid = valid[::-1, :].copy()

        # --- 4. Random 90-degree rotation ---
        k = np.random.randint(0, 4)
        if k > 0:
            chip = np.rot90(chip, k=k, axes=(2, 3)).copy()
            mask = np.rot90(mask, k=k).copy()
            valid = np.rot90(valid, k=k).copy()

        # --- 5. Color jitter (SAR) ---
        chip = self._color_jitter(chip)

        # --- 6. Gaussian blur ---
        if np.random.random() < 0.3:
            chip = self._gaussian_blur(chip)

        return chip, mask, valid

    def _random_crop(
        self,
        chip: np.ndarray,
        mask: np.ndarray,
        valid: np.ndarray,
        scale_range: tuple[float, float] = (0.8, 1.0),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RandomResizedCrop: crop a random region and resize back to original size."""
        T, C, H, W = chip.shape
        target_h, target_w = H, W

        # Sample crop size
        scale = np.random.uniform(*scale_range)
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        # Random top-left corner
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)

        # Crop
        chip_cropped = chip[:, :, top:top + crop_h, left:left + crop_w]
        mask_cropped = mask[top:top + crop_h, left:left + crop_w]
        valid_cropped = valid[top:top + crop_h, left:left + crop_w]

        # Resize back to target size using bilinear for chip, nearest for mask/valid
        import torch.nn.functional as F

        chip_tensor = torch.from_numpy(chip_cropped).float()
        chip_resized = F.interpolate(
            chip_tensor.view(T * C, 1, crop_h, crop_w),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False,
        ).view(T, C, target_h, target_w).numpy()

        mask_tensor = torch.from_numpy(mask_cropped).float().unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(
            mask_tensor, size=(target_h, target_w),
            mode='nearest',
        ).squeeze().long().numpy()

        valid_tensor = torch.from_numpy(valid_cropped).float().unsqueeze(0).unsqueeze(0)
        valid_resized = F.interpolate(
            valid_tensor, size=(target_h, target_w),
            mode='nearest',
        ).squeeze().bool().numpy()

        return chip_resized, mask_resized, valid_resized

    def _color_jitter(self, chip: np.ndarray) -> np.ndarray:
        """SAR color jitter: brightness offset ±2 dB, contrast scaling 0.8–1.2."""
        # Brightness: random offset applied identically to all timesteps/channels
        brightness = np.random.uniform(-2.0, 2.0)
        chip = chip + brightness

        # Contrast: random scaling per channel, same across timesteps
        for c in range(chip.shape[1]):
            contrast = np.random.uniform(0.8, 1.2)
            chip[:, c] = chip[:, c] * contrast

        return chip

    def _gaussian_blur(self, chip: np.ndarray) -> np.ndarray:
        """Gaussian blur with random sigma from [0.5, 1.5]."""
        from scipy.ndimage import gaussian_filter

        sigma = np.random.uniform(0.5, 1.5)
        result = np.empty_like(chip)
        for t in range(chip.shape[0]):
            for c in range(chip.shape[1]):
                result[t, c] = gaussian_filter(chip[t, c], sigma=sigma)
        return result
