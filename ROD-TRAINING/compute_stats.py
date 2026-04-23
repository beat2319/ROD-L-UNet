"""
Compute per-channel normalization statistics from training chips.

Uses Welford's online algorithm to avoid loading all chips into memory.

Usage:
    python compute_stats.py
    python compute_stats.py --manifest manifest.csv --output sar_normalization_stats.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(manifest_csv: str, output_json: str):
    df = pd.read_csv(manifest_csv)
    train_df = df[df["split"] == "train"]

    if train_df.empty:
        print("ERROR: No training samples in manifest.")
        return

    chip_paths = train_df["chip_path"].tolist()
    print(f"Computing stats from {len(chip_paths)} training chips...")

    # Welford's online algorithm for each channel
    # Channels: 0=VV_dB, 1=VH_dB, 2=RVI
    n_channels = 3
    count = np.zeros(n_channels)
    mean = np.zeros(n_channels)
    M2 = np.zeros(n_channels)

    for i, path in enumerate(chip_paths):
        chip = np.load(path)  # (6, 3, 256, 256)

        for c in range(n_channels):
            channel_data = chip[:, c].ravel()  # flatten temporal + spatial
            valid = channel_data[~np.isnan(channel_data)]

            for val in valid:
                count[c] += 1
                delta = val - mean[c]
                mean[c] += delta / count[c]
                delta2 = val - mean[c]
                M2[c] += delta * delta2

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(chip_paths)} chips")

    # Final variance / std
    std = np.zeros(n_channels)
    for c in range(n_channels):
        if count[c] > 1:
            std[c] = np.sqrt(M2[c] / count[c])
        else:
            std[c] = 1.0

    stats = {
        "vv_mean": float(mean[0]),
        "vv_std": float(std[0]),
        "vh_mean": float(mean[1]),
        "vh_std": float(std[1]),
        "rvi_mean": float(mean[2]),
        "rvi_std": float(std[2]),
        "num_chips_used": len(chip_paths),
        "num_pixels_per_channel": {f"ch{c}": int(count[c]) for c in range(n_channels)},
    }

    with open(output_json, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to {output_json}")
    print(f"  VV_dB:  mean={mean[0]:.4f}, std={std[0]:.4f}")
    print(f"  VH_dB:  mean={mean[1]:.4f}, std={std[1]:.4f}")
    print(f"  RVI:    mean={mean[2]:.4f}, std={std[2]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sar_normalization_stats.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    compute_stats(args.manifest, args.output)
