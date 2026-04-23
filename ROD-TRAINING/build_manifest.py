"""
Build training manifest: pair chips with masks, assign temporal train/val splits.

Split strategy: temporal (not geographic). Train on earlier years, validate on
later years. All islands appear in both splits.

Usage:
    python build_manifest.py
    python build_manifest.py --train-years 2016 2017 2018 2019 --val-years 2020 2021
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_sample_filename(filename: str) -> dict | None:
    """
    Parse island, date, patch_id from chip/mask filename.

    Patterns:
        positive_chip_{island}_{YYYY-MM-DD}_{patch_id}.npy
        positive_mask_{island}_{YYYY-MM-DD}_{patch_id}.npy
        negative_chip_{island}_{YYYY-MM-DD}_{patch_id}.npy
        negative_mask_{island}_{YYYY-MM-DD}_{patch_id}.npy

    Returns:
        Dict with label, island, date, patch_id or None if no match.
    """
    match = re.match(
        r"(positive|negative)_(chip|mask)_(\w+)_(\d{4}-\d{2}-\d{2})_(\d+)\.npy",
        filename,
    )
    if match:
        return {
            "label": match.group(1),
            "type": match.group(2),
            "island": match.group(3),
            "date": match.group(4),
            "patch_id": match.group(5),
            "year": int(match.group(4).split("-")[0]),
        }
    return None


def compute_acquisition_dates(label_date: str, repeat_days: int = 12) -> str:
    """
    Approximate the 6 acquisition dates for a chip based on label date.

    Chips are built as [closest, closest-1*repeat, ..., closest-5*repeat]
    (newest first in the saved file).

    Returns:
        JSON string of 6 dates in chip-native order (newest first).
    """
    from datetime import datetime, timedelta

    target = datetime.strptime(label_date, "%Y-%m-%d")
    dates = []
    for i in range(6):
        d = target - timedelta(days=i * repeat_days)
        dates.append(d.strftime("%Y-%m-%d"))
    return json.dumps(dates)


def build_manifest(
    data_root: str | Path,
    output_csv: str | Path,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    sentinel1_repeat_days: int = 12,
):
    data_root = Path(data_root)

    if train_years is None:
        train_years = [2016, 2017, 2018, 2019]
    if val_years is None:
        val_years = [2020, 2021]

    # Collect all chips
    chips = {}
    for chip_dir in [data_root / "positive_chips", data_root / "negative_chips"]:
        if not chip_dir.exists():
            continue
        for npy_file in sorted(chip_dir.rglob("*.npy")):
            info = parse_sample_filename(npy_file.name)
            if info and info["type"] == "chip":
                key = (info["label"], info["island"], info["date"], info["patch_id"])
                chips[key] = {
                    "chip_path": str(npy_file),
                    **info,
                }

    # Match masks to chips
    rows = []
    matched = 0
    unmatched_masks = 0

    for mask_dir in [data_root / "positive_masks", data_root / "negative_masks"]:
        if not mask_dir.exists():
            continue
        for npy_file in sorted(mask_dir.rglob("*.npy")):
            info = parse_sample_filename(npy_file.name)
            if not info or info["type"] != "mask":
                continue

            key = (info["label"], info["island"], info["date"], info["patch_id"])
            if key in chips:
                chip_info = chips[key]
                acq_dates = compute_acquisition_dates(
                    info["date"], sentinel1_repeat_days
                )

                # Assign split by year
                year = info["year"]
                if year in train_years:
                    split = "train"
                elif year in val_years:
                    split = "val"
                else:
                    continue  # skip years not in either split

                rows.append({
                    "chip_path": chip_info["chip_path"],
                    "mask_path": str(npy_file),
                    "island": info["island"],
                    "year": year,
                    "date": info["date"],
                    "patch_id": info["patch_id"],
                    "label": info["label"],
                    "split": split,
                    "acquisition_dates": acq_dates,
                })
                matched += 1
            else:
                unmatched_masks += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    # Summary
    print(f"Matched chip-mask pairs: {matched}")
    print(f"Unmatched masks (no chip): {unmatched_masks}")
    print(f"Train samples: {(df['split'] == 'train').sum()}")
    print(f"Val samples: {(df['split'] == 'val').sum()}")

    for split_name in ["train", "val"]:
        split_df = df[df["split"] == split_name]
        print(f"\n{split_name.upper()} distribution:")
        print(f"  By label: {dict(split_df['label'].value_counts())}")
        print(f"  By island: {dict(split_df['island'].value_counts())}")
        print(f"  By year: {dict(split_df['year'].value_counts())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training manifest")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../ROD-PROCESSING/data",
        help="Root of data directory with chip/mask subdirs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="manifest.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=[2016, 2017, 2018, 2019],
    )
    parser.add_argument(
        "--val-years",
        nargs="+",
        type=int,
        default=[2020, 2021],
    )
    args = parser.parse_args()

    build_manifest(args.data_root, args.output, args.train_years, args.val_years)
