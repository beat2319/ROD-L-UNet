"""
Pre-training data validation: verify chip/mask integrity before training.

Usage:
    python validate_data.py
    python validate_data.py --manifest manifest.csv --num-checks 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_data(manifest_csv: str, num_checks: int = 10):
    df = pd.read_csv(manifest_csv)
    print(f"Manifest: {len(df)} samples")
    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Val: {(df['split'] == 'val').sum()}")
    print()

    passed = 0
    failed = 0

    # Check 1: Chip-mask file existence
    print("Check 1: File existence...")
    missing_chips = 0
    missing_masks = 0
    for _, row in df.iterrows():
        if not Path(row["chip_path"]).exists():
            missing_chips += 1
        if not Path(row["mask_path"]).exists():
            missing_masks += 1
    if missing_chips == 0 and missing_masks == 0:
        print("  PASS: All files exist")
        passed += 1
    else:
        print(f"  FAIL: {missing_chips} missing chips, {missing_masks} missing masks")
        failed += 1

    # Check 2: Shape consistency (random sample)
    print("Check 2: Shape consistency...")
    sample = df.sample(n=min(num_checks, len(df)), random_state=42)
    shape_ok = True
    for _, row in sample.iterrows():
        chip = np.load(row["chip_path"])
        mask = np.load(row["mask_path"])
        if chip.shape != (6, 3, 256, 256):
            print(f"  FAIL: Chip shape {chip.shape} at {row['chip_path']}")
            shape_ok = False
            break
        if mask.shape != (256, 256):
            print(f"  FAIL: Mask shape {mask.shape} at {row['mask_path']}")
            shape_ok = False
            break
    if shape_ok:
        print(f"  PASS: All {len(sample)} samples have correct shapes")
        passed += 1
    else:
        failed += 1

    # Check 3: Class distribution
    print("Check 3: Class distribution...")
    class_counts = {0: 0, 1: 0, 255: 0}
    for _, row in df.iterrows():
        mask = np.load(row["mask_path"])
        vals, counts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, counts):
            v = int(v)
            if v in class_counts:
                class_counts[v] += c
    total = sum(class_counts.values())
    print(f"  Class 0 (healthy):  {class_counts[0]:>10d} ({100*class_counts[0]/total:.1f}%)")
    print(f"  Class 1 (mortality): {class_counts[1]:>10d} ({100*class_counts[1]/total:.1f}%)")
    print(f"  Class 255 (nodata): {class_counts[255]:>10d} ({100*class_counts[255]/total:.1f}%)")
    mortality_frac = class_counts[1] / (class_counts[0] + class_counts[1]) if (class_counts[0] + class_counts[1]) > 0 else 0
    print(f"  Mortality fraction (of valid pixels): {mortality_frac:.3f}")
    passed += 1

    # Check 4: NaN fraction per timestep
    print("Check 4: NaN fractions...")
    nan_ok = True
    for _, row in sample.iterrows():
        chip = np.load(row["chip_path"])
        for t in range(6):
            nan_frac = np.isnan(chip[t]).sum() / chip[t].size
            if nan_frac > 0.5:
                print(f"  WARNING: Timestep {t} has {nan_frac:.1%} NaN in {row['chip_path']}")
                nan_ok = False
    if nan_ok:
        print(f"  PASS: No timestep has >50% NaN")
        passed += 1
    else:
        print("  WARNING: Some timesteps have high NaN fractions")
        passed += 1  # Warning, not failure

    # Check 5: Temporal ordering after reversal
    print("Check 5: Temporal ordering...")
    ordering_ok = True
    for _, row in sample.iterrows():
        dates = json.loads(row["acquisition_dates"])
        # Dates in manifest are newest-first (chip-native order)
        # After reversal in dataset, they should be oldest-first
        # Verify they're in descending order as stored
        for i in range(len(dates) - 1):
            if dates[i] < dates[i + 1]:
                print(f"  FAIL: Dates not descending at index {i}: {dates}")
                ordering_ok = False
                break
        if not ordering_ok:
            break
    if ordering_ok:
        print(f"  PASS: Acquisition dates in expected chip-native order (newest first)")
        passed += 1
    else:
        failed += 1

    # Check 6: Temporal split correctness
    print("Check 6: Temporal split...")
    train_years = set(df[df["split"] == "train"]["year"].unique())
    val_years = set(df[df["split"] == "val"]["year"].unique())
    overlap = train_years & val_years
    if overlap:
        print(f"  FAIL: Year overlap between train and val: {overlap}")
        failed += 1
    else:
        train_islands = set(df[df["split"] == "train"]["island"].unique())
        val_islands = set(df[df["split"] == "val"]["island"].unique())
        all_in_both = train_islands == val_islands
        print(f"  Train years: {sorted(train_years)}, Val years: {sorted(val_years)}")
        print(f"  Train islands: {sorted(train_islands)}, Val islands: {sorted(val_islands)}")
        if all_in_both:
            print("  PASS: No year overlap, all islands in both splits")
        else:
            print("  WARNING: Not all islands in both splits")
        passed += 1

    # Check 7: No class-1 in negative masks
    print("Check 7: No mortality in negative masks...")
    neg_df = df[df["label"] == "negative"]
    has_mortality = False
    for _, row in neg_df.sample(n=min(num_checks, len(neg_df)), random_state=42).iterrows():
        mask = np.load(row["mask_path"])
        if 1 in mask:
            print(f"  FAIL: Class 1 found in negative mask {row['mask_path']}")
            has_mortality = True
            break
    if not has_mortality:
        print("  PASS: No class 1 in sampled negative masks")
        passed += 1
    else:
        failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        print("Fix issues before training.")
    else:
        print("All checks passed. Ready for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--num-checks",
        type=int,
        default=10,
        help="Number of random samples to check",
    )
    args = parser.parse_args()

    validate_data(args.manifest, args.num_checks)
