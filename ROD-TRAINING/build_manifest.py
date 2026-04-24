"""
Build training manifest: pair chips with masks, assign temporal train/val splits.

Split strategy: temporal (not geographic). Train on earlier years, validate on
later years. All islands appear in both splits.

Acquisition dates are computed using the same Sentinel-1 date discovery logic
as chip_sentinel1.py: discover actual S1 mosaic dates, find closest to the
label date, then walk backwards through the sorted date list to build the
6-timestep temporal stack.

Usage:
    python build_manifest.py
    python build_manifest.py --train-years 2016 2017 2018 2019 --val-years 2020 2021
"""

import argparse
import json
import re
from datetime import datetime, date
from pathlib import Path

import pandas as pd


# ============================================================================
# SENTINEL-1 DATE DISCOVERY (replicates chip_sentinel1.py logic)
# ============================================================================

def discover_sentinel1_dates(year: int, s1_base: Path) -> dict:
    """
    Discover all available Sentinel-1 dates for a given year.

    Scans year directory for date-encoded subdirectories:
        Sentinel1_124({year})/Sentinel1_124({YYYY-MM-DD})/

    Returns:
        dict mapping datetime.date -> Path, sorted by date ascending.
    """
    year_dir = s1_base / f"Sentinel1_124({year})"
    if not year_dir.exists():
        return {}

    dates = {}
    for item in sorted(year_dir.iterdir()):
        match = re.search(r'(\d{4}-\d{2}-\d{2})', item.name)
        if match:
            d = datetime.strptime(match.group(1), '%Y-%m-%d').date()
            dates[d] = item

    return dict(sorted(dates.items()))


def discover_all_relevant_dates(year: int, s1_base: Path) -> dict:
    """
    Discover Sentinel-1 dates for the target year and prior year.

    Loading the prior year ensures early-January patches can find
    December revisits for the temporal stack.
    """
    dates = {}
    for y in [year - 1, year]:
        year_dates = discover_sentinel1_dates(y, s1_base)
        dates.update(year_dates)
    return dict(sorted(dates.items()))


def find_closest_date(target_date: date, sorted_dates: list) -> tuple:
    """
    Find the closest available Sentinel-1 date to a target date.

    Returns:
        Tuple of (closest_date, index_in_sorted_list) or (None, -1).
    """
    if not sorted_dates:
        return None, -1

    closest = min(sorted_dates, key=lambda d: abs((d - target_date).days))
    idx = sorted_dates.index(closest)
    return closest, idx


def get_temporal_stack_dates(closest_idx: int, sorted_dates: list,
                             depth: int = 6) -> list | None:
    """
    Build the temporal stack of Sentinel-1 acquisition dates.

    Walks backwards from the closest date through the sorted date list,
    exactly as chip_sentinel1.py does when building chips.

    Returns:
        List of date strings in chip-native order (newest first),
        or None if insufficient prior dates are available.
    """
    if closest_idx < depth - 1:
        return None  # Not enough prior revisits

    stack = []
    for i in range(depth):
        idx = closest_idx - i
        d = sorted_dates[idx]
        stack.append(d.strftime("%Y-%m-%d"))
    return stack


def compute_acquisition_dates(label_date_str: str, year: int, s1_base: Path,
                              all_dates_cache: dict,
                              date_threshold_days: int = 14,
                              depth: int = 6) -> list | None:
    """
    Compute the actual acquisition dates for a chip by replicating the
    same date-matching logic used in chip_sentinel1.py.

    Args:
        label_date_str: The label date from the chip filename (YYYY-MM-DD).
        year: The label year (used for date discovery).
        s1_base: Path to Orbit124_Sentinel1 directory.
        all_dates_cache: Cache dict mapping year -> {date: path}.
        date_threshold_days: Max offset between label and closest S1 date.
        depth: Number of timesteps in the temporal stack.

    Returns:
        List of date strings in chip-native order (newest first),
        or None if dates cannot be resolved.
    """
    if year not in all_dates_cache:
        all_dates_cache[year] = discover_all_relevant_dates(year, s1_base)

    all_dates = all_dates_cache[year]
    if not all_dates:
        return None

    target_date = datetime.strptime(label_date_str, '%Y-%m-%d').date()
    sorted_dates = list(all_dates.keys())

    closest_date, closest_idx = find_closest_date(target_date, sorted_dates)
    if closest_date is None:
        return None

    # Check date offset threshold (same as chip_sentinel1.py DATE_THRESHOLD_DAYS)
    offset_days = abs((closest_date - target_date).days)
    if offset_days > date_threshold_days:
        return None

    return get_temporal_stack_dates(closest_idx, sorted_dates, depth)


# ============================================================================
# FILENAME PARSING
# ============================================================================

def parse_sample_filename(filename: str) -> dict | None:
    """
    Parse island, date, patch_id from chip/mask filename.

    Patterns:
        positive_chip_{island}_{YYYY-MM-DD}_{patch_id}.npy
        positive_mask_{island}_{YYYY-MM-DD}_{patch_id}.npy
        negative_chip_{island}_{YYYY-MM-DD}_{patch_id}.npy
        negative_mask_{island}_{YYYY-MM-DD}_{patch_id}.npy

    Returns:
        Dict with label, type, island, date, patch_id or None if no match.
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


# ============================================================================
# MANIFEST BUILDING
# ============================================================================

def build_manifest(
    data_root: str | Path,
    output_csv: str | Path,
    s1_base: str | Path,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
):
    data_root = Path(data_root)
    s1_base = Path(s1_base)

    if train_years is None:
        train_years = [2016, 2017, 2018, 2019]
    if val_years is None:
        val_years = [2020, 2021]

    # Cache for S1 date discovery per year
    all_dates_cache = {}

    # Collect all chips — use folder path for island/year labeling
    chips = {}
    for chip_dir in [data_root / "positive_chips", data_root / "negative_chips"]:
        if not chip_dir.exists():
            continue
        for npy_file in sorted(chip_dir.rglob("*.npy")):
            info = parse_sample_filename(npy_file.name)
            if info and info["type"] == "chip":
                # Validate against folder structure
                path_island = npy_file.parent.parent.name
                path_year = int(npy_file.parent.name)
                assert info["island"] == path_island, \
                    f"Island mismatch: filename={info['island']} vs path={path_island} in {npy_file}"
                assert info["year"] == path_year, \
                    f"Year mismatch: filename={info['year']} vs path={path_year} in {npy_file}"

                key = (info["label"], info["island"], info["date"], info["patch_id"])
                chips[key] = {
                    "chip_path": str(npy_file),
                    **info,
                }

    # Match masks to chips and compute real acquisition dates
    rows = []
    matched = 0
    unmatched_masks = 0
    skipped_no_dates = 0

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

                # Compute real acquisition dates using same logic as chipping
                acq_dates = compute_acquisition_dates(
                    label_date_str=info["date"],
                    year=info["year"],
                    s1_base=s1_base,
                    all_dates_cache=all_dates_cache,
                )

                if acq_dates is None:
                    skipped_no_dates += 1
                    continue

                acq_dates_json = json.dumps(acq_dates)

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
                    "acquisition_dates": acq_dates_json,
                })
                matched += 1
            else:
                unmatched_masks += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    # Summary
    print(f"Matched chip-mask pairs: {matched}")
    print(f"Unmatched masks (no chip): {unmatched_masks}")
    print(f"Skipped (no valid S1 temporal stack): {skipped_no_dates}")
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
        "--s1-base",
        type=str,
        default="../ROD-COLLECTION/attachments/Orbit124_Sentinel1",
        help="Path to Sentinel-1 mosaic base directory",
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

    build_manifest(
        args.data_root,
        args.output,
        args.s1_base,
        args.train_years,
        args.val_years,
    )
