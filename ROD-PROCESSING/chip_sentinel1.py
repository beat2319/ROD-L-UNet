"""
Sentinel-1 Chipping Script for ROD Conv-LSTM Training

Extracts temporal stacks of Sentinel-1 SAR chips (VV, VH, RVI) from patch
geometries produced by positive_patch.py and negative_patch.py.

Pipeline position:
  positive.py  -> positive_patch.py  -> [chip_sentinel1.py] -> training chips
  negative.py  -> negative_patch.py  -> [chip_sentinel1.py] -> training chips

Usage:
  python chip_sentinel1.py 2016
  python chip_sentinel1.py 2020 --label negative --workers 4
"""

import argparse
import json
import re
from datetime import datetime, date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from osgeo import gdal
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class ChipConfig:
    """Configuration for Sentinel-1 chipping."""

    # Spatial parameters (must match patch generation scripts)
    PATCH_SIZE_M = 2560
    TARGET_CRS = "EPSG:32604"
    PIXEL_RES = 10.0  # 10m -> 256x256 chips

    # Temporal parameters
    TEMPORAL_DEPTH = 6  # T0 through T-5

    # Sentinel-1 data paths
    SENTINEL1_BASE = Path("../ROD-COLLECTION/attachments/Orbit124_Sentinel1")

    # Input patch directories
    POSITIVE_PATCH_DIR = Path("./attachments/patch_bounds")
    NEGATIVE_PATCH_DIR = Path("./attachments/negative_patches")

    # Output directories
    POSITIVE_CHIP_DIR = Path("./data/positive_chips")
    NEGATIVE_CHIP_DIR = Path("./data/negative_chips")

    # dB conversion and clipping
    DB_CLIP_MIN = -30.0
    DB_CLIP_MAX = 10.0
    NODATA_VALUE = 0  # from mosaic_sentinel1.py (dstNodata=0)

    # Date matching
    DATE_THRESHOLD_DAYS = 14  # max offset for closest-date match

    # Processing
    NUM_WORKERS = 16


# ============================================================================
# DATE DISCOVERY AND MATCHING
# ============================================================================

def discover_sentinel1_dates(year: int, config: ChipConfig) -> dict:
    """
    Discover all available Sentinel-1 mosaic dates for a given year.

    Mosaics are expected directly in the year directory as:
      Sentinel1_124({year})/Sentinel1_124_{YYYY-MM-DD}_mosaic.tif

    Returns:
        dict mapping datetime.date -> Path, sorted by date ascending.
    """
    year_dir = config.SENTINEL1_BASE / f"Sentinel1_124({year})"
    if not year_dir.exists():
        return {}

    dates = {}
    for item in sorted(year_dir.iterdir()):
        match = re.search(r'(\d{4}-\d{2}-\d{2})_mosaic\.tif$', item.name)
        if match:
            d = datetime.strptime(match.group(1), '%Y-%m-%d').date()
            dates[d] = item

    return dict(sorted(dates.items()))


def discover_all_relevant_dates(year: int, config: ChipConfig) -> dict:
    """
    Discover Sentinel-1 dates for the target year and prior year.

    Loading the prior year ensures early-January patches can find
    December revisits for the temporal stack.
    """
    dates = {}
    for y in [year - 1, year]:
        year_dates = discover_sentinel1_dates(y, config)
        dates.update(year_dates)
    return dict(sorted(dates.items()))


def parse_geojson_filename(filename: str, label: str):
    """
    Parse island and date from a patch GeoJSON filename.

    Positive: patches_{island}_{YYYY-MM-DD}.geojson
    Negative: negative_{island}_{YYYY-MM-DD}.geojson

    Returns:
        Tuple of (island, date_str) or (None, None) if no match.
    """
    if label == "positive":
        match = re.match(r'patches_(\w+)_(\d{4}-\d{2}-\d{2})\.geojson', filename)
    else:
        match = re.match(r'negative_(\w+)_(\d{4}-\d{2}-\d{2})\.geojson', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


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


def get_temporal_stack(closest_idx: int, sorted_dates: list,
                       dates_to_paths: dict, depth: int = 6):
    """
    Build the temporal stack of Sentinel-1 mosaic paths.

    Args:
        closest_idx: Index of the closest date in sorted_dates.
        sorted_dates: Sorted list of available dates.
        dates_to_paths: Mapping of date -> file path.
        depth: Number of timesteps (including current).

    Returns:
        List of (date, path) tuples of length `depth`, or None if
        insufficient prior dates are available.
    """
    if closest_idx < depth - 1:
        return None  # Not enough prior revisits

    stack = []
    for i in range(depth):
        idx = closest_idx - i
        d = sorted_dates[idx]
        stack.append((d, dates_to_paths[d]))
    return stack


# ============================================================================
# GEOMETRY AND RASTER I/O
# ============================================================================

def extract_patch_bounds(feature: dict) -> tuple:
    """
    Extract bounding box from a GeoJSON feature.

    Positive patches have min_x/min_y/max_x/max_y properties.
    Negative patches only have geometry; extract bounds from coordinates.

    Returns:
        (min_x, min_y, max_x, max_y) tuple.
    """
    props = feature.get('properties', {})

    if 'min_x' in props:
        return (props['min_x'], props['min_y'],
                props['max_x'], props['max_y'])
    else:
        coords = feature['geometry']['coordinates'][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (min(xs), min(ys), max(xs), max(ys))


def clip_and_convert(mosaic_path: Path, bounds: tuple, config: ChipConfig) -> np.ndarray:
    """
    Open a Sentinel-1 mosaic, reproject if needed, clip to bounds,
    compute RVI, convert VV/VH to dB, and return as a 3-channel array.

    Args:
        mosaic_path: Path to the Sentinel-1 mosaic TIF (2 bands: VV, VH).
        bounds: (min_x, min_y, max_x, max_y) in EPSG:32604.
        config: Configuration object.

    Returns:
        float32 array of shape (3, H, W) with channels [VV_dB, VH_dB, RVI].
        Nodata pixels are NaN.
    """
    min_x, min_y, max_x, max_y = bounds

    with rasterio.open(mosaic_path) as src:
        src_crs = src.crs

        if src_crs is not None and src_crs.to_epsg() == 32604:
            # Already in target CRS - direct windowed read
            window = rasterio.windows.from_bounds(
                min_x, min_y, max_x, max_y,
                transform=src.transform
            )
            # Read both bands: band 1 = VV, band 2 = VH
            vv = src.read(1, window=window, boundless=True,
                          fill_value=config.NODATA_VALUE).astype(np.float32)
            vh = src.read(2, window=window, boundless=True,
                          fill_value=config.NODATA_VALUE).astype(np.float32)
        else:
            # Need reprojection - use gdal.Warp with in-memory output
            warp_options = gdal.WarpOptions(
                format='MEM',
                dstSRS='EPSG:32604',
                outputBounds=[min_x, min_y, max_x, max_y],
                xRes=config.PIXEL_RES,
                yRes=config.PIXEL_RES,
                dstNodata=config.NODATA_VALUE,
                resampleAlg='bilinear',
            )
            ds = gdal.Warp('', str(mosaic_path), options=warp_options)
            vv = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            vh = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
            ds = None  # Close in-memory dataset

    # --- Nodata mask (applied to linear values) ---
    nodata_mask = (vv == config.NODATA_VALUE) | (vh == config.NODATA_VALUE)
    nodata_mask |= (vv <= 0) | (vh <= 0)

    # --- RVI from linear values (before dB conversion) ---
    # RVI = (4 * VH) / (VV + VH)
    denominator = vv + vh
    rvi = np.full_like(vv, np.nan, dtype=np.float32)
    valid_linear = ~nodata_mask & (denominator > 0)
    rvi[valid_linear] = (4.0 * vh[valid_linear]) / denominator[valid_linear]

    # --- Convert VV and VH to dB ---
    vv_db = np.full_like(vv, np.nan, dtype=np.float32)
    vh_db = np.full_like(vh, np.nan, dtype=np.float32)

    vv_db[valid_linear] = 10.0 * np.log10(vv[valid_linear])
    vh_db[valid_linear] = 10.0 * np.log10(vh[valid_linear])

    # Clip dB to reasonable range
    vv_db = np.clip(vv_db, config.DB_CLIP_MIN, config.DB_CLIP_MAX)
    vh_db = np.clip(vh_db, config.DB_CLIP_MIN, config.DB_CLIP_MAX)

    # Stack channels: (3, H, W)
    chip = np.stack([vv_db, vh_db, rvi], axis=0)

    return chip


# ============================================================================
# PER-FILE PROCESSING
# ============================================================================

def process_geojson_file(args: tuple) -> dict:
    """
    Process all patches from a single GeoJSON file.

    Args:
        args: Tuple of (geojson_path, label, all_dates, config, year)

    Returns:
        Dict with processing statistics.
    """
    geojson_path, label, all_dates, config, year = args

    # Parse filename
    island, date_str = parse_geojson_filename(geojson_path.name, label)
    if island is None:
        return {'status': 'skip', 'reason': 'bad filename', 'file': geojson_path.name}

    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    sorted_dates = list(all_dates.keys())

    # Find closest Sentinel-1 date
    closest_date, closest_idx = find_closest_date(target_date, sorted_dates)

    if closest_date is None:
        return {'status': 'skip', 'reason': 'no Sentinel-1 dates', 'file': geojson_path.name}

    # Check date offset threshold
    offset_days = abs((closest_date - target_date).days)
    if offset_days > config.DATE_THRESHOLD_DAYS:
        return {
            'status': 'skip',
            'reason': f'date offset {offset_days}d > {config.DATE_THRESHOLD_DAYS}d threshold',
            'file': geojson_path.name
        }

    # Build temporal stack
    temporal_stack = get_temporal_stack(
        closest_idx, sorted_dates, all_dates, config.TEMPORAL_DEPTH
    )

    if temporal_stack is None:
        return {
            'status': 'skip',
            'reason': f'insufficient prior revisits (need {config.TEMPORAL_DEPTH}, have {closest_idx + 1})',
            'file': geojson_path.name
        }

    # Load patch geometries
    with open(geojson_path) as f:
        geojson = json.load(f)

    features = geojson['features']
    if not features:
        return {'status': 'skip', 'reason': 'no features', 'file': geojson_path.name}

    # Set output directory
    output_base = config.POSITIVE_CHIP_DIR if label == "positive" else config.NEGATIVE_CHIP_DIR
    output_dir = output_base / island / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "positive_chip" if label == "positive" else "negative_chip"
    success_count = 0
    fail_count = 0

    for j, feature in enumerate(features):
        bounds = extract_patch_bounds(feature)
        patch_id = feature.get('properties', {}).get('patch_id', j)

        # Stack all 6 timesteps
        timestep_arrays = []
        failed = False

        for t, (s1_date, mosaic_path) in enumerate(temporal_stack):
            try:
                arr = clip_and_convert(mosaic_path, bounds, config)
                timestep_arrays.append(arr)
            except Exception as e:
                failed = True
                break

        if failed:
            fail_count += 1
            continue

        # Stack into (6, 3, 256, 256)
        chip = np.stack(timestep_arrays, axis=0)

        # Validate shape
        expected_size = int(config.PATCH_SIZE_M / config.PIXEL_RES)  # 256
        if chip.shape != (config.TEMPORAL_DEPTH, 3, expected_size, expected_size):
            fail_count += 1
            continue

        # Save
        output_filename = f"{prefix}_{island}_{date_str}_{patch_id}.npy"
        output_path = output_dir / output_filename
        np.save(output_path, chip)
        success_count += 1

    return {
        'status': 'done',
        'file': geojson_path.name,
        'island': island,
        'date': date_str,
        'success': success_count,
        'failed': fail_count
    }


# ============================================================================
# MAIN
# ============================================================================

def get_geojson_files_for_year(patch_dir: Path, label: str, year: int) -> list:
    """Get all GeoJSON files from a patch directory matching the given year."""
    if label == "positive":
        pattern = "patches_*.geojson"
    else:
        pattern = "negative_*.geojson"

    files = []
    for f in sorted(patch_dir.glob(pattern)):
        _, date_str = parse_geojson_filename(f.name, label)
        if date_str:
            file_year = int(date_str.split('-')[0])
            if file_year == year:
                files.append(f)
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Sentinel-1 Chipping for ROD Conv-LSTM Training'
    )
    parser.add_argument('year', type=int, help='Year to process (e.g., 2016)')
    parser.add_argument(
        '--workers', type=int, default=6,
        help='Number of parallel workers (default: 6)'
    )
    parser.add_argument(
        '--label', choices=['positive', 'negative', 'both'],
        default='both', help='Which patches to process (default: both)'
    )
    args = parser.parse_args()

    year = args.year
    config = ChipConfig()
    config.NUM_WORKERS = args.workers

    print("=" * 80)
    print("SENTINEL-1 CHIPPING FOR ROD CONV-LSTM TRAINING")
    print("=" * 80)
    print(f"Year: {year}")
    print(f"Label: {args.label}")
    print(f"Workers: {config.NUM_WORKERS}")
    print(f"Chip shape: ({config.TEMPORAL_DEPTH}, 3, "
          f"{int(config.PATCH_SIZE_M / config.PIXEL_RES)}, "
          f"{int(config.PATCH_SIZE_M / config.PIXEL_RES)})")
    print(f"Channels: [VV_dB, VH_dB, RVI]")
    print()

    # Discover Sentinel-1 dates (current + prior year)
    all_dates = discover_all_relevant_dates(year, config)
    if not any(y == year for y in [d.year for d in all_dates.keys()]):
        print(f"ERROR: No Sentinel-1 mosaics found for {year}")
        return

    print(f"Found {len(all_dates)} Sentinel-1 mosaics "
          f"({sum(1 for d in all_dates if d.year == year)} in {year}, "
          f"{sum(1 for d in all_dates if d.year == year - 1)} in {year - 1})")
    print()

    # Collect work items
    labels = ['positive', 'negative'] if args.label == 'both' else [args.label]
    work_items = []

    for label in labels:
        patch_dir = config.POSITIVE_PATCH_DIR if label == "positive" else config.NEGATIVE_PATCH_DIR
        files = get_geojson_files_for_year(patch_dir, label, year)

        if not files:
            print(f"No {label} patch files found for {year}")
            continue

        print(f"Found {len(files)} {label} patch files for {year}")
        for f in files:
            work_items.append((f, label, all_dates, config, year))

    if not work_items:
        print("No work to do.")
        return

    print(f"\nTotal GeoJSON files to process: {len(work_items)}")
    print()

    # Process with multiprocessing
    total_success = 0
    total_failed = 0
    total_skipped = 0

    with Pool(processes=config.NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_geojson_file, work_items),
            total=len(work_items),
            desc="Processing GeoJSON files"
        ))

    # Summarize
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for r in results:
        if r['status'] == 'skip':
            print(f"  SKIP {r['file']}: {r['reason']}")
            total_skipped += 1
        elif r['status'] == 'done':
            total_success += r['success']
            total_failed += r['failed']

    print(f"\nChips saved: {total_success}")
    print(f"Patches failed: {total_failed}")
    print(f"Files skipped: {total_skipped}")
    print(f"\nOutput directories:")
    for label in labels:
        chip_dir = config.POSITIVE_CHIP_DIR if label == "positive" else config.NEGATIVE_CHIP_DIR
        print(f"  {chip_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
