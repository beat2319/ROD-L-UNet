"""
Negative Patch Mask Generation for ROD Conv-LSTM Training

Creates binary masks for negative (no-mortality) training patches:
  - Canopy (class 0) from tcc_negative rasters
  - Other/nodata (class 255) for everything else

All source data is aligned by {island}_{date} — no date matching needed.

Pipeline position:
  negative.py -> negative_patch.py -> [change_detection_masks_negative.py] -> training masks

Usage:
  python change_detection_masks_negative.py 2018
  python change_detection_masks_negative.py 2018 --workers 4
"""

import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from osgeo import gdal
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class NegativeMaskConfig:
    """Configuration for negative patch mask generation."""

    # Spatial parameters (must match patch generation scripts)
    PATCH_SIZE_M = 2560
    TARGET_CRS = "EPSG:32604"
    PIXEL_RES = 10.0  # 10m -> 256x256 chips
    MASK_SIZE = 256

    # Mask classes
    CLASS_HEALTHY = 0    # valid tcc_negative pixel
    NODATA_VALUE = 255   # other/nodata

    # Input directories
    NEGATIVE_PATCH_DIR = Path("./attachments/negative_patches")
    TCC_NEGATIVE_DIR = Path("./attachments/tcc_negative")

    # Output directory
    OUTPUT_DIR = Path("./data/negative_masks")

    # Processing
    NUM_WORKERS = 6


# ============================================================================
# FILE LOOKUP
# ============================================================================

def parse_negative_filename(filename: str):
    """
    Parse island and date from a negative patch GeoJSON filename.

    Pattern: negative_{island}_{YYYY-MM-DD}.geojson

    Returns:
        Tuple of (island, date_str) or (None, None) if no match.
    """
    match = re.match(r'negative_(\w+)_(\d{4}-\d{2}-\d{2})\.geojson', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_negative_files_for_year(year: int, config: NegativeMaskConfig) -> list:
    """Get all negative patch GeoJSON files matching the given year."""
    files = []
    for f in sorted(config.NEGATIVE_PATCH_DIR.glob("negative_*.geojson")):
        _, date_str = parse_negative_filename(f.name)
        if date_str:
            file_year = int(date_str.split('-')[0])
            if file_year == year:
                files.append(f)
    return files


def resolve_tcc_path(island: str, date_str: str, config: NegativeMaskConfig) -> Path:
    """Build direct path to tcc_negative file. No date matching needed."""
    return config.TCC_NEGATIVE_DIR / f"tcc_negative_{island}_{date_str}.tif"


# ============================================================================
# RASTERIZATION
# ============================================================================

def rasterize_canopy(tcc_path: Path, bounds: tuple, config: NegativeMaskConfig) -> np.ndarray:
    """
    Read tcc_negative raster, resample from ~30m to 10m, and create
    binary canopy mask within patch bounds.

    Returns:
        uint8 array of shape (256, 256). 0=canopy, 255=nodata.
    """
    min_x, min_y, max_x, max_y = bounds
    size = config.MASK_SIZE

    warp_options = gdal.WarpOptions(
        format='MEM',
        dstSRS=config.TARGET_CRS,
        outputBounds=[min_x, min_y, max_x, max_y],
        xRes=config.PIXEL_RES,
        yRes=config.PIXEL_RES,
        dstNodata=config.NODATA_VALUE,
        resampleAlg='nearest',
        width=size,
        height=size,
    )
    ds = gdal.Warp('', str(tcc_path), options=warp_options)
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # Any valid pixel (not 255) = canopy -> class 0
    canopy = np.full((size, size), config.NODATA_VALUE, dtype=np.uint8)
    valid_mask = data != 255
    canopy[valid_mask] = config.CLASS_HEALTHY

    return canopy


# ============================================================================
# GEOMETRY HELPER
# ============================================================================

def extract_patch_bounds(feature: dict) -> tuple:
    """
    Extract bounding box from a GeoJSON feature.

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


# ============================================================================
# PER-FILE PROCESSING
# ============================================================================

def process_negative_file(args: tuple) -> dict:
    """
    Process all patches from a single negative patch GeoJSON file.

    Returns:
        Dict with processing statistics.
    """
    geojson_path, config = args

    # Parse filename
    island, date_str = parse_negative_filename(geojson_path.name)
    if island is None:
        return {'status': 'skip', 'reason': 'bad filename', 'file': geojson_path.name}

    year = int(date_str.split('-')[0])

    # Resolve tcc_negative path
    tcc_path = resolve_tcc_path(island, date_str, config)

    if not tcc_path.exists():
        return {'status': 'skip', 'reason': f'no tcc_negative: {tcc_path.name}', 'file': geojson_path.name}

    # Load patch features
    with open(geojson_path) as f:
        geojson = json.load(f)

    features = geojson['features']
    if not features:
        return {'status': 'skip', 'reason': 'no features', 'file': geojson_path.name}

    # Set output directory
    output_dir = config.OUTPUT_DIR / island / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    size = config.MASK_SIZE
    success_count = 0
    fail_count = 0

    for j, feature in enumerate(features):
        bounds = extract_patch_bounds(feature)
        patch_id = feature.get('properties', {}).get('patch_id', j)

        try:
            # Binary mask: canopy (0) vs other (255)
            mask = rasterize_canopy(tcc_path, bounds, config)

            if mask.shape != (size, size):
                fail_count += 1
                continue

            output_filename = f"negative_mask_{island}_{date_str}_{patch_id}.npy"
            np.save(output_dir / output_filename, mask)
            success_count += 1

        except Exception:
            fail_count += 1

    return {
        'status': 'done',
        'file': geojson_path.name,
        'island': island,
        'date': date_str,
        'success': success_count,
        'failed': fail_count,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Negative Patch Mask Generation for ROD Training'
    )
    parser.add_argument('year', type=int, help='Year to process (e.g., 2018)')
    parser.add_argument(
        '--workers', type=int, default=6,
        help='Number of parallel workers (default: 6)'
    )
    args = parser.parse_args()

    config = NegativeMaskConfig()
    config.NUM_WORKERS = args.workers
    year = args.year

    print("=" * 80)
    print("NEGATIVE PATCH MASK GENERATION FOR ROD TRAINING")
    print("=" * 80)
    print(f"Year: {year}")
    print(f"Workers: {config.NUM_WORKERS}")
    print(f"Mask shape: ({config.MASK_SIZE}, {config.MASK_SIZE})")
    print(f"Classes: {config.CLASS_HEALTHY}=canopy, {config.NODATA_VALUE}=other")
    print()

    # Collect work items
    neg_files = get_negative_files_for_year(year, config)
    if not neg_files:
        print(f"No negative patch files found for {year}")
        return

    print(f"Found {len(neg_files)} negative patch files for {year}")

    work_items = [(f, config) for f in neg_files]

    # Process with multiprocessing
    total_success = 0
    total_failed = 0
    total_skipped = 0

    with Pool(processes=config.NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_negative_file, work_items),
            total=len(work_items),
            desc="Processing negative patches"
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

    print(f"\nMasks saved: {total_success}")
    print(f"Patches failed: {total_failed}")
    print(f"Files skipped: {total_skipped}")
    print(f"\nOutput directory: {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
