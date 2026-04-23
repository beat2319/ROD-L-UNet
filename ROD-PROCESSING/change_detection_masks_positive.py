"""
Change Detection Mask Generation for ROD Conv-LSTM Training

Creates per-patch semantic segmentation masks by compositing:
  - Healthy canopy (class 0) from tcc_negative rasters
  - Mortality/change (class 1) from dissolved positive detection polygons,
    ONLY where raw annual TCC confirms tree canopy exists (TCC > 0)
  - Other/nodata (class 255) for everything else, including parts of
    mortality polygons that fall outside trusted canopy

All source data is aligned by {island}_{date} — no date matching needed.

Pipeline position:
  positive.py -> positive_patch.py -> [change_detection_masks.py] -> training masks

Usage:
  python change_detection_masks.py 2018
  python change_detection_masks.py 2018 --workers 4
"""

import argparse
import json
import re
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
from osgeo import gdal
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import shape
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class MaskConfig:
    """Configuration for change detection mask generation."""

    # Spatial parameters (must match patch generation scripts)
    PATCH_SIZE_M = 2560
    TARGET_CRS = "EPSG:32604"
    PIXEL_RES = 10.0  # 10m -> 256x256 chips
    MASK_SIZE = 256

    # Mask classes
    CLASS_HEALTHY = 0    # valid tcc_negative pixel
    CLASS_MORTALITY = 1  # positive detection polygon
    NODATA_VALUE = 255   # other/nodata

    # Input directories
    PATCH_BOUNDS_DIR = Path("./attachments/patch_bounds")
    POSITIVE_DIR = Path("./attachments/positive")
    TCC_NEGATIVE_DIR = Path("./attachments/tcc_negative")
    TCC_DIR = Path("./attachments/tcc")

    # Output directory
    OUTPUT_DIR = Path("./data/positive_masks")

    # Processing
    NUM_WORKERS = 6


# ============================================================================
# FILE LOOKUP
# ============================================================================

def parse_patch_bounds_filename(filename: str):
    """
    Parse island and date from a patch_bounds GeoJSON filename.

    Pattern: patches_{island}_{YYYY-MM-DD}.geojson

    Returns:
        Tuple of (island, date_str) or (None, None) if no match.
    """
    match = re.match(r'patches_(\w+)_(\d{4}-\d{2}-\d{2})\.geojson', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_patch_bounds_files_for_year(year: int, config: MaskConfig) -> list:
    """Get all patch_bounds GeoJSON files matching the given year."""
    files = []
    for f in sorted(config.PATCH_BOUNDS_DIR.glob("patches_*.geojson")):
        _, date_str = parse_patch_bounds_filename(f.name)
        if date_str:
            file_year = int(date_str.split('-')[0])
            if file_year == year:
                files.append(f)
    return files


def resolve_paths(island: str, date_str: str, config: MaskConfig) -> tuple:
    """
    Build direct paths to positive and tcc_negative files for a given
    island + date. No date matching needed — all sources are aligned.

    Returns:
        (positive_path, tcc_path) — paths may not exist on disk.
    """
    positive_path = config.POSITIVE_DIR / f"positive_{island}_{date_str}.geojson"
    tcc_path = config.TCC_NEGATIVE_DIR / f"tcc_negative_{island}_{date_str}.tif"
    return positive_path, tcc_path


def resolve_tcc_raw_path(year: int, config: MaskConfig) -> Path:
    """
    Resolve path to raw annual TCC raster for the given year.

    Pattern: tcc_124_{year}.tif

    Returns:
        Path to raw TCC file. Caller must check .exists().
    """
    return config.TCC_DIR / f"tcc_124_{year}.tif"


# ============================================================================
# DISSOLVE POSITIVE DETECTIONS
# ============================================================================

def dissolve_positive(positive_path: Path):
    """
    Load positive detection GeoJSON and dissolve all polygon features
    into a single geometry.

    Returns:
        Dissolved shapely geometry (Polygon or MultiPolygon), or None
        if the file doesn't exist or has no features.
    """
    if not positive_path.exists():
        return None

    gdf = gpd.read_file(positive_path)
    if gdf.empty:
        return None

    return gdf.geometry.unary_union


# ============================================================================
# RASTERIZATION
# ============================================================================

def rasterize_canopy(tcc_path: Path, bounds: tuple, config: MaskConfig) -> np.ndarray:
    """
    Read tcc_negative raster, resample from ~30m to 10m, and create
    binary canopy mask within patch bounds.

    Args:
        tcc_path: Path to tcc_negative TIF (~30m resolution).
        bounds: (min_x, min_y, max_x, max_y) in EPSG:32604.
        config: MaskConfig.

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


def rasterize_mortality(geometry, transform, shape: tuple) -> np.ndarray:
    """
    Rasterize a dissolved mortality geometry into a binary mask.

    Args:
        geometry: Shapely geometry (Polygon or MultiPolygon).
        transform: Affine transform for the 256x256 grid.
        shape: (height, width) = (256, 256).

    Returns:
        uint8 array of shape (256, 256). 1=mortality, 0=elsewhere.
    """
    mask = rio_rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return mask


def rasterize_trusted_canopy(
    tcc_raw_path: Path, bounds: tuple, config: MaskConfig
) -> np.ndarray:
    """
    Read raw annual TCC raster, resample from ~30m to 10m via nearest-neighbor,
    and create a boolean mask indicating trusted canopy locations.

    Trusted canopy = TCC value > 0 AND TCC value != 255 (NoData).
    Raw TCC values: 0 = no canopy, 1-254 = canopy cover %, 255 = NoData.

    Args:
        tcc_raw_path: Path to raw TCC TIF (e.g., tcc_124_2018.tif).
        bounds: (min_x, min_y, max_x, max_y) in EPSG:32604.
        config: MaskConfig.

    Returns:
        Boolean array of shape (256, 256). True = trusted canopy.
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
    ds = gdal.Warp('', str(tcc_raw_path), options=warp_options)
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    trusted = (data > 0) & (data != 255)

    return trusted


# ============================================================================
# GEOMETRY HELPER (from chip_sentinel1.py)
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


# ============================================================================
# PER-FILE PROCESSING
# ============================================================================

def process_patch_bounds_file(args: tuple) -> dict:
    """
    Process all patches from a single patch_bounds GeoJSON file.

    Args:
        args: Tuple of (geojson_path, config)

    Returns:
        Dict with processing statistics.
    """
    geojson_path, config = args

    # Parse filename
    island, date_str = parse_patch_bounds_filename(geojson_path.name)
    if island is None:
        return {'status': 'skip', 'reason': 'bad filename', 'file': geojson_path.name}

    year = int(date_str.split('-')[0])

    # Resolve companion files
    positive_path, tcc_path = resolve_paths(island, date_str, config)

    # Resolve raw TCC for trusted canopy filtering
    tcc_raw_path = resolve_tcc_raw_path(year, config)

    # Check tcc_negative exists (required for canopy layer)
    if not tcc_path.exists():
        return {'status': 'skip', 'reason': f'no tcc_negative: {tcc_path.name}', 'file': geojson_path.name}

    # Dissolve positive detections (optional — may not exist)
    mortality_geom = dissolve_positive(positive_path)

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
        min_x, min_y, max_x, max_y = bounds
        patch_id = feature.get('properties', {}).get('patch_id', j)

        try:
            # Step 1: Rasterize canopy (valid tcc_negative -> class 0)
            mask = rasterize_canopy(tcc_path, bounds, config)

            # Step 1b: Compute trusted canopy from raw annual TCC
            trusted_canopy = None
            if tcc_raw_path.exists():
                trusted_canopy = rasterize_trusted_canopy(
                    tcc_raw_path, bounds, config
                )

            # Step 2: Rasterize mortality with trusted canopy filter
            if mortality_geom is not None:
                transform = rasterio.transform.from_bounds(
                    min_x, min_y, max_x, max_y, size, size
                )
                mort_mask = rasterize_mortality(
                    mortality_geom, transform, (size, size)
                )

                if trusted_canopy is not None:
                    # Mortality over trusted canopy -> class 1
                    mask[
                        (mort_mask == 1) & trusted_canopy
                    ] = config.CLASS_MORTALITY
                    # Mortality over non-canopy -> class 255 (ignore)
                    mask[
                        (mort_mask == 1) & ~trusted_canopy
                    ] = config.NODATA_VALUE
                else:
                    # No raw TCC available, fall back to old behavior
                    mask[mort_mask == 1] = config.CLASS_MORTALITY

            # Validate shape
            if mask.shape != (size, size):
                fail_count += 1
                continue

            # Save
            output_filename = f"positive_mask_{island}_{date_str}_{patch_id}.npy"
            np.save(output_dir / output_filename, mask)
            success_count += 1

        except Exception as e:
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
        description='Change Detection Mask Generation for ROD Training'
    )
    parser.add_argument('year', type=int, help='Year to process (e.g., 2018)')
    parser.add_argument(
        '--workers', type=int, default=6,
        help='Number of parallel workers (default: 6)'
    )
    args = parser.parse_args()

    config = MaskConfig()
    config.NUM_WORKERS = args.workers
    year = args.year

    print("=" * 80)
    print("CHANGE DETECTION MASK GENERATION FOR ROD TRAINING")
    print("=" * 80)
    print(f"Year: {year}")
    print(f"Workers: {config.NUM_WORKERS}")
    print(f"Mask shape: ({config.MASK_SIZE}, {config.MASK_SIZE})")
    print(f"Classes: {config.CLASS_HEALTHY}=canopy, {config.CLASS_MORTALITY}=mortality (TCC-filtered), {config.NODATA_VALUE}=other")
    print()

    # Collect work items
    patch_files = get_patch_bounds_files_for_year(year, config)
    if not patch_files:
        print(f"No patch_bounds files found for {year}")
        return

    print(f"Found {len(patch_files)} patch_bounds files for {year}")

    work_items = [(f, config) for f in patch_files]

    # Process with multiprocessing
    total_success = 0
    total_failed = 0
    total_skipped = 0

    with Pool(processes=config.NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_patch_bounds_file, work_items),
            total=len(work_items),
            desc="Processing patch_bounds files"
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
