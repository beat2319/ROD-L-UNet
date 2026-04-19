#!/usr/bin/env python3
"""
Mosaic Script for Orbit124_Sentinel1 GeoTIFFs

Stitches the 4 GeoTIFF tiles per date folder (from Google Earth Engine exports)
into a single mosaic using rasterio. No Orfeo ToolBox required.

Usage:
    python mosaic_sentinel1.py
    python mosaic_sentinel1.py  # with YEAR_FILTER set below
"""

import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge


# --- CONFIGURATION ---
INPUT_DIR = "./attachments/Orbit124_Sentinel1"
OUTPUT_FILENAME_PATTERN = "{date}_mosaic.tif"
NUM_WORKERS = 4              # Use all 4 cores
GDAL_NUM_THREADS = 8          # 8 GDAL threads per worker = 32 total threads
YEAR_FILTER = "2016"            # Set to "2016" etc. to filter, or None for all
COMPRESSION = "DEFLATE"
TILE_SIZE = 256
NODATA = 0


def find_date_folders_with_tifs(input_dir):
    """
    Walk directory tree and find all date folders (deepest directories)
    containing .tif files.

    Returns:
        dict: Mapping of date_folder_path -> list of .tif file paths
    """
    date_folders = defaultdict(list)

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    start_depth = len(input_path.parts)
    dir_contents = defaultdict(list)

    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        depth = len(root_path.parts) - start_depth

        tif_files = []
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                tif_files.append(root_path / f)

        if tif_files:
            dir_contents[root_path] = tif_files

    if not dir_contents:
        print("ERROR: No .tif or .tiff files found in directory tree.")
        sys.exit(1)

    max_depth = max(len(p.parts) - start_depth for p in dir_contents.keys())

    for dir_path, tif_files in dir_contents.items():
        depth = len(dir_path.parts) - start_depth
        if depth == max_depth:
            date_folders[dir_path] = tif_files

    return date_folders


def process_date_folder(args):
    """
    Worker function to mosaic all .tif files in a single date folder.

    Args:
        args: Tuple of (date_folder, tif_files, output_path)

    Returns:
        dict: Result with keys 'status', 'folder', 'message'
    """
    date_folder, tif_files, output_path = args

    # Enable GDAL threading for this worker
    os.environ['GDAL_NUM_THREADS'] = str(GDAL_NUM_THREADS)

    # Skip if output already exists
    if output_path.exists():
        return {
            'status': 'skip',
            'folder': str(date_folder),
            'message': 'Output already exists'
        }

    # Warn if not exactly 4 files
    if len(tif_files) != 4:
        return {
            'status': 'skip',
            'folder': str(date_folder),
            'message': f'Expected 4 GeoTIFFs, but found {len(tif_files)}'
        }

    sources = []
    try:
        # Open all input files
        for tif in tif_files:
            sources.append(rasterio.open(tif))

        # Merge using first-pixel-wins compositing
        mosaic_array, mosaic_transform = merge(sources, nodata=NODATA)

        # Build output profile from first source
        src_profile = sources[0].profile
        out_profile = src_profile.copy()
        out_profile.update({
            'driver': 'GTiff',
            'height': mosaic_array.shape[1],
            'width': mosaic_array.shape[2],
            'count': mosaic_array.shape[0],
            'dtype': 'float32',
            'transform': mosaic_transform,
            'nodata': NODATA,
            'compress': COMPRESSION,
            'BIGTIFF': 'YES',
            'tiled': True,
            'blockxsize': TILE_SIZE,
            'blockysize': TILE_SIZE,
        })

        # Write mosaic
        with rasterio.open(output_path, 'w', **out_profile) as dst:
            dst.write(mosaic_array.astype(np.float32))

        # Verify output
        if not output_path.exists():
            return {
                'status': 'fail',
                'folder': str(date_folder),
                'message': f'Output not found after write: {output_path}'
            }

        return {
            'status': 'success',
            'folder': str(date_folder),
            'message': f'Created {output_path}'
        }

    except Exception as e:
        return {
            'status': 'fail',
            'folder': str(date_folder),
            'message': f'Error: {str(e)}'
        }
    finally:
        for src in sources:
            src.close()


def main():
    print("\n" + "=" * 60)
    print("Orbit124_Sentinel1 Batch Mosaic Script")
    print(f"Configuration: {NUM_WORKERS} workers x {GDAL_NUM_THREADS} GDAL threads = {NUM_WORKERS * GDAL_NUM_THREADS} threads")
    print(f"Compression: {COMPRESSION}")
    print(f"NoData: {NODATA}")
    print("=" * 60 + "\n")

    # Find all date folders with .tif files
    date_folders = find_date_folders_with_tifs(INPUT_DIR)

    # Filter by year if specified
    if YEAR_FILTER:
        date_folders = {
            folder: files for folder, files in date_folders.items()
            if YEAR_FILTER in folder.name
        }
        if not date_folders:
            print(f"ERROR: No date folders found for year '{YEAR_FILTER}'")
            sys.exit(1)
        print(f"Filtered to year '{YEAR_FILTER}': {len(date_folders)} folders\n")

    total_folders = len(date_folders)
    total_files = sum(len(files) for files in date_folders.values())

    print(f"Found {total_folders} date folders with {total_files} total GeoTIFFs\n")

    # Prepare work items
    work_items = []
    for date_folder, tif_files in sorted(date_folders.items()):
        date_name = date_folder.name.replace('(', '_').replace(')', '')
        output_path = date_folder.parent / OUTPUT_FILENAME_PATTERN.format(date=date_name)
        work_items.append((date_folder, tif_files, output_path))

    # Process in parallel
    print(f"Processing with {NUM_WORKERS} parallel workers...\n")

    success_count = 0
    skip_count = 0
    fail_count = 0
    skipped_folders = []
    failed_folders = []

    with Pool(processes=NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_date_folder, work_items), 1):
            status = result['status'].upper()
            folder_name = Path(result['folder']).name
            print(f"[{i}/{total_folders}] {status}: {folder_name}")

            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skip':
                skip_count += 1
                skipped_folders.append(result)
            elif result['status'] == 'fail':
                fail_count += 1
                failed_folders.append(result)
                print(f"    -> {result['message']}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total date folders: {total_folders}")
    print(f"Successfully mosaiced: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")
    print("=" * 60)

    if failed_folders:
        print("\nFailed folders:")
        for r in failed_folders:
            print(f"  - {Path(r['folder']).name}: {r['message']}")

    if skipped_folders:
        print("\nSkipped folders:")
        for r in skipped_folders[:10]:
            print(f"  - {Path(r['folder']).name}: {r['message']}")
        if len(skipped_folders) > 10:
            print(f"  ... and {len(skipped_folders) - 10} more")


if __name__ == "__main__":
    main()
