#!/usr/bin/env python3
"""
Mosaic Script for Orbit124_Sentinel1 GeoTIFFs

This script finds GeoTIFF files in the deepest directories (date folders)
within the Orbit124_Sentinel1 folder and creates a mosaiced file for each
date folder using GDAL Python API with multiprocessing.

Optimized for 12-core CPU: 6 workers × 2 GDAL threads per worker = 12 cores total.

Usage:
    python mosaic_sentinel1.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
from osgeo import gdal


# --- CONFIGURATION ---
INPUT_DIR = "./attachments/Orbit124_Sentinel1"
OUTPUT_FILENAME_PATTERN = "{date}_mosaic.tif"  # {date} will be replaced with folder name
NODATA_VALUE = 0
NUM_WORKERS = 6
GDAL_NUM_THREADS = 2
COMPRESS = "LZW"  # Options: "LZW" (fast), "DEFLATE" (better but slower)
YEAR_FILTER = "2015"  # Set to a year string like "2024" to process only that year, or None for all


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

    # Count starting depth
    start_depth = len(input_path.parts)

    # Track directory depth and contents
    dir_contents = defaultdict(list)  # dir_path -> list of tif files

    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        depth = len(root_path.parts) - start_depth

        # Find all .tif/.tiff files in this directory
        tif_files = []
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                tif_files.append(root_path / f)

        if tif_files:
            dir_contents[root_path] = tif_files

    # Find maximum depth that has files
    if not dir_contents:
        print("ERROR: No .tif or .tiff files found in directory tree.")
        sys.exit(1)

    max_depth = max(len(p.parts) - start_depth for p in dir_contents.keys())

    # Collect only directories at max depth
    for dir_path, tif_files in dir_contents.items():
        depth = len(dir_path.parts) - start_depth
        if depth == max_depth:
            date_folders[dir_path] = tif_files

    return date_folders


def process_date_folder(args):
    """
    Worker function to process a single date folder.
    Runs in a separate process via multiprocessing.Pool.

    Args:
        args: Tuple of (date_folder, tif_files, output_path)

    Returns:
        dict: Result with keys 'status', 'folder', 'message'
    """
    date_folder, tif_files, output_path = args

    # Check if output already exists
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

    # Set GDAL config for threading
    gdal.SetConfigOption('GDAL_NUM_THREADS', str(GDAL_NUM_THREADS))

    # Build warp options
    # dstNodata: treat 0 as nodata (prevents black lines at tile edges - critical for ROD-ML)
    # creationOptions: compress output to save space and improve I/O
    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstNodata=NODATA_VALUE,
        creationOptions=[f'COMPRESS={COMPRESS}'],
        multithread=True
    )

    # Execute warp using GDAL Python API
    try:
        gdal.Warp(str(output_path), [str(f) for f in tif_files], options=warp_options)
        return {
            'status': 'success',
            'folder': str(date_folder),
            'message': f'Created {output_path}'
        }
    except Exception as e:
        return {
            'status': 'fail',
            'folder': str(date_folder),
            'message': f'gdal.Warp failed: {str(e)}'
        }


def main():
    print("\n" + "="*60)
    print("Orbit124_Sentinel1 Batch Mosaic Script")
    print(f"Configuration: {NUM_WORKERS} workers × {GDAL_NUM_THREADS} threads = {NUM_WORKERS * GDAL_NUM_THREADS} cores")
    print(f"Compression: {COMPRESS}")
    print("="*60 + "\n")

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

    # Prepare work items for multiprocessing
    work_items = []
    for date_folder, tif_files in sorted(date_folders.items()):
        date_name = date_folder.name.replace('(', '').replace(')', '')
        output_path = date_folder.parent / OUTPUT_FILENAME_PATTERN.format(date=date_name)
        work_items.append((date_folder, tif_files, output_path))

    # Process in parallel using multiprocessing.Pool
    print(f"Processing with {NUM_WORKERS} parallel workers...\n")

    success_count = 0
    skip_count = 0
    fail_count = 0
    skipped_folders = []
    failed_folders = []

    with Pool(processes=NUM_WORKERS) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_date_folder, work_items), 1):
            results.append(result)

            # Live progress update (unordered)
            status = result['status'].upper()
            folder_name = Path(result['folder']).name
            print(f"[{i}/{total_folders}] {status}: {folder_name}")

            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skip':
                skip_count += 1
                skipped_folders.append(result['folder'])
            elif result['status'] == 'fail':
                fail_count += 1
                failed_folders.append(result['folder'])

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total date folders: {total_folders}")
    print(f"Successfully mosaiced: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")
    print("="*60)

    if failed_folders:
        print("\nFailed folders:")
        for folder in failed_folders:
            print(f"  - {Path(folder).name}")

    if skipped_folders:
        print("\nSkipped folders:")
        for folder in skipped_folders[:10]:  # Show first 10
            print(f"  - {Path(folder).name}")
        if len(skipped_folders) > 10:
            print(f"  ... and {len(skipped_folders) - 10} more")


if __name__ == "__main__":
    main()
