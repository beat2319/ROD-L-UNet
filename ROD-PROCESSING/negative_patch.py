# conda activate solaris
"""
Negative Patch Bounds Generator for ROD Training Data

Creates 2560m x 2560m patch bounding boxes with 1280m overlap (50%)
from TCC negative rasters in attachments/tcc_negative/.

These patches are bounding boxes that will be used to clip Sentinel-1 raster data.
Output follows naming convention: negative_{island}_{date}.geojson
"""
import geopandas as gpd
import rasterio
from shapely.geometry import box
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class PatchConfig:
    """Configuration parameters for patch generation."""

    # Spatial parameters
    PATCH_SIZE_M = 2560      # 2560m x 2560m
    OVERLAP_M = 1280         # 1280m overlap (50%)
    STRIDE_M = PATCH_SIZE_M - OVERLAP_M  # 1280m stride

    # Nodata filtering
    NODATA_THRESHOLD = 0.15  # Maximum 15% nodata allowed per patch

    # I/O paths
    INPUT_DIR = Path("./attachments/tcc_negative")
    OUTPUT_DIR = Path("./data/negative_patches")
    POSITIVE_DIR = Path("./attachments/positive")

    # CRS
    TARGET_CRS = "EPSG:32604"  # UTM Zone 4N


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_filename(filename: str):
    """
    Parse island and date from negative filename.

    Args:
        filename: Input filename like 'tcc_negative_hawaii_2016-01-12.tif'

    Returns:
        Tuple of (island, date) where date is YYYY-MM-DD format
    """
    # Match pattern: tcc_negative_{island}_{YYYY-MM-DD}.tif
    match = re.match(r'tcc_negative_(\w+)_(\d{4}-\d{2}-\d{2})\.tif', filename)
    if match:
        island = match.group(1)
        date = match.group(2)
        return island, date
    return None, None


def get_positive_file(island: str, date: str, config: PatchConfig) -> Path | None:
    """
    Get path to corresponding positive file for a given island and date.

    Args:
        island: Island name (e.g., 'hawaii')
        date: Date string (YYYY-MM-DD format)
        config: Configuration object

    Returns:
        Path to positive file, or None if not found
    """
    positive_path = config.POSITIVE_DIR / f"positive_{island}_{date}.geojson"
    return positive_path if positive_path.exists() else None


def patch_intersects_positive(patch_box, positive_gdf) -> bool:
    """
    Check if a patch box intersects with any positive polygon.

    Args:
        patch_box: Shapely box geometry for the patch
        positive_gdf: GeoDataFrame with positive (ROD) polygons

    Returns:
        True if patch intersects with any positive polygon, False otherwise
    """
    patch_gdf = gpd.GeoDataFrame({'geometry': [patch_box]}, crs=positive_gdf.crs)
    # Use any intersection with the positive polygons
    return not patch_gdf.sjoin(positive_gdf, how="inner", predicate="intersects").empty


def generate_patch_grid(bounds, config: PatchConfig):
    """
    Generate a grid of patch boxes with specified overlap.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        config: Configuration object

    Yields:
        (patch_id, patch_box) for each grid cell
    """
    minx, miny, maxx, maxy = bounds

    # Calculate grid dimensions
    width = maxx - minx
    height = maxy - miny

    num_cols = int(np.ceil(width / config.STRIDE_M))
    num_rows = int(np.ceil(height / config.STRIDE_M))

    patch_id = 0

    for col in range(num_cols):
        x = minx + col * config.STRIDE_M
        for row in range(num_rows):
            y = miny + row * config.STRIDE_M

            # Create patch bounding box
            patch_box = box(
                x, y,
                x + config.PATCH_SIZE_M,
                y + config.PATCH_SIZE_M
            )

            yield patch_id, patch_box
            patch_id += 1


def build_patches(input_file, config: PatchConfig, positive_gdf=None):
    """
    Generate negative patch bounds from a raster file with nodata and positive filtering.

    Args:
        input_file: Path to input GeoTIFF
        config: Configuration object
        positive_gdf: Optional GeoDataFrame with positive (ROD) polygons to exclude

    Returns:
        Tuple of (GeoDataFrame with patch bounds, total_patches_generated, patches_filtered_nodata, patches_filtered_positive)
    """
    # Open raster and keep it open for windowed reads
    with rasterio.open(input_file) as src:
        bounds = src.bounds
        src_crs = src.crs
        nodata_value = src.nodata

        # Convert bounds to tuple
        bounds_tuple = (bounds.left, bounds.bottom, bounds.right, bounds.top)

        # Generate patch grid and filter by nodata and positive intersection
        polygons = []
        total_generated = 0
        patches_filtered_nodata = 0
        patches_filtered_positive = 0

        for patch_id, patch_box in generate_patch_grid(bounds_tuple, config):
            total_generated += 1
            # Get patch bounds
            patch_bounds = patch_box.bounds
            minx, miny, maxx, maxy = patch_bounds

            # Create window for this patch
            window = rasterio.windows.from_bounds(
                minx, miny, maxx, maxy,
                transform=src.transform
            )

            # Read the patch data (just first band for nodata check)
            patch_data = src.read(1, window=window, boundless=True, fill_value=nodata_value)

            # Calculate nodata percentage
            total_pixels = patch_data.size
            nodata_pixels = np.sum(patch_data == nodata_value)
            nodata_percentage = nodata_pixels / total_pixels

            # Only keep patch if nodata is below threshold
            if nodata_percentage < config.NODATA_THRESHOLD:
                # Check if patch intersects with any positive polygons
                if positive_gdf is not None and patch_intersects_positive(patch_box, positive_gdf):
                    patches_filtered_positive += 1
                else:
                    polygons.append(patch_box)
            else:
                patches_filtered_nodata += 1

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'geometry': polygons, 'label': "negative"},
        crs=src_crs
    )

    # Reproject to target CRS if needed
    if gdf.crs != config.TARGET_CRS:
        gdf = gdf.to_crs(config.TARGET_CRS)

    return gdf, total_generated, patches_filtered_nodata, patches_filtered_positive


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    print("=" * 80)
    print("NEGATIVE PATCH BOUNDS GENERATOR")
    print("=" * 80)
    print(f"Patch size: {PatchConfig.PATCH_SIZE_M}m x {PatchConfig.PATCH_SIZE_M}m")
    print(f"Overlap: {PatchConfig.OVERLAP_M}m (50%)")
    print(f"Stride: {PatchConfig.STRIDE_M}m")
    print(f"Nodata threshold: {PatchConfig.NODATA_THRESHOLD * 100}%")
    print(f"Target CRS: {PatchConfig.TARGET_CRS}")
    print()

    # Create output directory
    PatchConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {PatchConfig.OUTPUT_DIR}")
    print(f"Positive data directory: {PatchConfig.POSITIVE_DIR}")
    print()

    # Get all negative GeoTIFF files
    tif_files = sorted(PatchConfig.INPUT_DIR.glob("tcc_negative_*.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No negative GeoTIFF files found in {PatchConfig.INPUT_DIR}")

    print(f"Found {len(tif_files)} negative GeoTIFF files")
    print()

    # Track overall statistics
    total_files = 0
    total_patches_generated = 0
    total_patches_filtered_nodata = 0
    total_patches_filtered_positive = 0

    for tif_file in tqdm(tif_files, desc="Processing files"):
        # Parse filename
        island, date = parse_filename(tif_file.name)

        if island is None:
            print(f"  Skipping {tif_file.name}: could not parse filename")
            continue

        total_files += 1

        # Load corresponding positive file for intersection filtering
        positive_gdf = None
        positive_file = get_positive_file(island, date, PatchConfig)
        if positive_file:
            positive_gdf = gpd.read_file(positive_file)
            if positive_gdf.crs is None:
                positive_gdf = positive_gdf.set_crs(epsg=4326, allow_override=True)
            # Get raster CRS to reproject positive data
            with rasterio.open(tif_file) as src:
                raster_crs = src.crs
            # Reproject positive data to match raster CRS
            if positive_gdf.crs != raster_crs:
                positive_gdf = positive_gdf.to_crs(raster_crs)

        # Generate patches with nodata and positive filtering
        gdf, total_generated, filtered_nodata, filtered_positive = build_patches(
            tif_file, PatchConfig, positive_gdf
        )

        num_patches = len(gdf)
        total_patches_generated += num_patches
        total_patches_filtered_nodata += filtered_nodata
        total_patches_filtered_positive += filtered_positive

        # Save with naming structure: negative_{island}_{date}.geojson
        output_filename = f"negative_{island}_{date}.geojson"
        output_path = PatchConfig.OUTPUT_DIR / output_filename

        gdf.to_file(output_path, driver="GeoJSON")

        tqdm.write(f"  Saved: {output_filename} ({num_patches} patches, "
                   f"{filtered_nodata} nodata filtered, {filtered_positive} positive filtered)")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files processed: {total_files}")
    print(f"Total patches generated: {total_patches_generated}")
    print(f"Total patches filtered (nodata > {PatchConfig.NODATA_THRESHOLD * 100}%): {total_patches_filtered_nodata}")
    print(f"Total patches filtered (positive intersection): {total_patches_filtered_positive}")
    print(f"Total patches filtered overall: {total_patches_filtered_nodata + total_patches_filtered_positive}")
    print(f"Output directory: {PatchConfig.OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
