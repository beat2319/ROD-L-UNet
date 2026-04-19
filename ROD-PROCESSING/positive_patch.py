"""
Positive Patch Bounds Generator for ROD Training Data

Creates 2560m x 2560m patch bounding boxes with 1280m overlap (50%)
that intersect with ROD positive polygons from attachments/positive/.

These patches are bounding boxes that will be used to clip Sentinel-1 raster data.
Output follows naming convention from attachments/ohia_mortality: one GeoJSON per source file.
"""

import geopandas as gpd
import numpy as np
import re
from pathlib import Path
from shapely.geometry import box
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
    PIXEL_RES = 10.0         # 10m pixel resolution (matches Sentinel-1)

    # I/O paths
    INPUT_DIR = Path("./attachments/positive")
    OUTPUT_DIR = Path("./data/patch_bounds")

    # CRS
    TARGET_CRS = "EPSG:32604"  # UTM Zone 4N


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_filename(filename: str):
    """
    Parse island and date from positive filename.

    Args:
        filename: Input filename like 'positive_hawaii_2016-01-12.geojson'

    Returns:
        Tuple of (island, date) where date is YYYY-MM-DD format
    """
    # Match pattern: positive_{island}_{YYYY-MM-DD}.geojson
    match = re.match(r'positive_(\w+)_(\d{4}-\d{2}-\d{2})\.geojson', filename)
    if match:
        island = match.group(1)
        date = match.group(2)
        return island, date
    return None, None


def generate_patch_grid(bounds, config: PatchConfig):
    """
    Generate a grid of patch boxes with specified overlap.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        config: Configuration object

    Yields:
        (patch_id, patch_box, center_x, center_y) for each grid cell
    """
    minx, miny, maxx, maxy = bounds

    # Snap origins to pixel resolution grid
    # This ensures alignment with Sentinel-1 10m pixels
    minx = np.floor(minx / config.PIXEL_RES) * config.PIXEL_RES
    miny = np.floor(miny / config.PIXEL_RES) * config.PIXEL_RES

    # Calculate grid dimensions (now based on snapped bounds)
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

            center_x = x + config.PATCH_SIZE_M / 2
            center_y = y + config.PATCH_SIZE_M / 2

            yield patch_id, patch_box, center_x, center_y
            patch_id += 1


def patch_intersects_polygons(patch_box, polygons_gdf) -> bool:
    """
    Check if a patch box intersects with any polygon from the source GeoDataFrame.

    Args:
        patch_box: Shapely box geometry for the patch
        polygons_gdf: GeoDataFrame with source polygons

    Returns:
        True if patch intersects with any polygon, False otherwise
    """
    patch_gdf = gpd.GeoDataFrame({'geometry': [patch_box]}, crs=polygons_gdf.crs)
    # Use any intersection with the polygons
    return not patch_gdf.sjoin(polygons_gdf, how="inner", predicate="intersects").empty


def create_output_file(island: str, date: str, patch_features: list,
                      output_path: Path, config: PatchConfig):
    """
    Create output GeoJSON file containing all patch bounding boxes for a source file.

    Args:
        island: Island name (e.g., 'hawaii')
        date: Date string (YYYY-MM-DD format)
        patch_features: List of feature dictionaries for each patch
        output_path: Path to save the output file
        config: Configuration object
    """
    output_data = {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::32604'}},
        'features': patch_features
    }

    # Save to GeoJSON
    with open(output_path, 'w') as f:
        import json
        json.dump(output_data, f, indent=2)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    print("=" * 80)
    print("POSITIVE PATCH BOUNDS GENERATOR")
    print("=" * 80)
    print(f"Patch size: {PatchConfig.PATCH_SIZE_M}m x {PatchConfig.PATCH_SIZE_M}m")
    print(f"Overlap: {PatchConfig.OVERLAP_M}m (50%)")
    print(f"Stride: {PatchConfig.STRIDE_M}m")
    print(f"CRS: {PatchConfig.TARGET_CRS}")
    print()

    # Create output directory
    PatchConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {PatchConfig.OUTPUT_DIR}")
    print()

    # Get all positive GeoJSON files
    geojson_files = sorted(PatchConfig.INPUT_DIR.glob("positive_*.geojson"))

    if not geojson_files:
        raise FileNotFoundError(f"No positive GeoJSON files found in {PatchConfig.INPUT_DIR}")

    print(f"Found {len(geojson_files)} positive GeoJSON files")

    # Track overall statistics
    total_source_files = 0
    total_patches_generated = 0

    for geojson_file in tqdm(geojson_files, desc="Processing source files"):
        # Parse filename
        island, date = parse_filename(geojson_file.name)

        if island is None:
            print(f"  Skipping {geojson_file.name}: could not parse filename")
            continue

        total_source_files += 1

        # Load source polygons
        source_gdf = gpd.read_file(geojson_file)

        # Ensure CRS matches target
        if source_gdf.crs is None:
            source_gdf = source_gdf.set_crs(epsg=4326, allow_override=True)
        if source_gdf.crs != PatchConfig.TARGET_CRS:
            source_gdf = source_gdf.to_crs(PatchConfig.TARGET_CRS)

        if source_gdf.empty:
            print(f"  Skipping {geojson_file.name}: no polygons")
            continue

        # Get overall bounds from source file (or use study area bounds)
        # Using overall study area bounds ensures all patches use same grid
        minx, miny, maxx, maxy = source_gdf.total_bounds

        # Generate patch grid
        patch_features = []

        for patch_id, patch_box, center_x, center_y in generate_patch_grid((minx, miny, maxx, maxy), PatchConfig):
            # Check if patch intersects with ANY polygon from this source file
            if patch_intersects_polygons(patch_box, source_gdf):
                # Create feature for this patch
                patch_features.append({
                    'type': 'Feature',
                    'properties': {
                        'patch_id': patch_id,
                        'geometry': 'Polygon',
                        'center_x': center_x,
                        'center_y': center_y,
                        'min_x': patch_box.bounds[0],
                        'min_y': patch_box.bounds[1],
                        'max_x': patch_box.bounds[2],
                        'max_y': patch_box.bounds[3]
                    },
                    'geometry': patch_box.__geo_interface__
                })
                total_patches_generated += 1

        # Save all patches for this source file to one GeoJSON
        output_filename = f"patches_{island}_{date}.geojson"
        output_path = PatchConfig.OUTPUT_DIR / output_filename

        if patch_features:
            create_output_file(island, date, patch_features, output_path, PatchConfig)
            print(f"  Saved: {output_filename} ({len(patch_features)} patches)")
        else:
            print(f"  Skipping {output_filename}: no patches intersect polygons")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Source files processed: {total_source_files}")
    print(f"Total patches generated: {total_patches_generated}")
    print(f"Output directory: {PatchConfig.OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
