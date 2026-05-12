"""
Visualization script for the positive patch pipeline.
Generates visualization files for Molokai on 01/29/2019 in UTM Zone 4 (EPSG:32604).

Outputs:
1. Flightline (not buffered) - GeoJSON
2. Flightline 1 mile buffer - GeoJSON
3. All ohia mortality intersecting flightline buffer (±2 months) - GeoJSON
4. Polygon-only ohia mortality intersecting flightline buffer - GeoJSON
"""

import geopandas as gpd
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# Configuration
TARGET_ISLAND = 'Molokai'
TARGET_DATE = datetime(2019, 1, 29)
TARGET_DATE_STR = '2019-01-29'
START_DATE_MS = 1548785653000  # 01/29/2019 in milliseconds
UTM_ZONE_4 = 'EPSG:32604'

# Paths - using raw source files
FLIGHTLINE_PATH = '../ROD-COLLECTION/attachments/DMSM/flightlines/flightlines_2019.geojson'
OHIA_MORTALITY_PATH = '../ROD-COLLECTION/attachments/DMSM/ohia_mortality/ohia_mortality.geojson'

# Output directory
OUTPUT_DIR = './visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"Positive Patch Pipeline Visualization")
print(f"Island: {TARGET_ISLAND}")
print(f"Date: {TARGET_DATE_STR}")
print(f"CRS: {UTM_ZONE_4}")
print("=" * 60)

# =============================================================================
# STEP 1: Load and filter flightline data
# =============================================================================
print("\n[1/5] Loading and filtering flightline data...")

flightline = gpd.read_file(FLIGHTLINE_PATH)

# Convert START_DATE to datetime
flightline['START_DATE_DT'] = pd.to_datetime(flightline['START_DATE'], unit='ms')

# Filter for Molokai and 01/29/2019
flightline_filtered = flightline[
    (flightline['Island'] == TARGET_ISLAND) &
    (flightline['START_DATE'] == START_DATE_MS)
]

if len(flightline_filtered) == 0:
    raise ValueError(f"No flightline found for {TARGET_ISLAND} on {TARGET_DATE_STR}")

print(f"  Found {len(flightline_filtered)} flightline(s)")

# Convert to UTM Zone 4
flightline_filtered = flightline_filtered.set_crs(epsg=4326, allow_override=True)
flightline_filtered = flightline_filtered.to_crs(UTM_ZONE_4)

# Output 1: Save flightline (not buffered)
output_path = os.path.join(OUTPUT_DIR, f'flightline_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}.geojson')
flightline_filtered.to_file(output_path, driver='GeoJSON')
print(f"  Saved: {output_path}")

# =============================================================================
# STEP 2: Create 1-mile flightline buffer
# =============================================================================
print("\n[2/5] Creating 1-mile flightline buffer...")

# 1 mile = 1609.34 meters
flightline_buffer = flightline_filtered.copy()
flightline_buffer['geometry'] = flightline_buffer.geometry.buffer(1609.34)

# Output 2: Save flightline 1-mile buffer
output_path = os.path.join(OUTPUT_DIR, f'flightline_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}_1mile_buffer.geojson')
flightline_buffer.to_file(output_path, driver='GeoJSON')
print(f"  Saved: {output_path}")

# =============================================================================
# STEP 3: Load ohia mortality and intersect with flightline buffer (ALL types)
# =============================================================================
print("\n[3/5] Loading ohia mortality and intersecting with flightline buffer...")

ohia_mortality = gpd.read_file(OHIA_MORTALITY_PATH)

# Convert CREATED_DATE from milliseconds to datetime
ohia_mortality['CREATED_DATE_DT'] = pd.to_datetime(ohia_mortality['CREATED_DATE'], unit='ms')

# Calculate date range (±2 months)
date_range_start = TARGET_DATE - relativedelta(months=2)
date_range_end = TARGET_DATE + relativedelta(months=2)

print(f"  Date range: {date_range_start.strftime('%m/%d/%Y')} to {date_range_end.strftime('%m/%d/%Y')}")

# Filter by island and date range (ISLAND is uppercase)
ohia_mortality_filtered = ohia_mortality[
    (ohia_mortality['ISLAND'] == TARGET_ISLAND) &
    (ohia_mortality['CREATED_DATE_DT'] >= date_range_start) &
    (ohia_mortality['CREATED_DATE_DT'] <= date_range_end)
].copy()

print(f"  Found {len(ohia_mortality_filtered)} mortality records in date range")

# Convert to UTM Zone 4
ohia_mortality_filtered = ohia_mortality_filtered.set_crs(epsg=4326, allow_override=True)
ohia_mortality_filtered = ohia_mortality_filtered.to_crs(UTM_ZONE_4)

# Spatial join: keep any mortality feature that intersects the flightline buffer
mortality_intersected = gpd.sjoin(ohia_mortality_filtered, flightline_buffer, how='inner', predicate='intersects')

# Drop join columns from flightline buffer
join_cols = [col for col in mortality_intersected.columns if col.endswith('_right') or col in flightline_buffer.columns.drop('geometry')]
join_cols = [c for c in join_cols if c not in ohia_mortality_filtered.columns or c == 'index_right']
mortality_intersected = mortality_intersected.drop(columns=[c for c in mortality_intersected.columns if c in join_cols and c != 'geometry'])

print(f"  {len(mortality_intersected)} mortality features intersect flightline buffer")

# Output 3: Save all mortality intersecting flightline buffer
output_path = os.path.join(OUTPUT_DIR, f'ohia_mortality_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}_positive_all.geojson')
mortality_intersected.to_file(output_path, driver='GeoJSON')
print(f"  Saved: {output_path}")

# =============================================================================
# STEP 4: Filter to POLYGON-type mortality only
# =============================================================================
print("\n[4/5] Filtering to polygon-only mortality...")

mortality_polygon = mortality_intersected[mortality_intersected['AREA_TYPE'] == 'POLYGON'].copy()

if len(mortality_polygon) == 0:
    print("  Warning: No POLYGON-type mortality features found")
else:
    print(f"  {len(mortality_polygon)} polygon mortality features (removed {len(mortality_intersected) - len(mortality_polygon)} point features)")

    # Output 4: Save polygon-only mortality
    output_path = os.path.join(OUTPUT_DIR, f'ohia_mortality_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}_positive_polygon.geojson')
    mortality_polygon.to_file(output_path, driver='GeoJSON')
    print(f"  Saved: {output_path}")

# =============================================================================
# STEP 5: Summary statistics
# =============================================================================
print("\n[5/5] Summary statistics...")

if len(mortality_intersected) > 0:
    n_point = len(mortality_intersected[mortality_intersected['AREA_TYPE'] == 'POINT'])
    n_polygon = len(mortality_intersected[mortality_intersected['AREA_TYPE'] == 'POLYGON'])
    print(f"  Total mortality features: {len(mortality_intersected)}")
    print(f"    POINT-type: {n_point}")
    print(f"    POLYGON-type: {n_polygon}")

    if 'PERCENT_AFFECTED_FACTOR' in mortality_intersected.columns:
        print(f"\n  Severity breakdown:")
        for sev in sorted(mortality_intersected['PERCENT_AFFECTED_FACTOR'].unique()):
            count = len(mortality_intersected[mortality_intersected['PERCENT_AFFECTED_FACTOR'] == sev])
            print(f"    {sev}: {count} features")

    if 'ACRES' in mortality_intersected.columns:
        print(f"\n  Acreage (all features):")
        print(f"    Min: {mortality_intersected['ACRES'].min():.4f}")
        print(f"    Max: {mortality_intersected['ACRES'].max():.4f}")
        print(f"    Total: {mortality_intersected['ACRES'].sum():.4f}")

    if len(mortality_polygon) > 0 and 'ACRES' in mortality_polygon.columns:
        print(f"\n  Acreage (polygon only):")
        print(f"    Min: {mortality_polygon['ACRES'].min():.4f}")
        print(f"    Max: {mortality_polygon['ACRES'].max():.4f}")
        print(f"    Total: {mortality_polygon['ACRES'].sum():.4f}")
else:
    print("  No mortality features to summarize")

print("\n" + "=" * 60)
print("Visualization script complete!")
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("=" * 60)
