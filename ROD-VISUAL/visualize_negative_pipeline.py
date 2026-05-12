"""
Visualization script for the negative patch pipeline.
Generates visualization files for Molokai on 01/29/2019 in UTM Zone 4 (EPSG:32604).

Outputs:
1. Flightline (not buffered) - GeoJSON
2. Flightline 1 mile buffer - GeoJSON
3. TCC clipped to Molokai - GeoTIFF
4. TCC clipped to flightline buffer - GeoTIFF
5. Raw ohia mortality for that date (±2 months) - GeoJSON
6. 500m buffer on ohia mortality - GeoJSON
7. Inverse clip of 500m buffer from TCC clipped flightline buffer (75-100% canopy) - GeoTIFF
"""

import geopandas as gpd
import pandas as pd
import rioxarray
import numpy as np
import rasterio
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# Configuration
TARGET_ISLAND = 'Molokai'
TARGET_DATE = datetime(2019, 1, 29)
TARGET_DATE_STR = '2019-01-29'
START_DATE_MS = 1548785653000  # 01/29/2019 in milliseconds
UTM_ZONE_4 = 'EPSG:32604'
NODATA_VALUE = 255

# Paths - using raw source files as requested
FLIGHTLINE_PATH = '../ROD-COLLECTION/attachments/DMSM/flightlines/flightlines_2019.geojson'
TCC_PATH = '../ROD-COLLECTION/attachments/TCC/nlcd_tcc_hawaii_wgs84_v2023-5_20190101_20191231.tif'
OHIA_MORTALITY_PATH = '../ROD-COLLECTION/attachments/DMSM/ohia_mortality/ohia_mortality.geojson'
COASTLINE_PATH = '../ROD-COLLECTION/attachments/Coastline/coastline.geojson'

# Output directory
OUTPUT_DIR = './visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"Negative Patch Pipeline Visualization")
print(f"Island: {TARGET_ISLAND}")
print(f"Date: {TARGET_DATE_STR}")
print(f"CRS: {UTM_ZONE_4}")
print("=" * 60)

# =============================================================================
# STEP 1: Load and filter flightline data
# =============================================================================
print("\n[1/7] Loading and filtering flightline data...")

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
print("\n[2/7] Creating 1-mile flightline buffer...")

# 1 mile = 1609.34 meters
flightline_buffer = flightline_filtered.copy()
flightline_buffer['geometry'] = flightline_buffer.geometry.buffer(1609.34)

# Output 2: Save flightline 1-mile buffer
output_path = os.path.join(OUTPUT_DIR, f'flightline_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}_1mile_buffer.geojson')
flightline_buffer.to_file(output_path, driver='GeoJSON')
print(f"  Saved: {output_path}")

# =============================================================================
# STEP 3: Load and prepare TCC data
# =============================================================================
print("\n[3/7] Loading and preparing TCC data...")

tcc = rioxarray.open_rasterio(TCC_PATH)
print(f"  Original TCC CRS: {tcc.rio.crs}")

# Reproject to UTM Zone 4
tcc_utm4 = tcc.rio.reproject(UTM_ZONE_4)
print(f"  Reprojected to: {tcc_utm4.rio.crs}")

# =============================================================================
# STEP 4: Load Molokai coastline and clip TCC
# =============================================================================
print("\n[4/7] Clipping TCC to Molokai coastline...")

coastline = gpd.read_file(COASTLINE_PATH)
coastline_molokai = coastline[coastline['isle'] == TARGET_ISLAND].copy()
coastline_molokai = coastline_molokai.to_crs(UTM_ZONE_4)

# Clip TCC to Molokai coastline
tcc_molokai = tcc_utm4.rio.clip(coastline_molokai.geometry.values, all_touched=True)

# Output 3: Save TCC clipped to Molokai
output_path = os.path.join(OUTPUT_DIR, f'tcc_{TARGET_ISLAND.lower()}_2019.tif')
tcc_molokai.rio.to_raster(output_path)
print(f"  Saved: {output_path}")

# =============================================================================
# STEP 5: Clip TCC to flightline buffer
# =============================================================================
print("\n[5/7] Clipping TCC to flightline buffer...")

# Clip TCC to 1-mile flightline buffer
tcc_flightline_buffer = tcc_molokai.rio.clip(flightline_buffer.geometry.values, all_touched=True)

# Output 4: Save TCC clipped to flightline buffer
output_path = os.path.join(OUTPUT_DIR, f'tcc_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}_flightline_buffer.tif')
tcc_flightline_buffer.rio.to_raster(output_path)
print(f"  Saved: {output_path}")

# =============================================================================
# STEP 6: Load ohia mortality data (±2 months)
# =============================================================================
print("\n[6/7] Loading ohia mortality data...")

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

print(f"  Found {len(ohia_mortality_filtered)} mortality records")

# Convert to UTM Zone 4
ohia_mortality_filtered = ohia_mortality_filtered.to_crs(UTM_ZONE_4)

# Output 5: Save raw ohia mortality
output_path = os.path.join(OUTPUT_DIR, f'ohia_mortality_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}.geojson')
ohia_mortality_filtered.to_file(output_path, driver='GeoJSON')
print(f"  Saved: {output_path}")

# =============================================================================
# STEP 7: Create 500m mortality buffer
# =============================================================================
print("\n[7/7] Creating 500m mortality buffer and inverse clipping...")

if len(ohia_mortality_filtered) > 0:
    # Dissolve mortality polygons
    dissolved_mortality = ohia_mortality_filtered.dissolve()

    # Create 500m buffer
    mortality_buffer = dissolved_mortality.buffer(500)

    # Create GeoDataFrame with buffered geometry
    mortality_buffer_gdf = gpd.GeoDataFrame(geometry=mortality_buffer, crs=UTM_ZONE_4)
    dissolved_buffer = mortality_buffer_gdf.dissolve()

    # Output 6: Save 500m mortality buffer
    output_path = os.path.join(OUTPUT_DIR, f'ohia_mortality_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}_500m_buffer.geojson')
    dissolved_buffer.to_file(output_path, driver='GeoJSON')
    print(f"  Saved: {output_path}")

    # Inverse clip: keep areas OUTSIDE the 500m buffer
    tcc_inverse = tcc_flightline_buffer.rio.clip(dissolved_buffer.geometry.values, all_touched=True, invert=True)

    # Apply 75-100% canopy threshold
    pixels = np.asarray(tcc_inverse.values)
    pixels = pixels.astype('float32')

    # Mask pixels outside 75-100% range (set to nodata)
    masked_pixels = np.ma.masked_outside(pixels, 75, 100)

    # Prepare output metadata
    out_meta = {
        "driver": "GTiff",
        "height": tcc_inverse.rio.height,
        "width": tcc_inverse.rio.width,
        "count": tcc_inverse.rio.count,
        "dtype": str(tcc_inverse.dtype),
        "crs": str(tcc_inverse.rio.crs),
        "transform": tcc_inverse.rio.transform(),
        "nodata": NODATA_VALUE
    }

    # Output 7: Save inverse clipped TCC with 75-100% threshold
    output_path = os.path.join(OUTPUT_DIR, f'tcc_{TARGET_ISLAND.lower()}_{TARGET_DATE_STR}_inverse_clipped.tif')
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(masked_pixels.filled(NODATA_VALUE))
    print(f"  Saved: {output_path}")
else:
    print("  No mortality records found - skipping buffer and inverse clip")

print("\n" + "=" * 60)
print("Visualization script complete!")
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("=" * 60)
