"""
Clip GAO 2016 centers by flightline date (01/12/2016).

Takes the Hawaii 2016 flightlines GeoJSON, filters to the Jan 12 flightline(s),
reprojects to match the GAO centers CRS, and clips the point dataset.
"""

import geopandas as gpd
from pathlib import Path

# Paths relative to script location
BASE = Path(__file__).resolve().parent.parent
FLIGHTLINES = BASE / "ROD-PROCESSING" / "attachments" / "flightlines" / "flightlines_hawaii_2016_processed.geojson"
GAO_CENTERS = BASE / "ROD-COLLECTION" / "attachments" / "GAO" / "2016_htmasked_brown_clipped_centers.geojson"
OUTPUT = Path(__file__).resolve().parent / "gao_2016_clipped_01122016.geojson"

TARGET_DATE = "01/12/2016"

# Load data
print(f"Loading flightlines from {FLIGHTLINES}")
flightlines = gpd.read_file(FLIGHTLINES)
print(f"  CRS: {flightlines.crs}, Features: {len(flightlines)}")

print(f"\nLoading GAO centers from {GAO_CENTERS}")
centers = gpd.read_file(GAO_CENTERS)
print(f"  CRS: {centers.crs}, Features: {len(centers)}")

# Filter flightlines to target date
flightline_date = flightlines[flightlines["START_DATE_STR"] == TARGET_DATE]
print(f"\nFiltered flightlines to {TARGET_DATE}: {len(flightline_date)} feature(s)")

if len(flightline_date) == 0:
    print("ERROR: No flightlines found for the target date.")
    raise SystemExit(1)

# Reproject flightlines to match centers CRS
if flightline_date.crs != centers.crs:
    print(f"Reprojecting flightlines from {flightline_date.crs} → {centers.crs}")
    flightline_date = flightline_date.to_crs(centers.crs)
else:
    print("CRS already matches, no reprojection needed.")

# Clip centers by flightline polygons
clipped = gpd.clip(centers, flightline_date)
print(f"\nClipped result: {len(clipped)} points (of {len(centers)} total)")

# Save output
clipped.to_file(OUTPUT, driver="GeoJSON")
print(f"\nSaved to {OUTPUT}")
print(f"Output CRS: {clipped.crs}")
