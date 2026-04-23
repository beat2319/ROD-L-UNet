"""
Clip 2016 TCC raster by the 2016-01-12 flightline.

Filters the Hawaii 2016 flightlines GeoJSON to the Jan 12 flightline(s),
then masks the NLCD TCC raster to that geometry.
"""

import geopandas as gpd
import rasterio
import rasterio.mask
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent
FLIGHTLINES = BASE / "ROD-PROCESSING" / "attachments" / "flightlines" / "flightlines_hawaii_2016_processed.geojson"
TCC = BASE / "ROD-COLLECTION" / "attachments" / "TCC" / "nlcd_tcc_hawaii_wgs84_v2023-5_20160101_20161231.tif"
OUTPUT = Path(__file__).resolve().parent / "tcc_2016_clipped_01122016.tif"

TARGET_DATE = "01/12/2016"

# Load flightlines and filter to target date
print(f"Loading flightlines from {FLIGHTLINES}")
flightlines = gpd.read_file(FLIGHTLINES)
print(f"  CRS: {flightlines.crs}, Features: {len(flightlines)}")

flightline_date = flightlines[flightlines["START_DATE_STR"] == TARGET_DATE]
print(f"\nFiltered to {TARGET_DATE}: {len(flightline_date)} feature(s)")

if len(flightline_date).all() == 0:
    print("ERROR: No flightlines found for the target date.")
    raise SystemExit(1)

# Reproject flightlines to match raster CRS
with rasterio.open(TCC) as src:
    raster_crs = src.crs
    print(f"\nTCC raster CRS: {raster_crs}")

if flightline_date.crs != raster_crs:
    print(f"Reprojecting flightlines from {flightline_date.crs} -> {raster_crs}")
    flightline_date = flightline_date.to_crs(raster_crs)
else:
    print("CRS already matches.")

# Clip raster by flightline geometry
geometries = flightline_date.geometry.values

with rasterio.open(TCC) as src:
    out_image, out_transform = rasterio.mask.mask(src, geometries, crop=True)
    out_meta = src.meta.copy()

out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
})

with rasterio.open(OUTPUT, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"\nSaved clipped TCC to {OUTPUT}")
print(f"  Shape: {out_image.shape}")
print(f"  CRS: {out_meta['crs']}")
