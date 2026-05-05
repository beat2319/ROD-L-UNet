# Claude's Plan - Update negative.py

## Context

The current `negative.py` processes ROD (Rapid 'Ōhiʻa Death) data by:
- Taking yearly `ohia_mortality` files
- Buffering ROD detections by 500m
- Clipping to coastline
- Applying inverse clip to TCC raster

## Problem

The current approach is year-based, but we need to:
- Dissolve `ohia_mortality` records that fall within each flightline's date range (START_DATE to END_DATE)
- NOT filter by AREA_TYPE (include all record types)
- Follow processing steps from `ohia_mortality.py`

## Data Structures

### Flightlines (`attachments/flightlines/flightlines_{year}_processed.geojson`):
- START_DATE: milliseconds since epoch
- END_DATE: milliseconds since epoch
- START_DATE_STR: MM/DD/YYYY format
- geometry: already buffered 1 mile (1609.34m)

### Ohia Mortality (`../ROD-COLLECTION/attachments/DMSM/ohia_mortality/ohia_mortality.geojson`):
- CREATED_DATE: milliseconds since epoch
- detection_timestamp: MM/DD/YYYY format (derived from CREATED_DATE)
- AREA_TYPE: "POLYGON", "POINT", etc. (NO filtering - include all)
- geometry: polygon or point of mortality area

## Implementation Plan

### For each flightline in `attachments/flightlines/`:

1. Load flightline - already buffered (1 mile) in EPSG:32604
2. Determine year from flightline's START_DATE
3. Load `ohia_mortality` from `../ROD-COLLECTION/attachments/DMSM/ohia_mortality/ohia_mortality.geojson`
4. Filter by date range:
   - Convert CREATED_DATE (milliseconds) to datetime
   - Keep records where CREATED_DATE is between flightline.START_DATE and flightline.END_DATE
   - Include ALL AREA_TYPES (no "POLYGON" filter)
5. Dissolve filtered `ohia_mortality` into single geometry
6. Buffer dissolved `ohia_mortality` by 500m
7. Clip coastline to buffered `ohia_mortality` geometry
8. Load TCC raster for year
9. Reproject TCC to EPSG:32604
10. Apply inverse clip to TCC raster (keep everything OUTSIDE of clipped coastline area)
11. Threshold pixels (75-100% canopy cover)
12. Save output: `rod_masked_{year}_{flightline_OBJECTID}.tif`

## Key Changes

| Current | New |
|---------|------|
| Yearly dissolve (by YEAR column) | Dissolve by flightline date range filter |
| Separate yearly `ohia_mortality` files | Single source file, filtered by CREATED_DATE |
| Filter to POLYGON only | Include ALL AREA_TYPES |
| Separate yearly `ohia_mortality` files | Process each flightline individually |
| Fixed `rod_path` input | Process each flightline individually |

## Implementation Details

```python
# Pseudocode outline
import geopandas as gpd
from shapely.geometry import shape
import os
import fiona
import rioxarray
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS
import numpy as np

def negative_area(rod_path, tcc_path, coastline_path):
    # make buffer around rod detections
    rod = gpd.read_file(f'{rod_path}')
    rod_32604 = rod.to_crs("EPSG:32604")
    rod_buffer = gpd.GeoDataFrame(geometry=(rod_32604.buffer(500)))
    dissolved_rod = rod_buffer.dissolve()

    # dissolved coastline and ensure proper alignment
    coastline = gpd.read_file(f'{coastline_path}')
    coastline_32604 = coastline.to_crs("EPSG:32604")
    dissolved_coastline = coastline_32604.dissolve()

    negative_mask = gpd.clip(dissolved_coastline, dissolved_rod)

    with rasterio.open(f"{tcc_path}", 'r') as src:
        out_image, out_transform = rasterio.mask.mask(src, negative_mask.geometry, invert=True)
    
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    pixels = np.asarray(out_image)
    pixels = pixels.astype('float32')
    new_image = np.ma.masked_outside(pixels, 75, 100) # threshold for canopy cover

    with rasterio.open("rod_masked.tif", "w", **out_meta) as dest:
        dest.write(new_image)

if __name__ == "__main__":
    negative_area("../attachments/DMSM/ohia_mortality/ohia_mortality.geojson", "./data/masked.tif", "./data/coastline.geojson")
```

## Verification

Run `python3 negative.py` and check that:
- No errors occur
- Output rasters are generated (one per flightline)
- Rasters contain expected data (check bounds and pixel values)

Verify `ohia_mortality` filtering:
- Spot-check that mortality records outside date range are excluded
- Confirm records within date range are included
- Ensure all AREA_TYPES are present (not just POLYGON)

Validate output naming: `rod_masked_{year}_{OBJECTID}.tif` pattern

---
[[../../README.md]]
[[../../Inbox/claudes plan.md]]