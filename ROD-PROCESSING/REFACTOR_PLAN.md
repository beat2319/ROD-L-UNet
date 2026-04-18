# Refactoring Plan for negative.py

## Issue 1: Critical Bug - Scope Issue in try...except Block

**Current Problem (lines 149-162):**
```python
try:
    clipped = tcc.rio.clip(negative_mask.geometry.values, all_touched=True, invert=True)
    clipped = clipped.rio.clip(flightline_reproj.geometry.values, all_touched=True)
    out_image = clipped.values
except Exception as e:
    print(f"    Clip error: {e}")
    out_image = tcc.values  # Falls back to full TCC
```

**The Bug:** If clipping fails, `clipped` is never defined, but lines 165-174 use `clipped.rio.height`, `clipped.rio.width`, etc. This causes either a `NameError` or reuses metadata from the previous iteration.

**Fix:** Skip the flightline entirely if clipping fails:
```python
try:
    clipped = tcc.rio.clip(negative_mask.geometry.values, all_touched=True, invert=True)
    clipped = clipped.rio.clip(flightline_reproj.geometry.values, all_touched=True)
except Exception as e:
    print(f"    Clip error: {e} - skipping flightline {objectid}")
    continue
```

---

## Issue 2: Performance - Redundant I/O and Reprojections

**Current Problem:**
- Line 119: `tcc = rioxarray.open_rasterio(tcc_path)` - loaded for EVERY flightline
- Line 125: `coastline_reproj = coastline.to_crs(tcc_crs)` - reprojected for EVERY flightline

**Fix:** Implement a cache dictionary to store loaded rasters and reprojected coastline per year:

```python
# Initialize cache before the flightline loop
tcc_cache = {}  # Maps year -> {'tcc': raster, 'tcc_crs': CRS, 'coastline_reproj': GeoDataFrame}

# Inside the flightline loop:
if year not in tcc_cache:
    tcc = rioxarray.open_rasterio(tcc_path)
    tcc_crs = tcc.rio.crs
    coastline_reproj = coastline.to_crs(tcc_crs)
    tcc_cache[year] = {
        'tcc': tcc,
        'tcc_crs': tcc_crs,
        'coastline_reproj': coastline_reproj
    }
else:
    cached = tcc_cache[year]
    tcc = cached['tcc']
    tcc_crs = cached['tcc_crs']
    coastline_reproj = cached['coastline_reproj']
```

---

## Issue 3: NoData Handling

**Current Problem (lines 179-185):**
```python
new_image = np.ma.masked_outside(pixels, 75, 100)
# ...
dest.write(new_image.filled(0))  # 0 means both "NoData" and "0% canopy"
```

**Fix:** Use a dedicated NoData value outside the 0-100 range:

```python
# Use 255 as NoData (outside 0-100 valid range)
NODATA_VALUE = 255

# Update metadata
out_meta["nodata"] = NODATA_VALUE

# Fill with NoData value instead of 0
dest.write(new_image.filled(NODATA_VALUE))
```

---

## Issue 4: Prevent Filename Overwrites

**Current Problem (lines 182-183):**
```python
date_str = start_date.strftime('%m-%d-%Y')
output_path = os.path.join(output_dir, f"rod_masked_{date_str}.tif")
```

**Fix:** Include OBJECTID in filename for uniqueness:
```python
date_str = start_date.strftime('%m-%d-%Y')
output_path = os.path.join(output_dir, f"rod_masked_{objectid}_{date_str}.tif")
```

---

## Implementation Order

1. **Issue 2 (Performance)** - Refactor first to avoid loading rasters repeatedly
2. **Issue 4 (Filenames)** - Simple change, independent of others
3. **Issue 3 (NoData)** - Simple change, independent of others
4. **Issue 1 (Critical Bug)** - Must be fixed after performance refactor as it affects the same code section
