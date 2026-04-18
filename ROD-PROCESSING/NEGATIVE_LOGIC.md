# Negative Sample Generation Logic

## Overview

The `negative.py` script generates **negative (background) training samples** for the Rapid ʻŌhiʻa Death (ROD) deep learning pipeline. These samples represent healthy forest canopy that is temporally and spatially associated with ROD detection flightlines but contains no known mortality.

## Core Processing Logic

### 1. Spatial-temporal Filtering

For each flightline group (island + date combination):

1. **Date range calculation**: Extract mortality records within ±2 months of the flightline date
   - Rationale: ROD may develop between surveys; this buffer captures detections that may have occurred before/after the flightline
   - Ensures temporal alignment between the SAR data and mortality labels

2. **Island-specific filtering**: Apply skip rules per island configuration
   - Certain dates are excluded (e.g., poor data quality, incomplete surveys)
   - Maui 2019 has special handling (only specific dates allowed)

### 2. Negative Sample Selection

```
TCC raster
    ↓ exclude
coastline within 500m of ROD
    ↓ exclude
everything outside flightlines
    ↓ keep
75-100% canopy pixels
```

**Step-by-step:**

1. **Dissolve mortality records** - Combine all filtered ROD polygons into a single geometry
2. **Buffer by 500m** - Creates an exclusion zone around known ROD
3. **Clip coastline to buffer** - Extracts coastline segments **within** 500m of ROD (named confusingly as `negative_mask`, but this is the exclusion area)
4. **Inverse clip TCC raster** (`invert=True`) - Keeps TCC pixels **OUTSIDE** the coastal ROD exclusion zone
5. **Clip to flightline extent** - Constrains to actual SAR coverage (also removes ocean)
6. **Threshold TCC** - Keep only 75-100% canopy pixels (high-density forest)

**Note:** The variable name `negative_mask` refers to the *exclusion area* (coastal zones near ROD that we want to avoid), not the final negative sample. The inverse clip operation is what actually selects the negative samples by keeping pixels outside this exclusion zone.

### 3. Tree Canopy Cover Thresholding

After inverse clipping, apply a **75-100% canopy cover filter**:

```python
pixels = np.asarray(clipped.values)
pixels = pixels.astype('float32')
new_image = np.ma.masked_outside(pixels, 75, 100)
```

- This selects **high-density forest areas** only
- Ensures the model learns to distinguish ROD from healthy forest, not non-forest land
- NoData value (255) used for pixels outside valid TCC range

### 4. Output Format

- File naming: `tcc_negative_{island}_{YYYY-MM-DD}.tif`
- Values: 75-100 (high canopy), 255 (NoData)
- CRS: Reprojected to match source TCC raster

## Why This Design?

### Class Imbalance Problem

ROD pixels are **extremely rare** (<1% of pixels in the dataset). Without careful negative sampling:
- Models may learn to predict "background" for everything
- Loss functions need aggressive class weighting (pos_weight ~20.0+)
- Precision/recall metrics become unreliable

### "Hard Negative" Sampling

This approach selects **hard negatives**—healthy forest that:
1. **Has high canopy cover** (75-100%) - structurally similar to infected forest
2. **Is spatially proximate** to ROD detections (via flightline alignment)
3. **Is temporally aligned** (±2 month buffer)
4. **Is NOT within 500m** of confirmed ROD (exclusion zone)

By training on hard negatives, the model must learn subtle spectral/structural differences between healthy and infected ʻōhiʻa, rather than trivial distinctions (forest vs. urban).

### Coastline Constrained Search

Clipping to coastline ensures:
- Negative samples are within the island/study area extent
- Ocean areas are excluded (not relevant for forest disease detection)
- Computationally efficient—searches only valid land areas

## Supporting Research Context

From the study documentation:

> "To enforce a robust training signal, we use a 1:1 balance between positive (infected) and negative (healthy canopy) chips. Negative samples are restricted to NLCD Tree Canopy Cover (TCC) product with a 10% threshold to ensure the model is able to determine ROD from a healthy forest without interference from non-forested land."

The `negative.py` implementation extends this with:
- **Higher TCC threshold** (75-100% vs 10%) - focuses on dense forest
- **Spatiotemporal filtering** - flightline-specific, date-buffered sampling
- **500m exclusion zone** - removes ambiguous edge areas near known ROD

## Configuration

### Island-Specific Rules

```python
ISLAND_CONFIG = {
    "Hawaii": {
        "skip_dates": ["05/31/2017", "03/12/2019", "07/03/2019", "12/18/2020", "12/17/2021"],
        "date_buffer_months": 2
    },
    "Lanai": {
        "skip_dates": ["08/16/2019"],
        "date_buffer_months": 2
    },
    "Maui": {
        "skip_dates": ["03/22/2016"],
        "date_buffer_months": 2,
        "year_2019_allowed": ["01/09/2019", "02/08/2019"]  # Special rule for 2019
    },
    "Molokai": {
        "skip_dates": [],
        "date_buffer_months": 2
    }
}
```

### Key Parameters

| Parameter | Value | Rationale |
|-----------|--------|-----------|
| **Buffer distance** | 500m | Exclusion zone around known ROD |
| **Date buffer** | ±2 months | Temporal alignment tolerance |
| **TCC threshold** | 75-100% | High-density forest only |
| **NoData value** | 255 | Outside valid TCC range |

## Data Flow

```
Input:
  ├── ohia_mortality.geojson (all detections, timestamped)
  ├── coastline.geojson (island boundaries)
  ├── flightlines_*.geojson (SAR coverage, timestamps)
  └── tcc_124_*.tif (annual canopy cover)

Per (Island, Date) group:
  1. Filter ohia_mortality by date range (±2 months)
  2. Buffer filtered detections by 500m → exclusion zone
  3. Clip coastline to buffer → exclusion_mask (coastal ROD areas)
  4. Load TCC raster for year
  5. Reproject all to TCC CRS
  6. Inverse clip: keep TCC OUTSIDE exclusion_mask (removes coastal ROD areas)
  7. Clip to flightline extent → removes ocean, constrains to SAR coverage
  8. Threshold: 75-100% TCC (high-density forest only)
  9. Save: tcc_negative_{island}_{date}.tif

Output:
  └── data/tcc_negative_*.tif (healthy forest samples, no known ROD)
```

## References

- [[../../J-STARS/Documentation/Study Area and Data Acquisition.md|Study Area and Data Acquisition]] - Overview of 1:1 balanced training approach
- [[../../J-STARS/ROD-COLLECTION/ROD-COLLECTION.md|ROD-COLLECTION]] - Mortality data source
- [[negative.py|negative.py]] - Implementation
