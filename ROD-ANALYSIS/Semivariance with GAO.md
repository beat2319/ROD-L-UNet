# Semivariance Analysis with GAO

## Overview

Semivariance (variogram) analysis using [[DMSM Labeled Dataset|DMSM]] and [[Areas/University/Semester_2/WRTG_3035/Fukushima-Spatial-Analysis/Radiation Fallout and Kriging.md|Kriging]] methods with Global Airborne Observatory (GAO) point data.

## Purpose

Quantify spatial autocorrelation and structure in ROD detection data using variogram modeling to:
- Understand the spatial scale of ROD spread
- Determine optimal parameters for kriging interpolation
- Assess the range, sill, and nugget of spatial dependence

## Data Sources

| Source | Type | Description |
|--------|------|-------------|
| **GAO** | Point data | High-resolution optical spectral readings (yearly flights) |
| **S1** | Raster | Sentinel-1 SAR backscatter data |
| **DMSM** | Polygon | Ground-truth mortality labels (bi-monthly) |

## Variogram Components

From [[Areas/University/Semester_2/WRTG_3035/Fukushima-Spatial-Analysis/Radiation Fallout and Kriging.md|Kriging methodology]]:

$$ \gamma(h) = c_0 + c \left[ 1 - \exp\left( -\frac{h}{a} \right) \right] $$

### Parameters

| Parameter | Symbol | Interpretation |
|-----------|--------|----------------|
| **Nugget** | $c_0$ | Measurement error and micro-scale variation |
| **Sill** | $c_0 + c$ | Total variance where spatial correlation ends |
| **Range** | $a$ | Distance at which points become spatially independent |

## Analysis Workflow

### 1. Data Preparation

```r
# Extract GAO points with spectral values
gao_points <- st_coordinates(gao_data)
gao_spectral <- gao_data$spectral_value

# Extract corresponding S1 values at GAO locations
s1_values <- extract(s1_raster, gao_points)
```

### 2. Experimental Variogram

$$ \gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} [Z(s_i) - Z(s_i + h)]^2 $$

Where:
- $Z(s_i)$: Measured value at location $i$
- $h$: Distance lag (bin)
- $N(h)$: Number of point pairs at distance $h$

### 3. Model Fitting

| Fitting Method | Description | Use Case |
|----------------|-------------|----------|
| **Manual/Visual** | Adjust parameters by eye | Good when you lack lots of data (rip it for DMSM) |
| **Ordinary Least Squares** | Minimize squared errors | Standard approach |
| **Weighted Least Squares** | Weight by pair count per lag | Better for uneven lag densities |
| **Generalized Least Squares** | Accounts for correlation | More complex, computationally intensive |

> **Note**: Treat automatic fitting with caution when data is limited - visual inspection is recommended (see [[Areas/University/Semester_2/GEOG_4023/03-31-2026.md|03-31-2026 notes]]).

### 4. Cross-Validation

| Method | Purpose |
|--------|---------|
| **Leave-one-out** | Assess prediction accuracy |
| **Residual variogram** | Check if spatial structure remains |

## Interpretation

### Range Interpretation

| Range Value | Interpretation for ROD |
|-------------|------------------------|
| **Small (< 100m)** | Highly localized spread, possibly individual tree or stand level |
| **Medium (100-500m)** | Neighborhood-level spread, wind/insect dispersal range |
| **Large (> 500m)** | Regional patterns, may reflect environmental gradients |

### Sill Interpretation

- **High sill**: High variability in ROD detection across study area
- **Low sill**: Homogeneous ROD patterns

### Nugget Interpretation

- **High nugget**: Significant measurement error or micro-scale variation
- **Low nugget**: Reliable measurements, captured spatial structure

## Implementation

```r
library(gstat)
library(sp)

# Create spatial points dataframe
gao_spdf <- SpatialPointsDataFrame(
  coords = gao_points,
  data = data.frame(spectral = gao_spectral)
)

# Experimental variogram
v <- variogram(spectral ~ 1, gao_spdf)

# Fit exponential model
v_fit <- fit.variogram(v, vgm(
  model = "Exp",        # Exponential model
  psill = v$gamma[max(which(v$np > 0))],  # Initial sill
  range = v$dist[max(which(v$np > 0))],   # Initial range
  nugget = 0           # Initial nugget
))

# Plot
plot(v, model = v_fit)
```

## Related Analyses

- [[Spatial Dependency Analysis with DMSM]]: Complementary analysis using categorical DMSM labels
- [[Geographically Weighted Regression (GWR)]]: Modeling spatially varying relationships
- [[DMSM Labeled Dataset]]: Source of ground-truth labels

## References

- [[Areas/University/Semester_2/WRTG_3035/Fukushima-Spatial-Analysis/Radiation Fallout and Kriging.md]]: Kriging methodology and variogram modeling
- [[Areas/University/Semester_2/GEOG_4023/03-31-2026.md]]: Variogram fitting approaches
