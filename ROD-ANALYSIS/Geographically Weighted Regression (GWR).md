# Geographically Weighted Regression (GWR)

## Overview

Geographically Weighted Regression (GWR) for modeling spatially varying relationships between [[DMSM Labeled Dataset|DMSM]] ROD severity and [[Study Area and Data Acquisition|Sentinel-1 (S1)]] SAR features.

## Purpose

GWR allows regression coefficients to vary spatially, addressing:
- Spatial non-stationarity (relationships change across space)
- Local variation in ROD-S1 relationships
- Identification of regional drivers of ROD severity

## When to Use GWR

| Situation | Recommendation |
|-----------|----------------|
| Relationships vary spatially | **Use GWR** |
| Global model (OLS) has spatially autocorrelated residuals | **Use GWR** |
| Small, homogeneous study area | OLS may be sufficient |
| Very sparse data | GWR may overfit |

## Model Specification

### Standard GWR Model

$$ y_i = \beta_0(u_i, v_i) + \sum_{k=1}^{p} \beta_k(u_i, v_i) x_{ik} + \varepsilon_i $$

Where:
- $y_i$: Response (e.g., ROD severity) at location $i$
- $(u_i, v_i)$: Coordinates of location $i$
- $\beta_0(u_i, v_i)$: Intercept at location $i$
- $\beta_k(u_i, v_i)$: Coefficient for predictor $k$ at location $i$
- $x_{ik}$: Value of predictor $k$ at location $i$
- $\varepsilon_i$: Error term

### For ROD Analysis

$$ \text{Severity}_i = \beta_0(u_i, v_i) + \beta_1(u_i, v_i)\text{S1\_VV}_i + \beta_2(u_i, v_i)\text{S1\_VH}_i + \beta_3(u_i, v_i)\text{S1\_Ratio}_i + \varepsilon_i $$

---

## Data Options

### Option 1: GWR with DMSM (Recommended)

| Aspect | Details |
|--------|---------|
| **Response Variable** | DMSM severity (ordinal/continuous) |
| **Predictors** | S1 features (VV, VH, ratio, texture, temporal change) |
| **Spatial Unit** | Polygon centroids or rasterized grid cells |
| **Advantages** | Broader coverage, bi-monthly frequency, severity dimension |
| **Limitations** | Human observation error, variable accuracy |

**Handling DMSM Polygons:**
```r
# Option A: Centroid approach
centroids <- st_coordinates(st_centroid(dmsm_polygons))

# Option B: Area-weighted aggregation
# Calculate mean S1 values within each polygon
s1_by_polygon <- exact_extract(s1_raster, dmsm_polygons, fun = 'mean')

# Option C: Rasterize and sample
# Convert DMSM to raster grid (same resolution as S1)
```

### Option 2: GWR with GAO (Calibration)

| Aspect | Details |
|--------|---------|
| **Response Variable** | GAO spectral values |
| **Predictors** | S1 features |
| **Spatial Unit** | Points |
| **Advantages** | High accuracy, instrumental measurements |
| **Limitations** | Once yearly, smaller sample size |

**Use Case:** Calibration and validation of S1-ROD relationships

### Combined Approach (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Calibration with GAO                                │
│         - Establish S1-spectral relationship                 │
│         - Understand spatial variation                      │
│         - Validate model performance                        │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: Training with DMSM                                  │
│         - Use spatially-informed priors from GAO            │
│         - Train on broader DMSM coverage                     │
│         - Capture more spatial heterogeneity                 │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: Prediction to Full S1 Coverage                      │
│         - Generate ROD severity/risk map                    │
│         - Apply to areas without DMSM/GAO coverage          │
└─────────────────────────────────────────────────────────────┘
```

---

## GWR Parameters

### Bandwidth Selection

Critical decision determining local vs. global fit:

| Bandwidth Type | Description | When to Use |
|----------------|-------------|-------------|
| **Fixed (distance)** | Constant spatial extent for all locations | Evenly distributed data |
| **Adaptive (k-nearest)** | Constant number of neighbors, variable extent | Variable density data |

**For DMSM:** Use adaptive bandwidth (surveys have variable density)

```r
# Optimize bandwidth using AICc
bw <- bw.gwr(
  severity ~ vv + vh + ratio + texture,
  data = gwr_data,
  approach = "AICc",
  kernel = "bisquare",
  adaptive = TRUE
)
```

### Kernel Functions

| Kernel | Characteristics | Use Case |
|--------|----------------|----------|
| **Gaussian** | Smooth, infinite support | Standard choice |
| **Bisquare** | Bounded, zero outside bandwidth | Handles outliers better (recommended) |
| **Exponential** | Similar to Gaussian | Faster decay |

### Number of Neighbors (for adaptive bandwidth)

| k Neighbors | Spatial Extent | Trade-off |
|-------------|----------------|-----------|
| **Small (10-20)** | Very local | May overfit, high variance |
| **Medium (30-50)** | Balanced | Good starting point |
| **Large (100+)** | More global | May miss local variation |

**Recommendation:** Start with k = 30-50, optimize using AICc

---

## S1 Predictors for ROD

| S1 Feature | Description | ROD Relevance |
|------------|-------------|---------------|
| **VV** | Co-polarized backscatter | Canopy structure, density |
| **VH** | Cross-polarized backscatter | Volume scattering, structural complexity |
| **VH/VV Ratio** | Polarization ratio | Sensitive to structural decay |
| **Texture (GLCM)** | Spatial heterogeneity | Canopy uniformity, disturbance patterns |
| **Temporal Change** | ΔVV, ΔVH over time | Rapid canopy changes, disease progression |

---

## Model Diagnostics

### Global Diagnostics

| Metric | What It Checks | Interpretation |
|--------|----------------|----------------|
| **Global R²** | Overall model fit | Higher than OLS indicates GWR improvement |
| **AICc** | Model quality (vs. OLS) | Lower = better; if GWR AICc < OLS AICc, GWR preferred |
| **Adj. R²** | Adjusted for complexity | Compare across models |

### Local Diagnostics

| Metric | What It Checks | Interpretation |
|--------|----------------|----------------|
| **Local R²** | Local model fit | Identify areas of poor prediction |
| **Local coefficients** | Spatially varying relationships | Map coefficient surfaces |
| **Standard errors** | Uncertainty in local estimates | Confidence intervals for predictions |

### Spatial Autocorrelation Check

```r
# Check residuals for remaining spatial autocorrelation
moran.test(gwr_model$SDF$residual,
           nb2listw(knn2nb(knearneigh(gwr_data[, c("x", "y")], k = 5))))
```

- If significant: Model may not have captured all spatial structure
- If not significant: GWR successfully accounted for spatial dependence

---

## Validation Strategy

### 1. Spatial Cross-Validation

```r
library(sperrorest)

# Define spatial folds
folds <- partition_kmeans(gwr_data[, c("x", "y")], nfold = 5)

# Train on k-1 folds, predict on held-out fold
# Ensures validation tests spatial extrapolation
```

### 2. Hold-out Validation

- Withhold recent DMSM survey (temporal validation)
- Withhold subset of GAO points (if available)
- Compare predictions to ground truth

### 3. Comparison to Global Models

| Model | Expected Performance |
|-------|---------------------|
| **OLS** | Baseline, may have spatially autocorrelated residuals |
| **Spatial Lag (SAR)** | Better if spatial dependence is in response |
| **Spatial Error (SEM)** | Better if spatial dependence is in errors |
| **Random Forest** | Non-linear, may outperform if relationships complex |
| **GWR** | Best if relationships vary spatially |

---

## Implementation

```r
library(GWmodel)
library(sf)

# 1. Prepare data
dmsm_centroids <- st_coordinates(st_centroid(dmsm_polygons))
s1_values <- extract(s1_raster, dmsm_centroids)

gwr_data <- data.frame(
  severity = dmsm_polygons$severity,
  vv = s1_values[, "VV"],
  vh = s1_values[, "VH"],
  ratio = s1_values[, "VH"] / s1_values[, "VV"],
  texture = calculate_texture(s1_raster, dmsm_centroids),
  x = dmsm_centroids[, 1],
  y = dmsm_centroids[, 2]
)

# 2. Optimize bandwidth
bw <- bw.gwr(
  severity ~ vv + vh + ratio + texture,
  data = gwr_data,
  approach = "AICc",
  kernel = "bisquare",
  adaptive = TRUE
)

# 3. Run GWR
gwr_model <- gwr.basic(
  severity ~ vv + vh + ratio + texture,
  data = gwr_data,
  bw = bw,
  kernel = "bisquare",
  adaptive = TRUE
)

# 4. Diagnostics
print(gwr_model$lm$GW.diagnostic)
# Check: R2, AdjR2, AICc

# 5. Map local coefficients
# Extract coefficient surfaces for spatial interpretation
```

---

## Interpretation of Results

### Coefficient Surfaces

Mapping local coefficients reveals spatial patterns:

| Coefficient | High Positive Values | High Negative Values |
|-------------|---------------------|----------------------|
| **VV** | VV associated with severity | VV associated with health |
| **VH** | VH associated with severity | VH associated with health |
| **Ratio** | High ratio indicates severity | Low ratio indicates severity |

### Local R² Map

- **High local R²**: Model explains severity well in these areas
- **Low local R²**: Poor prediction, may need additional covariates
- Look for patterns in low R² areas (environmental gradients, data gaps)

### Prediction Map

Apply GWR model to full S1 coverage:
```r
# Predict severity across entire study area
full_s1 <- extract(s1_raster, prediction_grid)
predictions <- predict.gwr(gwr_model, full_s1)
```

---

## Addressing Challenges

### 1. Zero-Inflation

Many areas have zero severity (no ROD).

**Solutions:**
- **Two-part model**: Model presence/absence first, then severity given presence
- **Tobit model**: Censored regression for bounded severity
- **Zero-inflated spatial model**

### 2. Edge Effects

DMSM accuracy decreases near survey boundaries.

**Solutions:**
- Mask predictions outside 1-mile flightline buffer
- Use confidence bands that increase near edges
- Exclude edge areas from validation

### 3. Multicollinearity

S1 features may be correlated (VV, VH, ratio).

**Solutions:**
- Check local VIF (Variance Inflation Factor)
- Remove or combine highly correlated features
- Use ridge regression GWR variant

---

## Related Analyses

- [[Spatial Dependency Analysis with DMSM]]: Preliminary spatial autocorrelation analysis
- [[Semivariance with GAO]]: Variogram analysis for understanding spatial structure
- [[DMSM Labeled Dataset]]: Source dataset details

## References

- Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*
- [[Study Area and Data Acquisition]]: S1 data acquisition and preprocessing
