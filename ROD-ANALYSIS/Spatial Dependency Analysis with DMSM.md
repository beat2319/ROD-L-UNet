# Spatial Dependency Analysis with DMSM

## Overview

Spatial autocorrelation analysis using [[DMSM Labeled Dataset|DMSM]] polygon data to understand clustering patterns of Rapid 'Ōhiʻa Death (ROD) across the study area.

## Purpose

Quantify spatial dependency in ROD distribution using:
- **Moran's I**: For continuous/ordinal severity measures
- **Join Count Statistics**: For binary categorical data

## Data Characteristics

| Aspect | Details |
|--------|---------|
| **Data Source** | DMSM (Digital Mobile Sketch Mapping) helicopter surveys |
| **Frequency** | Bi-monthly |
| **Labels** | Binary (present/absent) and Ordinal (severity levels) |
| **Spatial Unit** | Polygons (hand-drawn, variable accuracy) |
| **Visibility Range** | 1-2 miles from flightline |

> **Caution**: *"Using DMSM labeled data has some concerns about being used to identify detections, and could be misleading in some areas"* - [[Projects/Current/J-STARS/ROD-WEB-APP/README|Brian Tucker]]

## Spatial Weights for Polygons

### Neighbor Definition Methods

| Method | Definition | When to Use |
|--------|------------|-------------|
| **Queen Contiguity** | Share any boundary point or vertex | Default for polygon adjacency |
| **Rook Contiguity** | Share only edges | When corner contact is not meaningful |
| **k-Nearest Neighbors** | k closest centroids | For irregular polygon arrangements |
| **Distance Threshold** | Centroids within set distance | When contiguity is not appropriate |

### Row Standardization

$$ w_{ij}^{std} = \frac{w_{ij}}{\sum_j w_{ij}} $$

Weights sum to 1 for each polygon, allowing interpretation as "average of neighbors."

---

## Moran's I

### Formula

$$ I = \frac{n}{\sum_i \sum_j w_{ij}} \times \frac{\sum_i \sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2} $$

Where:
- $x_i, x_j$: Values (e.g., severity) in polygons $i$ and $j$
- $\bar{x}$: Global mean
- $w_{ij}$: Spatial weight (1 if neighbors, 0 otherwise)
- $n$: Number of polygons

### When to Use

| Data Type | Recommended |
|-----------|-------------|
| **Binary** | Use Join Count instead |
| **Ordinal severity** | Moran's I (treat as 1, 2, 3...) |
| **Continuous** | Moran's I (ideal) |

### Interpretation

| Moran's I | Pattern | Meaning for ROD |
|-----------|---------|-----------------|
| $I > 0$, significant | **Clustering** | Similar severity values cluster together (hot/cold spots) |
| $I \approx 0$ | **Random** | No spatial pattern in ROD distribution |
| $I < 0$, significant | **Dispersion** | Unlike values adjacent (checkerboard pattern) |

### Moran Scatterplot

```
Severity Lag
    ↑
    │    HL     HH
    │  outlier  hot spot
    │
────┼────────────────→ Severity
    │
    │   LL     LH
    │  cold    outlier
    ↓
```

- **High-High (HH)**: Severe ROD surrounded by severe ROD (hot spot)
- **Low-Low (LL)**: Healthy areas surrounded by healthy areas (cold spot)
- **High-Low (HL)**: Severe area surrounded by healthy (possible new spread)
- **Low-High (LH)**: Healthy area surrounded by severe (possible resistance)

### Local Moran's I (LISA)

While global Moran's I gives one value for the entire map, LISA calculates I for each polygon to identify local clusters and outliers.

---

## Join Count Statistics

### Purpose

Specifically designed for **binary categorical data** - ideal for ROD presence/absence analysis.

### Join Types

| Join Type | Definition | Interpretation |
|-----------|------------|----------------|
| **BB (Black-Black)** | Two adjacent polygons both have ROD | Clustering of infected areas |
| **BW (Black-White)** | Adjacent polygons, one with/one without ROD | Interface between infected/healthy |
| **WW (White-White)** | Two adjacent polygons both without ROD | Clustering of healthy areas |

### Formula

For BB joins:
$$ J_{BB} = \frac{1}{2} \sum_i \sum_j w_{ij} x_i x_j $$

Where $x_i = 1$ if polygon $i$ has ROD, $0$ otherwise.

### Expected Values under CSR

$$ E[J_{BB}] = \frac{W}{2} \times \frac{n_B(n_B - 1)}{n(n - 1)} $$

Where:
- $W = \sum \sum w_{ij}$ (total number of neighbor pairs)
- $n_B$ = number of polygons with ROD
- $n$ = total number of polygons

### Interpretation

| Result | Pattern | Meaning |
|--------|---------|---------|
| **Observed BB > Expected BB** | Clustering | ROD areas cluster (contagious spread) |
| **Observed BB ≈ Expected BB** | Random | No spatial pattern |
| **Observed BB < Expected BB** | Dispersion | ROD areas separated from each other |

---

## Implementation

### Moran's I for Severity

```r
library(spdep)

# Create spatial weights matrix (Queen contiguity)
nb <- poly2nb(dmsm_polygons, queen = TRUE)
lw <- nb2listw(nb, style = "W")  # Row-standardized

# Calculate Moran's I
moran_result <- moran.test(
  dmsm_polygons$severity,
  lw
)

# Local Moran's I (LISA)
lisa_result <- localmoran(
  dmsm_polygons$severity,
  lw
)

# Create LISA map
dmsm_polygons$lisa_quadrant <- ifelse(
  dmsm_polygons$severity > mean(dmsm_polygons$severity) &
  lisa_result[,1] > 0, "HH",
  ifelse(
    dmsm_polygons$severity < mean(dmsm_polygons$severity) &
    lisa_result[,1] > 0, "LL",
    ifelse(
      dmsm_polygons$severity > mean(dmsm_polygons$severity) &
      lisa_result[,1] < 0, "HL", "LH"
    )
  )
)
```

### Join Count for Binary Presence

```r
library(spdep)

# Create binary variable (1 = ROD present, 0 = absent)
dmsm_polygons$rod_binary <- ifelse(dmsm_polygons$severity > 0, 1, 0)

# Calculate join counts
jc <- joincount.test(
  dmsm_polygons$rod_binary,
  nb2listw(nb, style = "B")  # Binary weights
)

# Print results
print(jc)
```

---

## Addressing DMSM Limitations

### 1. Detection Bias

Surveyors map mortality within 1-2 miles of helicopter, with varying visibility.

**Solutions:**
- Mask areas outside 1-mile flightline buffer
- Use flightline proximity as covariate
- Focus analysis on core survey areas

### 2. Human Error

*"Although aerial sketch surveys are invaluable to forest health monitoring, human error can cause high variability in data collection"* - [[DMSM Labeled Dataset|Odachi thesis]]

**Solutions:**
- Filter out small polygons (< minimum mortality area)
- Use centroids for point-based analysis
- Consider polygon buffers for robustness

### 3. Temporal Variability

Bi-monthly surveys may show temporal changes.

**Solutions:**
- Analyze each survey period separately
- Calculate spatiotemporal autocorrelation
- Track hot spot movement over time

---

## Related Analyses

- [[Semivariance with GAO]]: Variogram analysis using point data
- [[Geographically Weighted Regression (GWR)]]: Modeling spatially varying relationships
- [[DMSM Labeled Dataset]]: Source dataset details

## References

- [[Areas/University/Semester_2/GEOG_4023/02-24-2026.md]]: Spatial autocorrelation, Moran's I, and Join Count theory
- [[Projects/Current/J-STARS/ROD-WEB-APP/README]]: Expert feedback on DMSM limitations
- [[DMSM Labeled Dataset]]: DMSM methodology and characteristics
