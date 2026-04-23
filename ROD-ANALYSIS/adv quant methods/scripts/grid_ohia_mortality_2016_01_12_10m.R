#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(sf)
  library(terra)
  library(ggplot2)
})

analysis_date <- as.Date("2016-01-12")
analysis_date_label <- format(analysis_date, "%m/%d/%Y")
analysis_date_slug <- format(analysis_date, "%Y_%m_%d")

grid_resolution_m <- 10

flightline_path <- "flightlines_hawaii_2016_processed.geojson"
mortality_path <- "ohia_mortality_hawaii_2016.geojson"

output_dir <- file.path("outputs", "ohia_mortality_2016_01_12_10m")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

affected_geojson_path <- file.path(output_dir, "affected_cells_10m.geojson")
affected_csv_path <- file.path(output_dir, "affected_cells_10m.csv")
summary_csv_path <- file.path(output_dir, "grid_summary_2016_01_12.csv")
severity_raster_path <- file.path(output_dir, "severity_intensity_10m.tif")
map_png_path <- file.path(output_dir, "severity_intensity_10m_map.png")

temp_dir <- file.path(output_dir, "tmp")
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
terra::terraOptions(tempdir = temp_dir, memfrac = 0.6, progress = 1)

gdal_float_options <- c(
  "COMPRESS=DEFLATE",
  "PREDICTOR=3",
  "TILED=YES",
  "BIGTIFF=YES"
)

acres_per_m2 <- 1 / 4046.8564224

required_fields <- function(x, fields, label) {
  missing_fields <- setdiff(fields, names(x))
  if (length(missing_fields) > 0) {
    stop(
      label,
      " is missing required field(s): ",
      paste(missing_fields, collapse = ", "),
      call. = FALSE
    )
  }
}

read_valid_sf <- function(path) {
  if (!file.exists(path)) {
    stop("Missing input file: ", path, call. = FALSE)
  }

  x <- sf::st_read(path, quiet = TRUE)
  x <- sf::st_zm(x, drop = TRUE, what = "ZM")
  sf::st_make_valid(x)
}

align_down <- function(value, origin, resolution) {
  origin + floor((value - origin) / resolution) * resolution
}

align_up <- function(value, origin, resolution) {
  origin + ceiling((value - origin) / resolution) * resolution
}

make_candidate_cells <- function(feature, template, resolution, crs_obj) {
  feature_bbox <- sf::st_bbox(feature)
  template_xmin <- terra::xmin(template)
  template_ymin <- terra::ymin(template)

  local_xmin <- align_down(
    as.numeric(feature_bbox["xmin"]),
    template_xmin,
    resolution
  )
  local_xmax <- align_up(
    as.numeric(feature_bbox["xmax"]),
    template_xmin,
    resolution
  )
  local_ymin <- align_down(
    as.numeric(feature_bbox["ymin"]),
    template_ymin,
    resolution
  )
  local_ymax <- align_up(
    as.numeric(feature_bbox["ymax"]),
    template_ymin,
    resolution
  )

  if (local_xmax <= local_xmin || local_ymax <= local_ymin) {
    return(NULL)
  }

  local_bbox <- sf::st_as_sfc(
    sf::st_bbox(
      c(
        xmin = local_xmin,
        ymin = local_ymin,
        xmax = local_xmax,
        ymax = local_ymax
      ),
      crs = crs_obj
    )
  )

  local_grid <- sf::st_make_grid(
    local_bbox,
    cellsize = c(resolution, resolution),
    what = "polygons",
    square = TRUE
  )

  if (length(local_grid) == 0) {
    return(NULL)
  }

  intersects_feature <- lengths(
    sf::st_intersects(local_grid, sf::st_geometry(feature))
  ) > 0
  local_grid <- local_grid[intersects_feature]

  if (length(local_grid) == 0) {
    return(NULL)
  }

  center_xy <- sf::st_coordinates(sf::st_centroid(local_grid))
  cell_id <- terra::cellFromXY(template, center_xy[, c("X", "Y"), drop = FALSE])
  keep <- !is.na(cell_id)

  if (!any(keep)) {
    return(NULL)
  }

  sf::st_sf(
    cell_id = as.integer(cell_id[keep]),
    x = center_xy[keep, "X"],
    y = center_xy[keep, "Y"],
    geometry = local_grid[keep],
    crs = crs_obj
  )
}

build_priority_zones <- function(mortality_sf) {
  severity_factors <- sort(unique(mortality_sf$severity_factor), decreasing = TRUE)
  assigned_geom <- NULL
  zone_parts <- vector("list", length(severity_factors))

  for (i in seq_along(severity_factors)) {
    factor_value <- severity_factors[i]
    group <- mortality_sf[mortality_sf$severity_factor == factor_value, ]
    group_geom <- sf::st_make_valid(sf::st_union(sf::st_geometry(group)))

    if (is.null(assigned_geom)) {
      zone_geom <- group_geom
    } else {
      zone_geom <- suppressWarnings(sf::st_difference(group_geom, assigned_geom))
    }

    zone_geom <- suppressWarnings(
      sf::st_collection_extract(sf::st_make_valid(zone_geom), "POLYGON")
    )

    if (length(zone_geom) > 0) {
      zone_area <- as.numeric(sf::st_area(zone_geom))
      zone_geom <- zone_geom[zone_area > 0]
    }

    if (length(zone_geom) > 0) {
      severity_labels <- sort(unique(as.character(group$severity_label)))
      zone_parts[[i]] <- sf::st_sf(
        severity_factor = rep(factor_value, length(zone_geom)),
        dominant_severity = rep(paste(severity_labels, collapse = "; "), length(zone_geom)),
        geometry = zone_geom
      )
    }

    assigned_geom <- if (is.null(assigned_geom)) {
      group_geom
    } else {
      sf::st_make_valid(sf::st_union(c(assigned_geom, group_geom)))
    }
  }

  zone_parts <- zone_parts[!vapply(zone_parts, is.null, logical(1))]
  if (length(zone_parts) == 0) {
    stop("No severity zones were produced.", call. = FALSE)
  }

  do.call(rbind, zone_parts)
}

message("Reading input GeoJSON files...")
flightlines <- read_valid_sf(flightline_path)
mortality <- read_valid_sf(mortality_path)

required_fields(
  flightlines,
  c("START_DATE_STR"),
  "Flightline data"
)
required_fields(
  mortality,
  c(
    "detection_timestamp",
    "PERCENT_AFFECTED",
    "PERCENT_AFFECTED_FACTOR",
    "IMPACT_ACRES"
  ),
  "Mortality data"
)

if (is.na(sf::st_crs(flightlines))) {
  stop("Flightline data must have a projected CRS.", call. = FALSE)
}
if (sf::st_is_longlat(flightlines)) {
  stop("Flightline data must be projected in meters.", call. = FALSE)
}
if (is.na(sf::st_crs(mortality))) {
  stop("Mortality data must have a CRS.", call. = FALSE)
}
if (sf::st_crs(flightlines) != sf::st_crs(mortality)) {
  mortality <- sf::st_transform(mortality, sf::st_crs(flightlines))
}

analysis_crs <- sf::st_crs(flightlines)

flightline_day <- flightlines[flightlines$START_DATE_STR == analysis_date_label, ]
mortality_day <- mortality[mortality$detection_timestamp == analysis_date_label, ]

if (nrow(flightline_day) != 1) {
  stop(
    "Expected exactly 1 flightline for ",
    analysis_date_label,
    "; found ",
    nrow(flightline_day),
    ".",
    call. = FALSE
  )
}
if (nrow(mortality_day) != 96) {
  stop(
    "Expected exactly 96 mortality polygons for ",
    analysis_date_label,
    "; found ",
    nrow(mortality_day),
    ".",
    call. = FALSE
  )
}

flightline_day <- sf::st_make_valid(flightline_day)
mortality_day <- sf::st_make_valid(mortality_day)
mortality_day$severity_label <- as.character(mortality_day$PERCENT_AFFECTED)
mortality_day$severity_factor <- as.numeric(mortality_day$PERCENT_AFFECTED_FACTOR)
mortality_day$source_impact_acres <- as.numeric(mortality_day$IMPACT_ACRES)

if (any(is.na(mortality_day$severity_factor))) {
  stop("PERCENT_AFFECTED_FACTOR contains missing or non-numeric values.", call. = FALSE)
}

expected_severity_counts <- data.frame(
  severity_label = c(
    "Severe (30-50%)",
    "Moderate (11-29%)",
    "Light (4-10%)",
    "Very Light (1-3%)"
  ),
  severity_factor = c(0.40, 0.20, 0.07, 0.02),
  expected_count = c(8L, 27L, 37L, 24L),
  stringsAsFactors = FALSE
)

observed_severity_counts <- as.data.frame(
  table(
    severity_label = mortality_day$severity_label,
    severity_factor = mortality_day$severity_factor
  ),
  stringsAsFactors = FALSE
)
observed_severity_counts <- observed_severity_counts[
  observed_severity_counts$Freq > 0,
]
observed_severity_counts$severity_factor <- as.numeric(
  observed_severity_counts$severity_factor
)
names(observed_severity_counts)[names(observed_severity_counts) == "Freq"] <- "count"

severity_check <- merge(
  expected_severity_counts,
  observed_severity_counts,
  by = c("severity_label", "severity_factor"),
  all.x = TRUE
)
severity_check$count[is.na(severity_check$count)] <- 0L

if (any(severity_check$count != severity_check$expected_count)) {
  stop("January 12 severity counts do not match the expected validation counts.", call. = FALSE)
}

flightline_union <- sf::st_make_valid(sf::st_union(sf::st_geometry(flightline_day)))
flightline_sf <- sf::st_sf(geometry = flightline_union)

mortality_intersects_flightline <- lengths(
  sf::st_intersects(mortality_day, flightline_sf)
) > 0
if (!all(mortality_intersects_flightline)) {
  stop(
    "Only ",
    sum(mortality_intersects_flightline),
    " of ",
    nrow(mortality_day),
    " mortality polygons intersect the January 12 flightline.",
    call. = FALSE
  )
}

message("Clipping mortality polygons to the January 12 flightline...")
mortality_clipped <- suppressWarnings(
  sf::st_intersection(
    mortality_day[
      ,
      c(
        "severity_label",
        "severity_factor",
        "source_impact_acres"
      )
    ],
    flightline_sf
  )
)
mortality_clipped <- suppressWarnings(
  sf::st_collection_extract(sf::st_make_valid(mortality_clipped), "POLYGON")
)
mortality_clipped$clipped_area_m2 <- as.numeric(sf::st_area(mortality_clipped))
mortality_clipped <- mortality_clipped[
  !sf::st_is_empty(mortality_clipped) & mortality_clipped$clipped_area_m2 > 0,
]

if (nrow(mortality_clipped) == 0) {
  stop("No mortality polygon area remains after clipping to the flightline.", call. = FALSE)
}

message("Building highest-severity-priority mortality zones...")
severity_zones <- build_priority_zones(mortality_clipped)
severity_zones$total_zone_area_m2 <- as.numeric(sf::st_area(severity_zones))

flightline_bbox <- sf::st_bbox(flightline_sf)
template_xmin <- floor(as.numeric(flightline_bbox["xmin"]) / grid_resolution_m) *
  grid_resolution_m
template_xmax <- ceiling(as.numeric(flightline_bbox["xmax"]) / grid_resolution_m) *
  grid_resolution_m
template_ymin <- floor(as.numeric(flightline_bbox["ymin"]) / grid_resolution_m) *
  grid_resolution_m
template_ymax <- ceiling(as.numeric(flightline_bbox["ymax"]) / grid_resolution_m) *
  grid_resolution_m

template <- terra::rast(
  terra::ext(template_xmin, template_xmax, template_ymin, template_ymax),
  resolution = grid_resolution_m,
  crs = analysis_crs$wkt
)

message("Template cells in bounding rectangle: ", format(terra::ncell(template), big.mark = ","))

message("Creating candidate 10 m cells for mortality polygons...")
candidate_parts <- vector("list", nrow(mortality_clipped))
for (i in seq_len(nrow(mortality_clipped))) {
  candidate_parts[[i]] <- make_candidate_cells(
    mortality_clipped[i, ],
    template = template,
    resolution = grid_resolution_m,
    crs_obj = analysis_crs
  )

  if (i %% 20 == 0 || i == nrow(mortality_clipped)) {
    message("  processed ", i, " of ", nrow(mortality_clipped), " clipped polygons")
  }
}

candidate_parts <- candidate_parts[!vapply(candidate_parts, is.null, logical(1))]
if (length(candidate_parts) == 0) {
  stop("No candidate cells intersect the mortality polygons.", call. = FALSE)
}

candidate_cells <- do.call(rbind, candidate_parts)
candidate_cells <- candidate_cells[order(candidate_cells$cell_id), ]
candidate_cells <- candidate_cells[!duplicated(candidate_cells$cell_id), ]

message("Unique candidate cells: ", format(nrow(candidate_cells), big.mark = ","))

message("Clipping candidate cells to the flightline...")
survey_cells <- suppressWarnings(
  sf::st_intersection(candidate_cells, flightline_sf)
)
survey_cells <- suppressWarnings(
  sf::st_collection_extract(sf::st_make_valid(survey_cells), "POLYGON")
)
survey_cells$survey_area_m2 <- as.numeric(sf::st_area(survey_cells))
survey_cells <- survey_cells[
  !sf::st_is_empty(survey_cells) & survey_cells$survey_area_m2 > 0,
]

if (nrow(survey_cells) == 0) {
  stop("No candidate cells remain after clipping to the flightline.", call. = FALSE)
}

message("Intersecting survey cells with priority severity zones...")
cell_zone_intersections <- suppressWarnings(
  sf::st_intersection(
    survey_cells[, c("cell_id", "survey_area_m2")],
    severity_zones[, c("severity_factor", "dominant_severity")]
  )
)
cell_zone_intersections <- suppressWarnings(
  sf::st_collection_extract(sf::st_make_valid(cell_zone_intersections), "POLYGON")
)
cell_zone_intersections$piece_area_m2 <- as.numeric(
  sf::st_area(cell_zone_intersections)
)
cell_zone_intersections <- cell_zone_intersections[
  !sf::st_is_empty(cell_zone_intersections) &
    cell_zone_intersections$piece_area_m2 > 0,
]

if (nrow(cell_zone_intersections) == 0) {
  stop("No cell-zone intersections were produced.", call. = FALSE)
}

piece_df <- sf::st_drop_geometry(cell_zone_intersections)
piece_df$weighted_piece_m2 <- piece_df$piece_area_m2 * piece_df$severity_factor

cell_sums <- aggregate(
  cbind(
    mortality_area_m2 = piece_area_m2,
    weighted_damage_m2 = weighted_piece_m2
  ) ~ cell_id,
  data = piece_df,
  FUN = sum
)

cell_max_severity <- aggregate(
  severity_factor ~ cell_id,
  data = piece_df,
  FUN = max
)
names(cell_max_severity)[2] <- "max_severity_factor"

severity_area <- aggregate(
  piece_area_m2 ~ cell_id + severity_factor + dominant_severity,
  data = piece_df,
  FUN = sum
)
severity_area <- severity_area[
  order(
    severity_area$cell_id,
    -severity_area$piece_area_m2,
    -severity_area$severity_factor
  ),
]
dominant_severity <- severity_area[
  !duplicated(severity_area$cell_id),
  c("cell_id", "dominant_severity")
]

survey_cell_df <- unique(
  sf::st_drop_geometry(survey_cells)[
    ,
    c("cell_id", "x", "y", "survey_area_m2")
  ]
)

affected_df <- merge(survey_cell_df, cell_sums, by = "cell_id")
affected_df <- merge(affected_df, cell_max_severity, by = "cell_id")
affected_df <- merge(affected_df, dominant_severity, by = "cell_id")

affected_df$mortality_cover_fraction <- pmin(
  1,
  pmax(0, affected_df$mortality_area_m2 / affected_df$survey_area_m2)
)
affected_df$severity_intensity <- pmax(
  0,
  affected_df$weighted_damage_m2 / affected_df$survey_area_m2
)
affected_df$weighted_damage_acres <- affected_df$weighted_damage_m2 * acres_per_m2

affected_df <- affected_df[
  order(affected_df$cell_id),
  c(
    "cell_id",
    "x",
    "y",
    "survey_area_m2",
    "mortality_area_m2",
    "mortality_cover_fraction",
    "severity_intensity",
    "weighted_damage_m2",
    "weighted_damage_acres",
    "max_severity_factor",
    "dominant_severity"
  )
]

affected_cells <- merge(
  survey_cells[, "cell_id"],
  affected_df,
  by = "cell_id"
)
affected_cells <- affected_cells[order(affected_cells$cell_id), ]

if (any(is.na(affected_cells$severity_intensity))) {
  stop("Affected cells contain missing severity intensity values.", call. = FALSE)
}

message("Affected cells with mortality overlap: ", format(nrow(affected_cells), big.mark = ","))

message("Writing affected-cell GeoJSON and CSV...")
if (file.exists(affected_geojson_path)) {
  unlink(affected_geojson_path)
}
sf::st_write(affected_cells, affected_geojson_path, quiet = TRUE)
write.csv(sf::st_drop_geometry(affected_cells), affected_csv_path, row.names = FALSE)

message("Writing full 10 m severity GeoTIFF...")
survey_zero_path <- file.path(temp_dir, "survey_zero_10m.tif")
affected_points_path <- file.path(temp_dir, "affected_points_10m.tif")

if (file.exists(survey_zero_path)) {
  unlink(survey_zero_path)
}
if (file.exists(affected_points_path)) {
  unlink(affected_points_path)
}
if (file.exists(severity_raster_path)) {
  unlink(severity_raster_path)
}

flightline_burn <- flightline_sf
flightline_burn$survey_value <- 0
survey_zero <- terra::rasterize(
  terra::vect(flightline_burn),
  template,
  field = "survey_value",
  background = NA,
  touches = TRUE,
  filename = survey_zero_path,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S", gdal = gdal_float_options)
)

affected_points <- sf::st_as_sf(
  affected_df,
  coords = c("x", "y"),
  crs = analysis_crs,
  remove = FALSE
)
affected_raster <- terra::rasterize(
  terra::vect(affected_points),
  template,
  field = "severity_intensity",
  background = NA,
  filename = affected_points_path,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S", gdal = gdal_float_options)
)

severity_raster <- terra::cover(
  affected_raster,
  survey_zero,
  filename = severity_raster_path,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S", gdal = gdal_float_options)
)

output_raster <- terra::rast(severity_raster_path)
if (!isTRUE(all.equal(as.numeric(terra::res(output_raster)), c(grid_resolution_m, grid_resolution_m)))) {
  stop("Output raster resolution is not 10 m.", call. = FALSE)
}
if (terra::crs(output_raster) != terra::crs(template)) {
  stop("Output raster CRS does not match the template CRS.", call. = FALSE)
}

message("Rendering overview map...")
map_df <- sf::st_drop_geometry(affected_cells)
map_plot <- ggplot() +
  geom_sf(data = flightline_sf, fill = NA, color = "#303030", linewidth = 0.2) +
  geom_point(
    data = map_df,
    aes(x = x, y = y, color = severity_intensity),
    size = 0.1,
    alpha = 0.8
  ) +
  scale_color_gradientn(
    colors = c("#f7fbff", "#fee08b", "#f46d43", "#a50026"),
    name = "Severity\nintensity"
  ) +
  coord_sf(crs = analysis_crs, expand = FALSE) +
  labs(
    title = "Ohia Mortality Severity Intensity, 10 m Grid",
    subtitle = paste0(
      "Hawaii flightline ",
      analysis_date_label,
      " | affected cells only"
    ),
    x = "Easting (m)",
    y = "Northing (m)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "#e6e6e6", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(
  filename = map_png_path,
  plot = map_plot,
  width = 10,
  height = 8,
  dpi = 300
)

message("Writing summary and QA metrics...")
affected_outside_flightline_n <- sum(
  lengths(sf::st_covered_by(affected_cells, flightline_sf)) == 0
)

source_mortality_area_m2 <- as.numeric(sum(sf::st_area(mortality_day)))
clipped_mortality_area_m2 <- sum(mortality_clipped$clipped_area_m2)
priority_zone_area_m2 <- sum(severity_zones$total_zone_area_m2)
overlap_removed_m2 <- clipped_mortality_area_m2 - priority_zone_area_m2
source_impact_acres_total <- sum(mortality_day$source_impact_acres, na.rm = TRUE)
grid_weighted_damage_acres_total <- sum(affected_cells$weighted_damage_acres, na.rm = TRUE)

summary_items <- list()
add_summary <- function(metric, value) {
  summary_items[[length(summary_items) + 1]] <<- data.frame(
    metric = metric,
    value = as.character(value),
    stringsAsFactors = FALSE
  )
}

add_summary("analysis_date", as.character(analysis_date))
add_summary("analysis_date_source_label", analysis_date_label)
add_summary("flightline_features_selected", nrow(flightline_day))
add_summary("mortality_features_selected", nrow(mortality_day))
add_summary("mortality_features_intersecting_flightline", sum(mortality_intersects_flightline))
add_summary("crs", analysis_crs$input)
add_summary("grid_resolution_m", grid_resolution_m)
add_summary("template_cells_in_bounding_rectangle", terra::ncell(template))
add_summary("template_rows", terra::nrow(template))
add_summary("template_cols", terra::ncol(template))
add_summary("flightline_area_m2", as.numeric(sf::st_area(flightline_sf)))
add_summary("source_mortality_area_m2", source_mortality_area_m2)
add_summary("clipped_mortality_area_m2", clipped_mortality_area_m2)
add_summary("priority_zone_area_m2", priority_zone_area_m2)
add_summary("overlap_removed_by_priority_m2", overlap_removed_m2)
add_summary("candidate_cells", nrow(candidate_cells))
add_summary("affected_cells", nrow(affected_cells))
add_summary("affected_cells_outside_flightline", affected_outside_flightline_n)
add_summary("source_impact_acres_total", source_impact_acres_total)
add_summary("grid_weighted_damage_acres_total", grid_weighted_damage_acres_total)
add_summary(
  "grid_minus_source_impact_acres",
  grid_weighted_damage_acres_total - source_impact_acres_total
)
add_summary("severity_intensity_min", min(affected_cells$severity_intensity, na.rm = TRUE))
add_summary("severity_intensity_mean", mean(affected_cells$severity_intensity, na.rm = TRUE))
add_summary("severity_intensity_max", max(affected_cells$severity_intensity, na.rm = TRUE))
add_summary("raster_path", severity_raster_path)
add_summary("raster_resolution_x_m", terra::res(output_raster)[1])
add_summary("raster_resolution_y_m", terra::res(output_raster)[2])
add_summary("raster_ncell", terra::ncell(output_raster))
add_summary("affected_geojson_path", affected_geojson_path)
add_summary("affected_csv_path", affected_csv_path)
add_summary("map_png_path", map_png_path)

for (i in seq_len(nrow(observed_severity_counts))) {
  severity_count_metric <- paste0(
    "severity_count_",
    gsub("[^A-Za-z0-9]+", "_", observed_severity_counts$severity_label[i]),
    "_factor_",
    observed_severity_counts$severity_factor[i]
  )
  add_summary(severity_count_metric, observed_severity_counts$count[i])
}

summary_df <- do.call(rbind, summary_items)
write.csv(summary_df, summary_csv_path, row.names = FALSE)

if (affected_outside_flightline_n != 0) {
  stop("Some affected cells are not covered by the flightline.", call. = FALSE)
}

message("Done.")
message("  Raster: ", severity_raster_path)
message("  Affected GeoJSON: ", affected_geojson_path)
message("  Affected CSV: ", affected_csv_path)
message("  Summary: ", summary_csv_path)
message("  Map: ", map_png_path)
