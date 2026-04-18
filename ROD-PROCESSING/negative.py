import pandas as pd
import geopandas as gpd
import os
import rioxarray
from dateutil.relativedelta import relativedelta
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS
import numpy as np
from datetime import datetime

# NoData value - outside the 0-100 valid TCC range
NODATA_VALUE = 255

# Create output directory if it doesn't exist
output_dir = './attachments/tcc_negative'
os.makedirs(output_dir, exist_ok=True)

# Load coastline (keep original CRS for now, will reproject to match TCC)
coastline = gpd.read_file("../ROD-COLLECTION/attachments/Coastline/coastline.geojson")

# Load ohia_mortality (full dataset) from ROD-COLLECTION/attachments/DMSM/
ohia_mortality = gpd.read_file("../ROD-COLLECTION/attachments/DMSM/ohia_mortality/ohia_mortality.geojson")

# Convert CREATED_DATE from milliseconds to datetime
ohia_mortality['CREATED_DATE_DT'] = pd.to_datetime(ohia_mortality['CREATED_DATE'], unit='ms')

# Add date string column to ohia_mortality in MM/DD/YYYY format
ohia_mortality['CREATED_DATE_STR'] = ohia_mortality['CREATED_DATE_DT'].dt.strftime('%m/%d/%Y')

# Load all flightlines
flightline_dir = './attachments/flightlines'
flightline_files = [f for f in os.listdir(flightline_dir) if f.endswith('_processed.geojson')]

print(f"Found {len(flightline_files)} flightline files: {flightline_files}")

# Load all flightlines into a single GeoDataFrame
all_flightlines = []
for flightline_file in flightline_files:
    flightline_path = os.path.join(flightline_dir, flightline_file)
    gdf = gpd.read_file(flightline_path)
    all_flightlines.append(gdf)

all_flightlines = pd.concat(all_flightlines, ignore_index=True)

# Convert START_DATE to datetime and add formatted date string
all_flightlines['START_DATE_DT'] = pd.to_datetime(all_flightlines['START_DATE'], unit='ms')
all_flightlines['DATE_STR'] = all_flightlines['START_DATE_DT'].dt.strftime('%m/%d/%Y')

# Group by (Island, DATE_STR)
flightline_groups = all_flightlines.groupby(['Island', 'DATE_STR'])

print(f"Total flightlines: {len(all_flightlines)}")
print(f"Unique (Island, Date) groups: {len(flightline_groups)}")

# Cache for TCC rasters and reprojected coastline per year
# Maps year -> {'tcc': raster, 'tcc_crs': CRS, 'coastline_reproj': GeoDataFrame}
tcc_cache = {}

# Process each (Island, Date) group
for (island, date_str), group_df in flightline_groups:
    year = int(date_str.split('/')[-1])

    print(f"\nProcessing {island} on {date_str} - {len(group_df)} flightlines")

    # Get the date range for filtering ohia mortality (±2 months)
    flight_date = datetime.strptime(date_str, '%m/%d/%Y')
    start_date = flight_date - relativedelta(months=2)
    end_date = flight_date + relativedelta(months=2)

    # Filter ohia_mortality within the date range
    filtered_rod = ohia_mortality[
        (ohia_mortality['CREATED_DATE_DT'] >= start_date) &
        (ohia_mortality['CREATED_DATE_DT'] <= end_date)
    ]

    print(f"  Matching date range: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}")
    print(f"  Found {len(filtered_rod)} ohia_mortality records in this range")

    if len(filtered_rod) == 0:
        print(f"  Skipping {island} on {date_str} - no matching mortality records")
        continue

    # Get TCC raster path for the year
    tcc_path = f"./attachments/tcc/tcc_124_{year}.tif"

    if not os.path.exists(tcc_path):
        print(f"  Warning: TCC raster for year {year} not found: {tcc_path}")
        continue

    # Load TCC and reproject coastline (cached per year for performance)
    if year not in tcc_cache:
        print(f"  Loading TCC raster for year {year} into cache...")
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

    print(f"  TCC CRS: {tcc_crs}")

    # Reproject filtered ohia_mortality to match TCC CRS
    filtered_rod_reproj = filtered_rod.to_crs(tcc_crs)

    # Dissolve filtered ohia_mortality
    dissolved_rod = filtered_rod_reproj.dissolve()

    # Buffer dissolved ohia_mortality by 500m
    rod_buffer = dissolved_rod.buffer(500)

    # Create GeoDataFrame with buffered geometry
    rod_buffer_gdf = gpd.GeoDataFrame(geometry=rod_buffer, crs=tcc_crs)
    dissolved_buffer = rod_buffer_gdf.dissolve()

    # Clip coastline to buffered ohia_mortality
    negative_mask = gpd.clip(coastline_reproj, dissolved_buffer)

    print(f"  TCC bounds: {tcc.rio.bounds()}")
    print(f"  Negative mask bounds: {negative_mask.total_bounds}")

    # Dissolve all flightlines in this group
    flightlines_reproj = group_df.to_crs(tcc_crs)
    dissolved_flightlines = flightlines_reproj.dissolve()

    try:
        # Inverse clip: use all_touched=True and invert=True
        # invert=True keeps areas OUTSIDE the geometry
        clipped = tcc.rio.clip(negative_mask.geometry.values, all_touched=True, invert=True)

        # Clip by dissolved flightlines (all flightlines in this group combined)
        clipped = clipped.rio.clip(dissolved_flightlines.geometry.values, all_touched=True)

    except Exception as e:
        # If clip fails, skip this group entirely to avoid corrupted output
        print(f"  Clip error: {e} - skipping {island} on {date_str}")
        continue

    # Prepare output metadata using CLIPPED raster properties
    out_meta = {
        "driver": "GTiff",
        "height": clipped.rio.height,
        "width": clipped.rio.width,
        "count": clipped.rio.count,
        "dtype": str(clipped.dtype),
        "crs": str(clipped.rio.crs),
        "transform": clipped.rio.transform(),
        "nodata": NODATA_VALUE  # Use dedicated NoData value
    }

    # Threshold pixels (75-100% canopy cover)
    pixels = np.asarray(clipped.values)
    pixels = pixels.astype('float32')
    new_image = np.ma.masked_outside(pixels, 25, 100)

    # Save output raster with island and date format (island_YYYY-MM-DD)
    output_date = datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
    output_path = os.path.join(output_dir, f"tcc_negative_{island.lower()}_{output_date}.tif")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(new_image.filled(NODATA_VALUE))

    print(f"  Saved: {output_path}")

print("\nProcessing complete!")
