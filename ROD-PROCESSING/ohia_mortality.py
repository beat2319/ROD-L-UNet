import pandas as pd
import geopandas as gpd
import os

# Create output directory if it doesn't exist
output_dir = './attachments/ohia_mortality'
os.makedirs(output_dir, exist_ok=True)

coastline = gpd.read_file("../ROD-COLLECTION/attachments/Coastline/coastline.geojson")
ohia_mortality = gpd.read_file("../ROD-COLLECTION/attachments/DMSM/ohia_mortality/ohia_mortality.geojson")

# Convert milliseconds to datetime objects
# We use unit='ms' because your timestamp has 13 digits
ohia_mortality['detection_timestamp'] = pd.to_datetime(ohia_mortality['CREATED_DATE'], unit='ms').dt.strftime('%m/%d/%Y')

# Optional: Remove the old column if you don't need it
ohia_mortality = ohia_mortality.drop(columns=['CREATED_DATE'])

ohia_mortality_polygon = ohia_mortality[ohia_mortality['AREA_TYPE'] == "POLYGON"]

# Reproject to EPSG:32604 (WGS 84 / UTM zone 4N)
ohia_mortality_polygon = ohia_mortality_polygon.set_crs(epsg=4326, allow_override=True)
ohia_mortality_polygon = ohia_mortality_polygon.to_crs(epsg=32604)
coastline = coastline.to_crs(epsg=32604)

# Mapping between ohia_mortality ISLAND property and coastline isle property
island_name_map = {
    'Hawaii': 'Hawaii',
    'Kauai': 'Kauai',
    'Lanai': 'Lanai',
    'Maui': 'Maui',
    'Molokai': 'Molokai',
    'Oahu': 'Oahu',
}

# orbit_124 islands (exclude Kauai, Oahu)
orbit_124_islands = ['Hawaii', 'Lanai', 'Maui', 'Molokai']

values_124 = ['Hawaii', 'kahoolawe', 'Lanai', 'Maui', 'Molokai']
orbit_124 = coastline[coastline['isle'].isin(values_124)]

ohia_mortality_124 = gpd.clip(ohia_mortality_polygon, orbit_124)

# Extract and save separate files for each island-year combination
if 'YEAR' in ohia_mortality_124.columns and 'ISLAND' in ohia_mortality_124.columns:
    years = ohia_mortality_124['YEAR'].unique()
    print(f"Found {len(years)} unique years: {sorted(years)}")

    for year in years:
        year_data = ohia_mortality_124[ohia_mortality_124['YEAR'] == year]
        islands = year_data['ISLAND'].unique()

        for island in islands:
            # Skip islands not in orbit_124
            if island not in orbit_124_islands:
                continue

            island_data = year_data[year_data['ISLAND'] == island]
            island_name_lower = island.lower()
            output_path = os.path.join(output_dir, f'ohia_mortality_{island_name_lower}_{year}.geojson')
            island_data.to_file(output_path, driver='GeoJSON')
            print(f"Saved {output_path} with {len(island_data)} features")
else:
    print("Error: 'YEAR' or 'ISLAND' column not found in the dataset")
    print(f"Available columns: {ohia_mortality_124.columns.tolist()}")